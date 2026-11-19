"""Benchmark of text representations for the salary-interval model.

Every variant uses the same rows, the same grouped stratified 5-fold CV and the
same CatBoost MultiQuantile settings as `python -m ML.salary_model --final`
(employer name dropped); only the text block changes. Results go to
ML/experiments/<id>/<variant>/ and summary.csv; finished variants are skipped
on restart, so the run can be interrupted and resumed.

    python -X utf8 -m ML.text_features.run                       # all variants
    python -X utf8 -m ML.text_features.run --only A0_no_text B4_handcrafted
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ML.salary_model.calibration import oof_variant
from ML.salary_model.config import (
    CATBOOST_PARAMS,
    CATEGORICAL_COLUMNS,
    DATA_DIR,
    EXPERIMENTS_DIR,
    FINAL_DROP_COLUMNS,
    QUANTILES,
    RANDOM_SEED,
    ROOT,
    TARGET_COLUMNS,
    TFIDF_PARAMS,
)
from ML.salary_model.data import load_dataset, prepare_categories
from ML.salary_model.metrics import pooled_metrics
from ML.salary_model.splits import assign_cv_splits, cv_fold_split, duplicate_groups, group_positions

from .embeddings import e5_embeddings
from .conditions import condition_features
from .handcrafted import benefit_vocabulary, handcrafted_features
from .redaction import content_text, employer_templates, redact, target_in_text

N_FOLDS = 5
E5_MODEL = "intfloat/multilingual-e5-base"
WORD_TFIDF_PARAMS = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 3, "max_features": 100_000,
                     "sublinear_tf": True, "lowercase": True}
TEXT_FIELDS = ("title", "description", "benefits")


def V(name, blocks=(), redaction="strict", data="01_location_monthly.csv", fields=TEXT_FIELDS, dims=64,
      catboost_text=False, cv="duplicate", weights="none", note=""):
    return {"name": name, "blocks": list(blocks), "redaction": redaction, "data": data, "fields": list(fields),
            "dims": dims, "catboost_text": catboost_text, "cv": cv, "weights": weights, "note": note}


# Stage 1 (done 2026-11-17, folds grouped by duplicates as in ML.salary_model).
STAGE1 = [
    V("A0_no_text", redaction="none", note="categorical card parameters only"),
    V("A1_prod_char", ["tfidf_char"], "prod", note="current production text block (reproduction)"),
    V("A4_strict_char", ["tfidf_char"], "strict", note="production block on strictly redacted text"),
    V("B4_handcrafted", ["handcrafted"], note="keyword flags + text stats + perks, no TF-IDF"),
    V("B5_strict_char_handcrafted", ["tfidf_char", "handcrafted"]),
    V("A2_raw_char", ["tfidf_char"], "raw", note="NO redaction: leakage ceiling"),
    V("A3_homoglyph_char", ["tfidf_char"], "homoglyph"),
    V("B1_title_char", ["tfidf_char"], fields=("title",), dims=32),
    V("B2_strict_word", ["tfidf_word"]),
    V("B7_e5", ["e5"]),
    V("B8_e5_handcrafted", ["e5", "handcrafted"]),
]

# Stage 2: no employer memorization. Folds grouped by employer (every test
# employer is unseen); text reduced to job conditions (conditions.py) or to
# "content" text without employer templates, names and decoration.
STAGE2 = [
    V("C0_no_text_by_employer", redaction="none", cv="employer", note="reference for C*"),
    V("C1_prod_char_by_employer", ["tfidf_char"], "prod", cv="employer", note="production text block"),
    V("C2_conditions_by_employer", ["conditions"], cv="employer", note="closed list of job conditions only"),
    V("C3_conditions_content_char_by_employer", ["conditions", "tfidf_char"], "content", cv="employer",
      note="conditions + TF-IDF of de-templated, de-branded text"),
    V("C5_old_best_by_employer", ["tfidf_char", "handcrafted"], cv="employer", note="stage-1 best (B5) on unseen employers"),
    V("C4_conditions_weighted_by_employer", ["conditions"], cv="employer", weights="employer_sqrt",
      note="C2 with rows weighted 1/sqrt(ads of the employer): chains do not dominate"),
    V("C6_no_text_weighted_by_employer", redaction="none", cv="employer", weights="employer_sqrt",
      note="reference for C4"),
    V("B11_conditions", ["conditions"], note="conditions on duplicate-grouped folds (compare with A0/B5)"),
]

# Not run by default (--include-optional): slow or superseded.
OPTIONAL = [
    V("B6_catboost_text", [], catboost_text=True, note="CatBoost built-in text processing (~25 min per fold)"),
    V("B3a_strict_char_svd16", ["tfidf_char"], dims=16),
    V("B3b_strict_char_svd128", ["tfidf_char"], dims=128),
    V("R0_remote_no_text", redaction="none", data="02_remote_monthly.csv", cv="employer"),
    V("R2_remote_conditions", ["conditions"], data="02_remote_monthly.csv", cv="employer"),
]
VARIANTS = STAGE1 + STAGE2


def normalize_employer(name) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    name = re.sub(r"[«»\"'“”„]", " ", name.lower().replace("ё", "е"))
    name = re.sub(r"\b(ооо|оао|зао|пао|ао|ип|нко|ано|гк|тд|тк)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def assign_employer_cv_splits(df: pd.DataFrame, features: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Like splits.assign_cv_splits, but a group is the whole employer.

    All ads of one (normalized) employer share a fold and an inner
    train/validation side, so test employers are unseen. Rows without an
    employer name fall back to the duplicate group."""
    duplicate_group, stratum = duplicate_groups(df, features)
    employer = df["employer_name_observed"].map(normalize_employer)
    group = ("employer:" + employer).where(employer != "", duplicate_group)
    groups = group_positions(group, stratum, RANDOM_SEED)
    groups["fold"] = np.minimum((groups["position"] * n_folds).astype(int), n_folds - 1)
    inner = group_positions(group, stratum, RANDOM_SEED + 1).set_index("duplicate_group")["position"]
    by_group = groups.set_index("duplicate_group")
    assignment = df[["source", "vacancy_id", "data_file"]].copy()
    assignment["duplicate_group"] = group
    assignment["stratum"] = stratum
    assignment["fold"] = group.map(by_group["fold"])
    assignment["inner_position"] = group.map(inner)
    assert assignment["fold"].notna().all() and assignment.groupby("duplicate_group")["fold"].nunique().max() == 1
    assert set(assignment["fold"]) == set(range(n_folds)), "A fold is empty"
    return assignment


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


class Dataset:
    """Rows, folds and every redacted text of one experiment CSV (built once)."""

    def __init__(self, name: str, log: Log):
        self.path = DATA_DIR / name
        self.df, self.features, self.targets, self.excluded = load_dataset(self.path)
        self.assignment = assign_cv_splits(self.df, self.features, N_FOLDS)
        self._assignments = {}
        self.employer = self.df["employer_name_observed"].map(normalize_employer).reset_index(drop=True)
        self._conditions = None
        self.categorical = [c for c in CATEGORICAL_COLUMNS if c not in FINAL_DROP_COLUMNS]
        self.cats = prepare_categories(self.features, self.categorical).reset_index(drop=True)
        self.benefits = self.features["benefits"].fillna("").astype(str).reset_index(drop=True)
        self._texts, self._handcrafted, self._e5 = {}, {}, {}
        self.log = log
        log(f"{name}: {len(self.df)} rows, {self.targets.notna().sum().to_dict()} with targets")

    def assignment_for(self, cv: str) -> pd.DataFrame:
        if cv == "duplicate":
            return self.assignment
        if cv not in self._assignments:
            self._assignments[cv] = assign_employer_cv_splits(self.df, self.features, N_FOLDS)
            sizes = self._assignments[cv].groupby("fold").size().to_dict()
            self.log(f"employer-grouped folds: {sizes}")
        return self._assignments[cv]

    def fields(self, level: str) -> dict[str, pd.Series]:
        """title / description / benefits at a redaction level."""
        if level not in self._texts:
            t0 = time.time()
            if level == "content":
                strict = self.fields("strict")
                templates = employer_templates(strict["description"].tolist(), self.employer.tolist())
                self.log(f"content: {len(templates)} employer template sentences removed from descriptions")
                self._texts[level] = {
                    "title": pd.Series([content_text(t, set(), e) for t, e in zip(strict["title"], self.employer)]),
                    "description": pd.Series([content_text(d, templates, e)
                                              for d, e in zip(strict["description"], self.employer)]),
                    "benefits": self.benefits,
                }
                return self._texts[level]
            if level == "prod":  # exactly what ML.salary_model feeds to TF-IDF
                title, description = self.features["title"], self.features["description"]
            else:
                title, description = self.df["title"], self.df["description"]
            self._texts[level] = {
                "title": title.map(lambda v: redact(v, level)).fillna("").astype(str).reset_index(drop=True),
                "description": description.map(lambda v: redact(v, level)).fillna("").astype(str).reset_index(drop=True),
                "benefits": self.benefits,
            }
            self.log(f"redaction {level}: {time.time() - t0:.0f} s")
        return self._texts[level]

    def text(self, level: str, fields: list[str]) -> pd.Series:
        parts = self.fields(level)
        return pd.Series(["\n".join(p for p in row if p).strip() for row in zip(*(parts[f] for f in fields))])

    def handcrafted(self, level: str) -> pd.DataFrame:
        if level not in self._handcrafted:
            parts = self.fields(level)
            vocab = benefit_vocabulary(self.benefits.replace("", np.nan))
            self._handcrafted[level] = handcrafted_features(
                parts["title"], parts["description"], self.df["description"].fillna("").reset_index(drop=True),
                self.benefits, vocab).reset_index(drop=True)
            self.benefit_vocab = vocab
        return self._handcrafted[level]

    def conditions(self) -> pd.DataFrame:
        """Job-condition features, always from strictly redacted text."""
        if self._conditions is None:
            parts = self.fields("strict")
            self._conditions = condition_features(parts["title"], parts["description"], self.benefits).reset_index(drop=True)
        return self._conditions

    def weights(self, scheme: str) -> np.ndarray | None:
        if scheme == "none":
            return None
        if scheme == "employer_sqrt":
            counts = self.employer.map(self.employer.value_counts()).to_numpy(dtype=float)
            return 1.0 / np.sqrt(counts)
        raise ValueError(f"Unknown weights {scheme}")

    def e5(self, level: str, fields: list[str]) -> np.ndarray:
        key = (level, tuple(fields))
        if key not in self._e5:
            self._e5[key] = e5_embeddings(self.text(level, fields).tolist(), E5_MODEL,
                                          ROOT / "ML" / ".cache" / "text_features", self.log)
        return self._e5[key]

    def leakage_audit(self) -> pd.DataFrame:
        rows = []
        bounds = list(zip(self.targets["salary_lower_rub"], self.targets["salary_upper_rub"]))
        for level in ("raw", "prod", "homoglyph", "strict"):
            parts = self.fields(level)
            hits = np.array([target_in_text(t + "\n" + d, list(b))
                             for t, d, b in zip(parts["title"], parts["description"], bounds)])
            rows.append({"data": self.path.name, "redaction": level, "rows": len(hits),
                         "rows_with_target_amount_in_text": int(hits.sum()), "share": float(hits.mean()),
                         "description_chars": int(parts["description"].str.len().sum())})
        return pd.DataFrame(rows)


def dense_blocks(ds: Dataset, variant: dict, train: np.ndarray) -> tuple[np.ndarray | None, list[str], dict]:
    """Text blocks fitted on train rows of the fold only."""
    level, fields, dims = variant["redaction"], variant["fields"], variant["dims"]
    arrays, names, info = [], [], {}
    for block in variant["blocks"]:
        if block in ("tfidf_char", "tfidf_word"):
            texts = ds.text(level, fields)
            params = TFIDF_PARAMS if block == "tfidf_char" else WORD_TFIDF_PARAMS
            vectorizer = TfidfVectorizer(**params)
            vectorizer.fit(texts[train])
            matrix = vectorizer.transform(texts)
            svd = TruncatedSVD(n_components=min(dims, matrix.shape[1] - 1), random_state=RANDOM_SEED)
            svd.fit(matrix[train])
            arrays.append(svd.transform(matrix))
            names += [f"{block}_svd_{i:03d}" for i in range(arrays[-1].shape[1])]
            info[f"{block}_vocabulary"] = len(vectorizer.vocabulary_)
            info[f"{block}_explained_variance"] = float(svd.explained_variance_ratio_.sum())
        elif block == "e5":
            embeddings = ds.e5(level, fields)
            pca = PCA(n_components=dims, random_state=RANDOM_SEED).fit(embeddings[train])
            arrays.append(pca.transform(embeddings))
            names += [f"e5_pca_{i:03d}" for i in range(dims)]
            info["e5_explained_variance"] = float(pca.explained_variance_ratio_.sum())
        elif block == "conditions":
            frame = ds.conditions()
            frame = frame.loc[:, frame.loc[train].notna().any().to_numpy()]  # rare numbers may be absent in a fold
            arrays.append(frame.to_numpy(dtype=float))
            names += list(frame.columns)
        elif block in ("handcrafted", "handcrafted_nostyle"):
            frame = ds.handcrafted(level)
            if block == "handcrafted_nostyle":
                frame = frame.loc[:, ~frame.columns.str.startswith("txt_")]
            arrays.append(frame.to_numpy(dtype=float))
            names += list(frame.columns)
        else:
            raise ValueError(f"Unknown block {block}")
    return (np.hstack(arrays) if arrays else None), names, info


def fit_bound(backend, frame, y_log, split, cat_cols, text_cols, params, weights=None):
    """Fit one bound; returns (raw log P25/P50/P75 for all rows, best iteration, importances)."""
    train = (split == "train") & ~np.isnan(y_log)
    valid = (split == "validation") & ~np.isnan(y_log)
    if backend == "catboost":
        from catboost import CatBoostRegressor, Pool
        model = CatBoostRegressor(**params)
        kwargs = {"cat_features": cat_cols, "text_features": text_cols or None}
        w_train = None if weights is None else weights[train] / weights[train].mean()
        w_valid = None if weights is None else weights[valid] / weights[valid].mean()
        model.fit(Pool(frame.loc[train], y_log[train], weight=w_train, **kwargs),
                  eval_set=Pool(frame.loc[valid], y_log[valid], weight=w_valid, **kwargs), use_best_model=True)
        predicted = np.asarray(model.predict(Pool(frame, **kwargs)), dtype=float).reshape(len(frame), len(QUANTILES))
        return np.sort(predicted, axis=1), model.get_best_iteration() + 1, model.get_feature_importance()
    # Smoke-test backend (no CatBoost): one HistGradientBoosting per quantile.
    from sklearn.ensemble import HistGradientBoostingRegressor
    numeric = frame.copy()
    for column in cat_cols:
        numeric[column] = numeric[column].astype("category").cat.codes
    numeric = numeric.drop(columns=text_cols or [])
    predicted = np.column_stack([
        HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=60, random_state=RANDOM_SEED)
        .fit(numeric.loc[train], y_log[train], sample_weight=None if weights is None else weights[train])
        .predict(numeric) for q in QUANTILES])
    return np.sort(predicted, axis=1), 60, np.zeros(frame.shape[1])


def run_variant(ds: Dataset, variant: dict, out: Path, backend: str, log: Log) -> None:
    params = dict(CATBOOST_PARAMS, verbose=0, thread_count=max(1, (os.cpu_count() or 2) - 1))
    oof_parts, fold_rows, importances = [], [], []
    assignment = ds.assignment_for(variant.get("cv", "duplicate"))
    for fold in range(N_FOLDS):
        t0 = time.time()
        split = cv_fold_split(assignment, fold, N_FOLDS).to_numpy()
        train = split == "train"
        dense, names, info = dense_blocks(ds, variant, train)
        frame = ds.cats.copy()
        if dense is not None:
            frame = pd.concat([frame, pd.DataFrame(dense, columns=names)], axis=1)
        text_cols = []
        if variant["catboost_text"]:
            parts = ds.fields(variant["redaction"])
            for field in variant["fields"]:
                frame[f"text_{field}"] = parts[field].replace("", "__EMPTY__")
                text_cols.append(f"text_{field}")
        t_features = time.time() - t0
        test = split == "test"
        oof = pd.DataFrame({"row": np.flatnonzero(test), "fold": fold,
                            "employer": ds.employer.to_numpy()[test],
                            "profession": ds.features["profession"].fillna("Без профессии").to_numpy()[test],
                            "city": ds.features["city"].fillna("__MISSING__").to_numpy()[test]})
        for target in TARGET_COLUMNS:
            y = ds.targets[target].to_numpy(dtype=float)
            y_log = np.log1p(y)
            raw, best, importance = fit_bound(backend, frame, y_log, split, ds.categorical, text_cols, params,
                                              ds.weights(variant.get("weights", "none")))
            valid = (split == "validation") & ~np.isnan(y)
            shifts = np.array([np.quantile(y_log[valid] - raw[valid, i], q) for i, q in enumerate(QUANTILES)])
            calibrated = np.expm1(np.sort(raw + shifts, axis=1))
            oof[f"{target}_actual"] = y[test]
            for i, q in enumerate((25, 50, 75)):
                oof[f"{target}_p{q}"] = calibrated[test, i]
                oof[f"{target}_rawlog_p{q}"] = raw[test, i]
            fold_rows.append({"fold": fold, "target": target, "best_iteration": best,
                              "n_train": int((train & ~np.isnan(y)).sum()), "n_features": frame.shape[1]} | info)
            importances.append(pd.DataFrame({"feature": frame.columns, "importance": importance,
                                             "fold": fold, "target": target}))
        oof_parts.append(oof)
        log(f"  {variant['name']} fold {fold + 1}/{N_FOLDS}: {time.time() - t0:.0f} s "
            f"(features {t_features:.0f} s), trees " + "/".join(str(r["best_iteration"]) for r in fold_rows[-2:]))

    oof = pd.concat(oof_parts, ignore_index=True)
    oof = oof.join(ds.df[["source", "vacancy_id"]], on="row")
    oof["duplicate_group"] = assignment["duplicate_group"].to_numpy()[oof["row"]]
    metrics = []
    for how in ("raw", "fold_validation", "profession_crossfit"):
        variant_oof = oof_variant(oof, how)
        metrics.append(pooled_metrics(variant_oof, []).assign(calibration=how))
        if how == "profession_crossfit":
            crossfit = variant_oof
            by_profession = pooled_metrics(variant_oof, ["profession"]).assign(calibration=how)
    metrics = pd.concat(metrics, ignore_index=True).assign(variant=variant["name"])
    vdir = out / variant["name"]
    vdir.mkdir(parents=True, exist_ok=True)
    keep = ["source", "vacancy_id", "row", "fold", "duplicate_group", "employer", "profession", "city"]
    keep += [c for c in crossfit.columns if c.startswith("salary_") and "rawlog" not in c]
    crossfit[keep].to_csv(vdir / "oof_profession_crossfit.csv", index=False, encoding="utf-8")
    metrics.to_csv(vdir / "metrics.csv", index=False, encoding="utf-8")
    by_profession.assign(variant=variant["name"]).to_csv(vdir / "metrics_by_profession.csv", index=False, encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(vdir / "folds.csv", index=False, encoding="utf-8")
    imp = pd.concat(importances).groupby(["feature", "target"], as_index=False)["importance"].mean()
    imp.to_csv(vdir / "feature_importance.csv", index=False, encoding="utf-8")
    (vdir / "variant.json").write_text(json.dumps(variant, ensure_ascii=False, indent=2), encoding="utf-8")
    head = metrics[metrics.calibration == "profession_crossfit"]
    log(f"DONE {variant['name']}: " + "; ".join(
        f"{r.target}: pinball {r.pinball_log:.4f}, median APE {r.median_ape:.4f}, cover {r.coverage_p25_p75:.3f}"
        for r in head.itertuples()))


def write_summary(out: Path) -> None:
    frames = [pd.read_csv(p) for p in sorted(out.glob("*/metrics.csv"))]
    if frames:
        summary = pd.concat(frames, ignore_index=True)
        summary.to_csv(out / "summary.csv", index=False, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXPERIMENTS_DIR / "2026-11-17_text_features")
    parser.add_argument("--only", nargs="*", help="variant names to run (default: all)")
    parser.add_argument("--backend", choices=["catboost", "sklearn"], default="catboost",
                        help="sklearn = fast smoke test without CatBoost")
    parser.add_argument("--max-rows", type=int, default=None, help="smoke test on a subset")
    parser.add_argument("--include-optional", action="store_true", help="also run slow/superseded variants")
    args = parser.parse_args(argv)

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")
    pool = VARIANTS + (OPTIONAL if args.include_optional or args.only else [])
    variants = [v for v in pool if not args.only or v["name"] in args.only]
    log(f"start: {len(variants)} variants, backend {args.backend}, cpu {os.cpu_count()}")

    passport = out / "experiment.json"
    if passport.exists():
        meta = json.loads(passport.read_text(encoding="utf-8"))
        meta["variants"] = VARIANTS + OPTIONAL
        meta["status"] = "running"
        meta["stage2"] = ("no employer memorization: folds grouped by normalized employer (cv=employer); text reduced "
                          "to a closed list of job conditions (conditions.py) or to content text without employer "
                          "templates, names and decoration (redaction.content_text). Templates are learnt from texts "
                          "and employer names of all rows, never from salaries.")
        meta["code_git_commit"] = git_commit()
        passport.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not passport.exists():
        passport.write_text(json.dumps({
            "experiment_id": out.name,
            "question": "Which text features help the salary-interval model, and how much of the gain is salary leakage?",
            "evaluation": "grouped stratified 5-fold CV (same folds as ML.salary_model --final); "
                          "headline = pooled out-of-fold metrics with profession cross-fit calibration",
            "code_git_commit": git_commit(),
            "uncommitted_changes": True,
            "catboost": {k: v for k, v in CATBOOST_PARAMS.items() if k != "verbose"},
            "tfidf_char": TFIDF_PARAMS, "tfidf_word": WORD_TFIDF_PARAMS, "e5_model": E5_MODEL,
            "dropped_columns": FINAL_DROP_COLUMNS,
            "variants": VARIANTS + OPTIONAL,
            "datasets": {},
            "status": "running",
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    datasets: dict[str, Dataset] = {}
    for variant in variants:
        if (out / variant["name"] / "metrics.csv").exists():
            log(f"skip {variant['name']} (already done)")
            continue
        try:
            if variant["data"] not in datasets:
                ds = Dataset(variant["data"], log)
                if args.max_rows:
                    keep = ds.df.sample(min(args.max_rows, len(ds.df)), random_state=0).index.sort_values()
                    ds.df, ds.features, ds.targets = (x.loc[keep].reset_index(drop=True) for x in (ds.df, ds.features, ds.targets))
                    ds.assignment = assign_cv_splits(ds.df, ds.features, N_FOLDS)
                    ds.cats = prepare_categories(ds.features, ds.categorical).reset_index(drop=True)
                    ds.benefits = ds.features["benefits"].fillna("").astype(str).reset_index(drop=True)
                    ds.employer = ds.df["employer_name_observed"].map(normalize_employer).reset_index(drop=True)
                datasets[variant["data"]] = ds
                audit_path = out / f"leakage_audit_{Path(variant['data']).stem}.csv"
                if not audit_path.exists():
                    audit = ds.leakage_audit()
                    audit.to_csv(audit_path, index=False, encoding="utf-8")
                    log("leakage audit: " + "; ".join(f"{r.redaction} {r.share:.3f}" for r in audit.itertuples()))
                    ds.assignment.assign(**ds.df[["source", "vacancy_id"]]).to_csv(
                        out / f"folds_{Path(variant['data']).stem}.csv", index=False, encoding="utf-8")
                meta = json.loads(passport.read_text(encoding="utf-8"))
                meta["datasets"][variant["data"]] = {"path": str(ds.path.relative_to(ROOT)), "sha256": sha256(ds.path),
                                                     "rows_after_filters": len(ds.df),
                                                     "excluded": ds.excluded["reason"].value_counts().to_dict()}
                passport.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            log(f"variant {variant['name']}: blocks {variant['blocks'] or '-'}, redaction {variant['redaction']}, "
                f"fields {variant['fields']}, dims {variant['dims']}, catboost_text {variant['catboost_text']}, "
                f"cv {variant.get('cv', 'duplicate')}")
            t0 = time.time()
            run_variant(datasets[variant["data"]], variant, out, args.backend, log)
            log(f"{variant['name']} took {(time.time() - t0) / 60:.1f} min")
        except Exception as error:
            log(f"FAILED {variant['name']}: {type(error).__name__}: {error}")
            (out / f"FAILED_{variant['name']}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_summary(out)
    meta = json.loads(passport.read_text(encoding="utf-8"))
    meta["status"] = "finished"
    passport.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("all done")


if __name__ == "__main__":
    main()
