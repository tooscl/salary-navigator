"""Several models on salary normalization v1 targets (log monthly gross).

Folds are grouped by employer (all ads of an employer are test together, so it
is unseen), stratified by profession x city. Every model predicts P25 / P50 / P75
of log(month gross); one calibration step for all models (per-quantile shift
learnt on the fold's validation split), so models differ only in what they learn.

    python -X utf8 -m ML.normalized_salary.run                 # all models
    python -X utf8 -m ML.normalized_salary.run --only M0_segment_median M2_hgb
    python -X utf8 -m ML.normalized_salary.run --skip-catboost  # sklearn only
    python -X utf8 -m ML.normalized_salary.run --fast           # lighter CatBoost, separate folder
    python -X utf8 -m ML.normalized_salary.run --final          # train M4 on all rows -> ML/models/

Results: ML/experiments/2026-11-18_normalized_v1/. Finished models are skipped.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import traceback

import numpy as np
import pandas as pd

from .data import (CARD_COLUMNS, META_CATEGORICAL, META_NUMERIC, ROOT, TARGET, build, condition_block,
                   write_dataset, DATASET_PATH)

OUT = ROOT / "ML" / "experiments" / "2026-11-18_normalized_v1"
N_FOLDS = 5
VALIDATION_SHARE = 0.10
SEED = 42
QUANTILES = (0.25, 0.50, 0.75)
CATBOOST_PARAMS = {
    "loss_function": "MultiQuantile:alpha=0.25,0.5,0.75", "eval_metric": "MultiQuantile:alpha=0.25,0.5,0.75",
    "iterations": 1500, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 5.0, "random_seed": SEED,
    "random_strength": 1.0, "od_type": "Iter", "od_wait": 100, "verbose": 0, "allow_writing_files": False,
}
# --fast: shallower trees, faster learning, no categorical combinations (the slow part of
# CatBoost on ~40 categorical columns). Results go to a separate folder.
FAST_PARAMS = {"depth": 6, "learning_rate": 0.1, "iterations": 800, "od_wait": 50, "max_ctr_complexity": 1}


def M(name, kind, card=True, meta=True, conditions=False, note=""):
    return {"name": name, "kind": kind, "card": card, "meta": meta, "conditions": conditions, "note": note}


MODELS = [
    M("M0_segment_median", "segment", note="P25/P50/P75 of profession x city in train (fallback profession, all)"),
    M("M1_ridge", "ridge", conditions=True, note="linear model on one-hot card + format + conditions"),
    M("M2_hgb", "hgb", conditions=True, note="sklearn HistGradientBoosting quantile, card + format + conditions"),
    M("M3_catboost_card", "catboost", note="CatBoost MultiQuantile, card parameters + pay format"),
    M("M4_catboost_card_conditions", "catboost", conditions=True, note="M3 + job conditions from text"),
    M("M5_catboost_no_format", "catboost", meta=False, conditions=True,
      note="M4 without pay-format features (unit, shape, tax, shift length)"),
]


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ----------------------------------------------------------------- folds
def group_positions(group: pd.Series, stratum: pd.Series, weight: pd.Series, seed: int) -> pd.Series:
    """Position in [0, 1] of each group inside its stratum after a seeded shuffle."""
    g = (pd.DataFrame({"group": group, "stratum": stratum, "w": weight})
         .groupby("group", as_index=False).agg(stratum=("stratum", "first"), w=("w", "sum"))
         .sort_values("group", kind="stable"))
    g["key"] = np.random.default_rng(seed).random(len(g))
    g = g.sort_values(["stratum", "key"], kind="stable")
    total = g.groupby("stratum")["w"].transform("sum")
    g["position"] = (g.groupby("stratum")["w"].cumsum() - g["w"] / 2) / total
    return g.set_index("group")["position"]


def employer_folds(rows: pd.DataFrame) -> pd.DataFrame:
    group = ("employer:" + rows["employer_key"]).where(rows["employer_key"] != "", "vacancy:" + rows["vacancy_key"])
    stratum = rows["profession_resolved"].fillna("—") + " | " + rows["city_resolved"].fillna("—")
    outer = group_positions(group, stratum, rows["row_weight"], SEED)
    inner = group_positions(group, stratum, rows["row_weight"], SEED + 1)
    folds = pd.DataFrame({"group": group, "stratum": stratum})
    folds["fold"] = np.minimum((group.map(outer) * N_FOLDS).astype(int), N_FOLDS - 1)
    folds["inner"] = group.map(inner)
    assert folds.groupby("group")["fold"].nunique().max() == 1
    return folds


def fold_split(folds: pd.DataFrame, fold: int) -> np.ndarray:
    split = np.full(len(folds), "train", dtype=object)
    split[folds["inner"].to_numpy() < VALIDATION_SHARE / (1 - 1 / N_FOLDS)] = "validation"
    split[folds["fold"].to_numpy() == fold] = "test"
    return split


# ----------------------------------------------------------------- features
def feature_frame(rows: pd.DataFrame, conditions: pd.DataFrame | None, spec: dict):
    cats, nums = [], []
    parts = []
    if spec["card"]:
        parts.append(rows[CARD_COLUMNS].fillna("__MISSING__").astype(str))
        cats += CARD_COLUMNS
    if spec["meta"]:
        parts.append(rows[META_CATEGORICAL].fillna("__MISSING__").astype(str))
        parts.append(rows[META_NUMERIC].astype(float))
        cats += META_CATEGORICAL
        nums += META_NUMERIC
    if spec["conditions"]:
        parts.append(conditions.astype(float))
        nums += list(conditions.columns)
    return pd.concat(parts, axis=1).reset_index(drop=True), cats, nums


# ----------------------------------------------------------------- models
def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.cumsum(w) - w / 2
    return float(np.interp(q * w.sum(), cum, v))


def fit_predict(spec, X, cats, nums, y, w, split, rows, log):
    """Raw log P25/P50/P75 for every row, plus a small info dict."""
    train, valid = split == "train", split == "validation"
    info = {}
    if spec["kind"] == "segment":
        prof = rows["profession_resolved"].fillna("—").to_numpy()
        city = rows["city_resolved"].fillna("—").to_numpy()
        frame = pd.DataFrame({"prof": prof, "city": city, "y": y, "w": w})
        tr = frame[train]
        table = {}
        for keys, level in ((["prof", "city"], "cell"), (["prof"], "prof")):
            for key, part in tr.groupby(keys):
                if part["w"].sum() >= 20:
                    table[(level,) + (key if isinstance(key, tuple) else (key,))] = [
                        weighted_quantile(part["y"], part["w"], q) for q in QUANTILES]
        overall = [weighted_quantile(tr["y"], tr["w"], q) for q in QUANTILES]
        out = np.array([table.get(("cell", p, c)) or table.get(("prof", p)) or overall for p, c in zip(prof, city)])
        return out, info
    if spec["kind"] == "ridge":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cats),
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums),
        ])
        model = make_pipeline(pre, Ridge(alpha=10.0))
        model.fit(X[train], y[train], ridge__sample_weight=w[train])
        p50 = model.predict(X)
        return np.column_stack([p50, p50, p50]), info
    if spec["kind"] == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import OrdinalEncoder
        Xn = X.copy()
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=200,
                             encoded_missing_value=np.nan)
        Xn[cats] = enc.fit(X.loc[train, cats]).transform(X[cats])
        mask = [c in cats for c in Xn.columns]
        preds = []
        for q in QUANTILES:
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05, max_iter=600,
                                                  max_leaf_nodes=63, min_samples_leaf=30, categorical_features=mask,
                                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
                                                  random_state=SEED)
            model.fit(Xn[train], y[train], sample_weight=w[train])
            preds.append(model.predict(Xn))
            info[f"iterations_q{int(q * 100)}"] = int(model.n_iter_)
        return np.sort(np.column_stack(preds), axis=1), info
    if spec["kind"] == "catboost":
        from catboost import CatBoostRegressor, Pool
        params = dict(CATBOOST_PARAMS, thread_count=max(1, (os.cpu_count() or 2) - 1)) | spec.get("catboost", {})
        model = CatBoostRegressor(**params)
        model.fit(Pool(X[train], y[train], cat_features=cats, weight=w[train] / w[train].mean()),
                  eval_set=Pool(X[valid], y[valid], cat_features=cats, weight=w[valid] / w[valid].mean()),
                  use_best_model=True)
        pred = np.asarray(model.predict(Pool(X, cat_features=cats)), dtype=float).reshape(len(X), 3)
        info["trees"] = model.get_best_iteration() + 1
        info["importance"] = dict(zip(X.columns, model.get_feature_importance()))
        return np.sort(pred, axis=1), info
    raise ValueError(spec["kind"])


# ----------------------------------------------------------------- metrics
def pinball(actual, predicted, q):
    r = actual - predicted
    return np.maximum(q * r, (q - 1) * r)


def row_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    a = np.log(oof["actual"].to_numpy())
    loss = sum(pinball(a, np.log(oof[f"p{int(q * 100)}"].to_numpy()), q) for q in QUANTILES) / 3
    return oof.assign(loss=loss, ape=np.abs(oof["p50"] - oof["actual"]) / oof["actual"],
                      inside=((oof["actual"] >= oof["p25"]) & (oof["actual"] <= oof["p75"])).astype(float))


def summarize(frame: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    def one(part):
        w = part["weight"].to_numpy()
        emp = part.groupby("group").apply(lambda g: np.average(g["loss"], weights=g["weight"]), include_groups=False)
        return pd.Series({"vacancies": w.sum(), "rows": len(part), "loss": np.average(part["loss"], weights=w),
                          "loss_employer_balanced": emp.mean(),
                          "median_ape": weighted_quantile(part["ape"], w, 0.5),
                          "coverage_p25_p75": np.average(part["inside"], weights=w),
                          "median_salary": float(np.exp(weighted_quantile(np.log(part["actual"]), w, 0.5)))})
    if not by:
        return one(frame).to_frame().T
    return frame.groupby(by).apply(one, include_groups=False).reset_index()


def paired(a: pd.DataFrame, b: pd.DataFrame, rng, n=2000) -> dict:
    m = a[["row", "group", "weight", "loss"]].merge(b[["row", "loss"]], on="row", suffixes=("_a", "_b"))
    m["la"], m["lb"] = m["loss_a"] * m["weight"], m["loss_b"] * m["weight"]
    g = m.groupby("group")[["la", "lb"]].sum()
    la, lb = g["la"].to_numpy(), g["lb"].to_numpy()
    draws = [lb[i].sum() / la[i].sum() - 1 for i in (rng.integers(0, len(la), len(la)) for _ in range(n))]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"rel_change": lb.sum() / la.sum() - 1, "ci_low": lo, "ci_high": hi}


# ----------------------------------------------------------------- run
def run_model(spec, rows, conditions, folds, out, log):
    X, cats, nums = feature_frame(rows, conditions if spec["conditions"] else None, spec)
    y = np.log(rows[TARGET].astype(float).to_numpy())
    w = rows["row_weight"].to_numpy(dtype=float)
    parts, info_rows, importance = [], [], []
    for fold in range(N_FOLDS):
        t0 = time.time()
        split = fold_split(folds, fold)
        raw, info = fit_predict(spec, X, cats, nums, y, w, split, rows, log)
        valid, test = split == "validation", split == "test"
        shifts = [weighted_quantile(y[valid] - raw[valid, i], w[valid], q) for i, q in enumerate(QUANTILES)]
        pred = np.exp(np.sort(raw + np.array(shifts), axis=1))
        parts.append(pd.DataFrame({
            "row": np.flatnonzero(test), "fold": fold, "group": folds["group"].to_numpy()[test],
            "vacancy_key": rows["vacancy_key"].to_numpy()[test], "weight": w[test],
            "profession": rows["profession_resolved"].fillna("Без профессии").to_numpy()[test],
            "city": rows["city_resolved"].fillna("—").to_numpy()[test],
            "salary_unit": rows["salary_unit"].to_numpy()[test], "salary_shape": rows["salary_shape"].to_numpy()[test],
            "actual": np.exp(y[test]), "p25": pred[test, 0], "p50": pred[test, 1], "p75": pred[test, 2]}))
        imp = info.pop("importance", None)
        if imp:
            importance.append(pd.Series(imp, name=fold))
        info_rows.append({"fold": fold, "seconds": round(time.time() - t0, 1), "n_train": int((split == "train").sum()),
                          "n_validation": int(valid.sum()), "n_test": int(test.sum()),
                          "shift_p25": shifts[0], "shift_p50": shifts[1], "shift_p75": shifts[2]} | info)
        log(f"  {spec['name']} fold {fold + 1}/{N_FOLDS}: {time.time() - t0:.0f} s " + json.dumps(info))
    oof = row_metrics(pd.concat(parts, ignore_index=True))
    mdir = out / spec["name"]
    mdir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(mdir / "oof.csv", index=False, encoding="utf-8")
    pd.DataFrame(info_rows).to_csv(mdir / "folds.csv", index=False, encoding="utf-8")
    if importance:
        imp = pd.concat(importance, axis=1).mean(axis=1).sort_values(ascending=False)
        imp.rename("importance").to_csv(mdir / "feature_importance.csv", encoding="utf-8")
    (mdir / "model.json").write_text(json.dumps(spec | {"categorical": cats, "numeric_count": len(nums)},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    s = summarize(oof).iloc[0]
    log(f"DONE {spec['name']}: loss {s.loss:.4f}, employer-balanced {s.loss_employer_balanced:.4f}, "
        f"median APE {s.median_ape:.3f}, coverage {s.coverage_p25_p75:.3f}")


def report(out: Path, log) -> None:
    oofs = {p.parent.name: pd.read_csv(p) for p in sorted(out.glob("*/oof.csv"))}
    if not oofs:
        return
    overall = pd.concat([summarize(o).assign(model=n) for n, o in oofs.items()], ignore_index=True)
    overall.to_csv(out / "summary.csv", index=False, encoding="utf-8")
    pd.concat([summarize(o, ["profession"]).assign(model=n) for n, o in oofs.items()]).to_csv(
        out / "summary_by_profession.csv", index=False, encoding="utf-8")
    pd.concat([summarize(o, ["salary_unit"]).assign(model=n) for n, o in oofs.items()]).to_csv(
        out / "summary_by_unit.csv", index=False, encoding="utf-8")
    rng = np.random.default_rng(7)
    rows = []
    for ref in ("M0_segment_median", "M3_catboost_card", "M2_hgb"):
        if ref not in oofs:
            continue
        for name, o in oofs.items():
            if name != ref:
                rows.append({"reference": ref, "model": name} | paired(oofs[ref], o, rng))
    if rows:
        pd.DataFrame(rows).to_csv(out / "paired_comparison.csv", index=False, encoding="utf-8")
    log("summary:\n" + overall[["model", "loss", "loss_employer_balanced", "median_ape", "coverage_p25_p75"]]
        .round(4).to_string(index=False))


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--fast", action="store_true", help="lighter CatBoost (depth 6, no categorical combinations)")
    parser.add_argument("--final", nargs="?", const="M4_catboost_card_conditions", default=None,
                        help="train this model on all rows and save it to ML/models/ (default M4)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or (OUT.with_name(OUT.name + "_fast") if args.fast else OUT)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")
    specs = [m | ({"catboost": FAST_PARAMS} if args.fast and m["kind"] == "catboost" else {}) for m in MODELS
             if (not args.only or m["name"] in args.only) and not (args.skip_catboost and m["kind"] == "catboost")]
    log(f"start: {[m['name'] for m in specs]}, cpu {os.cpu_count()}")

    rows, meta = build(print_stats=False)
    dataset_sha = write_dataset(rows)
    rows = rows[rows["in_model_range"]].reset_index(drop=True)
    log(f"dataset: {meta['normalized_vacancies']} normalized vacancies, {meta['normalized_rows']} rows; in model range "
        f"{meta['vacancies_in_model_range']} vacancies, {meta['rows_in_model_range']} rows")
    t0 = time.time()
    conditions = condition_block(rows)
    log(f"conditions: {conditions.shape[1]} features, {time.time() - t0:.0f} s")
    if args.final:
        from .final import train_final
        spec = next(m for m in MODELS if m["name"] == args.final)
        spec = spec | ({"catboost": FAST_PARAMS} if args.fast and spec["kind"] == "catboost" else {})
        train_final(spec, rows, conditions, out, log, CATBOOST_PARAMS, meta)
        return
    folds = employer_folds(rows)
    log("employer folds (vacancies): " + str(pd.Series(rows['row_weight'].to_numpy()).groupby(folds['fold'].to_numpy())
                                         .sum().round().astype(int).to_dict()))
    passport = json.loads((out / "experiment.json").read_text(encoding="utf-8")) if (out / "experiment.json").exists() else {}
    (out / "experiment.json").write_text(json.dumps(passport | {
        "experiment_id": out.name,
        "question": "How well do different models predict salary normalized by rules v1 on unseen employers?",
        "target": f"log({TARGET}); P25/P50/P75",
        "evaluation": "5-fold CV grouped by normalized employer, stratified by profession x city; rows weighted "
                      "1/number of shift-length variants; calibration = per-quantile shift on the fold's validation",
        "dataset": meta | {"file": str(DATASET_PATH.relative_to(ROOT)), "file_sha256": dataset_sha},
        "models": specs, "catboost": CATBOOST_PARAMS | (FAST_PARAMS if args.fast else {}), "code_git_commit": git_commit(), "uncommitted_changes": True,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    for spec in specs:
        if (out / spec["name"] / "oof.csv").exists():
            log(f"skip {spec['name']} (done)")
            continue
        try:
            t0 = time.time()
            log(f"model {spec['name']}: {spec['note']}")
            run_model(spec, rows, conditions, folds, out, log)
            log(f"{spec['name']} took {(time.time() - t0) / 60:.1f} min")
        except Exception as error:
            log(f"FAILED {spec['name']}: {type(error).__name__}: {error}")
            (out / f"FAILED_{spec['name']}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report(out, log)
    log("all done")


if __name__ == "__main__":
    main()
