"""Final model: one of the compared models trained on ALL rows, saved to ML/models/.

    python -X utf8 -m ML.normalized_salary.run --final                # M4 (CatBoost)
    python -X utf8 -m ML.normalized_salary.run --final M2_hgb         # sklearn alternative

Needs the cross-validation run of that model first (its folds.csv): the number of
trees is the mean best iteration of the folds, the calibration shifts are the mean
fold shifts. No rows are held out, so quality is the cross-validation estimate.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd

from utils.salary_normalization import NOTEBOOK_CONFIG, RULES_VERSION

from .data import CARD_COLUMNS, META_CATEGORICAL, META_NUMERIC, MAX_MONTHLY, MIN_MONTHLY, ROOT, TARGET

MODELS_DIR = ROOT / "ML" / "models"


def model_dir_for(experiment_dir: Path, model_name: str) -> Path:
    return MODELS_DIR / f"{experiment_dir.name}_{model_name.split('_')[0]}"


def train_final(spec: dict, rows: pd.DataFrame, conditions: pd.DataFrame, experiment_dir: Path, log,
                catboost_params: dict, dataset_meta: dict) -> Path:
    from .run import QUANTILES, feature_frame, summarize

    cv_dir = experiment_dir / spec["name"]
    folds = pd.read_csv(cv_dir / "folds.csv")
    shifts = [float(folds[f"shift_p{int(q * 100)}"].mean()) for q in QUANTILES]
    X, cats, nums = feature_frame(rows, conditions if spec["conditions"] else None, spec)
    y = np.log(rows[TARGET].astype(float).to_numpy())
    w = rows["row_weight"].to_numpy(dtype=float)
    out = model_dir_for(experiment_dir, spec["name"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = {}
    if spec["kind"] == "catboost":
        from catboost import CatBoostRegressor, Pool
        trees = int(round(folds["trees"].mean()))
        params = {k: v for k, v in catboost_params.items() if k not in ("od_type", "od_wait")}
        params |= spec.get("catboost", {}) | {"iterations": trees, "thread_count": max(1, (os.cpu_count() or 2) - 1)}
        params.pop("od_wait", None)
        model = CatBoostRegressor(**params)
        model.fit(Pool(X, y, cat_features=cats, weight=w / w.mean()))
        model.save_model(str(out / "model.cbm"))
        importance = pd.Series(model.get_feature_importance(), index=X.columns).sort_values(ascending=False)
        info = {"kind": "catboost", "file": "model.cbm", "trees": trees, "params": params}
    elif spec["kind"] == "hgb":
        import joblib
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import OrdinalEncoder
        Xn = X.copy()
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=200,
                                 encoded_missing_value=np.nan).fit(X[cats])
        Xn[cats] = encoder.transform(X[cats])
        mask = [c in cats for c in Xn.columns]
        models = []
        for q in QUANTILES:
            iterations = int(round(folds[f"iterations_q{int(q * 100)}"].mean()))
            models.append(HistGradientBoostingRegressor(
                loss="quantile", quantile=q, learning_rate=0.05, max_iter=iterations, max_leaf_nodes=63,
                min_samples_leaf=30, categorical_features=mask, early_stopping=False, random_state=42,
            ).fit(Xn, y, sample_weight=w))
        joblib.dump({"encoder": encoder, "models": models}, out / "model.joblib")
        importance = None
        info = {"kind": "hgb", "file": "model.joblib"}
    else:
        raise ValueError(f"No final training for {spec['kind']}")
    seconds = time.time() - t0

    cv = summarize(pd.read_csv(cv_dir / "oof.csv")).iloc[0].to_dict()
    schema = {
        "target": TARGET, "target_transform": "log", "quantiles": list(QUANTILES),
        "feature_order": list(X.columns), "categorical": cats, "numeric": nums,
        "card_columns": CARD_COLUMNS if spec["card"] else [],
        "pay_format_columns": (META_CATEGORICAL + META_NUMERIC) if spec["meta"] else [],
        "condition_columns": list(conditions.columns) if spec["conditions"] else [],
        "missing_category": "__MISSING__",
        "normalization": {"rules_version": RULES_VERSION, "config": NOTEBOOK_CONFIG.__dict__,
                          "model_range_rub_month": [MIN_MONTHLY, MAX_MONTHLY]},
    }
    (out / "input_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps(
        {"method": "per-quantile shift of predicted log salary, mean of the 5 CV fold validation shifts",
         "shifts_log": dict(zip(["p25", "p50", "p75"], shifts))}, indent=2), encoding="utf-8")
    if importance is not None:
        importance.rename("importance").to_csv(out / "feature_importance.csv", encoding="utf-8")
    files = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    (out / "model_info.json").write_text(json.dumps({
        "model": spec["name"], "note": spec["note"], "experiment": experiment_dir.name,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "training_seconds": round(seconds, 1),
        "rows": len(rows), "vacancies": int(rows["vacancy_key"].nunique()), "dataset": dataset_meta,
        "cv_metrics_unseen_employers": cv, "files_bytes": files,
    } | info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"FINAL {spec['name']}: saved to {out.relative_to(ROOT)} in {seconds:.0f} s; files {files}")
    return out
