"""Salary corridor for vacancies with the saved final model.

    import pandas as pd
    from ML.normalized_salary.predict import load_model, predict_corridor

    model = load_model()                      # ML/models/2026-11-18_normalized_v1_M4
    vacancies = pd.read_csv("Data/dataset/vacancies.csv", dtype=str).head(50)
    result = predict_corridor(model, vacancies)

Input: rows in the shared dataset schema (Data/dataset/vacancies.csv columns).
The salary is normalized by rules v1 first: its pay format is a model input and
the normalized amount is compared with the corridor. Vacancies that rules v1
cannot normalize (no salary, only "до Y", shift without hours, piecework) get no
prediction. Output: one row per vacancy and shift-length variant.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import ROOT, TARGET, condition_block, prepare_rows

DEFAULT_MODEL = ROOT / "ML" / "models" / "2026-11-18_normalized_v1_M4"


def load_model(model_dir: Path = DEFAULT_MODEL) -> dict:
    model_dir = Path(model_dir)
    info = json.loads((model_dir / "model_info.json").read_text(encoding="utf-8"))
    schema = json.loads((model_dir / "input_schema.json").read_text(encoding="utf-8"))
    calibration = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    if info["kind"] == "catboost":
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor()
        estimator.load_model(str(model_dir / info["file"]))
    else:
        import joblib
        estimator = joblib.load(model_dir / info["file"])
    return {"dir": model_dir, "info": info, "schema": schema, "calibration": calibration, "estimator": estimator}


def features(model: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Model inputs in the saved column order (missing categories -> '__MISSING__')."""
    schema = model["schema"]
    missing = schema["missing_category"]
    columns = {c: (rows[c].fillna(missing).astype(str) if c in rows else pd.Series(missing, index=rows.index))
               for c in schema["categorical"]}
    for c in schema["numeric"]:
        if c not in schema["condition_columns"]:
            columns[c] = pd.to_numeric(rows[c], errors="coerce")
    if schema["condition_columns"]:
        block = condition_block(rows.reset_index(drop=True)).set_index(rows.index)
        columns |= {c: block[c].astype(float) for c in schema["condition_columns"]}
    return pd.DataFrame(columns)[schema["feature_order"]].reset_index(drop=True)


def predict_corridor(model: dict, vacancies: pd.DataFrame) -> pd.DataFrame:
    rows = prepare_rows(vacancies.astype(str).where(vacancies.notna()), print_stats=False).reset_index(drop=True)
    if rows.empty:
        return rows
    X = features(model, rows)
    estimator, kind = model["estimator"], model["info"]["kind"]
    if kind == "catboost":
        from catboost import Pool
        raw = np.asarray(estimator.predict(Pool(X, cat_features=model["schema"]["categorical"])), dtype=float)
        raw = raw.reshape(len(X), 3)
    else:
        Xn = X.copy()
        cats = model["schema"]["categorical"]
        Xn[cats] = estimator["encoder"].transform(X[cats])
        raw = np.column_stack([m.predict(Xn) for m in estimator["models"]])
    shifts = np.array([model["calibration"]["shifts_log"][k] for k in ("p25", "p50", "p75")])
    corridor = np.exp(np.sort(np.sort(raw, axis=1) + shifts, axis=1))
    out = rows[["source", "id", "title", "salary_raw", "salary_unit", "salary_shape", "salary_tax", "hours_variant",
                "profession_resolved", "city_resolved", TARGET, "in_model_range"]].copy()
    out["p25"], out["p50"], out["p75"] = corridor[:, 0], corridor[:, 1], corridor[:, 2]
    actual = out[TARGET].astype(float)
    out["verdict"] = np.select([actual < out["p25"], actual > out["p75"]], ["ниже рынка", "выше рынка"], "в рынке")
    return out
