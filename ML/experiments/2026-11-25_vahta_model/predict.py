"""Predict P25/P50/P75 for one explicitly selected published-rate cohort."""
import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from train import make_features, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV in the shared vacancy schema")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="Trained .cbm file")
    args = parser.parse_args()
    cohort, variant = args.model.stem.split("__", 1)
    raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    unit, tax, quote_kind = cohort.split("_", 2)
    schema = json.loads((args.model.parent / "feature_schema.json").read_text(encoding="utf-8"))
    model = CatBoostRegressor()
    model.load_model(str(args.model))
    sidecar = args.model.with_suffix(".calibration.json")
    radius = 0.0
    calibration = {}
    if sidecar.exists():
        calibration = json.loads(sidecar.read_text(encoding="utf-8"))
        variant, radius = calibration["column_variant"], calibration["log_radius"]
    cols = schema[variant]
    features, _ = make_features(raw, with_nlp=variant == "plus_kostya_nlp")
    if calibration.get("tfidf"):
        import joblib
        from tfidf import TEXT, safe_text, transform

        encoder = joblib.load(args.model.with_suffix(".encoder.joblib"))
        cleaned = raw[TEXT].agg("\n".join, axis=1).map(safe_text)
        features = features.join(transform(cleaned, encoder))
        cols = cols + encoder[2]
    for i in model.get_cat_feature_indices():
        features[cols[i]] = features[cols[i]].astype(object)
    out = raw[["id"]].copy()
    out["unit"], out["tax"], out["quote_kind"] = unit, tax, quote_kind
    p = predict(model, features, cols)
    if radius:
        from calibrate import widen
        p = widen(p, radius)
    out[["p25", "p50", "p75"]] = p
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
