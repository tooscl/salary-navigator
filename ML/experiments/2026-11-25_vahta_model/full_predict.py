"""Predict a requested published quote without inferring a pay period from the amount."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor

from calibrate import widen
from full_shift import KNOWN_LIMITS, plausible_mask
from full_train import CONTEXT, quantile_base, ridge_predict, sparse_features, text_transform, with_context
from tfidf import TEXT, safe_text
from train import HERE, make_features, predict


def cat_predict(folder, variant, features, text):
    cols = json.loads((folder / f"{variant}.columns.json").read_text(encoding="utf-8"))
    if any(c.startswith("text_") for c in cols):
        vocab, encoders = joblib.load(folder / "text_encoders.joblib")
        for sparse, (svd, names) in zip(text_transform(text, vocab), encoders, strict=True):
            features = features.join(pd.DataFrame(svd.transform(sparse), index=features.index, columns=names))
    model = CatBoostRegressor()
    model.load_model(str(folder / f"{variant}.cbm"))
    p = predict(model, features, cols)
    calibration = folder / f"{variant}.calibration.json"
    if calibration.exists():
        rule = json.loads(calibration.read_text(encoding="utf-8"))
        limits = rule.get("limits") or {rule["request_unit"]: rule["labeled_amount_range"]}
        if not features.unit.isin(limits).all():
            raise ValueError("The calibrated specialist has unsupported requested units")
        p = widen(p, rule["log_radius"])
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", type=Path, default=HERE / "full_results")
    parser.add_argument("--variant", default="card_char")
    parser.add_argument("--unit")
    parser.add_argument("--tax", choices=["net", "gross", "unknown", "conflict"])
    parser.add_argument("--quote-kind", choices=["central", "floor", "ceiling"])
    args = parser.parse_args()
    rows = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    if rows.empty:
        parser.error("Input CSV is empty")
    cols = ["График", "id", "military", *TEXT]
    if not all(c in rows for c in cols) or not rows["График"].eq("Вахта").all():
        parser.error("Use VAHTA rows in the shared vacancy schema")
    use_nlp = args.variant in {"full_text", "rules_nlp", "blend"}
    x, _ = make_features(rows, with_nlp=use_nlp)
    x["military_type"] = rows.military.str.casefold().isin({"true", "1", "yes", "да"}).map(
        {True: "yes", False: "no"}).astype(object)
    text = rows[TEXT].agg("\n".join, axis=1).map(safe_text)
    out = rows[["id"]].copy()
    if args.variant == "format":
        model, encoder, card = joblib.load(args.models / "format_classifier.joblib")
        vocab, _ = joblib.load(args.models / "text_encoders.joblib")
        matrix, _ = sparse_features(x, text_transform(text, vocab), card, encoder)
        out["predicted_ad_format"] = model.predict(matrix)
    else:
        for c in CONTEXT:
            override = getattr(args, c)
            if override is not None:
                rows[c] = override
            if c not in rows or rows[c].eq("").any():
                parser.error(f"Provide --{c.replace('_', '-')} or a column named {c}")
        x = with_context(x, rows)
        labeled = pd.read_csv(args.models / "labels.csv", keep_default_na=False, na_values={"y": [""]})
        labeled = labeled.loc[labeled.y.notna()]
        if args.variant == "shift_plausible_calibrated":
            labeled = labeled.loc[plausible_mask(labeled, {"shift": KNOWN_LIMITS["shift"]})]
        elif args.variant == "known_plausible_calibrated":
            labeled = labeled.loc[plausible_mask(labeled, KNOWN_LIMITS)]
        support = labeled.groupby(CONTEXT).size()
        for c in CONTEXT:
            out[c] = rows[c]
        out["reference_offers"] = [int(support.get(tuple(v), 0)) for v in rows[CONTEXT].to_numpy()]
        if args.variant.startswith("specialist_") and not rows.unit.eq(args.variant.removeprefix("specialist_")).all():
            parser.error("The selected specialist and requested pay unit do not match")
        if args.variant == "empirical":
            p = quantile_base(labeled, rows)
        elif args.variant in {"ridge", "blend"}:
            bundle, rcols = joblib.load(args.models / "ridge.joblib")
            p = ridge_predict(bundle, x, text, rcols, rows.unit)
            if args.variant == "blend":
                p = (p * cat_predict(args.models, "full_text", x, text)) ** .5
        else:
            p = cat_predict(args.models, args.variant, x, text)
        p[out.reference_offers.eq(0)] = float("nan")
        out[["p25", "p50", "p75"]] = p
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
