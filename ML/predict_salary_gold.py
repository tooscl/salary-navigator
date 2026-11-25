"""Apply a saved shift/net quantile bundle to vacancy features in a CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import salary_gold as g


def predict_rows(frame, item, median_table=None):
    bundle = item["model"]
    meta = item["metadata"]
    if missing := set(g.KEY) - set(frame):
        raise ValueError(f"Required input columns missing: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("No input vacancies")
    if frame[g.KEY[:2]].fillna("").apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("profession_resolved and city_resolved must be populated")
    if not (frame.unit.eq(meta["unit"]) & frame.tax.eq(meta["tax"])).all():
        raise ValueError(f"This bundle expects unit={meta['unit']} and tax={meta['tax']}")
    d = g.add_workload(frame)
    result = frame[[c for c in ["source", "id", *g.KEY] if c in frame]].copy()
    result[["p25_rub", "p50_rub", "p75_rub"]] = g.predict(d, bundle, median_table=median_table)
    if not np.isfinite(result[["p25_rub", "p50_rub", "p75_rub"]]).all().all():
        raise ValueError("No valid estimate for the requested salary context")
    if bundle["scale"] is not None:
        _, level = g.reference(d, bundle["scale"] if median_table is None else median_table)
        result["anchor_level"] = pd.Series(level, index=result.index).map({0: "city_profession", 1: "profession", 2: "unit_tax"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Trusted local .joblib file")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--median-source", type=Path, help="Audited canonical-format CSV of later observations")
    parser.add_argument("--median-from", help="Earliest observed day to include, YYYY-MM-DD (UTC)")
    parser.add_argument("--median-through", help="Latest observed day to include, YYYY-MM-DD (UTC)")
    args = parser.parse_args()
    if not (bool(args.median_source) == bool(args.median_from) == bool(args.median_through)):
        parser.error("Provide --median-source, --median-from and --median-through together")
    if args.input.resolve() == args.output.resolve() or args.model.resolve() == args.output.resolve():
        parser.error("--output must be separate from the input and model")
    item = joblib.load(args.model)
    input_rows = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    table = None
    if args.median_source:
        bundle = item["model"]
        if bundle["scale"] is None:
            parser.error("Only the relative bundle accepts refreshed medians")
        if args.output.resolve() == args.median_source.resolve():
            parser.error("--output must be separate from the median source")
        start, end = (pd.Timestamp(value, tz="UTC") for value in (args.median_from, args.median_through))
        if start > end:
            parser.error("--median-from must precede or equal --median-through")
        raw = pd.read_csv(args.median_source, dtype=str, keep_default_na=False)
        if raw.duplicated(["source", "id"]).any():
            parser.error("Duplicate vacancy IDs in the median source")
        potential = g.prepare(raw, assign_folds=False)
        dates = pd.to_datetime(potential.observed_at, errors="coerce", utc=True).dt.floor("D")
        matching = potential.unit.eq(item["metadata"]["unit"]) & potential.tax.eq(item["metadata"]["tax"])
        audited = potential.loc[potential.y.notna() & dates.between(start, end) & matching].assign(label=lambda d: d.y)
        table, updated = g.refresh_medians(bundle["scale"], audited)
        print(f"Refreshed medians for {len(updated)} cells; lower-support cells retain the fitted anchor")
    result = predict_rows(input_rows, item, table)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(result)} forecasts to {args.output}")


if __name__ == "__main__":
    main()
