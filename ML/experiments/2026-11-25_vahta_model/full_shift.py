"""Sensitivity experiment for shift quotes within a fixed plausible amount range."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import fit_calibrated, widen
from full_train import CONTEXT, quantile_base, with_context
from train import HERE, LIMITS, fit, predict, score

KNOWN_LIMITS = {**LIMITS, "duty_day": LIMITS["day"]}


def plausible_mask(labels, limits):
    amounts = labels[["low", "high"]].apply(pd.to_numeric, errors="coerce")
    low, high = amounts.min(axis=1), amounts.max(axis=1)
    keep = pd.Series(False, index=labels.index)
    for unit, (minimum, maximum) in limits.items():
        keep |= labels.unit.eq(unit) & low.ge(minimum) & high.le(maximum)
    return keep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "full_results")
    parser.add_argument("--known-units", action="store_true",
                        help="Pool all explicit units and apply unit-specific plausibility ranges")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels, features, _, _ = pd.read_pickle(args.cache / "features.local.pkl")
    limits = KNOWN_LIMITS if args.known_units else {"shift": KNOWN_LIMITS["shift"]}
    prefix = "known_plausible" if args.known_units else "shift_plausible"
    keep = plausible_mask(labels, limits)
    rows = labels.loc[keep]
    x = with_context(features.loc[keep], rows)
    cols = [c for c in x if not c.startswith("nlp_")]
    predictions = {k: np.empty((len(rows), 3)) for k in
                   ("empirical", "rules_plausible", "rules_plausible_split_raw", "rules_plausible_calibrated")}
    for fold in sorted(rows.fold.unique()):
        test = rows.fold.eq(fold)
        a, b = rows.loc[~test], rows.loc[test]
        assert set(a.group).isdisjoint(b.group)
        predictions["empirical"][test] = quantile_base(a, b)
        model = fit(x.loc[a.index], cols, a.y)
        predictions["rules_plausible"][test] = predict(model, x.loc[b.index], cols)
        model, radius = fit_calibrated(x.loc[a.index], a, cols, 42 + int(fold))
        p = predict(model, x.loc[b.index], cols)
        predictions["rules_plausible_split_raw"][test] = p
        predictions["rules_plausible_calibrated"][test] = widen(p, radius)

    metrics, outputs = [], []
    for variant, p in predictions.items():
        valid = np.isfinite(p).all(axis=1)
        for kind in ["all", *rows.quote_kind.unique()]:
            m = (rows.quote_kind.eq(kind) if kind != "all" else np.ones(len(rows), dtype=bool)) & valid
            metrics.append(dict(variant=variant, quote_kind=kind,
                                **score(rows.y.to_numpy()[m], p[m], rows.group.to_numpy()[m])))
        if args.known_units:
            for unit in rows.unit.unique():
                m = rows.unit.eq(unit) & valid
                if m.sum() >= 10:
                    metrics.append(dict(variant=variant, quote_kind=f"unit:{unit}",
                                        **score(rows.y.to_numpy()[m], p[m], rows.group.to_numpy()[m])))
        out = rows[["id", "group", "fold", "unit", "quote_kind", "y"]].copy()
        out["variant"] = variant
        out[["p25", "p50", "p75"]] = p
        outputs.append(out)
    pd.concat(outputs).to_csv(args.out / f"{prefix}_oof.csv", index=False)
    summary = pd.DataFrame(metrics)
    summary.to_csv(args.out / f"{prefix}_metrics.csv", index=False)
    model, radius = fit_calibrated(x, rows, cols, 42)
    model.save_model(str(args.out / f"{prefix}_calibrated.cbm"))
    (args.out / f"{prefix}_calibrated.columns.json").write_text(
        json.dumps(cols, ensure_ascii=False), encoding="utf-8")
    (args.out / f"{prefix}_calibrated.calibration.json").write_text(
        json.dumps({"log_radius": radius, "limits": limits}),
        encoding="utf-8")
    print(summary.loc[summary.quote_kind.eq("all")].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
