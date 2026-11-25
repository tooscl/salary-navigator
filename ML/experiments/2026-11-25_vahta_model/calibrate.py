"""Check split conformal calibration using separate source groups in each outer fold."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from train import HERE, ROOT, fit, make_features, predict, score


def widen(p, log_radius):
    return p * np.exp([-log_radius, 0, log_radius])


def fit_calibrated(x, rows, cols, seed):
    split = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    train, calibration = next(split.split(x, rows.y, rows.group))
    assert not set(rows.iloc[train].group) & set(rows.iloc[calibration].group)
    model = fit(x.iloc[train], cols, rows.y.iloc[train])
    p = np.log(predict(model, x.iloc[calibration], cols))
    y = np.log(rows.y.to_numpy()[calibration])
    residuals = np.maximum(p[:, 0] - y, y - p[:, 2])
    k = min(len(residuals) - 1, int(np.ceil((len(residuals) + 1) * .50)) - 1)
    return model, max(0., float(np.sort(residuals)[k]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "Data/dataset/vacancies.csv")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    folder = args.results
    sets = json.loads((folder / "feature_schema.json").read_text(encoding="utf-8"))
    old = pd.read_csv(folder / "predictions_oof.csv", dtype={"id": str})
    rows = old.loc[old.variant.eq("role_median")].copy().set_index("id")
    raw = pd.read_csv(args.data, dtype=str, keep_default_na=False).set_index("id").loc[rows.index]
    x, _ = make_features(raw)
    metrics, outputs = [], []
    variants = {"shift_net_exact_or_range": "card", "shift_net_lower_bound": "plus_m4_conditions"}
    for cohort, variant in variants.items():
        print("Calibrating", cohort, flush=True)
        d = rows.loc[rows.cohort.eq(cohort)].copy()
        matrix = x.loc[d.index]
        cols = sets[variant]
        preds = np.empty((len(d), 3))
        raw_preds = np.empty_like(preds)
        for fold in sorted(d.fold.unique()):
            test = d.fold.eq(fold)
            model, radius = fit_calibrated(matrix.loc[~test], d.loc[~test], cols, 42 + int(fold))
            p = predict(model, matrix.loc[test], cols)
            raw_preds[test], preds[test] = p, widen(p, radius)
        for label, p in ((variant + "_split_raw", raw_preds), (variant + "_calibrated", preds)):
            metrics.append(dict(cohort=cohort, variant=label, **score(d.y.to_numpy(), p, d.group.to_numpy())))
            out = d[["group", "y", "origin", "fold", "cohort"]].copy()
            out["variant"] = label
            out[["p25", "p50", "p75"]] = p
            outputs.append(out)
        model, radius = fit_calibrated(matrix, d, cols, 42)
        stem = folder / f"{cohort}__{variant}_calibrated"
        model.save_model(str(stem.with_suffix(".cbm")))
        stem.with_suffix(".calibration.json").write_text(
            json.dumps({"column_variant": variant, "log_radius": radius}, indent=2), encoding="utf-8"
        )
    summary = pd.DataFrame(metrics)
    summary.to_csv(folder / "metrics_calibrated.csv", index=False)
    pd.concat(outputs).to_csv(folder / "predictions_calibrated_oof.csv")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
