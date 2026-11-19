"""Paired comparison of the text-feature variants (run after ML.text_features.run).

Per vacancy and salary bound: loss = mean pinball of log P25/P50/P75 after
profession cross-fit calibration (the headline metric of ML.salary_model).
Differences between variants are bootstrapped over duplicate groups, so
near-identical ads cannot shrink the confidence interval.

    python -X utf8 -m ML.text_features.analyze
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from ML.salary_model.config import EXPERIMENTS_DIR, QUANTILES, TARGET_COLUMNS

BOOTSTRAP = 2000
SEED = 7


def row_losses(oof: pd.DataFrame) -> pd.DataFrame:
    """Long table: row, target, loss (NaN-free)."""
    parts = []
    for target in TARGET_COLUMNS:
        actual = np.log1p(oof[f"{target}_actual"].to_numpy(dtype=float))
        loss = np.zeros(len(oof))
        for q in QUANTILES:
            predicted = np.log1p(oof[f"{target}_p{int(q * 100)}"].to_numpy(dtype=float))
            residual = actual - predicted
            loss += np.maximum(q * residual, (q - 1) * residual) / len(QUANTILES)
        ape = np.abs(oof[f"{target}_actual"] - oof[f"{target}_p50"]) / oof[f"{target}_actual"]
        frame = pd.DataFrame({"row": oof["row"], "group": oof["duplicate_group"], "profession": oof["profession"],
                              "employer": oof["employer"] if "employer" in oof else np.nan,
                              "target": target, "loss": loss, "ape": ape,
                              "inside": ((oof[f"{target}_actual"] >= oof[f"{target}_p25"])
                                         & (oof[f"{target}_actual"] <= oof[f"{target}_p75"])).astype(float)})
        parts.append(frame[~np.isnan(actual)])
    return pd.concat(parts, ignore_index=True)


def load(out: Path) -> dict[str, pd.DataFrame]:
    return {p.parent.name: row_losses(pd.read_csv(p)) for p in sorted(out.glob("*/oof_profession_crossfit.csv"))}


def paired(a: pd.DataFrame, b: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Relative change of mean loss b vs a (negative = b better), with a 95 % group-bootstrap CI."""
    m = a.merge(b, on=["row", "target"], suffixes=("_a", "_b"))
    groups = m.groupby("group_a")[["loss_a", "loss_b"]].sum()
    counts = m.groupby("group_a").size().to_numpy()
    la, lb, n = groups["loss_a"].to_numpy(), groups["loss_b"].to_numpy(), counts
    point = lb.sum() / la.sum() - 1
    draws = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(la), len(la))
        draws.append(lb[idx].sum() / la[idx].sum() - 1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"n_rows": int(n.sum()), "loss_a": la.sum() / n.sum(), "loss_b": lb.sum() / n.sum(),
            "rel_change": point, "ci_low": low, "ci_high": high, "p_worse_or_equal": float(np.mean(np.array(draws) >= 0))}


def family(name: str) -> str:
    """Variants are compared only on the same rows and folds."""
    return name[0] if name[0] in "RC" else "AB"


def compare(losses: dict[str, pd.DataFrame], reference: str, targets=("both",) + tuple(TARGET_COLUMNS)) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for name, frame in losses.items():
        if name == reference or family(name) != family(reference):
            continue
        for target in targets:
            sel = (lambda f: f) if target == "both" else (lambda f, t=target: f[f["target"] == t])
            rows.append({"reference": reference, "variant": name, "target": target}
                        | paired(sel(losses[reference]), sel(frame), rng))
    return pd.DataFrame(rows)


def summary_table(losses: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per variant and bound: mean loss, employer-balanced loss (every employer
    weighs the same, so big chains do not dominate), median APE, P25-P75 coverage."""
    rows = []
    for name, frame in losses.items():
        for target, part in frame.groupby("target"):
            balanced = np.nan
            if part["employer"].notna().any():
                balanced = part.groupby("employer")["loss"].mean().mean()
            rows.append({"variant": name, "target": target, "n": len(part), "loss": part["loss"].mean(),
                         "loss_employer_balanced": balanced, "median_ape": part["ape"].median(),
                         "coverage_p25_p75": part["inside"].mean()})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXPERIMENTS_DIR / "2026-11-17_text_features")
    args = parser.parse_args(argv)
    losses = load(args.out)
    summary_table(losses).to_csv(args.out / "variant_metrics.csv", index=False, encoding="utf-8")
    references = ("A0_no_text", "A1_prod_char", "C0_no_text_by_employer", "C1_prod_char_by_employer",
                  "C6_no_text_weighted_by_employer", "R0_remote_no_text")
    tables = [compare(losses, ref) for ref in references if ref in losses]
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(args.out / "paired_comparison.csv", index=False, encoding="utf-8")
    view = result[result["target"] == "both"].copy()
    for c in ("rel_change", "ci_low", "ci_high"):
        view[c] = (100 * view[c]).round(2)
    print(view[["reference", "variant", "loss_b", "rel_change", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
