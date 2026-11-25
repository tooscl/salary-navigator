"""Train and compare routing schemes on the same employer-grouped folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import salary_routing as r

SCHEMES = ("global", "by_mode", "by_segment", "hybrid", "empirical", "manual")
FOUR = ("global", "by_mode", "by_segment", "hybrid")
BROAD = ("global", "by_mode", "hybrid", "empirical", "manual")
HIGH_SHIFT = 50_000


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_masks(d):
    for name, mask in [("all", d["mode"].notna()), ("gold", d.priority)]:
        yield name, mask
    for mode in r.MODES:
        yield mode, d["mode"].eq(mode)
    yield "remote_explicit", d["mode"].eq("remote") & d.mode_evidence.eq("work_format")
    yield "watch_shift_le_50000", d["mode"].eq("watch") & d.unit.eq("shift") & d.y.le(HIGH_SHIFT)


def full_coverage(d, scheme, names):
    for scope, mask in count_masks(d):
        if scope == "watch_shift_le_50000":
            continue
        selected = mask.to_numpy(dtype=bool)
        eligible = d.eligible.to_numpy(dtype=bool) & selected
        context = r.usable(d).to_numpy(dtype=bool) & selected
        scored = names != "abstain"
        yield dict(scope=scope, scheme=scheme, total=selected.sum(), contextual=context.sum(),
                   eligible=eligible.sum(), scored=(scored & selected).sum(),
                   scored_eligible=(scored & eligible).sum())


def quality_metrics(pred):
    def eligible_scores(d):
        result = r.gold.score(d, d[r.QUANTILES].to_numpy(dtype=float))
        selected = d.p50.notna()
        if selected.any():
            block = d.loc[selected]
            error = r.gold.losses(block.y, block[r.QUANTILES].to_numpy(dtype=float)).mean(axis=1)
            result["scaled_pinball"] = (error / block.scale.to_numpy(dtype=float)).mean()
            result["scaled_p50_mae"] = ((block.p50 - block.y).abs() / block.scale).mean()
        return result

    records, monetary = [], []
    wide = pred.pivot(index=["source", "id"], columns="scheme", values="p50")
    common = {
        "common_four": set(wide.index[wide[list(FOUR)].notna().all(axis=1)]),
        "common_broad": set(wide.index[wide[list(BROAD)].notna().all(axis=1)]),
    }
    for scheme, block in pred.groupby("scheme", sort=False):
        keys = pd.MultiIndex.from_frame(block[["source", "id"]])
        samples = [("available", block)]
        samples.extend((name, block.loc[keys.isin(ids)]) for name, ids in common.items()
                       if name != "common_broad" or scheme in BROAD)
        for sample, d in samples:
            for scope, mask in count_masks(d):
                selected = d.loc[mask]
                if selected.empty:
                    continue
                score = eligible_scores(selected)
                combined = {k: v for k, v in score.items()
                            if k in {"n", "available_n", "groups", "coverage", "scaled_pinball", "scaled_p50_mae"}}
                records.append(dict(scheme=scheme, sample=sample, scope=scope, **combined))
                for (unit, tax), pay in selected.groupby(["unit", "tax"]):
                    monetary.append(dict(scheme=scheme, sample=sample, scope=scope, unit=unit, tax=tax,
                                         **eligible_scores(pay)))
    return pd.DataFrame(records), pd.DataFrame(monetary)


def comparisons(pred):
    records = []
    for scheme in ("by_mode", "by_segment", "hybrid", "empirical", "manual"):
        candidates = pred.loc[pred.scheme.isin([scheme, "global"]) & pred.p50.notna()].copy()
        candidates["loss"] = r.gold.losses(candidates.y, candidates[r.QUANTILES].to_numpy(dtype=float)).mean(axis=1)
        candidates["scaled_loss"] = candidates.loss / candidates.scale
        for scope, mask in count_masks(candidates):
            c = candidates.loc[mask]
            wide = c.pivot(index=["source", "id", "group"], columns="scheme", values="scaled_loss")
            if not {scheme, "global"}.issubset(wide.columns):
                continue
            wide = wide.dropna()
            if wide.empty:
                continue
            diff = wide[scheme] - wide["global"]
            groups = diff.groupby(level="group").agg(["sum", "size"])
            indices = np.random.default_rng(171).integers(len(groups), size=(1500, len(groups)))
            boot = groups["sum"].to_numpy()[indices].sum(axis=1) / groups["size"].to_numpy()[indices].sum(axis=1)
            lo, hi = np.quantile(boot, [.025, .975]) if len(groups) >= 5 else (np.nan, np.nan)
            records.append(dict(scheme=scheme, reference="global", scope=scope, n=len(diff), groups=len(groups),
                                scaled_delta=diff.mean(), ci_low=lo, ci_high=hi,
                                relative_change_pct=100 * diff.mean() / wide["global"].mean()))
    return pd.DataFrame(records)


def run_fold(d, fold, policy):
    train, val, test = r.gold.split(d, fold)
    assert not set(train.group) & set(val.group)
    assert not set(train.group) & set(test.group)
    assert not set(val.group) & set(test.group)
    train, val = (p.loc[p.eligible] for p in (train, val))
    test = test.copy()
    scales = train.groupby(["unit", "tax"]).y.median()
    test["scale"] = pd.MultiIndex.from_frame(test[["unit", "tax"]]).map(scales)
    bundles = {"global": r.make_model(train, val, {}, 240), "empirical": r.make_empirical(train)}
    for mode in r.MODES:
        t, v = (df[df["mode"].eq(mode)] for df in (train, val))
        bundles[f"mode_{mode}"] = r.make_model(t, v, {"mode": mode}, 200)
    local, support = r.specialists(train, val, fold)
    bundles.update(local)
    print("fold", fold, "train/validation/test labels", len(train), len(val), test.eligible.sum(),
          "specialists", len(local), flush=True)
    preds = {name: r.predict_bundle(test, model) for name, model in bundles.items()}
    by_segment = pd.Series("abstain", index=test.index)
    for name, model in local.items():
        by_segment.loc[r.match(test, model["filters"])] = name
    _, requested = r.planned_rules(test, policy)
    choices = {
        "global": (pd.Series("global", index=test.index), []),
        "by_mode": ("mode_" + test["mode"], []),
        "by_segment": (by_segment, []),
        "hybrid": (by_segment.replace("abstain", "global"), ["global"]),
        "empirical": (pd.Series("empirical", index=test.index), []),
        "manual": (requested, policy["fallback"]),
    }
    labelled_predictions, coverage, unresolved = [], [], []
    rule_fields = sorted((r.RULE_FIELDS - set(r.CORE_COLUMNS)) & set(test.columns))
    fields = r.CORE_COLUMNS + ["mode_evidence", "mode_conflict", "group", "priority", "y", "shape", "date", "scale", *rule_fields]
    for scheme, (preference, fallback) in choices.items():
        values, names, reason = r.routable(test, preds, preference, fallback)
        coverage.extend(full_coverage(test, scheme, names))
        eligible = test.eligible.to_numpy(dtype=bool)
        part = test.loc[eligible, fields].copy()
        part[r.QUANTILES] = values[eligible]
        labelled_predictions.append(part.assign(scheme=scheme, fold=fold, approach=names[eligible],
                                                reference_or_reason=reason[eligible]))
        counts = pd.DataFrame({"mode":test["mode"].to_numpy(), "unit":test.unit.to_numpy(),
                               "tax":test.tax.to_numpy(), "eligible":eligible, "reason":reason,
                               "approach":names}).groupby(["mode", "unit", "tax", "eligible", "reason", "approach"]).size()
        unresolved.append(counts.rename("n").reset_index().assign(scheme=scheme, fold=fold))
    ids = []
    for kind, part in [("train", train), ("validation", val), ("test", test.loc[test.eligible])]:
        ids.append(part[r.CORE_COLUMNS + ["group"]].assign(fold=fold, part=kind))
    return bundles, support, labelled_predictions, coverage, unresolved, ids, test


def cell_comparison(pred):
    """Compare segment experts against other routes on the same cell and IDs."""
    rows = pred.loc[pred.profession_resolved.ne("") & pred.city_resolved.ne("")]
    paired_ids = set(pd.MultiIndex.from_frame(
        rows.loc[rows.scheme.eq("by_segment") & rows.p50.notna(), ["source", "id"]]
    ))
    rows = rows.loc[pd.MultiIndex.from_frame(rows[["source", "id"]]).isin(paired_ids)]
    records = []
    for key, block in rows.groupby(r.CELL + ["scheme"]):
        score = r.gold.score(block, block[r.QUANTILES].to_numpy(dtype=float))
        records.append(dict(zip(r.CELL + ["scheme"], key), **score))
    return pd.DataFrame(records)


def large_error_examples(result):
    rows = result.loc[result.scheme.isin(FOUR) & result.p50.notna()].copy()
    rows["error_p50"] = rows.p50 - rows.y
    rows["abs_error_p50"] = rows.error_p50.abs()
    return rows.sort_values("abs_error_p50", ascending=False).groupby(
        ["scheme", "mode", "unit", "tax"]
    ).head(10)


def saved_examples(test, out, models, policy):
    masks = [r.gold.cell(test, key) & test.eligible & test.unit.eq("shift") for key in r.gold.FIRST]
    masks.extend([
        test["mode"].eq("watch") & test.eligible & test.unit.eq("shift"),
        test["mode"].eq("remote") & test.eligible & test.mode_evidence.eq("work_format"),
        test["mode"].eq("remote") & test.eligible & test.mode_evidence.eq("run_only"),
        test["mode"].eq("ordinary") & test.unit.eq("unspecified"),
        test["mode"].eq("ordinary") & test.unit.eq("shift") & test.tax.eq("unknown"),
        test["mode"].eq("ordinary") & test.unit.eq("hour") & test.eligible,
    ])
    catalog = r.load_catalog(models)["models"]
    for mode in r.MODES:
        local = next((item for name, item in catalog.items()
                      if name.startswith("segment_") and item["scope"]["mode"] == mode
                      and (r.match(test, item["scope"]) & test.eligible).any()), None)
        if local:
            masks.append(r.match(test, local["scope"]) & test.eligible)
    columns = list(dict.fromkeys(r.CORE_COLUMNS + ["mode_evidence", "mode_conflict", *r.FEATURES, *r.gold.WORKLOAD]))
    rows = pd.concat([test.loc[mask].sort_values("id").head(1) for mask in masks], ignore_index=True)
    rows = rows.drop_duplicates(["source", "id"])
    fields = [c for c in columns if c in rows and c not in ("Рабочих дней (строго)", "Часов (строго)")]
    example = rows[fields]
    example.to_csv(out / "example_vacancies.csv", index=False)
    runtime = r.predict_vacancies(example, policy, folder=models)
    runtime.to_csv(out / "example_routes.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=r.EXPERIMENT_DIR)
    parser.add_argument("--models", type=Path, default=r.MODEL_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    raw, hash_value = r.gold.load_snapshot()
    data = r.prepare(raw)
    policy = r.default_policy()
    print("canonical", len(raw), "civilian", len(data), "labelled", data.eligible.sum(), flush=True)
    predictions, coverage, support, counts, ids = [], [], [], [], []
    for fold in range(3):
        models, segments, holdout, folds_coverage, reasons, fold_ids, test = run_fold(data, fold, policy)
        if fold == 0:
            evidence = f"../../experiments/{r.RUN}/README.md"
            catalog = {name: r.catalog_item(args.models, name, bundle, evidence) for name, bundle in models.items()}
            write_json(args.models / "catalog.json", {"version": 1, "trained_on_fold": fold, "models": catalog})
            saved_examples(test, args.output, args.models, policy)
        support.extend(segments)
        predictions.extend(holdout)
        coverage.extend(folds_coverage)
        counts.extend(reasons)
        ids.extend(fold_ids)

    out = args.output
    result = pd.concat(predictions, ignore_index=True)
    assert not result.duplicated(["source", "id", "scheme"]).any()
    assert result.groupby(["source", "id"]).scheme.nunique().eq(len(SCHEMES)).all()
    result.to_csv(out / "predictions_oof.csv", index=False)
    pd.concat(ids, ignore_index=True).to_csv(out / "split_ids.csv", index=False)
    aggregate, native = quality_metrics(result)
    aggregate.to_csv(out / "quality.csv", index=False)
    native.to_csv(out / "quality_by_unit.csv", index=False)
    totals = pd.DataFrame(coverage).groupby(["scope", "scheme"], as_index=False).sum()
    totals["route_coverage"] = totals.scored / totals.total
    totals["context_coverage"] = totals.scored / totals.contextual
    totals["eligible_coverage"] = totals.scored_eligible / totals.eligible
    totals.to_csv(out / "coverage.csv", index=False)
    summary = totals.merge(aggregate.query("sample == 'available'"), on=["scope", "scheme"], how="left")
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(counts, ignore_index=True).to_csv(out / "coverage_reasons.csv", index=False)
    pd.DataFrame(support).to_csv(out / "specialist_support.csv", index=False)
    comparisons(result).to_csv(out / "paired_vs_global.csv", index=False)
    available = data.groupby(["mode", "mode_evidence", "unit", "tax", "shape"], as_index=False).agg(
        n=("id", "size"), eligible=("eligible", "sum"), employers=("group", "nunique"))
    available.to_csv(out / "data_support.csv", index=False)
    large_error_examples(result).to_csv(out / "largest_errors.csv", index=False)
    cell_comparison(result).to_csv(out / "cell_comparison.csv", index=False)
    write_json(out / "routing_policy.json", policy)
    write_json(out / "experiment.json", {
        "name": r.RUN, "canonical_rows": len(raw), "validated_civilian_rows": len(data),
        "comparison_rows": int(data.eligible.sum()), "dataset_sha256": hash_value,
        "split": "3 employer/duplicate-group folds; 15% hash validation from remaining groups",
        "empirical_support": {"context": r.MIN_CONTEXT, "cell_or_profession": [10, 5]},
        "specialist_fit_support": r.MIN_SEGMENT, "specialist_min_validation": r.MIN_SEGMENT_VAL,
        "feature_columns": r.FEATURES, "catalog": f"ML/models/{r.RUN}/catalog.json",
    })
    print("Saved", len(result), "OOF decisions from", len(SCHEMES), "schemes", flush=True)


if __name__ == "__main__":
    main()
