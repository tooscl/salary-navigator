"""Run the gold salary benchmark from the repository root."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn

import salary_gold as g

RUN = "2026-11-18_priority_salary_gold"
VARIANTS = {
    "chewby": dict(method="gbr", columns=g.CHEWBY),
    "cat_matched": dict(method="cat", columns=g.CHEWBY),
    "cat_core": dict(method="cat", columns=g.CORE_FEATURES),
    "cat_detail": dict(method="cat", columns=g.DETAIL),
    "ssr_text": dict(method="cat", columns=g.DETAIL, text=True),
    "cat_relative": dict(method="cat", columns=g.DETAIL, relative=True),
}
DRAW_COUNT = 2000


def current_support(data):
    rows = []
    masks = {f"{p} × {c}": g.cell(data, (p, c)) for p, c in g.PRIORITY}
    masks.update(g.subgroup_masks(data))
    for name, mask in masks.items():
        subset = data.loc[mask & data.unit.eq("shift") & data.tax.eq("net")].copy()
        stats = g.pass_metrics(subset, subset.y)
        workload = subset[subset.factor.notna()]
        norm = g.pass_metrics(workload, workload.y * workload.factor)
        rows.append({
            "scope": name, **stats,
            "to_30": g.additional_points(stats, 30), "to_50": g.additional_points(stats, 50),
            "strict_shift_points": norm["n"], "strict_employers": norm["employers"],
            "strict_q1": norm["q1"], "strict_median": norm["median"], "strict_q3": norm["q3"],
            "strict_to_30_by_volume": max(0, 30 - norm["n"]),
            "strict_to_50_by_volume": max(0, 50 - norm["n"]),
            "strict_to_50_with_coverage": g.additional_points(norm, 50),
        })
    return pd.DataFrame(rows)


def add_predictions(output, d, values, *, name, fold, branch="main", target=None, **extras):
    if not len(d):
        return
    p = d[["source", "id", "group", "profession_resolved", "city_resolved", "unit", "tax", "priority", "shape",
           "factor", "date", "y"]].copy()
    p["actual"] = p.y if target is None else np.asarray(target).reshape(-1)
    p["metric_unit"] = "month" if branch in {"normalization", "v1_gross_176h"} else p.unit
    p["metric_tax"] = "gross" if branch == "v1_gross_176h" else p.tax
    p[["p25", "p50", "p75"]] = values
    for k, v in dict(model=name, fold=fold, branch=branch, **extras).items():
        p[k] = v
    output.append(p)


def scores_by_scope(d):
    gold = d.priority
    yield "ordinary", ~gold
    yield "gold", gold
    for p, c in g.PRIORITY:
        yield f"{p} × {c}", g.cell(d, (p, c))


def eval_predictions(p):
    records = []
    for (branch, model, unit, tax), d in p.groupby(["branch", "model", "unit", "tax"], sort=False):
        for scope, mask in scores_by_scope(d):
            rows = d.loc[mask]
            if rows.empty:
                continue
            res = g.score(rows, rows[["p25", "p50", "p75"]].to_numpy(dtype=float), target="actual")
            records.append(dict(branch=branch, model=model, unit=unit, tax=tax, scope=scope,
                                metric_unit=rows.metric_unit.iloc[0], metric_tax=rows.metric_tax.iloc[0], **res))
    return pd.DataFrame(records)


def fold_score(p):
    records = []
    for fold, part in p.groupby("fold"):
        rows = eval_predictions(part)
        records.append(rows.assign(fold=fold))
    return pd.concat(records, ignore_index=True)


def paired(p, branch, alternative, reference, scope, mask):
    d = p.loc[p.branch.eq(branch) & mask & p.model.isin([reference, alternative])].copy()
    d["loss"] = g.losses(d.actual, d[["p25", "p50", "p75"]].to_numpy(dtype=float)).mean(axis=1)
    v = d.pivot(index=["source", "id", "group"], columns="model", values="loss")
    if not {reference, alternative}.issubset(v):
        return
    v = v.dropna(subset=[reference, alternative])
    if not len(v):
        return
    difference = v[alternative] - v[reference]
    groups = difference.groupby(level="group").agg(["size", "sum"])
    rng = np.random.default_rng(491)
    indices = rng.integers(len(groups), size=(DRAW_COUNT, len(groups)))
    boot = groups["sum"].to_numpy()[indices].sum(axis=1) / groups["size"].to_numpy()[indices].sum(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(branch=branch, scope=scope, alternative=alternative, reference=reference, n=len(v),
                groups=len(groups), delta=difference.mean(), ci_low=lo, ci_high=hi,
                change_pct=100 * difference.mean() / v[reference].mean())


def compare(p):
    records = []
    first = [f"{a} × {b}" for a, b in g.FIRST]
    combinations = [
        ("main", "chewby", "empirical"), ("main", "cat_matched", "chewby"),
        ("main", "cat_detail", "cat_core"), ("main", "ssr_text", "cat_detail"),
        ("main", "cat_relative", "cat_detail"), ("main", "cat_detail", "empirical"),
        ("main", "ssr_text", "empirical"), ("main", "cat_local", "cat_detail"),
        ("normalization", "strict_month_target", "native_to_month"),
        ("normalization", "strict_month_pooled", "strict_month_target"),
        ("temporal", "updated_anchor", "frozen_anchor"),
    ]
    for branch, model, ref in combinations:
        for scope, mask in scores_by_scope(p):
            if scope not in ["ordinary", "gold", *first]:
                continue
            result = paired(p, branch, model, ref, scope, mask & p.unit.eq("shift"))
            if result:
                records.append(result)
    return pd.DataFrame(records)


def compare_pay_types(p):
    records = []
    for unit in ("hour", "week"):
        for scope, mask in [("gold", p.priority), ("ordinary", ~p.priority)]:
            for alt, ref in [
                ("unit_cat", "unit_empirical"),
                ("pooled_month_with_unit", "unit_empirical"),
                ("pooled_month_with_unit", "unit_cat"),
            ]:
                result = paired(p, "pay_type", alt, ref, scope, mask & p.unit.eq(unit))
                if result:
                    if result["groups"] < 5:
                        result.update(ci_low=np.nan, ci_high=np.nan)
                    records.append(dict(unit=unit, **result))
    return pd.DataFrame(records)


def evaluate_subgroups(data, p):
    records = []
    masks = g.subgroup_masks(data)
    for subgroup, mask in masks.items():
        anchor = data.loc[mask, ["profession_resolved", "city_resolved"]].drop_duplicates()
        if len(anchor) != 1:
            continue
        role, city = anchor.iloc[0]
        subset_ids = set(data.loc[mask, "id"])
        parent = p[p.branch.eq("main") & g.cell(p, (role, city)) & p.unit.eq("shift")]
        child = parent.id.isin(subset_ids)
        for tag, block in [("selected", parent.loc[child]), ("remainder", parent.loc[~child])]:
            for model, rows in block.groupby("model"):
                values = rows[["p25", "p50", "p75"]].to_numpy(dtype=float)
                records.append(dict(subgroup=subgroup, part=tag, model=model, **g.score(rows, values)))
    return pd.DataFrame(records)


def normalization(train, val, test, out, study, fold):
    config = VARIANTS["cat_detail"]
    nt, nv, ne = (d.loc[d.factor.notna()].copy() for d in (train, val, test))
    for d in (nt, nv, ne):
        d["monthly"] = d.y * d.factor
    same = [p.loc[p.unit.eq("shift")].copy() for p in (nt, nv, ne)]
    a, b, c = same
    base = g.fit(a, b, **config)
    normal = g.fit(a.assign(label=a.monthly), b.assign(label=b.monthly), **config)
    pooled = g.fit(nt.assign(label=nt.monthly), nv.assign(label=nv.monthly), **config)
    # Native shift and normalized pairs are compared on identical held-out
    # rows in monthly rubles. The pooled branch changes training data only.
    pred = {
        "native_to_month": g.predict(c, base) * c.factor.to_numpy(dtype=float)[:, None],
        "strict_month_target": g.predict(c, normal),
        "strict_month_pooled": g.predict(c, pooled),
    }
    for name, values in pred.items():
        add_predictions(out, c, values, name=name, fold=fold, branch="normalization", target=c.monthly)
    for label, parts in [("paired_shift", same), ("all_strict", [nt, nv, ne])]:
        for tag, part in zip(("train", "validation", "test"), parts):
            for name, mask in scores_by_scope(part):
                if mask.any():
                    study.append(dict(fold=fold, stage=tag, scope=name, set=label, n=mask.sum(),
                                      groups=part.loc[mask, "group"].nunique()))


def dates_on(series, start, end=None):
    first = pd.Timestamp(start, tz="UTC")
    return series.ge(first) if end is None else series.between(first, pd.Timestamp(end, tz="UTC"))


def temporal_study(shifts, out, reports, ids):
    d = shifts.copy()
    # observed_at refers to the current salary representation in the snapshot.
    d["date"] = pd.to_datetime(d.observed_at, errors="coerce", utc=True).dt.floor("D")
    old = d[d.date.le(pd.Timestamp("2026-10-29", tz="UTC"))]
    recent = d[dates_on(d.date, "2026-10-30", "2026-11-13") & d.priority]
    future = d[dates_on(d.date, "2026-11-14") & d.priority]
    old_no_future = old[~old.group.isin(future.group)]
    recent_no_future = recent[~recent.group.isin(future.group)]
    bucket = old_no_future.group.map(lambda key: int(key[:8], 16) % 100)
    train, val = old_no_future[bucket.ge(15)], old_no_future[bucket.lt(15)]
    assert not set(train.group).intersection(future.group)
    assert not set(recent_no_future.group).intersection(future.group)
    bundle = g.fit(train, val, **VARIANTS["cat_relative"])
    table = bundle["scale"]
    # Fresh anchors must pass the support gate at their own city/profession;
    # they never replace a parent/cohort level with a differently mixed sample.
    updated_table, new_cell = g.refresh_medians(table, recent_no_future)
    covered = future[g.KEY].apply(tuple, axis=1).isin(new_cell)
    for scope, _ in scores_by_scope(d):
        if scope == "ordinary":
            continue
        masks = {
            "old_raw": old, "fit": train, "validation": val, "update_raw": recent,
            "update": recent_no_future, "future": future, "future_with_fresh_anchor": future.loc[covered],
        }
        entry = {"scope": scope}
        for name, frame in masks.items():
            filtered = dict(scores_by_scope(frame))[scope]
            entry[name] = filtered.sum()
            entry[name + "_groups"] = frame.loc[filtered, "group"].nunique()
        if scope != "gold":
            targets = future.loc[dict(scores_by_scope(future))[scope], g.KEY].drop_duplicates()
            if len(targets):
                key = tuple(targets.iloc[0])
                old_value = table[0][1].get(key)
                new_value = new_cell.get(key)
                entry.update(median_old=None if old_value is None else old_value[0],
                             median_updated=None if new_value is None else new_value[0])
        reports.append(entry)
    for name, part in [("fit", train), ("validation", val), ("median_update", recent_no_future), ("future", future)]:
        ids.append(part[["source", "id", "group", "date"]].assign(part=name))
    matched = future.loc[covered]
    for name, medians in [("frozen_anchor", table), ("updated_anchor", updated_table)]:
        add_predictions(out, matched, g.predict(matched, bundle, median_table=medians),
                        name=name, fold=-1, branch="temporal")


def pay_type_comparisons(data, out, fold):
    supported = data.loc[data.factor.notna()]
    a, b, c = (df.assign(label=df.y * df.factor) for df in g.split(supported, fold))
    # The unit remains a feature, and evaluation always returns to native rubles.
    pooled = g.fit(a, b, **VARIANTS["cat_detail"])
    add_predictions(out, c, g.predict(c, pooled) / c.factor.to_numpy(dtype=float)[:, None],
                    name="pooled_month_with_unit", fold=fold, branch="pay_type")
    for unit in ("hour", "week"):
        q, r, s = g.split(data.loc[data.unit.eq(unit)], fold)
        baseline, _ = g.reference(s, g.reference_table(q))
        add_predictions(out, s, baseline, name="unit_empirical", fold=fold, branch="pay_type")
        if len(q) >= 10 and q.group.nunique() >= 5 and len(s):
            candidate = g.fit(q, r, **VARIANTS["cat_detail"], iterations=100)
            add_predictions(out, s, g.predict(s, candidate), name="unit_cat", fold=fold, branch="pay_type")


def save_bundle(folder, name, model, sample, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    item = dict(model=model, metadata=metadata)
    path = folder / f"{name}.joblib"
    joblib.dump(item, path, compress=3)
    reload = joblib.load(path)["model"]
    estimate = g.predict(sample, model)
    assert np.isfinite(estimate).all()
    np.testing.assert_allclose(g.predict(sample, reload), estimate, atol=1e-8)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=g.ROOT / "ML/experiments" / RUN)
    parser.add_argument("--models", type=Path, default=g.ROOT / "ML/models" / RUN)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    raw, sha = g.load_snapshot()
    d = g.prepare(raw)
    d["label"] = d.y
    eligible = d[d.evaluation_fold.ge(0) & d.tax.eq("net")].copy()
    shifts = eligible[eligible.unit.eq("shift")]
    predictions, split_ids, norm_support, fit_log, permutation, saved = [], [], [], [], [], []
    temporal_report, temporal_ids = [], []
    current_support(d).to_csv(out / "support.csv", index=False)
    gold = eligible[eligible.priority]
    columns = ["source", "id", "group", "date", "profession_resolved", "city_resolved", "shape", "unit", "tax",
               "y", "factor", "evaluation_fold"]
    gold[columns].assign(**g.subgroup_masks(gold)).to_csv(out / "gold.csv", index=False)
    d.loc[d.priority].groupby(["shape", "unit", "tax"], as_index=False).agg(rows=("id", "size"), point_n=("y", "count")).to_csv(
        out / "gold_formats.csv", index=False)
    print(f"{len(raw)} canonical; {len(eligible)} model-eligible; {len(gold)} gold", flush=True)

    for fold in range(3):
        train, val, test = g.split(shifts, fold)
        a, b, c = g.split(eligible, fold)
        assert not (set(a.group) & set(b.group) or set(a.group) & set(c.group) or set(b.group) & set(c.group))
        for tag, part in zip(("train", "validation", "test"), (a, b, c)):
            split_ids.append(part[columns].assign(fold=fold, part=tag, priority=part.priority))
        lookup = g.reference_table(train)
        p, how = g.reference(test, lookup)
        add_predictions(predictions, test, p, name="empirical", fold=fold, reference_level=how)
        if fold == 0:
            empirical = dict(method="empirical", reference=lookup, scale=None, columns=[])
            saved.append(save_bundle(args.models, "empirical", empirical, test.head(8),
                                    dict(fold=fold, unit="shift", tax="net", dataset=sha, train_n=len(train))))
        for name, config in VARIANTS.items():
            print(f"fold {fold} {name}: {len(train)}/{len(val)}/{len(test)}", flush=True)
            model = g.fit(train, val, **config)
            values = g.predict(test, model)
            add_predictions(predictions, test, values, name=name, fold=fold)
            if name == "cat_detail":
                add_predictions(predictions, test, g.predict(test, model, calibrate=False), name="cat_detail_uncalibrated", fold=fold)
                test_gold = test.loc[test.priority]
                base_loss = g.losses(test_gold.y, g.predict(test_gold, model)).mean()
                rng = np.random.default_rng(89 + fold)
                for field in ["profession_resolved", "city_resolved", "График", "Занятость", "Опыт работы",
                              "Количество рабочих часов в день", "Способ оформления", "Медкнижка"]:
                    copy = test_gold.copy()
                    order = copy.groupby(["profession_resolved", "city_resolved"]).groups.values() if field not in g.KEY else [copy.index]
                    for ids in order:
                        copy.loc[ids, field] = rng.permutation(copy.loc[ids, field].to_numpy())
                    score = g.losses(copy.y, g.predict(copy, model)).mean()
                    permutation.append(dict(fold=fold, field=field, n=len(copy), delta_pinball=score-base_loss))
            fit_log.append(dict(branch="main", fold=fold, model=name, train=model["training_n"], validation=len(val),
                                test=len(test), trees=getattr(model["models"], "tree_count_", 80)))
            if fold == 0:
                if name in {"chewby", "cat_detail", "ssr_text", "cat_relative"}:
                    saved.append(save_bundle(args.models, name, model, test.head(8),
                                            dict(fold=fold, unit="shift", tax="net", dataset=sha,
                                                 train_n=model["training_n"], validation_n=len(val))))
        for key in g.FIRST:
            lt, lv, le = [part.loc[g.cell(part, key)] for part in (train, val, test)]
            if len(lt) < 12 or lt.group.nunique() < 5 or not len(le):
                continue
            model = g.fit(lt, lv, **VARIANTS["cat_detail"], iterations=120)
            add_predictions(predictions, le, g.predict(le, model), name="cat_local", fold=fold)
            fit_log.append(dict(branch="local", fold=fold, model="cat_local", scope=" × ".join(key),
                                train=len(lt), validation=len(lv), test=len(le), trees=model["models"].tree_count_))
        print(f"fold {fold} strict normalization and rare pay", flush=True)
        normalization(a, b, c, predictions, norm_support, fold)
        pay_type_comparisons(eligible, predictions, fold)
    print("temporal re-anchoring", flush=True)
    temporal_study(shifts, predictions, temporal_report, temporal_ids)
    results = pd.concat(predictions, ignore_index=True)
    assert not results.duplicated(["branch", "model", "source", "id"]).any()
    summary = eval_predictions(results)
    results.to_csv(out / "predictions.csv", index=False)
    summary.to_csv(out / "metrics.csv", index=False)
    summary.loc[summary.scope.eq("gold")].to_csv(out / "metrics_gold.csv", index=False)
    fold_score(results).to_csv(out / "metrics_folds.csv", index=False)
    compare(results).to_csv(out / "paired_comparisons.csv", index=False)
    compare_pay_types(results).to_csv(out / "pay_type_paired.csv", index=False)
    pd.DataFrame(norm_support).to_csv(out / "normalization_support.csv", index=False)
    evaluate_subgroups(d, results).to_csv(out / "subgroups.csv", index=False)
    pd.DataFrame(permutation).to_csv(out / "feature_permutation.csv", index=False)
    pd.DataFrame(fit_log).to_csv(out / "fits.csv", index=False)
    pd.DataFrame(temporal_report).to_csv(out / "median_update_support.csv", index=False)
    pd.concat(temporal_ids, ignore_index=True).to_csv(out / "median_update_ids.csv", index=False)
    split_table = pd.concat(split_ids, ignore_index=True)
    split_table.to_csv(out / "split_ids.csv", index=False)
    # Error analysis uses the same OOF rows and preserves raw native units.
    error_rows = results[(results.branch.eq("main")) & results.priority].copy()
    shape_scores = []
    for (model, shape), block in error_rows.groupby(["model", "shape"]):
        shape_scores.append(dict(model=model, salary_form=shape, **g.score(block, block[["p25", "p50", "p75"]].to_numpy())))
    pd.DataFrame(shape_scores).to_csv(out / "salary_form_errors.csv", index=False)
    error_rows["error_p50"] = error_rows.p50 - error_rows.actual
    error_rows["absolute_error_p50"] = error_rows.error_p50.abs()
    keys = ["model", "profession_resolved", "city_resolved"]
    error_rows.sort_values("absolute_error_p50", ascending=False).groupby(keys).head(3).to_csv(out / "largest_errors.csv", index=False)
    run = {
        "experiment_id": RUN, "status": "executed_local",
        "dataset": {"file_path": "Data/dataset/vacancies.csv", "file_sha256": sha, "rows": len(raw)},
        "code_git_commit": None, "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=g.ROOT, text=True).strip(),
        "uncommitted_changes": True,
        "environment": dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                            scikit_learn=sklearn.__version__, catboost=catboost.__version__, joblib=joblib.__version__),
        "split": {"rule": "3-fold GroupKFold over eligible rows; 15 percent hash validation of remaining groups",
                  "ids_file": "split_ids.csv", "gold_file": "gold.csv"},
        "preprocessing_version": "priority_gold_v1", "parameters": VARIANTS,
        "metrics": {"all": "metrics.csv", "gold": "metrics_gold.csv", "folds": "metrics_folds.csv"},
        "model_path": args.models.resolve().relative_to(g.ROOT).as_posix() if args.models.resolve().is_relative_to(g.ROOT) else args.models.as_posix(),
        "models": [path.name for path in saved], "models_fold": 0,
    }
    (out / "experiment.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} metric rows and {len(saved)} validated model bundles to {out}", flush=True)


if __name__ == "__main__":
    main()
