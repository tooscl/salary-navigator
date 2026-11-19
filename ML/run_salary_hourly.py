"""Paired study of direct hourly targets versus shift forecasts divided by hours."""
from __future__ import annotations

import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

import salary_gold as g
import salary_routing as r
from run_salary_gold import VARIANTS

RUN = "2026-11-18_salary_hourly"
OUT = g.ROOT / "ML/experiments" / RUN
MODELS = g.ROOT / "ML/models" / RUN
ROUTING = ("global", "by_mode", "by_segment", "hybrid", "empirical")
FOUR = ROUTING[:4]
POLICIES = {"exact": {"exact"}, "single_range": {"exact", "closed_range"}}
CARDS = ["кассир", "продавец", "комплектовщик"]
HOURS = "Количество рабочих часов в день"
NUMBER = r"\d+(?:[.,]\d+)?"
HOUR_RE = re.compile(rf"({NUMBER})\s*(?:час(?:а|ов)?|ч\.?)?")
RANGE_RE = re.compile(rf"(?:от\s+)?({NUMBER})\s*(?:[-–—]|до)\s*({NUMBER})\s*(?:час(?:а|ов)?|ч\.?)?")


def parse_hours(value):
    if pd.isna(value):
        return "missing", np.nan
    text = re.sub(r"\s+", " ", str(value).casefold()).strip()
    if not text:
        return "missing", np.nan
    match = HOUR_RE.fullmatch(text)
    if match:
        hours = float(match[1].replace(",", "."))
        return ("exact", hours) if 1 <= hours <= 24 else ("invalid", np.nan)
    match = RANGE_RE.fullmatch(text)
    if match:
        low, high = (float(x.replace(",", ".")) for x in match.groups())
        if 1 <= low <= high <= 24:
            return ("exact" if low == high else "closed_range", (low + high) / 2)
        return "invalid", np.nan
    if re.search(r"(?:больше|меньше|(?:^|[,; ])до |(?:^|[,; ])от )", text):
        return "open_or_mixed", np.nan
    return ("multiple" if re.search(r"[,;/]", text) else "unrecognized"), np.nan


def prepare(raw):
    rows = r.prepare(raw)
    hours = pd.DataFrame(rows[HOURS].map(parse_hours).tolist(), index=rows.index, columns=["hours_status", "hours"])
    rows = rows.join(hours)
    rows["y_hour"] = rows.y / rows.hours
    rows["shift_net"] = rows.unit.eq("shift") & rows.tax.eq("net")
    rows["point_shift"] = rows.shift_net & rows.eligible
    prior = pd.read_csv(r.LEGACY_GOLD / "split_ids.csv", dtype={"id": str}, usecols=["source", "id", "fold", "part", "group"])
    old = prior.loc[prior.part.eq("test")].set_index(["source", "id"])
    known = old.reindex(pd.MultiIndex.from_frame(rows[["source", "id"]]))
    good = known.fold.notna().to_numpy()
    assert np.array_equal(known.group.to_numpy()[good], rows.group.to_numpy()[good])
    rows["old_gold_fold"] = known.fold.to_numpy()
    return rows


def split(d, fold, *, gold=False):
    if gold:
        d = d.loc[d.old_gold_fold.notna()].assign(evaluation_fold=lambda frame: frame.old_gold_fold.astype(int))
    train, val, test = g.split(d, fold)
    assert set(train.group).isdisjoint(val.group) and set(train.group).isdisjoint(test.group)
    assert set(val.group).isdisjoint(test.group)
    return train, val, test


def add(out, d, p, *, phase, hours_policy, target, model, fold, approach=None):
    d2 = d[["source", "id", "group", "mode", "mode_evidence", "priority", "profession_resolved",
            "city_resolved", "hours_status", "hours", "unit", "tax", "y", "y_hour"]].copy()
    d2[r.QUANTILES] = p
    d2["approach"] = model if approach is None else approach
    out.append(d2.assign(phase=phase, hours_policy=hours_policy, target=target, model=model, fold=fold))


def source_targets(t, v, target):
    col = "y" if target == "shift_then_divide" else "y_hour"
    return t.assign(label=t[col]), v.assign(label=v[col])


def scores_to_hour(values, d, target):
    return values / d.hours.to_numpy(dtype=float)[:, None] if target == "shift_then_divide" else values


def fit_gold(train, val, holdout, target, policy, fold, out):
    t, v = source_targets(train, val, target)
    for name in ("chewby", "cat_detail", "ssr_text"):
        bundle = g.fit(t, v, **VARIANTS[name])
        add(out, holdout, scores_to_hour(g.predict(holdout, bundle), holdout, target),
            phase="gold_models", hours_policy=policy, target=target, model=name, fold=fold)
        if fold == 0 and target == "direct_hour":
            joblib.dump(bundle, MODELS / f"{policy}_gold_{name}.joblib", compress=3)
    q, _ = g.reference(holdout, g.reference_table(t))
    add(out, holdout, scores_to_hour(q, holdout, target),
        phase="gold_models", hours_policy=policy, target=target, model="empirical", fold=fold)


def build_routing(t, v, fold):
    bundles = {"global": r.make_model(t, v, {}, 240), "empirical": r.make_empirical(t)}
    for mode in r.MODES:
        for_mode = t[t["mode"].eq(mode)]
        if len(for_mode) < r.MIN_CONTEXT[0] or for_mode.group.nunique() < r.MIN_CONTEXT[1]:
            continue
        bundles[f"mode_{mode}"] = r.make_model(for_mode, v[v["mode"].eq(mode)], {"mode": mode}, 200)
    local, support = r.specialists(t, v, fold)
    bundles.update(local)
    return bundles, support


def fit_routing(t, v, test, target, policy, fold, out, coverage, inventory):
    t, v = source_targets(t, v, target)
    bundles, support = build_routing(t, v, fold)
    inventory.extend(dict(**row, hours_policy=policy, target=target) for row in support)
    forecasts = {name: r.predict_bundle(test, model) for name, model in bundles.items()}
    local = pd.Series("abstain", index=test.index)
    for name, artifact in bundles.items():
        if name.startswith("segment_"):
            local.loc[r.match(test, artifact["filters"])] = name
    methods = {
        "global": (pd.Series("global", index=test.index), []),
        "by_mode": ("mode_" + test["mode"], []),
        "by_segment": (local, []),
        "hybrid": (local.replace("abstain", "global"), ["global"]),
        "empirical": (pd.Series("empirical", index=test.index), []),
    }
    valid = test.point_shift.to_numpy(dtype=bool)
    for model, (pref, fallback) in methods.items():
        p, names, _ = r.routable(test, forecasts, pref, fallback)
        p = scores_to_hour(p, test, target)
        known = np.isfinite(p).all(axis=1)
        for scope, mask in scopes(test):
            if scope == "watch_tail_le_50000":
                continue
            m = mask.to_numpy(dtype=bool)
            coverage.append(dict(hours_policy=policy,target=target,model=model,scope=scope,
                                 convertible_context=(m).sum(), scored_context=(m & known).sum(),
                                 labelled=(m & valid).sum(), scored_labelled=(m & known & valid).sum()))
        add(out, test.loc[valid], p[valid], phase="routing", hours_policy=policy, target=target, model=model,
            fold=fold, approach=names[valid])
    if fold == 0 and target == "direct_hour":
        for name, artifact in bundles.items():
            joblib.dump(artifact, MODELS / f"{policy}_route_{name}.joblib", compress=3)


def fit_native_extension(shift_train, shift_val, test_shift, full, fold, out):
    native = full.loc[full.eligible & full.unit.eq("hour") & full.tax.eq("net")].assign(hours=1., y_hour=lambda d: d.y)
    nt, nv, ne = split(native, fold)
    t = pd.concat([shift_train, nt])
    v = pd.concat([shift_val, nv])
    t, v = source_targets(t, v, "direct_hour")
    bundle = g.fit(t, v, method="cat", columns=r.FEATURES, iterations=240)
    test = pd.concat([test_shift, ne])
    p = g.predict(test, bundle)
    add(out, test, p, phase="extension", hours_policy="single_range", target="hour_plus_native",
        model="global", fold=fold)


def predict_hourly(frame, hours_policy="exact", policy=None, folder=MODELS):
    """Load fitted scenario packages and return hourly forecasts for observed shift inputs."""
    if hours_policy not in POLICIES:
        raise ValueError(f"Unsupported hours policy: {hours_policy}")
    prefix = f"{hours_policy}_route_"
    files = {path.stem.removeprefix(prefix): path for path in Path(folder).glob(f"{prefix}*.joblib")}
    if policy is None:
        policy = {"version": 1, "rules": [], "default": "global", "fallback": ["empirical"]}
    r.validate_policy(policy, files)
    data = r.normalize_input(frame)
    duration = pd.DataFrame(data[HOURS].map(parse_hours).tolist(), index=data.index,
                            columns=["hours_status", "hours"])
    supported = data.unit.eq("shift") & data.tax.eq("net") & duration.hours_status.isin(POLICIES[hours_policy])
    rule, preference = r.planned_rules(data, policy)
    forecasts = {}
    for name in dict.fromkeys([*preference, *policy.get("fallback", [])]):
        if name == "abstain":
            continue
        values, note = r.predict_bundle(data, joblib.load(files[name]))
        values[~supported.to_numpy()] = np.nan
        forecasts[name] = values, note
    q, approaches, reason = r.routable(data, forecasts, preference, policy.get("fallback", []))
    result = data.reindex(columns=r.CORE_COLUMNS).join(duration)
    result["rule"] = rule
    result["approach"] = approaches
    result[["p25_hour", "p50_hour", "p75_hour"]] = q
    result["reason"] = np.where(
        ~data.unit.eq("shift") | ~data.tax.eq("net"), "shift_net_required",
        np.where(~duration.hours_status.isin(POLICIES[hours_policy]), "unsupported_hours", reason),
    )
    result["hour_target"] = duration.hours_status.map(
        {"closed_range": "range_midpoint_scenario", "exact": "exact_hours"}).fillna("unavailable")
    result["limit"] = np.where(data["mode"].eq("watch"), "verify_shift_vs_rotation", "published_offers")
    return result


def scopes(d):
    yield "all", d["mode"].notna()
    yield "gold", d.priority
    yield "outside_gold", ~d.priority
    for mode in r.MODES:
        yield mode, d["mode"].eq(mode)
    yield "remote_explicit", d["mode"].eq("remote") & d.mode_evidence.eq("work_format")
    yield "watch_tail_le_50000", d["mode"].eq("watch") & d.y.le(50_000)
    for name, key in zip(CARDS, g.FIRST):
        role, city = key
        yield f"{name}_спб", d.priority & d.profession_resolved.eq(role) & d.city_resolved.eq(city)


def previous_predictions(predictions):
    """Restrict prior full-pool holdouts to the new identical test IDs."""
    keys = ["source","id","fold","model"]
    new = []
    for phase in ("gold_models","routing"):
        old = (pd.read_csv(r.LEGACY_GOLD / "predictions.csv", dtype={"id":str})
                   .query("branch=='main' and model in ['empirical','chewby','ssr_text','cat_detail']")
               if phase == "gold_models" else
               pd.read_csv(r.EXPERIMENT_DIR / "predictions_oof.csv", dtype={"id":str}).rename(columns={"scheme":"model"})
                 .loc[lambda d: d.model.isin(ROUTING)])
        current = predictions.loc[predictions.phase.eq(phase) & predictions.target.eq("direct_hour")]
        paired = current.drop(columns=r.QUANTILES + ["approach"]).merge(
            old[keys + r.QUANTILES], on=keys, how="left", validate="many_to_one", indicator=True)
        assert paired["_merge"].eq("both").all()
        paired = paired.drop(columns="_merge")
        paired[r.QUANTILES] = paired[r.QUANTILES].div(paired.hours, axis=0)
        new.append(paired.assign(target="previous_full_then_divide", approach=paired.model))
    return pd.concat(new, ignore_index=True)


def summary(p):
    records = []
    for (phase, policy), data in p.groupby(["phase", "hours_policy"]):
        sample_masks = {"available": set(data.id)}
        if phase == "routing":
            wide = data.loc[data.target.ne("previous_full_then_divide")].pivot(
                index="id", columns=["target", "model"], values="p50")
            for name, methods in [("common_four", FOUR), ("common_broad", ("global","by_mode","hybrid","empirical"))]:
                columns = [(target, model) for target in ("direct_hour","shift_then_divide") for model in methods]
                sample_masks[name] = set(wide.index[wide.reindex(columns=pd.MultiIndex.from_tuples(columns)).notna().all(axis=1)])
        for sample, ids in sample_masks.items():
            sliced = data.loc[data.id.isin(ids)]
            for (target, model, unit), block in sliced.groupby(["target","model","unit"]):
                if sample == "common_broad" and model == "by_segment":
                    continue
                for scope, mask in scopes(block):
                    if mask.any():
                        x = block.loc[mask]
                        scores = g.score(x,x[r.QUANTILES].to_numpy(dtype=float), target="y_hour")
                        records.append(dict(phase=phase,hours_policy=policy,target=target,model=model,source_unit=unit,
                                            metric_unit="hour",sample=sample,scope=scope,**scores))
    return pd.DataFrame(records)


def pair(a, b, label):
    keys = ["source","id","group"]
    d = a.merge(b[keys+r.QUANTILES], on=keys, suffixes=("_a","_b"), validate="one_to_one").dropna(
        subset=["p50_a","p50_b"])
    if not len(d):
        return None
    loss = [g.losses(d.y_hour,d[[f"{q}_{side}" for q in r.QUANTILES]].to_numpy(dtype=float)).mean(axis=1)
            for side in ("a","b")]
    groups = d.assign(diff=loss[0]-loss[1]).groupby("group")["diff"].agg(["sum","size"])
    ci = [np.nan,np.nan]
    if len(groups)>=5:
        idx = np.random.default_rng(51).integers(len(groups), size=(1500,len(groups)))
        ci = np.quantile(groups["sum"].to_numpy()[idx].sum(axis=1)/groups["size"].to_numpy()[idx].sum(axis=1),[.025,.975])
    return dict(comparison=label,n=len(d),groups=len(groups),first_pinball=loss[0].mean(),second_pinball=loss[1].mean(),
                delta=(loss[0]-loss[1]).mean(),ci_low=ci[0],ci_high=ci[1])


def comparisons(p):
    rows=[]
    for (phase, policy), data in p.groupby(["phase","hours_policy"]):
        for scope, mask in scopes(data):
            s=data.loc[mask & data.unit.eq("shift")]
            for model, part in s.groupby("model"):
                for ref in ("shift_then_divide","previous_full_then_divide"):
                    z=pair(part[part.target.eq("direct_hour")], part[part.target.eq(ref)], f"direct_hour vs {ref}")
                    if z:
                        rows.append(dict(phase=phase,hours_policy=policy,scope=scope,model=model,**z))
            if phase == "gold_models":
                for target in ("direct_hour", "shift_then_divide"):
                    same = s[s.target.eq(target)]
                    result = pair(same[same.model.eq("ssr_text")], same[same.model.eq("chewby")],
                                  f"ssr_text vs chewby ({target})")
                    if result:
                        rows.append(dict(phase=phase, hours_policy=policy, scope=scope, model="ssr_text", **result))
            direct=s[s.target.eq("direct_hour")]
            baseline="empirical" if phase=="gold_models" else "global"
            for model in set(direct.model)-{baseline}:
                z=pair(direct[direct.model.eq(model)], direct[direct.model.eq(baseline)],f"{model} vs {baseline}")
                if z:
                    rows.append(dict(phase=phase,hours_policy=policy,scope=scope,model=model,**z))
    ext=p[(p.phase=="extension") & p.unit.eq("shift")]
    direct=p[(p.phase=="routing") & p.target.eq("direct_hour") & p.model.eq("global") & p.hours_policy.eq("single_range")]
    for scope,mask in scopes(ext):
        z=pair(ext.loc[mask], direct, "plus native hours vs shift-derived hours only")
        if z:
            rows.append(dict(phase="extension",hours_policy="single_range",scope=scope,model="global",**z))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)
    raw, digest=g.load_snapshot()
    gold_digest = json.loads((r.LEGACY_GOLD / "experiment.json").read_text(encoding="utf8"))["dataset"]["file_sha256"]
    routing_digest = json.loads((r.EXPERIMENT_DIR / "experiment.json").read_text(encoding="utf8"))["dataset_sha256"]
    if digest != gold_digest or digest != routing_digest:
        raise ValueError("Historical comparisons require the same canonical dataset as both previous studies")
    data=prepare(raw)
    points=data.loc[data.point_shift].copy()
    audit=[]
    for scope,mask in scopes(data):
        d=data.loc[mask]
        sizes=d.loc[d.point_shift,"hours_status"].value_counts()
        overlap=(d.point_shift & d.priority & d.hours_status.isin(POLICIES["single_range"]) & d.factor.notna()).sum()
        audit.append(dict(scope=scope,all_rows=len(d),point_shift=d.point_shift.sum(),
                          native_hour=(d.eligible & d.unit.eq("hour") & d.tax.eq("net")).sum(),
                          monthly_strict=(d.point_shift & d.factor.notna()).sum(),gold_month_hour_overlap=overlap,
                          **sizes.to_dict()))
    pd.DataFrame(audit).fillna(0).to_csv(OUT/"hours_support.csv",index=False)
    points[["source","id","group","mode","priority","profession_resolved","city_resolved",HOURS,
            "hours_status","hours","y","y_hour","factor","evaluation_fold","old_gold_fold"]].to_csv(
                OUT/"shift_hour_inputs.csv",index=False)
    results,coverage,inventory,splits=[],[],[],[]
    for policy,allowed in POLICIES.items():
        admitted=data.loc[data.shift_net & data.hours_status.isin(allowed)]
        for fold in range(3):
            left=split(admitted.loc[admitted.point_shift],fold,gold=True)
            train_all,val_all,test=split(admitted,fold)
            train,val=(x.loc[x.point_shift] for x in (train_all,val_all))
            assert set(train.group).isdisjoint(test.group)
            for phase, parts in [("gold_models",left),("routing",(train,val,test[test.point_shift]))]:
                for name,d in zip(("train","validation","test"),parts):
                    splits.append(d[["source","id","group"]].assign(phase=phase,hours_policy=policy,fold=fold,part=name))
            print(policy,fold,"gold experiment",len(left[2]),"routing test",test.point_shift.sum(),flush=True)
            for target in ("shift_then_divide","direct_hour"):
                fit_gold(*left,target,policy,fold,results)
                fit_routing(train,val,test,target,policy,fold,results,coverage,inventory)
            if policy=="single_range":
                fit_native_extension(train,val,test.loc[test.point_shift],data,fold,results)
    p=pd.concat(results,ignore_index=True)
    p=pd.concat([p,previous_predictions(p)],ignore_index=True)
    assert not p.duplicated(["phase","hours_policy","target","model","source","id"]).any()
    p.to_csv(OUT/"predictions_oof.csv",index=False)
    summary(p).to_csv(OUT/"metrics.csv",index=False)
    comparisons(p).to_csv(OUT/"paired.csv",index=False)
    counts=pd.DataFrame(coverage).groupby(["hours_policy","target","model","scope"],as_index=False).sum()
    totals=pd.DataFrame(audit).set_index("scope").all_rows
    counts["all_civilian"]=counts.scope.map(totals)
    counts["coverage_all"]=counts.scored_context/counts.all_civilian
    counts["coverage_labelled"]=counts.scored_labelled/counts.labelled
    counts.to_csv(OUT/"coverage.csv",index=False)
    pd.DataFrame(inventory).to_csv(OUT/"specialist_support.csv",index=False)
    pd.concat(splits,ignore_index=True).to_csv(OUT/"split_ids.csv",index=False)
    (OUT/"experiment.json").write_text(json.dumps(dict(dataset_sha256=digest,rows=len(raw),
       targets=["shift_then_divide","direct_hour","previous_full_then_divide","hour_plus_native"],
       policies={k:sorted(v) for k,v in POLICIES.items()},folds={"gold_models":"PR #179","routing":"PR #180"},
       scoring="RUB per hour net on paired IDs"),indent=2)+"\n",encoding="utf8")
    print("Saved",len(p),"paired-study predictions",flush=True)


if __name__=="__main__":
    main()
