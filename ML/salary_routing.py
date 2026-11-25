"""Apply and evaluate salary forecasts with explicit, editable routing rules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import salary_gold as gold

RUN = "2026-11-18_salary_routing"
ROOT = gold.ROOT
EXPERIMENT_DIR = ROOT / "ML/experiments" / RUN
MODEL_DIR = ROOT / "ML/models" / RUN
MODES = ("ordinary", "watch", "remote")
CONTEXT = ["mode", "unit", "tax"]
CELL = CONTEXT + ["profession_resolved", "city_resolved"]
RULE_FIELDS = set(CELL) | {"График", "Занятость", "Медкнижка", "Способ оформления", "Количество рабочих часов в день"}
FEATURES = list(dict.fromkeys(["mode", *gold.DETAIL]))
CORE_COLUMNS = ["source", "id", *CELL]
UNITS = frozenset(gold.COMPARABLE_UNITS)
TAXES = frozenset(gold.KNOWN_TAX)
QUANTILES = ["p25", "p50", "p75"]
MIN_CONTEXT = (8, 4)
MIN_SEGMENT = (30, 10)
MIN_SEGMENT_VAL = 5
LEGACY_GOLD = ROOT / "ML/experiments/2026-11-18_priority_salary_gold"


def work_modes(raw):
    """Use declared fields first, then clearly identified collection provenance."""
    schedule_watch = gold.is_flag(raw["График"], r"вахт")
    title_watch = gold.is_flag(raw.title, r"вахт")
    format_text = raw["Формат работы"].astype("string").fillna("").str.strip()
    format_remote = gold.is_flag(format_text, r"удал")
    run_remote = gold.is_flag(raw.run_id, r"remote") & format_text.eq("")
    watch = schedule_watch | title_watch
    remote = format_remote | run_remote
    def conditions(*masks):
        return [mask.to_numpy(dtype=bool) for mask in masks]
    return pd.DataFrame({
        "mode": np.select(conditions(watch, remote), ["watch", "remote"], default="ordinary"),
        "mode_evidence": np.select(conditions(schedule_watch, title_watch, format_remote, run_remote),
                                   ["schedule", "title_only", "work_format", "run_only"], default="residual"),
        "mode_conflict": watch & remote,
    }, index=raw.index)


def prepare(raw):
    groups = gold.make_groups(raw)
    civilian = (
        raw.source.eq("avito") & raw.analysis_status.eq("validated") & raw.export_eligibility.eq("included")
        & ~raw.military.str.casefold().isin(["true", "1", "yes", "да"])
    )
    rows = raw.loc[civilian].copy()
    rows["group"] = groups.loc[rows.index]
    rows = gold.native_salary(rows.join(work_modes(rows)))
    rows["label"] = rows.y
    rows["eligible"] = rows.y.notna() & rows.unit.isin(UNITS) & rows.tax.isin(TAXES)
    rows["priority"] = rows.index.isin(gold.priority_vacancies(raw).index)
    rows["date"] = pd.to_datetime(rows.observed_at, utc=True, errors="coerce").dt.floor("D")
    # Keep whole observed employers and duplicates together, even with no salary label.
    eligible = rows.loc[rows.eligible].sort_values(["source", "id"])
    group_folds = {}
    for fold, (_, test) in enumerate(GroupKFold(n_splits=3).split(eligible, groups=eligible.group)):
        group_folds.update(dict.fromkeys(eligible.iloc[test].group, fold))
    rows["evaluation_fold"] = [
        group_folds.get(key, int(key[:8], 16) % 3) for key in rows.group
    ]
    return rows


def usable(frame):
    # Match contracts rather than guessing tax, pay duration, or work mode.
    return frame["mode"].isin(MODES) & frame.unit.isin(UNITS) & frame.tax.isin(TAXES)


def match(frame, filters):
    mask = pd.Series(True, index=frame.index)
    for col, value in filters.items():
        if col not in frame:
            return pd.Series(False, index=frame.index)
        mask &= frame[col].eq(value).fillna(False)
    return mask


def segment_key(frame):
    return frame[CELL].fillna("").astype(str).agg(" | ".join, axis=1)


def support(train, keys, n, groups):
    counts = train.groupby(keys, dropna=False).agg(n=("id", "size"), groups=("group", "nunique"))
    return counts.loc[counts.n.ge(n) & counts.groups.ge(groups)]


def catalog_item(folder, name, bundle, evidence):
    path = folder / f"{name}.joblib"
    joblib.dump(bundle, path, compress=3)
    return {
        "artifact": path.name, "scope": bundle["filters"], "method": bundle["kind"],
        "contexts": [list(key) for key in bundle["contexts"]],
        "fit_points": bundle["fit_points"], "evidence": evidence,
    }


def make_model(train, validation, filters, iteration_limit, *, depth=6, text=False):
    bundle = gold.fit(train, validation, method="cat", columns=FEATURES, iterations=iteration_limit, depth=depth, text=text)
    return {
        "kind": "catboost", "filters": filters, "model": bundle,
        "contexts": list(support(train, CONTEXT, *MIN_CONTEXT).index),
        "fit_points": len(train),
    }


def make_empirical(train):
    """Use only mode-compatible quartiles and show geographic/role fallback."""
    levels = [CELL, CONTEXT + ["profession_resolved"], CONTEXT]
    tables = []
    for i, fields in enumerate(levels):
        lookup = {}
        for key in support(train, fields, *(MIN_CONTEXT if i == 2 else (10, 5))).index:
            # Skip geographically or professionally undefined groups at finer levels.
            if any(not str(value).strip() for value in key[len(CONTEXT):]):
                continue
            lookup[key] = None
        quantiles = train.groupby(fields).label.quantile(list(gold.ALPHAS)).unstack()
        for key in lookup:
            lookup[key] = quantiles.loc[key].to_numpy(dtype=float)
        tables.append((fields, lookup))
    return {
        "kind": "empirical", "filters": {}, "tables": tables,
        "contexts": list(tables[-1][1]),
        "fit_points": len(train),
    }


def predict_bundle(frame, bundle):
    values = np.full((len(frame), 3), np.nan)
    detail = np.full(len(frame), "outside_validated_context", dtype=object)
    mask = (match(frame, bundle["filters"]) & usable(frame)
            & frame[CONTEXT].apply(tuple, axis=1).isin(bundle["contexts"]))
    mask = mask.to_numpy(dtype=bool)
    if not mask.any():
        return values, detail
    if bundle["kind"] == "empirical":
        names = ("city_profession", "profession_within_mode", "mode_unit_tax")
        for position, row in zip(np.flatnonzero(mask), frame.loc[mask, CELL].itertuples(index=False, name=None)):
            for name, (fields, lookup) in zip(names, bundle["tables"]):
                estimate = lookup.get(row[:len(fields)])
                if estimate is not None:
                    values[position] = estimate
                    detail[position] = name
                    break
    else:
        values[mask] = gold.predict(frame.loc[mask], bundle["model"])
        detail[mask] = "conditional_model"
    return values, detail


def normalize_input(frame):
    needed = set(CONTEXT)
    if missing := needed.difference(frame):
        raise ValueError(f"Required columns missing: {', '.join(sorted(missing))}")
    data = frame.copy().reset_index(drop=True)
    for col in CELL:
        if col not in data:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str).str.strip()
    if "mode_evidence" not in data:
        data["mode_evidence"] = "provided"
    if "mode_conflict" in data:
        data["mode_conflict"] = data["mode_conflict"].astype(str).str.casefold().isin({"true", "1", "yes", "да"})
    else:
        data["mode_conflict"] = False
    return gold.add_workload(data)


def default_policy():
    evidence = "../2026-11-18_priority_salary_gold/README.md"
    rules = []
    for label, (role, city) in zip(("cashier", "seller", "picker"), gold.FIRST):
        rules.append({
            "id": f"spb_{label}_statistic",
            "when": dict(mode="ordinary", unit="shift", tax="net", profession_resolved=role, city_resolved=city),
            "approach": "empirical",
            "why": "Квартили как рабочий ориентир до независимой проверки модели на новых работодателях.",
            "evidence": evidence,
        })
    return {"version": 1, "rules": rules, "default": "global", "fallback": ["empirical"]}


def validate_policy(policy, catalog):
    if policy.get("version") != 1:
        raise ValueError("Expected route policy version 1")
    rules = policy.get("rules", [])
    ids = [rule["id"] for rule in rules]
    if len(ids) != len(set(ids)) or "default" in ids or any(not x for x in ids):
        raise ValueError("Rule ids must be distinct and cannot use reserved id 'default'")
    for rule in rules:
        if not rule["when"] or set(rule["when"]) - RULE_FIELDS:
            raise ValueError(f"Invalid routing filters for {rule['id']}")
        if any(not isinstance(value, str) or not value for value in rule["when"].values()):
            raise ValueError(f"Use nonempty exact-match strings for {rule['id']}")
    names = [policy["default"], *policy.get("fallback", []), *(rule["approach"] for rule in rules)]
    if unknown := set(names) - set(catalog) - {"abstain"}:
        raise ValueError(f"Unrecognized approach: {', '.join(sorted(unknown))}")


def planned_rules(frame, policy):
    ids = pd.Series("default", index=frame.index)
    selected = pd.Series(policy["default"], index=frame.index)
    for rule in policy.get("rules", []):
        chosen = ids.eq("default") & match(frame, rule["when"])
        ids.loc[chosen] = rule["id"]
        selected.loc[chosen] = rule["approach"]
    return ids, selected


def rejection_reason(frame):
    return np.select(
        [~frame[col].isin(supported).to_numpy(dtype=bool) for col, supported in
         zip(CONTEXT, (MODES, UNITS, TAXES))],
        ["unknown_mode", "unsupported_or_unknown_unit", "unknown_tax"],
        default="no_supported_approach",
    )


def routable(frame, predictions, preference, fallbacks):
    """Reuse computed forecasts in offline comparison and interactive inference."""
    names = np.full(len(frame), "abstain", dtype=object)
    values = np.full((len(frame), 3), np.nan)
    details = rejection_reason(frame).astype(object)
    for requested in preference.unique():
        pending = preference.eq(requested).to_numpy(dtype=bool) & usable(frame).to_numpy(dtype=bool)
        for approach in dict.fromkeys([requested, *fallbacks]):
            if approach == "abstain":
                details[pending] = "policy_abstain"
                break
            if approach not in predictions or not pending.any():
                continue
            forecast, note = predictions[approach]
            chosen = pending & np.isfinite(forecast).all(axis=1)
            values[chosen], names[chosen], details[chosen] = forecast[chosen], approach, note[chosen]
            pending = pending & ~chosen
    return values, names, details


def load_catalog(folder=MODEL_DIR):
    return json.loads((Path(folder) / "catalog.json").read_text(encoding="utf-8"))


def predict_vacancies(frame, policy, folder=MODEL_DIR):
    """Apply reviewed local artefacts and an explicit routing policy without fitting."""
    folder = Path(folder)
    catalog = load_catalog(folder)["models"]
    validate_policy(policy, catalog)
    data = normalize_input(frame)
    rule, pref = planned_rules(data, policy)
    forecasts = {}
    for name in dict.fromkeys([*pref, *policy.get("fallback", [])]):
        if name != "abstain":
            bundle = joblib.load(folder / catalog[name]["artifact"])
            forecasts[name] = predict_bundle(data, bundle)
    values, names, notes = routable(data, forecasts, pref, policy.get("fallback", []))
    out = data.reindex(columns=CORE_COLUMNS + ["mode_evidence", "mode_conflict"]).copy()
    out["segment"] = segment_key(data)
    out["rule"] = rule
    out["requested"] = pref
    out["approach"] = names
    out[QUANTILES] = values
    out["reference_or_reason"] = notes
    out["fallback_used"] = out.approach.ne(out.requested) & out.approach.ne("abstain")
    limits = []
    for row in out.itertuples(index=False):
        detail = []
        if row.approach == "abstain":
            detail.append(str(row.reference_or_reason))
        else:
            detail.append("published_offers")
            if not row.city_resolved or not row.profession_resolved:
                detail.append("missing_segment_fields")
            if row.mode == "watch":
                detail.append("verify_single_shift_vs_full_rotation")
            if row.mode_evidence in {"run_only", "title_only", "residual"}:
                detail.append(f"mode_evidence:{row.mode_evidence}")
            if row.mode_conflict:
                detail.append("mode_conflict")
            if row.fallback_used:
                detail.append("fallback_used")
        limits.append("; ".join(detail))
    out["limitations"] = limits
    return out


def replay_policy(predictions, policy, catalog):
    """Score a new map against existing held-out model forecasts, without fitting."""
    validate_policy(policy, catalog)
    missing = set().union(*(rule["when"] for rule in policy.get("rules", []))) - set(predictions)
    if missing:
        raise ValueError(f"OOF inputs do not contain: {', '.join(sorted(missing))}")
    keys = ["source", "id"]
    base = predictions.loc[predictions.scheme.eq("global")].set_index(keys)
    data = base.reset_index()
    rule, preference = planned_rules(data, policy)
    source = predictions.loc[predictions.scheme.isin(["global", "by_mode", "by_segment", "empirical"])]
    oof = {}
    for name in dict.fromkeys([*preference, *policy.get("fallback", [])]):
        if name == "abstain":
            continue
        available = source.loc[source.approach.eq(name)].set_index(keys)
        if available.empty:
            raise ValueError(f"Missing OOF forecasts for {name}; evaluate its package on the common split first")
        aligned = available.reindex(base.index)
        oof[name] = aligned[QUANTILES].to_numpy(dtype=float), aligned.reference_or_reason.to_numpy()
    values, approaches, notes = routable(data, oof, preference, policy.get("fallback", []))
    data[QUANTILES] = values
    return data.assign(rule=rule, requested=preference, approach=approaches,
                       reference_or_reason=notes, scheme="replayed_policy")


def specialty_name(key):
    digest = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:12]
    return f"segment_{digest}"


def specialists(train, validation, fold, *, depth=6, text=False):
    """Train every sufficiently supported exact segment without consulting the test."""
    trained, records = {}, []
    candidates = train[(train.profession_resolved.ne("")) & (train.city_resolved.ne(""))]
    fit_counts = support(candidates, CELL, *MIN_SEGMENT)
    for key, row in fit_counts.sort_index().iterrows():
        filters = dict(zip(CELL, key))
        t, v = (d.loc[match(d, filters)] for d in (train, validation))
        name = specialty_name(key)
        ready = len(v) >= MIN_SEGMENT_VAL
        records.append(dict(fold=fold, approach=name, **filters, train=len(t), train_groups=row.groups,
                            validation=len(v), trained=ready))
        if ready:
            trained[name] = make_model(t, v, filters, iteration_limit=100, depth=depth, text=text)
    return trained, records
