"""Grouped validation for native pay quotes in rotation vacancies."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE / "source_rules")]
import kostya_nlp
import kostya_rules as rules
import vahta_extensions as extensions
from ML.text_features.conditions import condition_features
from ML.text_features.redaction import strict_redact
from utils.salary_normalization import _parse_salary

ALPHAS = np.array([0.25, 0.50, 0.75])
LIMITS = {"shift": (300, 50_000), "day": (300, 50_000),
          "hour": (50, 10_000), "week": (1_000, 300_000),
          "month": (10_000, 1_000_000), "rotation": (3_000, 2_000_000)}
CARD = [
    "profession_resolved", "region", "Занятость", "Способ оформления",
    "Количество рабочих часов в день", "Частота выплат", "Сфера деятельности компании",
    "Смены", "Опыт работы", "Медкнижка", "Объект",
    "Количество рабочих дней в неделю", "Тип предприятия", "Длительность вахты",
    "Инструменты", "Тип работ", "Тип организации",
    "Категория водительских прав", "Автомобиль",
]
NUMERIC_RULES = {"common__shift_hours", "loader__max_load_kg", "driver__capacity_tons"}
NLP = {
    "driver__vehicle_arrangement": kostya_nlp.extract_vehicle_arrangement_nlp,
    "picker__picking_technology": kostya_nlp.extract_picking_technology_nlp,
    "loader__max_load_kg": kostya_nlp.extract_manual_load_nlp,
}


def salary_tax(text):
    if re.search(r"на\s+руки|\bnet\b", text, re.I):
        return "net"
    if re.search(r"до\s+вычета|\bgross\b", text, re.I):
        return "gross"
    return "unknown"


def same_quote(left, right):
    return (left["low"], left["high"], left["shape"]) == (
        right["low"], right["high"], right["shape"])


def pick_target(row, parsed):
    """Allow one explicit unit and tax basis, with no conflicting pay evidence."""
    raw = row.salary_raw
    if row.piecework:
        return None, "piecework_context"
    card, _ = _parse_salary(raw)
    if card:
        card = dict(card, tax=salary_tax(raw))
        if card["shape"] == "from":
            card["high"] = card["low"]
    rates = {}
    for item in parsed["rate_options"]["candidates"]:
        rate = item["value"]
        if (rate["unit"] in LIMITS and rate["shape"] != "to"
                and not rate["conditional"] and not rate["bonus_context"]):
            key = tuple(rate[k] for k in ("unit", "low", "high", "shape"))
            if key in rates and rate["tax"] == "unknown":
                continue
            if (key in rates and rates[key]["tax"] != "unknown"
                    and rates[key]["tax"] != rate["tax"]):
                return None, "tax_conflict"
            rates[key] = rate
    if len(rates) > 1:
        return None, "multiple_primary_rates"
    body = next(iter(rates.values()), None)
    if card and card["unit"] != "unspecified":
        if body and (body["unit"] != card["unit"] or not same_quote(card, body)):
            return None, "card_body_conflict"
        quote = card.copy()
        origin = "card_confirmed" if body else "card_only"
    elif body and (not card or same_quote(card, body)):
        if not card and raw.strip().casefold() not in {"", "не указана", "зарплата не указана"}:
            return None, "unsupported_card"
        quote = body.copy()
        origin = "text_recovered"
    else:
        return None, "missing_or_ambiguous_period"
    if body and body["tax"] != "unknown":
        if quote["tax"] not in ("unknown", body["tax"]):
            return None, "tax_conflict"
        quote["tax"] = body["tax"]
    if quote["tax"] == "unknown":
        return None, "unknown_tax"
    if quote["unit"] in {"shift", "day", "hour"}:
        amounts = {quote["low"], quote["high"]}
        if (any(r["value"]["total_rub"] in amounts for r in parsed["rotation_totals"]["candidates"])
                or any(r["value"] in amounts for r in parsed["non_shift_amounts"]["candidates"])):
            return None, "rotation_period_conflict"
    minimum, maximum = LIMITS[quote["unit"]]
    if not minimum <= quote["low"] <= quote["high"] <= maximum:
        return None, "outside_unit_bounds"
    return dict(quote, value=(quote["low"] + quote["high"]) / 2, origin=origin), None


def groups(raw):
    """Transitive grouping by observed employer, profile, or duplicate description."""
    parent = list(range(len(raw)))
    seen = {}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(raw.itertuples(index=False)):
        profile = row.employer_profile_url.split("?")[0].rstrip("/")
        values = (row.employer_key, profile, row.description_key)
        for kind, value in enumerate(values):
            if value:
                previous = seen.setdefault((kind, value), i)
                parent[root(i)] = root(previous)
    return [root(i) for i in range(len(raw))]


def display_category(value):
    if value is None:
        return "_missing"
    if isinstance(value, (list, set, tuple)):
        return "|".join(sorted(map(str, value)))
    return str(value)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def feature_row(card, parsed, nlp_values):
    cleaned = rules.clean_description(card["description"])
    safe = strict_redact(cleaned)
    fs = rules.extract_description(safe)
    output = {key: card[key] for key in CARD}
    for key, v in fs.items():
        value = v["value"] if v["status"] == "resolved" else None
        output["rule_" + key] = number(value) if key in NUMERIC_RULES else display_category(value)
        output["status_" + key] = v["status"]
    for key in ("paid_hours", "vahta_work_duration", "rest_duration", "workplace"):
        result = parsed[key]
        value = result["value"] if result["status"] == "resolved" else None
        output["vahta_status_" + key] = result["status"]
        if key == "paid_hours":
            output["vahta_paid_hours"] = number(value)
        if key in {"vahta_work_duration", "rest_duration"}:
            amount, _, unit = (value or "").partition(" ")
            output["vahta_num_" + key] = number(amount)
            output["vahta_unit_" + key] = unit
    for key, v in nlp_values.items():
        value = v["value"] if v["status"] == "resolved" else None
        output["nlp_" + key] = number(value) if key in NUMERIC_RULES else display_category(value)
        output["nlp_status_" + key] = v["status"]
    return output


def make_features(rows, parsed=None, with_nlp=False):
    if parsed is None:
        parsed = {i: extensions.extract(text) for i, text in rows.description.items()}
    nlp_features = {}
    if with_nlp:
        import spacy
        nlp = spacy.load("ru_core_news_sm", disable=["ner"])
        texts = {i: rules.normalize_fragment(strict_redact(rules.clean_description(v)))
                 for i, v in rows.description.items()}
        unique = list(dict.fromkeys(texts.values()))
        by_text = {text: {name: fn(doc) for name, fn in NLP.items()}
                   for text, doc in zip(unique, nlp.pipe(unique, batch_size=24), strict=True)}
        nlp_features = {i: by_text[text] for i, text in texts.items()}
    card_features = pd.DataFrame.from_dict(
        {i: feature_row(row, parsed[i], nlp_features.get(i, {}))
         for i, row in rows.to_dict("index").items()}, orient="index"
    )
    conds = condition_features(rows.title.map(strict_redact),
                               rows.description.map(strict_redact),
                               rows["Что получают работники"])
    X = pd.concat([card_features, conds], axis=1)
    for name in X.select_dtypes(include=["object", "string"]).columns:
        X[name] = X[name].fillna("_missing").replace("", "_missing").astype(object)
    return X, list(card_features.columns)


def baseline(train, test):
    global_q = np.quantile(train.y, ALPHAS)
    by_role = {key: np.quantile(group.y, ALPHAS)
               for key, group in train.groupby("profession_resolved")
               if key and len(group) >= 15 and group.group.nunique() >= 4}
    return np.vstack([by_role.get(key, global_q) for key in test.profession_resolved])


def fit(frame, cols, y):
    model = CatBoostRegressor(
        loss_function="MultiQuantile:alpha=0.25,0.5,0.75",
        iterations=120, depth=4, learning_rate=0.08, l2_leaf_reg=6,
        border_count=24, one_hot_max_size=24,
        random_seed=42, thread_count=4, allow_writing_files=False, verbose=False,
    )
    model.fit(frame[cols], np.log(y), cat_features=[c for c in cols if frame[c].dtype == object])
    return model


def predict(model, frame, cols):
    return np.exp(np.sort(model.predict(frame[cols]), axis=1))


def score(y, p, group):
    error = np.abs(y - p[:, 1])
    ape = error / y
    inside = (p[:, 0] <= y) & (y <= p[:, 2])
    delta = np.log(y[:, None]) - np.log(p)
    loss = np.maximum(ALPHAS * delta, (ALPHAS - 1) * delta).mean(axis=1)
    by_group = pd.DataFrame({"group": group, "ape": ape, "loss": loss}).groupby("group").mean()
    return {"n": len(y), "groups": len(by_group), "mape": ape.mean(),
            "group_balanced_mape": by_group.ape.mean(), "median_ape": np.median(ape),
            "mae": error.mean(), "coverage": inside.mean(),
            "median_width_rel_p50": np.median((p[:, 2] - p[:, 0]) / p[:, 1]),
            "pinball_log": loss.mean(), "group_balanced_pinball_log": by_group.loss.mean(),
            "within_20pct": (ape <= .20).mean()}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "Data/dataset/vacancies.csv")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--without-nlp", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    raw = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    watch = raw["График"].eq("Вахта")
    eligible = raw.loc[watch & ~raw.military.str.casefold().isin({"true", "1", "yes", "да"})].reset_index(drop=True)
    eligible["employer_key"] = eligible.employer_name_observed.map(
        lambda x: " ".join(rules.normalize_fragment(x).split()))
    eligible["description_key"] = eligible.description.map(
        lambda x: " ".join(rules.normalize_fragment(rules.clean_description(x)).split()))
    eligible["group"] = groups(eligible)
    eligible["piecework"] = eligible.description_key.str.replace(
        r"не\s+сдельн\w*|не\s+сделка|без\s+сделки", "", regex=True
    ).str.contains(r"сдельн|\bсделк|при выполнении\s+норм|по выработ")
    parsed = [extensions.extract(text) for text in eligible.description]
    exclusions = Counter()
    targets = []
    for i, (row, add) in enumerate(zip(eligible.itertuples(index=False), parsed, strict=True)):
        quote, reason = pick_target(row, add)
        if quote is None:
            exclusions[reason] += 1
        else:
            targets.append({"index": i, "id": row.id, "group": row.group,
                            "profession_resolved": row.profession_resolved, **quote})
    frame = pd.DataFrame(targets)
    frame = frame.rename(columns={"value": "y"})
    cohort_stats = frame.groupby(["unit", "tax", "shape"]).size().to_dict()
    meta = dict(snapshot_sha256=sha256(args.data), rows=len(raw), watch_card=int(watch.sum()),
                excluded_title_only=int((~watch & raw.title.str.contains("вахт", case=False, regex=False)).sum()),
                watch_no_military=len(eligible),
                accepted=len(frame), rejected=dict(exclusions),
                by_unit_tax_shape={"|".join(key): v for key, v in cohort_stats.items()},
                rules_commit="34cff91b3", rules=rules.VERSION, extensions=extensions.VERSION)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    (args.out / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.drop(columns="index").to_csv(args.out / "target_quotes.csv", index=False)
    # Lower bounds describe a quoted floor. Keep them separate from exact/range quotes.
    frame["quote_kind"] = np.where(frame["shape"].eq("from"), "lower_bound", "exact_or_range")
    cohorts = [(key, d.copy()) for key, d in frame.groupby(["unit", "tax", "quote_kind"])
               if len(d) >= 60 and d.group.nunique() >= 12]
    idx = sorted({i for _, d in cohorts for i in d["index"]})
    selected = eligible.iloc[idx]
    if not len(selected):
        return
    X, card_features = make_features(selected, parsed, with_nlp=not args.without_nlp)
    sets = {
        "card": CARD,
        "kostya_vahta": [c for c in card_features if not c.startswith("nlp_")],
        "plus_m4_conditions": [c for c in X if not c.startswith("nlp_")],
    }
    if not args.without_nlp:
        sets["plus_kostya_nlp"] = list(X.columns)
    all_metrics, all_predictions = [], []
    for key, d in cohorts:
        name = "_".join(key)
        d = d.set_index("index")
        matrix = X.loc[d.index]
        splitter = GroupKFold(n_splits=5)
        print(f"Fitting {name}: {len(d)} offers, {d.group.nunique()} groups", flush=True)
        preds = {v: np.empty((len(d), 3)) for v in ("role_median", *sets)}
        fold_ids = np.full(len(d), -1)
        for fold, (train, test) in enumerate(splitter.split(matrix, d.y, d.group), 1):
            print(f"  fold {fold} / 5", flush=True)
            assert not set(d.iloc[train].group) & set(d.iloc[test].group)
            fold_ids[test] = fold
            preds["role_median"][test] = baseline(d.iloc[train], d.iloc[test])
            for variant, cols in sets.items():
                model = fit(matrix.iloc[train], cols, d.y.iloc[train])
                preds[variant][test] = predict(model, matrix.iloc[test], cols)
        assert (fold_ids > 0).all()
        results = []
        for variant, p in preds.items():
            results.append(dict(cohort=name, variant=variant, **score(d.y.to_numpy(), p, d.group.to_numpy())))
            rows = d[["id", "group", "y", "profession_resolved", "origin"]].copy()
            rows["fold"], rows["variant"], rows["cohort"] = fold_ids, variant, name
            rows[["p25", "p50", "p75"]] = p
            all_predictions.append(rows)
        all_metrics.extend(results)
        # Retain the full rules model and, if different, the best learned OOF variant.
        full = next(reversed(sets))
        best = min((s for s in results if s["variant"] in sets), key=lambda s: s["pinball_log"])["variant"]
        for variant in dict.fromkeys([full, best]):
            fit(matrix, sets[variant], d.y).save_model(str(args.out / f"{name}__{variant}.cbm"))
    pd.DataFrame(all_metrics).to_csv(args.out / "metrics.csv", index=False)
    pd.concat(all_predictions).to_csv(args.out / "predictions_oof.csv", index=False)
    (args.out / "feature_schema.json").write_text(json.dumps(sets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(all_metrics).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
