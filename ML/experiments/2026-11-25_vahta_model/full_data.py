"""Build full-coverage VAHTA labels for published pay, keeping unknown units separate."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tfidf import TEXT, safe_text
from train import HERE, ROOT, _parse_salary, extensions, groups, make_features, rules, salary_tax, sha256
from utils.salary_normalization import _clean_text


def card_quote(value):
    cleaned = _clean_text(value)
    upper = bool(re.match(r"^до\s+", cleaned))
    if upper:
        cleaned = re.sub(r"^до\s+", "", cleaned, count=1)
    rotation = bool(re.search(r"\s+(?:за|на)\s+вахту\b", cleaned))
    if rotation:
        cleaned = re.sub(r"\s+(?:за|на)\s+вахту\b", "", cleaned)
    parsed, _ = _parse_salary(cleaned)
    if not parsed:
        return None
    if upper and parsed["shape"] != "exact":
        return None
    parsed["tax"] = salary_tax(value)
    if upper:
        parsed.update(shape="to", high=parsed["low"], low=None)
    if rotation:
        parsed["unit"] = "rotation"
    return parsed


def body_matches(card, body):
    lo = None if body["shape"] == "to" else body["low"]
    hi = None if body["shape"] == "from" else body["high"]
    c_lo = None if card["shape"] == "to" else card["low"]
    c_hi = None if card["shape"] == "from" else card["high"]
    return card["shape"] == body["shape"] and (lo, hi) == (c_lo, c_hi)


def target_quote(value, parsed):
    card = card_quote(value)
    clean = _clean_text(value)
    rates = [item["value"] for item in parsed["rate_options"]["candidates"]
             if not item["value"]["conditional"] and not item["value"]["bonus_context"]]
    # A number of advertised salaries only carry a matching explicit period in the description.
    if card:
        matches = [r for r in rates if body_matches(card, r)]
        units = {r["unit"] for r in matches}
        if card["shape"] == "exact" and any(
            r["value"]["total_rub"] == card["low"] for r in parsed["rotation_totals"]["candidates"]
        ):
            units.add("rotation")
        context = "card_explicit" if card["unit"] != "unspecified" else "card_unknown_period"
        if card["unit"] == "unspecified" and len(units) == 1:
            card["unit"] = next(iter(units))
            context = "period_matched_from_text"
        elif len(units) > 1 or (units and card["unit"] not in units):
            context = "card_text_period_conflict"
            card["unit"] = "conflict"
        if card["unit"] in {"shift", "day", "hour"}:
            amounts = {card["low"], card["high"]}
            if any(r["value"]["total_rub"] in amounts for r in parsed["rotation_totals"]["candidates"]) or any(
                r["value"] in amounts for r in parsed["non_shift_amounts"]["candidates"]
            ):
                context = "card_text_period_conflict"
                card["unit"] = "conflict"
        matching_taxes = {r["tax"] for r in matches if r["unit"] == card["unit"] and r["tax"] != "unknown"}
        if card["tax"] == "unknown" and len(matching_taxes) == 1:
            card["tax"] = next(iter(matching_taxes))
        elif matching_taxes and (len(matching_taxes) > 1 or card["tax"] not in matching_taxes):
            card["tax"] = "conflict"
        return card, context
    if clean in {"", "не указана", "зарплата не указана"}:
        candidates = {(r["low"], r["high"], r["unit"], r["shape"], r["tax"]) for r in rates}
        if len(candidates) == 1:
            lo, hi, unit, shape, tax = candidates.pop()
            return dict(low=None if shape == "to" else lo, high=None if shape == "from" else hi,
                        unit=unit, shape=shape, tax=tax), "description_unique_rate"
    return None, "piecework_no_amount" if "сдельн" in clean else "no_parsable_amount"


def prepare(data, output, without_nlp=False):
    raw = pd.read_csv(data, dtype=str, keep_default_na=False)
    d = raw.loc[raw["График"].eq("Вахта")].reset_index(drop=True)
    d["employer_key"] = d.employer_name_observed.map(lambda x: " ".join(rules.normalize_fragment(x).split()))
    d["description_key"] = d.description.map(
        lambda x: " ".join(rules.normalize_fragment(rules.clean_description(x)).split()))
    d["group"] = groups(d)
    d["fold"] = -1
    for fold, (_, test) in enumerate(GroupKFold(n_splits=5).split(d, groups=d.group), 1):
        d.loc[test, "fold"] = fold
    parsed = [extensions.extract(x) for x in d.description]
    records = []
    for i, value in enumerate(d.salary_raw):
        quote, source = target_quote(value, parsed[i])
        if quote is None:
            quote = {"unit": "no_amount", "tax": "unknown", "shape": "no_amount", "low": np.nan, "high": np.nan}
        records.append(dict(quote, source_kind=source))
    labels = pd.concat([d[["id", "group", "fold", "military", "profession_resolved"]], pd.DataFrame(records)], axis=1)
    labels["military"] = np.where(labels.military.str.casefold().isin({"true", "1", "yes", "да"}), "yes", "no")
    # Use distinct prediction targets for central offers, quoted floors, and quoted ceilings.
    labels["y"] = np.select([labels["shape"].eq("range"), labels["shape"].eq("to")],
                            [(labels.low + labels.high) / 2, labels.high], default=labels.low)
    labels["quote_kind"] = labels["shape"].map(
        {"range": "central", "exact": "central", "from": "floor", "to": "ceiling", "no_amount": "no_amount"})
    assert labels.id.is_unique and labels.groupby("group").fold.nunique().max() == 1
    output.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output / "labels.csv", index=False)
    summary = dict(sha256=sha256(data), cards=len(labels), source_groups=int(labels.group.nunique()),
                   labeled=int(labels.y.notna().sum()), military=int(labels.military.eq("yes").sum()),
                   by_unit_tax_shape=labels.groupby(["unit", "tax", "quote_kind"]).size().to_dict(),
                   target_sources=labels.source_kind.value_counts().to_dict())
    summary["by_unit_tax_shape"] = {"|".join(k): v for k, v in summary["by_unit_tax_shape"].items()}
    (output / "dataset.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    x, card_and_rules = make_features(d, parsed, with_nlp=not without_nlp)
    x["military_type"] = labels.military.astype(object)
    text = d[TEXT].agg("\n".join, axis=1).map(safe_text)
    pd.to_pickle((labels, x, text, card_and_rules), output / "features.local.pkl")
    print(f"Prepared {len(x)} feature rows, {len(x.columns)} columns", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "Data/dataset/vacancies.csv")
    parser.add_argument("--out", type=Path, default=HERE / "full_results")
    parser.add_argument("--without-nlp", action="store_true")
    args = parser.parse_args()
    prepare(args.data, args.out, args.without_nlp)
