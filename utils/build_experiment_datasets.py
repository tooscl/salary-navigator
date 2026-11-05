"""Build four clean, mutually exclusive experiment datasets from vacancies.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = Path("Data/dataset/vacancies.csv")
OUTPUT_DIRECTORY = Path("ML/data/experiments")
RULES_VERSION = "experiment_datasets_v2"
# A non-rate amount whose largest number is below this is almost surely a rate
# per shift or hour written without "за смену"/"в час" (median ~4 000 rub,
# 11-12 h shifts, weekly payouts), not a monthly salary.
MIN_MONTHLY_AMOUNT = 10_000

DATASETS = (
    ("01_location_monthly", "01_location_monthly.csv"),
    ("02_remote_monthly", "02_remote_monthly.csv"),
    ("03_watch", "03_watch.csv"),
    ("04_other", "04_other.csv"),
)

CORE_COLUMNS = [
    "source", "vacancy_id", "source_url", "run_id", "run_ids", "observed_at", "first_seen_at", "last_seen_at",
    "city", "city_source", "region", "profession", "segments", "title", "description", "location_observed",
    "employer_name_observed",
    "date_raw", "salary_raw", "salary_lower_rub", "salary_upper_rub", "salary_value_kind", "salary_period",
    "salary_period_evidence", "salary_tax_basis", "salary_parse_status", "work_schedule", "employment_type",
    "work_format", "registration_method", "work_hours_per_day", "pay_frequency", "experience", "watch_duration_days",
    "analysis_status", "export_eligibility", "full_parameters_read", "listing_rank", "listing_promoted", "views_total",
    "views_today", "experiment_dataset", "work_mode", "pay_mode", "extra_attributes_json",
]

SOURCE_COLUMNS = {
    "source": "source", "vacancy_id": "id", "source_url": "source_url", "run_id": "run_id", "run_ids": "run_ids",
    "observed_at": "observed_at", "first_seen_at": "first_seen_at", "last_seen_at": "last_seen_at", "city": "city",
    "region": "region", "profession": "profession", "segments": "segments", "title": "title",
    "description": "description", "location_observed": "location_observed", "employer_name_observed": "employer_name_observed",
    "date_raw": "date_raw", "salary_raw": "salary_raw", "work_schedule": "График", "employment_type": "Занятость",
    "work_format": "Формат работы", "registration_method": "Способ оформления",
    "work_hours_per_day": "Количество рабочих часов в день", "pay_frequency": "Частота выплат",
    "experience": "Опыт работы", "watch_duration_days": "Длительность вахты", "analysis_status": "analysis_status",
    "export_eligibility": "export_eligibility", "full_parameters_read": "full_parameters_read",
    "listing_rank": "listing_rank", "listing_promoted": "listing_promoted", "views_total": "views_total",
    "views_today": "views_today",
}

RAW_CORE_COLUMNS = set(SOURCE_COLUMNS.values())
AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[ \u00a0]\d{3})+|\d{2,6})(?!\d)")
SPACE_RE = re.compile(r"[\t\f\v ]+")


def clean_value(value: str | None) -> str:
    """Normalise whitespace without flattening paragraph breaks in descriptions."""
    value = (value or "").replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(SPACE_RE.sub(" ", line).strip() for line in value.split("\n")).strip()


def inline(value: str | None) -> str:
    return " ".join(clean_value(value).split())


def numbers_from_salary(value: str) -> list[int]:
    return [int(item.replace(" ", "").replace("\u00a0", "")) for item in AMOUNT_RE.findall(value)]


def rate_period(value: str) -> str:
    text = inline(value).casefold()
    if any(marker in text for marker in ("в час", "за час", "/час", "почас")):
        return "hour"
    if any(marker in text for marker in ("за смен", "в смен", "за день", "в день", "за сут", "в сут")):
        return "day_or_shift"
    if any(marker in text for marker in ("в неделю", "за неделю", "еженедел")):
        return "week"
    if any(marker in text for marker in ("в месяц", "за месяц", "/мес", "ежемесяч")):
        return "month"
    return "unknown"


def is_variable_pay(value: str) -> bool:
    text = inline(value).casefold()
    return any(marker in text for marker in ("сдель", "процент", "%", "бонус", "преми"))


FEDERAL_CITIES = {"Москва", "Санкт-Петербург", "Севастополь"}
REGION_PART_RE = re.compile(r"обл\.|област|край|республик|автономн|\bао\b", re.IGNORECASE)
DISTRICT_PART_RE = re.compile(r"р-н|район|г\.\s*о\.|округ|поселение|муниципальн", re.IGNORECASE)
CITY_PREFIX_RE = re.compile(r"^(г\.|город)\s*", re.IGNORECASE)
# Metro / travel-time tail glued to the name: "Одинцово Одинцово 16–20 мин."
TRAVEL_TAIL_RE = re.compile(r"\s+\S.*?(\d+[–-]\d+|от \d+|до \d+)\s*мин.*$")


def city_from_location(location: str | None) -> str:
    """Settlement from an Avito address.

    'Москва, Кленовый б-р ...'                        -> 'Москва'
    'Свердловская обл., Екатеринбург, ул. ...'         -> 'Екатеринбург'
    'Московская обл., Ленинский г.о., Видное, ...'     -> 'Видное'
    """
    parts = [part.strip() for part in inline(location).split(",") if part.strip()]
    if not parts:
        return ""
    if parts[0] in FEDERAL_CITIES:
        return parts[0]
    start = 1 if REGION_PART_RE.search(parts[0]) and len(parts) > 1 else 0
    for part in parts[start:]:
        if DISTRICT_PART_RE.search(part):
            continue
        return TRAVEL_TAIL_RE.sub("", CITY_PREFIX_RE.sub("", part)).strip()
    return ""


def resolved_city(row: dict[str, str]) -> tuple[str, str]:
    """(city, city_source). The enrichment leaves `city` empty for addresses it
    has no reviewed placement for (Колпино, Шушары, Видное, ...); the address
    itself is still on the card, so the settlement is taken from it."""
    city = inline(row.get("city"))
    if city:
        return city, "card"
    city = city_from_location(row.get("location_observed"))
    return (city, "location_observed") if city else ("", "missing")


def is_military(row: dict[str, str]) -> bool:
    """Enrichment flag (military_rule_version): SVO / rear-service contracts whose
    amounts include one-off sign-up payments, not a market salary for the job."""
    return inline(row.get("military")).casefold() == "true"


def work_mode(row: dict[str, str]) -> str:
    options = {part.strip() for part in inline(row.get("Формат работы")).split(",") if part.strip()}
    if options == {"Удалённо"}:
        return "remote_only"
    if "Удалённо" in options:
        return "remote_capable"
    if options:
        return "non_remote_explicit"
    return "unknown_city_anchored" if resolved_city(row)[0] else "unknown_unanchored"


def pay_mode(row: dict[str, str]) -> str:
    raw = inline(row.get("salary_raw"))
    amounts = numbers_from_salary(raw)
    if not raw or "зарплата не указана" in raw.casefold() or not amounts:
        return "unknown_or_unparseable"
    if inline(row.get("График")) == "Вахта":
        return "watch"
    if is_military(row):
        return "military_contract"
    if is_variable_pay(raw):
        return "variable_or_mixed"
    period = rate_period(raw)
    if period != "unknown":
        return period
    if max(amounts) < MIN_MONTHLY_AMOUNT:
        return "low_amount_likely_rate"
    return "non_rate_amount"


def is_assumed_monthly(row: dict[str, str]) -> bool:
    """A plain amount without a period, treated as monthly: no watch, no
    military contract, no variable pay, no rate marker, not a tiny amount."""
    return pay_mode(row) == "non_rate_amount"


def dataset_for(row: dict[str, str]) -> str:
    mode = work_mode(row)
    if inline(row.get("График")) == "Вахта":
        return "03_watch"
    if not is_assumed_monthly(row):
        return "04_other"
    # Remote-capable ads ("Удалённо, Гибрид", "Удалённо, В офисе") are the
    # same market as remote-only ones: agents, sales, assistants; the exact
    # options stay in work_format.
    if mode in {"remote_only", "remote_capable"}:
        return "02_remote_monthly"
    if resolved_city(row)[0]:
        return "01_location_monthly"
    return "04_other"


def salary_fields(row: dict[str, str], dataset: str) -> dict[str, str]:
    raw = clean_value(row.get("salary_raw"))
    amounts = numbers_from_salary(raw)
    text = inline(raw).casefold()
    if not amounts:
        kind, lower, upper, status = "unknown", "", "", "no_amount"
    elif text.startswith("от "):
        kind, lower, upper, status = "lower_bound", str(amounts[0]), "", "amount_parsed"
    elif text.startswith("до "):
        kind, lower, upper, status = "upper_bound", "", str(amounts[0]), "amount_parsed"
    elif len(amounts) == 1:
        kind, lower, upper, status = "point", str(amounts[0]), str(amounts[0]), "amount_parsed"
    else:
        kind, lower, upper, status = "range", str(min(amounts)), str(max(amounts)), "amount_parsed"

    if dataset in {"01_location_monthly", "02_remote_monthly"}:
        period, evidence = "month", "assumed_from_non_rate_listing"
        status = "amount_parsed_month_assumed" if amounts else status
    elif dataset == "03_watch":
        period, evidence = "watch", "work_schedule"
    else:
        period, evidence = rate_period(raw), "salary_raw" if rate_period(raw) != "unknown" else "unknown"
    tax_basis = "net" if "на руки" in text else ("gross" if "до вычета" in text else "unknown")
    return {
        "salary_lower_rub": lower, "salary_upper_rub": upper, "salary_value_kind": kind,
        "salary_period": period, "salary_period_evidence": evidence, "salary_tax_basis": tax_basis,
        "salary_parse_status": status,
    }


def output_row(row: dict[str, str], dataset: str, source_headers: list[str]) -> dict[str, str]:
    result = {column: clean_value(row.get(source_column)) for column, source_column in SOURCE_COLUMNS.items()}
    result.update(salary_fields(row, dataset))
    result["city"], result["city_source"] = resolved_city(row)
    result["experiment_dataset"] = dataset
    result["work_mode"] = work_mode(row)
    result["pay_mode"] = pay_mode(row)
    extra = {header: clean_value(row.get(header)) for header in source_headers if header not in RAW_CORE_COLUMNS and clean_value(row.get(header))}
    result["extra_attributes_json"] = json.dumps(extra, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {column: result.get(column, "") for column in CORE_COLUMNS}


def csv_bytes(rows: list[dict[str, str]], columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def dictionary_bytes() -> bytes:
    rows = []
    for column in CORE_COLUMNS:
        source = SOURCE_COLUMNS.get(column, "derived")
        rows.append({"column_name": column, "source_column": source, "description": "Derived experiment field" if source == "derived" else "Cleaned canonical dataset field"})
    return csv_bytes(rows, ["column_name", "source_column", "description"])


def build(root: Path = ROOT) -> dict[str, bytes]:
    source_path = root / SOURCE_PATH
    source_bytes = source_path.read_bytes()
    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        source_headers = reader.fieldnames or []
        source_rows = list(reader)
    keys = [(row.get("source"), row.get("id")) for row in source_rows]
    if len(keys) != len(set(keys)) or any(not source or not vacancy_id for source, vacancy_id in keys):
        raise ValueError("Source data must have unique non-empty source + id keys")

    grouped = {name: [] for name, _ in DATASETS}
    for row in source_rows:
        dataset = dataset_for(row)
        grouped[dataset].append(output_row(row, dataset, source_headers))
    if sum(len(rows) for rows in grouped.values()) != len(source_rows):
        raise ValueError("Every source row must be assigned exactly once")

    outputs = {filename: csv_bytes(grouped[name], CORE_COLUMNS) for name, filename in DATASETS}
    outputs["column_dictionary.csv"] = dictionary_bytes()
    manifest = {
        "schema_version": 1, "rules_version": RULES_VERSION,
        "source": {"path": str(SOURCE_PATH).replace("\\", "/"), "sha256": hashlib.sha256(source_bytes).hexdigest(), "rows": len(source_rows), "key": ["source", "id"]},
        "datasets": [{"name": name, "path": f"ML/data/experiments/{filename}", "rows": len(grouped[name]), "sha256": hashlib.sha256(outputs[filename]).hexdigest()} for name, filename in DATASETS],
        "schema": {"columns": CORE_COLUMNS, "encoding": "utf-8-sig", "extra_attributes": "Non-empty source columns not promoted to the clean schema."},
    }
    outputs["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return outputs


def write_or_check(root: Path, check: bool) -> None:
    outputs = build(root)
    directory = root / OUTPUT_DIRECTORY
    for name, data in outputs.items():
        path = directory / name
        if check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Outdated {path.relative_to(root)}; run python utils/build_experiment_datasets.py")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    rows = sum(
        sum(1 for _ in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        for name, data in outputs.items()
        if name.endswith(".csv") and name != "column_dictionary.csv"
    )
    print(f"{rows} vacancies written to {OUTPUT_DIRECTORY}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when derived outputs differ")
    args = parser.parse_args()
    write_or_check(ROOT, args.check)


if __name__ == "__main__":
    main()
