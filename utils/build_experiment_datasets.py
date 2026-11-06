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
RULES_VERSION = "experiment_datasets_v3"
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


# ---------------------------------------------------------------------------
# City from an Avito address
# ---------------------------------------------------------------------------
# Gazetteer: the 1 117 Russian cities of hflabs/city (DaData), CC BY-SA 4.0,
# https://github.com/hflabs/city. Columns: city, region, region_type, population.
CITIES_PATH = Path(__file__).resolve().parent / "data" / "russian_cities.csv"
# Real cities missing from hflabs/city, added by hand (same columns).
CITIES_EXTRA_PATH = Path(__file__).resolve().parent / "data" / "russian_cities_extra.csv"
FEDERAL_CITIES = {"москва": "Москва", "санкт-петербург": "Санкт-Петербург", "севастополь": "Севастополь"}
REGION_PART_RE = re.compile(r"обл\.|област|край|республик|автономн|\bао\b|\bаобл\b", re.IGNORECASE)
DISTRICT_PART_RE = re.compile(r"р-н|район|г\.\s*о\.|округ|поселение|муниципальн|сельсовет|направление", re.IGNORECASE)
# A part that is a street, building or similar: the settlement comes before it.
STREET_PART_RE = re.compile(
    r"(^|\s)(ул\.|улица|пр\.|просп|пр-т|проспект|ш\.|шоссе|пер\.|переулок|б-р|бульвар|наб\.|набережная|пл\.|площадь"
    r"|проезд|туп\.|тупик|аллея|линия|мкр|микрорайон|кв-л|квартал|тракт|дор\.|дорога|стр\.|к\d|д\.\s*\d)",
    re.IGNORECASE,
)
CITY_PREFIX_RE = re.compile(r"^(г\.|г\s|город\s)\s*", re.IGNORECASE)
# Rural settlements kept as they are when no gazetteer city is found.
SETTLEMENT_RE = re.compile(
    r"^(пос\.|посёлок|поселок|п\.|пгт|рп|р\.п\.|с\.|село|дер\.|д\.|деревня|ст-ца|станица|х\.|хутор|аул|сл\.|слобода)\s",
    re.IGNORECASE,
)
# Scraper notes and metro hints that are not part of the address.
SERVICE_NOTE_RE = re.compile(r"\([^)]*\)|;.*$|\bрядом\b.*$", re.IGNORECASE)
# Metro / travel-time tail: 'Казань Кремлёвская до 5 мин. Площадь Тукая 16–20 мин.'
# is cut at the first time marker, leaving 'Казань Кремлёвская'.
TRAVEL_TIME_RE = re.compile(r"\s(до\s|от\s)?\d+([–-]\d+)?\s*мин.*$", re.IGNORECASE)
NON_ADDRESS_PARTS = {"россия", "рф"}

_CITIES: dict[str, list[tuple[str, str, int]]] | None = None


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").strip(" .").split())


def gazetteer() -> dict[str, list[tuple[str, str, int]]]:
    """normalized city name -> [(name, normalized region, population)], largest first."""
    global _CITIES
    if _CITIES is None:
        cities: dict[str, list[tuple[str, str, int]]] = {}
        for path in (CITIES_PATH, CITIES_EXTRA_PATH):
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    entry = (row["city"], _norm(row["region"]), int(row["population"] or 0))
                    cities.setdefault(_norm(row["city"]), []).append(entry)
        _CITIES = {key: sorted(value, key=lambda e: -e[2]) for key, value in cities.items()}
    return _CITIES


# 'муниципальное образование Город Екатеринбург', 'г.о. Волгоград' -> the city name.
MUNICIPAL_PREFIX_RE = re.compile(
    r"^(муниципальное образование|городской округ|г\.\s*о\.|го)\s+(город\s+|г\.\s*)?", re.IGNORECASE
)
# A word followed by one of these is a district name ('Советский район'),
# not a city, even if a city of that name exists.
DISTRICT_SUFFIX_RE = re.compile(r"^(район|г\.\s*о\.|городской|муниципальный|сельсовет|округ)\b", re.IGNORECASE)
# A lowercase street type right after the word means a street ('Московский пр.'),
# while a capitalised word is a metro station ('Екатеринбург Площадь 1905 года').
STREET_SUFFIX_RE = re.compile(r"^(ул\.|улица|пр\.|просп|пр-т|ш\.|шоссе|пер\.|б-р|наб\.|пл\.|проезд|туп\.|аллея|линия|тракт)")


def _gazetteer_city(part: str, region_text: str) -> str:
    """Longest run of leading words of `part` that is a known city ('Нижний
    Новгород Горьковская 5–10 мин.' -> 'Нижний Новгород'). Homonyms are told
    apart by the region named in the address, else the most populous wins."""
    text = CITY_PREFIX_RE.sub("", MUNICIPAL_PREFIX_RE.sub("", part.strip()))
    words = text.split()
    region = _norm(region_text)
    for size in range(min(3, len(words)), 0, -1):
        candidates = gazetteer().get(_norm(" ".join(words[:size])))
        if not candidates:
            continue
        rest = " ".join(words[size:])
        if DISTRICT_SUFFIX_RE.match(rest) or STREET_SUFFIX_RE.match(rest):
            return ""  # 'Советский район р-н Советский', 'Московский пр.'
        in_region = [c for c in candidates if region and c[1].split()[0] in region]
        return (in_region or candidates)[0][0]
    return ""


def _looks_like_name(part: str) -> bool:
    """A bare settlement name not in the gazetteer ('Луганск')."""
    words = part.split()
    return (
        0 < len(words) <= 3
        and words[0][:1].isupper()
        and not any(ch.isdigit() for ch in part)
        and not DISTRICT_PART_RE.search(part)
        and not SETTLEMENT_RE.match(part)
    )


def locate_city(location: str | None) -> tuple[str, str]:
    """(settlement, kind) from an Avito address; kind is 'city' (gazetteer),
    'settlement' (village / township not in the gazetteer) or '' (not found).

    'Москва, Кленовый б-р ...'                                  -> ('Москва', 'city')
    'Свердловская обл., Екатеринбург, ул. ...'                   -> ('Екатеринбург', 'city')
    'Московская обл., Ленинский г.о., Видное, ...'               -> ('Видное', 'city')
    'Нижегородская обл., Нижний Новгород Горьковская 5–10 мин.'  -> ('Нижний Новгород', 'city')
    'Ростовская обл., Аксайский р-н, пос. Койсуг, ул. ...'      -> ('пос. Койсуг', 'settlement')
    """
    text = SERVICE_NOTE_RE.sub("", inline(location))
    parts = [part.strip() for part in text.split(",") if part.strip() and _norm(part) not in NON_ADDRESS_PARTS]
    if not parts:
        return "", ""
    first_words = _norm(CITY_PREFIX_RE.sub("", parts[0])).split()
    for size in (2, 1):  # 'Санкт-Петербург Адмиралтейская' still starts with the city
        if " ".join(first_words[:size]) in FEDERAL_CITIES:
            return FEDERAL_CITIES[" ".join(first_words[:size])], "city"
    region_text = parts[0] if REGION_PART_RE.search(parts[0]) else ""
    # Scan up to the first street-like part: a gazetteer city at the start of a
    # part wins at once; villages and unknown names are kept as fallbacks.
    settlements, others = [], []
    for part in parts[1 if region_text else 0 :]:
        part = TRAVEL_TIME_RE.sub("", part).strip()
        if not part:
            continue
        if SETTLEMENT_RE.match(part):
            settlements.append(part)
            continue
        city = _gazetteer_city(part, region_text)
        if city:
            return city, "city"
        if STREET_PART_RE.search(part):
            break
        others.append(part)
    if settlements:
        words = []
        for word in settlements[0].split():
            if any(ch.isdigit() for ch in word):
                break
            words.append(word)
        return " ".join(words[:4]), "settlement"
    for part in others:
        if _looks_like_name(part):
            return CITY_PREFIX_RE.sub("", part).strip(), "unlisted"
    return "", ""


def city_from_location(location: str | None) -> str:
    return locate_city(location)[0]


def resolved_city(row: dict[str, str]) -> tuple[str, str]:
    """(city, city_source). The enrichment leaves `city` empty for addresses it
    has no reviewed placement for (Колпино, Шушары, Видное, ...); the address
    itself is still on the card, so the settlement is taken from it:
    'location_observed' = a gazetteer city, 'location_settlement' = a village or
    township, 'missing' = nothing usable."""
    city = inline(row.get("city"))
    if city:
        return city, "card"
    city, kind = locate_city(row.get("location_observed"))
    if kind == "city":
        return city, "location_observed"
    if kind == "settlement":
        return city, "location_settlement"
    if kind == "unlisted":
        return city, "location_unlisted"
    return "", "missing"


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
