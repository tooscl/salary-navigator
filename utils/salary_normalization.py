"""Нормализация опубликованных зарплат, версия 1 (правила Analytics/salary_normalization_v1.ipynb).

Перенос функции normalize_salaries из ноутбука без изменения правил, чтобы её
можно было импортировать в эксперименты:

    from utils.salary_normalization import SalaryConfig, normalize_salaries, NOTEBOOK_CONFIG
    df_normalized = normalize_salaries(df_raw, config=NOTEBOOK_CONFIG)

По умолчанию результат совпадает с ноутбуком: исходные колонки + две новые
(salary_month_gross_normalized, salary_hour_gross_normalized). С
return_details=True добавляются служебные колонки для обучения:
salary_unit, salary_shape, salary_tax, hours_variant, hours_variant_index,
hours_variant_count, source_position. Правила описаны в ноутбуке.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

import numpy as np
import pandas as pd

RULES_VERSION = "salary_normalization_v1"


TARGET_COLUMNS = ["salary_month_gross_normalized", "salary_hour_gross_normalized"]
DETAIL_COLUMNS = ["salary_unit", "salary_shape", "salary_tax", "hours_variant", "hours_variant_index",
                  "hours_variant_count", "source_position"]
HOURS_COLUMN = "Количество рабочих часов в день"

# годовая стандартная шкала основной налоговой базы.
TAX_BRACKETS = [
    (2_400_000, 0.13),
    (5_000_000, 0.15),
    (20_000_000, 0.18),
    (50_000_000, 0.20),
    (math.inf, 0.22),
]

UNITS = {
    "за смену": "shift",
    "в смену": "shift",
    "в час": "hour",
    "за час": "hour",
    "в неделю": "week",
    "за неделю": "week",
    "за день": "day",
    "в день": "day",
    "в месяц": "month",
    "за месяц": "month",
    "месяц": "month",
    "сдельно": "piecework",
}

SALARY_RE = re.compile(
    r"(?:(от|до)\s*)?(\d[\d ]*(?:[.,]\d{1,2})?)"
    r"(?:\s*(?:[—–-]|до)\s*(\d[\d ]*(?:[.,]\d{1,2})?))?"
    r"\s*(?:₽|руб\.?|рублей|rub)"
    r"(?:\s+(" + "|".join(UNITS) + r"))?"
    r"(?:\s+(на руки|до вычета налогов|net|gross))?"
)


@dataclass(frozen=True)
class SalaryConfig:
    hours_per_month: float = 176.0
    weeks_per_year: float = 52.0
    unspecified_as_month: bool = True
    split_range_by_hours: bool = True
    impute_missing_hours: bool = True


def _clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\ufeff", " ")).strip().casefold()


def _parse_salary(value):
    text = _clean_text(value)
    if not text or text in {"зарплата не указана", "не указана"}:
        return None, "зарплата не указана"

    if "сдельн" in text and SALARY_RE.fullmatch(text) is None:
        return None, "сдельная оплата"

    match = SALARY_RE.fullmatch(text)
    if match is None:
        return None, "формат зарплаты не распознан"

    bound, first, second, unit_text, tax_text = match.groups()
    first = float(first.replace(" ", "").replace(",", "."))
    second = float(second.replace(" ", "").replace(",", ".")) if second else None

    if first <= 0 or not math.isfinite(first):
        return None, "некорректная зарплата"
    if second is not None and (second < first or not math.isfinite(second)):
        return None, "некорректная вилка"

    unit = UNITS.get(unit_text, "unspecified")
    tax = "net" if tax_text in {"на руки", "net"} else "gross"

    if unit == "piecework":
        return None, "сдельная оплата"
    if bound == "до":
        return None, "только верхняя граница «до»"
    if second is not None:
        shape, low, high = "range", first, second
    elif bound == "от":
        shape, low, high = "from", first, np.nan
    else:
        shape, low, high = "exact", first, first

    return {
        "low": low,
        "high": high,
        "shape": shape,
        "unit": unit,
        "tax": tax,
    }, None


def _parse_hours(value):
    """Возвращает список середин закрытых интервалов часов."""
    text = _clean_text(value)
    if not text:
        return [], "часы не указаны"

    result = []
    for token in dict.fromkeys(re.split(r"\s*[,;]\s*", text)):
        token = re.sub(r"\s*(?:часа|часов|час|ч\.)$", "", token).strip()
        interval = re.fullmatch(
            r"(?:от\s*)?(\d+(?:\.\d+)?)\s*(?:[–—-]|до)\s*(\d+(?:\.\d+)?)",
            token,
        )
        exact = re.fullmatch(r"\d+(?:\.\d+)?", token)
        open_interval = re.fullmatch(r"(?:до|меньше|больше|от)\s*\d+(?:\.\d+)?", token)

        if interval:
            low, high = map(float, interval.groups())
            if not 0 < low <= high <= 24:
                return [], "некорректные часы"
            result.append((low + high) / 2)
        elif exact:
            hours = float(token)
            if not 0 < hours <= 24:
                return [], "некорректные часы"
            result.append(hours)
        elif open_interval:
            # открытый интервал
            result.append(np.nan)
        else:
            return [], "формат часов не распознан"

    result.sort(key=lambda value: math.inf if pd.isna(value) else value)
    return result, None


def _fit_hours_lookup(raw):
    """средние часы с использованием полей города и профессии"""
    donors = []
    for row in raw.to_dict("records"):
        salary, _ = _parse_salary(row.get("salary_raw"))
        hours, hours_error = _parse_hours(row.get(HOURS_COLUMN))
        city = _clean_text(row.get("city_resolved"))
        profession = _clean_text(row.get("profession_resolved"))
        schedule = _clean_text(row.get("График"))

        if (
            salary is not None
            and salary["unit"] in {"shift", "day"}
            and hours_error is None
            and len(hours) == 1
            and pd.notna(hours[0])
            and profession
        ):
            donors.append((city, profession, schedule, hours[0]))

    if not donors:
        return {"city_profession_schedule": {}, "profession_schedule": {}, "profession": {}}

    table = pd.DataFrame(
        donors,
        columns=["city_resolved", "profession_resolved", "График", "hours"],
    )
    return {
        "city_profession_schedule": (
            table.loc[table["city_resolved"].ne("")]
            .groupby(["city_resolved", "profession_resolved", "График"])["hours"]
            .mean()
            .to_dict()
        ),
        "profession_schedule": (
            table.groupby(["profession_resolved", "График"])["hours"]
            .mean()
            .to_dict()
        ),
        "profession": table.groupby("profession_resolved")["hours"].mean().to_dict(),
    }


def _find_imputed_hours(row, lookup):
    """Ищет часы от более точной resolved-группы к более широкой."""
    city = _clean_text(row.get("city_resolved"))
    profession = _clean_text(row.get("profession_resolved"))
    schedule = _clean_text(row.get("График"))

    if not profession:
        return None

    return (
        lookup["city_profession_schedule"].get((city, profession, schedule))
        or lookup["profession_schedule"].get((profession, schedule))
        or lookup["profession"].get(profession)
    )


def _annual_net_to_gross(annual_net):
    """восстановление gross из годовой суммы net по прогрессивной шкале."""
    remaining = float(annual_net)
    gross = 0.0
    lower = 0.0

    for upper, rate in TAX_BRACKETS:
        net_capacity = (upper - lower) * (1 - rate)
        if remaining <= net_capacity:
            return gross + remaining / (1 - rate)
        gross += upper - lower
        remaining -= net_capacity
        lower = upper

    return gross


def normalize_salaries(raw, config=None, print_stats=True, return_details=False):
    """Возвращает преобразованный DataFrame: исходные колонки + две зарплаты.

    В результат входят только строки, для которых рассчитаны обе зарплаты.
    При нескольких вариантах часов одна исходная вакансия может дать несколько строк.
    """
    config = config or SalaryConfig()

    required = [
        "salary_raw",
        HOURS_COLUMN,
        "city_resolved",
        "profession_resolved",
        "График",
    ]
    missing_columns = [column for column in required if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"В DataFrame отсутствуют колонки: {missing_columns}")
    if not raw.columns.is_unique:
        raise ValueError("В DataFrame повторяются названия колонок")
    if any(column in raw.columns for column in TARGET_COLUMNS):
        raise ValueError("Передайте исходный DataFrame без нормализованных колонок")
    if config.hours_per_month <= 0 or config.weeks_per_year <= 0:
        raise ValueError("Коэффициенты времени должны быть положительными")

    hours_lookup = _fit_hours_lookup(raw) if config.impute_missing_hours else {}
    output_positions = []
    normalized_values = []
    details = []
    excluded = Counter()
    skipped_hour_variants = 0
    included_source_rows = set()

    for position, row in enumerate(raw.to_dict("records")):
        salary, error = _parse_salary(row.get("salary_raw"))
        if salary is None:
            excluded[error] += 1
            continue

        unit = salary["unit"]
        if unit == "unspecified":
            if config.unspecified_as_month:
                unit = "month"
            else:
                excluded["период зарплаты не указан"] += 1
                continue

        hours_options = [np.nan]
        if unit in {"shift", "day"}:
            hours_options, hours_error = _parse_hours(row.get(HOURS_COLUMN))
            if hours_error == "часы не указаны" and config.impute_missing_hours:
                imputed = _find_imputed_hours(row, hours_lookup)
                hours_options = [imputed] if imputed is not None else []
                hours_error = None if imputed is not None else "нет часов и среднего по группе"

            if hours_error is not None or not hours_options:
                excluded[hours_error or "часы не найдены"] += 1
                continue

        option_count = len(hours_options)
        valid_for_source = 0

        for option_index, hours in enumerate(hours_options):
            if unit in {"shift", "day"} and pd.isna(hours):
                skipped_hour_variants += 1
                continue

            low = salary["low"]
            high = salary["high"]

            if (
                salary["shape"] == "range"
                and option_count > 1
                and config.split_range_by_hours
            ):
                width = (high - low) / option_count
                low = low + width * option_index
                high = low + width

            representative = (
                (low + high) / 2
                if salary["shape"] == "range"
                else low
            )

            if unit in {"shift", "day"}:
                monthly = representative / hours * config.hours_per_month
            elif unit == "hour":
                monthly = representative * config.hours_per_month
            elif unit == "week":
                monthly = representative * config.weeks_per_year / 12
            elif unit == "month":
                monthly = representative
            else:
                continue

            if salary["tax"] == "net":
                monthly = _annual_net_to_gross(monthly * 12) / 12

            hourly = monthly / config.hours_per_month
            output_positions.append(position)
            normalized_values.append((monthly, hourly))
            details.append((unit if salary["unit"] != "unspecified" else "unspecified", salary["shape"], salary["tax"],
                            hours, option_index, option_count, position))
            valid_for_source += 1

        if valid_for_source:
            included_source_rows.add(position)
        else:
            excluded["нет пригодного варианта часов"] += 1

    result = raw.iloc[output_positions].copy().reset_index(drop=True)
    result[TARGET_COLUMNS] = pd.DataFrame(
        normalized_values,
        columns=TARGET_COLUMNS,
        dtype=float,
    )

    if return_details:
        result[DETAIL_COLUMNS] = pd.DataFrame(details, columns=DETAIL_COLUMNS)

    if print_stats:
        print(f"Исходных вакансий: {len(raw):,}")
        print(f"Нормализовано исходных вакансий: {len(included_source_rows):,}")
        print(f"Исключено исходных вакансий: {len(raw) - len(included_source_rows):,}")
        print(f"Строк в результате с учётом вариантов часов: {len(result):,}")
        if excluded:
            print("\nПричины исключения:")
            for reason, count in excluded.most_common():
                print(f"  {reason}: {count:,}")

    return result


# Параметры, с которыми ноутбук применяет функцию (ячейка «Применение»).
NOTEBOOK_CONFIG = SalaryConfig(
    hours_per_month=176.0,
    unspecified_as_month=True,
    split_range_by_hours=True,
    impute_missing_hours=False,
)
