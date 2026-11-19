"""Experiment dataset: vacancies normalized by salary normalization v1, plus model features.

Scope (the notebook's intent): no watch work (График = Вахта), no remote work
(any "Удалённо" in Формат работы; the notebook compares with "Удаленно" and so
keeps them), no military ads. Normalization = utils.salary_normalization with the
notebook's parameters. A vacancy with several shift-length variants gives one row
per variant; each such row weighs 1 / number of variants, so every vacancy
counts once in training and in the metrics.

Rows outside 10 000 - 1 000 000 rub per month are kept in the dataset file but
not used by the models (``in_model_range``): below are mostly shift rates
written without a period (rule "no period = month"), above are shift rates
divided by short shifts.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

from utils.salary_normalization import NOTEBOOK_CONFIG, RULES_VERSION, normalize_salaries

ROOT = Path(__file__).resolve().parents[2]
VACANCIES = ROOT / "Data" / "dataset" / "vacancies.csv"
DATASET_PATH = ROOT / "ML" / "data" / "experiments" / "05_normalized_v1.csv"
TARGET = "salary_month_gross_normalized"
MIN_MONTHLY, MAX_MONTHLY = 10_000, 1_000_000

# Avito card parameters known before publication (Russian names as in vacancies.csv).
CARD_COLUMNS = [
    "city_resolved", "region", "profession_resolved", "График", "Занятость", "Способ оформления",
    "Количество рабочих часов в день", "Частота выплат", "Сфера деятельности компании", "Опыт работы",
    "В том числе для кандидатов", "Медкнижка", "Количество рабочих дней в неделю", "Тип предприятия",
    "Режим работы", "Смены", "Способ доставки", "Стажировка", "Категория водительских прав",
    "Статус самозанятости", "Вид доставки", "Ассортимент", "Разряд", "Тип работ", "Техника",
    "Можно заканчивать в 17:00", "Стаж вождения", "Возраст автомобиля", "Навык мелкого ремонта автомобиля",
    "С чем надо работать", "Формат работы", "Объект", "Специализация", "Тип организации", "Формат заведения",
    "Компания предоставляет", "Инструменты",
]
# How the pay is written in the ad (no amounts): unit, shape, tax basis, shift-length variant.
META_CATEGORICAL = ["salary_unit", "salary_shape", "salary_tax"]
META_NUMERIC = ["hours_variant"]
BENEFITS = "Что получают работники"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_employer(name) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    name = re.sub(r"[«»\"'“”„]", " ", name.lower().replace("ё", "е"))
    name = re.sub(r"\b(ооо|оао|зао|пао|ао|ип|нко|ано|гк|тд|тк)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def scope(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    watch = df["График"].eq("Вахта")
    remote = df["Формат работы"].fillna("").str.contains("Удал")
    military = df["military"].fillna("False").eq("True")
    keep = ~(watch | remote | military)
    counts = {"vacancies": len(df), "watch": int(watch.sum()), "remote": int((remote & ~watch).sum()),
              "military": int((military & ~watch & ~remote).sum()), "in_scope": int(keep.sum())}
    return df[keep].reset_index(drop=True), counts


def prepare_rows(vacancies: pd.DataFrame, print_stats: bool = False) -> pd.DataFrame:
    """Normalize vacancies (rules v1) and add the columns the models need.

    Input: rows in the shared dataset schema (Data/dataset/vacancies.csv). Output:
    one row per vacancy and shift-length variant, with all source columns, the
    normalized salary, pay-format details, weights and keys."""
    rows = normalize_salaries(vacancies, NOTEBOOK_CONFIG, print_stats=print_stats, return_details=True)
    y = rows[TARGET].astype(float)
    rows["in_model_range"] = y.between(MIN_MONTHLY, MAX_MONTHLY)
    rows["row_weight"] = 1.0 / rows["hours_variant_count"].astype(float)
    rows["employer_key"] = rows["employer_name_observed"].map(normalize_employer)
    rows["vacancy_key"] = rows["source"].astype(str) + ":" + rows["id"].astype(str)
    return rows


def build(vacancies: Path = VACANCIES, print_stats: bool = True) -> tuple[pd.DataFrame, dict]:
    """Normalized rows with all source columns + details + weights + model-range flag."""
    raw = pd.read_csv(vacancies, dtype=str)
    in_scope, counts = scope(raw)
    rows = prepare_rows(in_scope, print_stats=print_stats)
    meta = {
        "input": {"path": str(vacancies.relative_to(ROOT)), "sha256": sha256(vacancies), "rows": len(raw)},
        "rules_version": RULES_VERSION,
        "normalization_config": NOTEBOOK_CONFIG.__dict__,
        "scope": counts,
        "normalized_vacancies": int(rows["vacancy_key"].nunique()),
        "normalized_rows": len(rows),
        "model_range_rub_month": [MIN_MONTHLY, MAX_MONTHLY],
        "rows_in_model_range": int(rows["in_model_range"].sum()),
        "vacancies_in_model_range": int(rows.loc[rows["in_model_range"], "vacancy_key"].nunique()),
        "rows_by_unit": rows["salary_unit"].value_counts().to_dict(),
    }
    return rows, meta


def write_dataset(rows: pd.DataFrame, path: Path = DATASET_PATH) -> str:
    """Compact CSV: keys, details and targets (features are read from vacancies.csv)."""
    columns = ["source", "id", "hours_variant_index", "hours_variant_count", "hours_variant", "salary_unit",
               "salary_shape", "salary_tax", "salary_raw", "profession_resolved", "city_resolved", "employer_key",
               TARGET, "salary_hour_gross_normalized", "row_weight", "in_model_range"]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows[columns].to_csv(path, index=False, encoding="utf-8")
    return sha256(path)


def condition_block(rows: pd.DataFrame) -> pd.DataFrame:
    """Job conditions from strictly redacted text (ML/text_features)."""
    from ML.text_features.conditions import condition_features
    from ML.text_features.redaction import strict_redact

    vacancies = rows.drop_duplicates("vacancy_key").set_index("vacancy_key")
    title = vacancies["title"].map(strict_redact)
    description = vacancies["description"].map(strict_redact)
    block = condition_features(title, description, vacancies[BENEFITS].fillna(""))
    return block.loc[rows["vacancy_key"]].reset_index(drop=True)
