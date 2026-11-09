"""Rebuild the 15 agreed ML segment passports from the pinned dataset snapshot."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data/dataset/vacancies.csv"
OUTPUT = ROOT / "Analytics/salary_segment_results/priority_segment_passports.md"
EXPECTED_SHA256 = "835f5ac1c0ba1770d7fdd0718261f3445bff11acec653c795daa3c250d4e29d7"
PROFESSIONS = ("Кассир", "Продавец", "Грузчик", "Комплектовщик", "Водитель грузового транспорта")
CITIES = ("Москва", "Санкт-Петербург", "Екатеринбург")
CORE = ("График", "Занятость", "Опыт работы", "Частота выплат", "Сфера деятельности компании")
ROLE_FIELDS = {
    "Кассир": ("Медкнижка", "Ассортимент", "Режим работы"),
    "Продавец": ("Ассортимент", "Медкнижка", "Режим работы"),
    "Грузчик": ("Количество рабочих часов в день", "Смены", "Тип работ"),
    "Комплектовщик": ("Количество рабочих часов в день", "Смены", "Что получают работники"),
    "Водитель грузового транспорта": ("Категория водительских прав", "Техника", "Стаж вождения"),
}
SALARY_RE = re.compile(
    r"(?:(от|до)\s+)?(\d[\d\s\u00a0]*)"
    r"(?:\s*[—–-]\s*(\d[\d\s\u00a0]*))?\s*₽"
    r"(?:\s+(за смену|в час|за час|в неделю|за неделю|за день|в день|в месяц|за месяц|сдельно))?"
    r"(?:\s+(на руки|до вычета налогов))?", re.IGNORECASE,
)
UNITS = {"за смену": "shift", "в час": "hour", "за час": "hour", "в неделю": "week",
         "за неделю": "week", "за день": "day", "в день": "day", "в месяц": "month",
         "за месяц": "month", "сдельно": "piecework"}


def salary_parts(value: object) -> tuple[str, str, str, float | None]:
    if pd.isna(value) or not str(value).strip() or str(value).casefold() == "зарплата не указана":
        return "missing", "unspecified", "unknown", None
    match = SALARY_RE.fullmatch(re.sub(r"\s+", " ", str(value).replace("\ufeff", " ")).strip())
    if not match:
        return "unrecognized", "unspecified", "unknown", None
    bound, first, second, unit, tax = match.groups()
    first_number = float(re.sub(r"\D", "", first))
    second_number = float(re.sub(r"\D", "", second)) if second else None
    shape = "range" if second_number is not None else bound.casefold() if bound else "exact"
    shape = {"от": "from", "до": "to"}.get(shape, shape)
    point = (first_number + second_number) / 2 if second_number is not None else first_number if shape == "exact" else None
    period = UNITS.get(unit.casefold(), "unspecified") if unit else "unspecified"
    tax_base = {"на руки": "net", "до вычета налогов": "gross"}.get(tax.casefold(), "unknown") if tax else "unknown"
    return shape, period, tax_base, point


def observed(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("").fillna(False)


def views(series: pd.Series) -> pd.Series:
    # The source column contains both "15634 просмотров" and plain digits.
    return pd.to_numeric(series.astype("string").str.extract(r"^\s*([\d\s\u00a0]+?)(?:\s+просмотр\w*|\s*$)", expand=False)
                         .str.replace(r"\s", "", regex=True), errors="coerce")


def fmt(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.0f}".replace(",", " ")


def rate(series: pd.Series) -> str:
    return f"{100 * observed(series).mean():.1f}%"


def main() -> None:
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Dataset SHA-256 changed: {digest}")
    df = pd.read_csv(DATA, low_memory=False).drop_duplicates(["source", "id"], keep="last")
    df = df.loc[df["profession_resolved"].isin(PROFESSIONS) & df["city_resolved"].isin(CITIES)].copy()
    vahta = df["График"].astype("string").fillna("").str.contains("вахт", case=False) | df["title"].astype("string").fillna("").str.contains("вахт", case=False)
    remote = df["Формат работы"].astype("string").fillna("").str.contains("удал", case=False) | df["run_id"].astype("string").fillna("").str.contains("remote", case=False)
    military = df["military"].fillna(False).astype(str).str.casefold().isin(("true", "1", "да", "yes"))
    df = df.loc[~(vahta | remote | military)].copy()
    df[["shape", "period", "tax", "point"]] = pd.DataFrame(df["salary_raw"].map(salary_parts).tolist(), index=df.index)
    df["view_count"] = views(df["views_total"])

    lines = ["# Паспорта 15 согласованных ячеек", "",
             "Снимок: `Data/dataset/vacancies.csv`, SHA-256 `" + digest + "`; дедупликация `source × id`, последняя строка. Источник наблюдений — карточки Avito в этом CSV. Скрипт: `Analytics/build_priority_segment_passports.py`.", "",
             "Отбор: только пять согласованных профессий × три города; исключены вахта (поле «График» или заголовок), удалённая работа (поле «Формат работы» или run_id) и `military=True`. Профессия и город — `profession_resolved` и `city_resolved`. Объём ячейки включает вакансии без опубликованной зарплаты; зарплатные разрезы включают только распознанные опубликованные суммы.", "",
             "`exact` — опубликованная сумма; `range` — середина закрытой вилки как аналитический показатель. `from/to` — односторонние границы, без точечного значения. Q1, медиана и Q3 показаны лишь при ≥30 точечных значениях внутри одной пары период × налоговая база. `unspecified` — неизвестный период, не месяц; `unknown` — неизвестная налоговая база. Периоды и базы не объединены и не нормализованы. Все суммы в рублях за указанный период.", "",
             "Просмотры — `views_total` из карточки на момент наблюдения, не скорость спроса и не число соискателей; возраст публикаций не выравнивался. В таблицах указаны число наблюдаемых значений и медиана только среди них; «—» означает отсутствие наблюдений, а не ноль. Наблюдаемый 0 остаётся нулём. Пять Core-полей здесь — параметры карточки Авито из исследования инфополей; заполненность означает наличие непустого значения, а не подтверждённую пригодность признака для модели. Доли могут отличаться от общей матрицы исследования из-за исключений и другого знаменателя.", "",
             "Первый кандидат: **Кассир × Санкт-Петербург**; альтернатива: **Продавец × Санкт-Петербург**. Это согласованный пул, а не зарплатный топ сменных ячеек. Курьеры в него не входят.", ""]
    for profession in PROFESSIONS:
        for city in CITIES:
            g = df.loc[df["profession_resolved"].eq(profession) & df["city_resolved"].eq(city)]
            published = g["shape"].isin(("exact", "range", "from", "to"))
            points = g["shape"].isin(("exact", "range")) & g["point"].gt(0)
            lines.extend([f"## {profession} × {city}", "",
                          f"Объём после исключений: **{len(g)}**; опубликованная распознанная зарплата: **{published.sum()}**; точечных значений: **{points.sum()}**. Просмотры по всей ячейке: **{g['view_count'].notna().sum()}/{len(g)}**, медиана наблюдаемых **{fmt(g['view_count'].median())}**.", "",
                          "Заполненность Core: " + "; ".join(f"{field} {rate(g[field])}" for field in CORE) + ".", "",
                          "Профессиональные поля: " + "; ".join(f"{field} {rate(g[field])}" for field in ROLE_FIELDS[profession]) + ".", "",
                          "| Период | Налоговая база | Опубликовано | exact | range | from/to | Точки | Q1 | Медиана | Q3 | Просмотры: n / медиана |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for (period, tax), h in g.loc[published].groupby(["period", "tax"], sort=True):
                values = h.loc[h["shape"].isin(("exact", "range")) & h["point"].gt(0), "point"].astype(float)
                quantiles = values.quantile([.25, .5, .75]) if len(values) >= 30 else None
                viewed = h["view_count"].dropna()
                view_text = f"{len(viewed)} / {fmt(viewed.median())}" if len(viewed) else "—"
                lines.append(f"| {period} | {tax} | {len(h)} | {(h['shape'] == 'exact').sum()} | {(h['shape'] == 'range').sum()} | {h['shape'].isin(('from', 'to')).sum()} | {len(values)} | {fmt(quantiles.loc[.25]) if quantiles is not None else '—'} | {fmt(quantiles.loc[.5]) if quantiles is not None else '—'} | {fmt(quantiles.loc[.75]) if quantiles is not None else '—'} | {view_text} |")
            if not published.any():
                lines.append("| — | — | 0 | 0 | 0 | 0 | 0 | — | — | — | — |")
            limits = []
            if (~published).any():
                limits.append(f"{(~published).sum()} без распознанной опубликованной зарплаты")
            unknown_period = (g.loc[published, "period"] == "unspecified").sum()
            if unknown_period:
                limits.append(f"{unknown_period} с неизвестным периодом")
            unknown_tax = (g.loc[published, "tax"] == "unknown").sum()
            if unknown_tax:
                limits.append(f"{unknown_tax} с неизвестной налоговой базой")
            if not g["view_count"].notna().all():
                limits.append(f"просмотры отсутствуют у {g['view_count'].isna().sum()} вакансий")
            lines.extend(["", "Ограничения: " + ("; ".join(limits) if limits else "не выявлены по перечисленным полям") + ". Для обучения выбрать единый период и налоговую базу, а разбиение делать с учётом работодателя.", ""])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}; {len(df)} eligible rows")


if __name__ == "__main__":
    main()
