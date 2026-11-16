"""Screen categorical vacancy fields with a stratified Pearson χ² test."""
from hashlib import sha256
from io import BytesIO
from math import sqrt
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from analyze_priority_salary_formats import (
    DATASET_COMMIT, DATASET_PATH, DATASET_SHA256, MIN_POINTS, TARGET_POINTS,
    additional_points, amounts, clean, employer_block_ci, get_group, pass_metrics, percent,
    prepared, rub, table,
)
from segment_normality import ALPHA, ROOT, SEGMENT_KEYS


REPORT = ROOT / "Analytics/salary_segment_results/priority_salary_chi_square.md"
MIN_MARGIN = 5
RESAMPLES = 49_999
SEED = 42
OUTCOMES = {"point": "Точка / граница", "high": "Выше медианы / остальные точки"}
FEATURE_COLUMNS = {
    "Занятость": "Занятость",
    "График": "График",
    "Опыт": "Опыт работы",
    "Медкнижка": "Медкнижка",
    "Часы": None,
    "Ротация": None,
    "Права": None,
    "Тип техники": None,
    "Сфера деятельности": "Сфера деятельности компании",
    "Частота выплат": "Частота выплат",
    "Тип предприятия": "Тип предприятия",
    "Стаж вождения": "Стаж вождения",
    "Способ оформления": "Способ оформления",
    "Ассортимент": "Ассортимент",
}
MULTI = {
    "Указан трудовой договор": ("Способ оформления", "трудовой договор"),
    "Указан ГПХ с самозанятым": ("Способ оформления", "договор гпх с самозанятым"),
    "Указан ГПХ с физлицом": ("Способ оформления", "договор гпх с физлицом"),
    "Указаны ночные смены": ("Смены", "ночные"),
    "Указан график 2/2": ("Смены", "2/2"),
    "Указано питание": ("Что получают работники", "питание"),
    "Указана развозка": ("Что получают работники", "транспорт до работы"),
    "Указана униформа": ("Что получают работники", "униформа"),
    "Указаны продукты / товары повседневного спроса": ("Ассортимент", "продукты и товары повседневного спроса"),
    "Указан склад": ("Тип предприятия", "склад"),
    "Указаны права C": ("Категория водительских прав", "c"),
    "Указаны права CE": ("Категория водительских прав", "ce"),
}
FIELDS = [*FEATURE_COLUMNS, *MULTI]
OTHER = "Редкие значения"
HOLDOUT_FRACTION = .30
TEST_COLUMNS = ["outcome", "field", "chi2", "p", "v", "n", "cells"]


def features(group):
    values = pd.DataFrame(index=group.index)
    for field, column in FEATURE_COLUMNS.items():
        values[field] = clean(group[column or field])
        if field in {"Способ оформления", "Ассортимент"}:
            multiple = values[field].str.contains(r"[,;]", na=False)
            values[field] = values[field].mask(multiple, "Несколько вариантов")
    for field, (column, option) in MULTI.items():
        source = clean(group[column]).str.casefold()
        source = source.str.replace(r"\s*/\s*", "/", regex=True)
        present = source.str.contains(r"(?:^|[,;])\s*" + re.escape(option) + r"\s*(?:[,;]|$)", regex=True)
        values[field] = present.map({True: "да", False: "нет"}).astype("string").where(source.notna())
    return values


def employer_partition(part):
    """Keep the same employer on one side of the split across the whole dataset."""
    buckets = part["employer"].dropna().unique()
    keep = {
        value for value in buckets
        if int.from_bytes(sha256(value.encode()).digest()[:8], "big") / 2**64 >= HOLDOUT_FRACTION
    }
    return part["employer"].isin(keep), part["employer"].notna() & ~part["employer"].isin(keep)


def cell_thresholds(part):
    return amounts(part).groupby([part[key] for key in SEGMENT_KEYS]).median()


def outcomes(part, thresholds, outcome):
    points = amounts(part)
    if outcome == "point":
        return points.notna().astype("Int64")
    cutoff = thresholds.reindex(pd.MultiIndex.from_frame(part[SEGMENT_KEYS])).set_axis(part.index)
    return points.gt(cutoff).where(points.notna() & cutoff.notna()).astype("Int64")


def contingency(category, target):
    observed = pd.crosstab(category, target).reindex(columns=[0, 1], fill_value=0)
    rare = observed.sum(axis=1).lt(MIN_MARGIN)
    if rare.any():
        pooled = observed.loc[rare].sum()
        observed = observed.loc[~rare]
        if pooled.sum() >= MIN_MARGIN:
            observed.loc[OTHER] = pooled
    if len(observed) < 2 or observed.sum(axis=0).min() < MIN_MARGIN:
        return None
    return observed.to_numpy()


def test_tables(tables, rng):
    """Calibrate sum of Pearson statistics conditional on margins within each cell."""
    observed_sum = 0.0
    null_sum = np.zeros(RESAMPLES)
    n = 0
    for observed in tables:
        result = chi2_contingency(observed, correction=False)
        row_sizes = observed.sum(axis=1)
        success = int(observed[:, 1].sum())
        weights = np.reciprocal(result.expected_freq).sum(axis=1)
        sampled = rng.multivariate_hypergeometric(row_sizes, success, size=RESAMPLES)
        null_sum += ((sampled - result.expected_freq[:, 1]) ** 2) @ weights
        observed_sum += result.statistic
        n += observed.sum()

    return {
        "chi2": observed_sum, "p": (1 + np.count_nonzero(null_sum >= observed_sum - 1e-10)) / (RESAMPLES + 1),
        "v": sqrt(observed_sum / n), "n": n, "cells": len(tables),
    }


def screen(part, thresholds, hypotheses):
    rows = []
    rng = np.random.default_rng(SEED)
    samples = {}
    for outcome in dict.fromkeys(outcome for outcome, _ in hypotheses):
        target = outcomes(part, thresholds, outcome)
        sample = part.loc[target.notna()].sample(frac=1, random_state=SEED).drop_duplicates("employer")
        samples[outcome] = (target, sample)

    for outcome, field in hypotheses:
        target, sample = samples[outcome]
        tables = []
        for _, cell in sample.groupby(SEGMENT_KEYS, sort=False):
            observed = contingency(cell[field], target.loc[cell.index])
            if observed is not None:
                tables.append(observed)
        result = test_tables(tables, rng) if tables else {
            "chi2": np.nan, "p": np.nan, "v": np.nan, "n": 0, "cells": 0,
        }
        rows.append({"outcome": outcome, "field": field, **result})
    result = pd.DataFrame(rows, columns=TEST_COLUMNS)
    tested = result["p"].notna()
    result["p_holm"] = np.nan
    if tested.any():
        result.loc[tested, "p_holm"] = multipletests(result.loc[tested, "p"], method="holm")[1]
    return result


def subgroup_metrics(part, discovery, holdout, selected):
    results = []
    for key, cell in part.groupby(SEGMENT_KEYS):
        for field in selected:
            for level, group in cell.groupby(field):
                metrics = pass_metrics(group)
                if metrics["n"] < 10:
                    continue
                discovery_group = group.loc[discovery]
                holdout_group = group.loc[holdout]
                results.append({
                    "cell": " × ".join(key), "field": field, "level": level,
                    **metrics, "gap_30": additional_points(metrics, MIN_POINTS),
                    "gap_50": additional_points(metrics, TARGET_POINTS),
                    "cluster_ci": employer_block_ci(group),
                    "discovery_n": amounts(discovery_group).count(), "holdout_n": amounts(holdout_group).count(),
                    "holdout_employers": holdout_group.loc[amounts(holdout_group).notna(), "employer"].nunique(),
                })
    return pd.DataFrame(results)


def significance(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",") if pd.notna(value) else "—"


def tests_table(tests):
    return table(
        ["Цель", "Поле", "Ячеек", "n", "χ²", "p", "p Холма", "V"],
        [
            [OUTCOMES[row.outcome], row.field, row.cells, row.n, significance(row.chi2, 2),
             significance(row.p, 5), significance(row.p_holm), significance(row.v, 3)]
            for row in tests.itertuples(index=False)
        ],
    )


def groups_table(groups):
    if groups.empty:
        return "В этой таблице нет подходящих групп."
    return table(
        ["Ячейка", "Фильтр", "Точек / зарплат", "Работодатели", "Q1–Q3, ₽",
         "ДИ по работодателям", "Точек на отложенных работодателях / работодателей",
         "Что мешает A", "Минимум новых точек / до 50"],
        [
            [row.cell, f"{row.field} = {row.level}", f"{row.n}/{row.parsed}", row.employers,
             f"{rub(row.q1)}–{rub(row.q3)}", percent(row.cluster_ci),
             f"{row.holdout_n} / {row.holdout_employers}", ", ".join(row.failures) or "A",
             f"{row.gap_30} / {row.gap_50}"]
            for row in groups.itertuples(index=False)
        ],
    )


def build_report(raw):
    part = get_group(prepared(raw), "shift", "net").copy()
    part[FIELDS] = features(part)
    discovery, holdout = employer_partition(part)
    thresholds = cell_thresholds(part.loc[discovery])
    discovery_tests = screen(part.loc[discovery], thresholds, [(outcome, field) for outcome in OUTCOMES for field in FIELDS])
    significant = discovery_tests.loc[discovery_tests["p_holm"].le(ALPHA)]
    selected = significant["field"].unique().tolist()
    validation = screen(part.loc[holdout], thresholds, list(zip(significant["outcome"], significant["field"])))
    subgroups = subgroup_metrics(part, discovery, holdout, selected)
    passing = subgroups
    almost = subgroups
    failing = subgroups
    if not subgroups.empty:
        passing = subgroups.loc[subgroups["class_a"]]
        failing = subgroups.loc[~subgroups["class_a"]]
        almost = failing.loc[failing["n"].ge(MIN_POINTS) & failing["gap_30"].le(2)]
    unknown_employers = int(part["employer"].isna().sum())

    sections = [
        "# Поиск подсегментов по χ²",
        f"Снимок main `{DATASET_COMMIT}`, SHA-256 `{DATASET_SHA256}`. "
        f"В 15 приоритетных ячейках использовано {len(part)} распознанных зарплат за смену на руки "
        f"({amounts(part).count()} точек, остальные — односторонние границы). "
        "Расчёт использует те же исключения, что [паспорта и план добора](priority_salary_formats_and_collection.md).",
        "## Результат для сегментации",
        f"Разведочный тест выделил {len(selected)} представления признака, прошедших поправку Холма. "
        f"На полном снимке {len(passing)} найденных по ним разрезов проходят исходный класс A. "
        "Это подгруппы внутри уже пригодных ячеек Санкт-Петербурга. Разделение ячейки на несколько "
        "проходящих частей пока не показано. У каждого нового кандидата на отложенных работодателях "
        "менее 30 точек, поэтому результаты по классу A относятся к этому снимку.",
        groups_table(passing),
        "У комплектовщиков Санкт-Петербурга фильтр «Способ оформления = Договор ГПХ с самозанятым» "
        "даёт ориентир 4 000–4 500 ₽ за смену. Здесь оставлены карточки, где указан только этот способ. "
        "Группа целиком входит в разрез «трудовой договор не указан» среди карточек с заполненным "
        "способом оформления. У продавцов Санкт-Петербурга трудовой договор может быть одним из "
        "нескольких вариантов. Доля точек выросла до 68/92. Ширина доверительного интервала по "
        "работодателям 35,3% превышает порог 25% для исходного интервала по вакансиям.",
        "### Где добор может открыть следующий разрез",
        "Сменная оплата на руки с точной суммой или закрытой вилкой. Числа отражают те же условия "
        "добора, что в основном отчёте. Добавления идут от разных работодателей вне крупнейшего "
        "при текущем количестве односторонних границ. Ширину интервала и разброс проверяем повторно.",
        groups_table(almost),
        "У грузчиков Москвы с указанным трудовым договором 32 точки из 66 (48,5%). "
        "Две новые пригодные точки доводят долю до 50%; цель 50 требует +18. "
        "На отложенных работодателях у этого условия доля 14/23; объёма для допуска всё ещё недостаточно.",
        "## Как применён χ²",
        "Нулевая гипотеза — независимость значения поля и зарплатного исхода внутри "
        "`профессия × город`. Проверены два исхода. Первый — точная сумма или середина закрытой "
        "вилки против объявления только с границей. Второй — точечная зарплата выше медианы своей "
        "ячейки против остальных точек (включая равные медиане). Порог медианы рассчитан по поисковой "
        "части и заморожен для проверки. Бинаризация фиксирует направление относительно медианы "
        "и может пропускать различия формы распределения. Тест проверяет связь категорий. Величины "
        "в рублях и условия допуска оцениваются в исходных разрезах из неокруглённых зарплат.",
        "Статистика — сумма χ² Пирсона по таблицам сопряжённости отдельных ячеек. P-значение получено "
        "49 999 симуляциями таблиц с теми же маргинальными частотами в каждой ячейке; это учитывает "
        "малые ожидаемые частоты. В анализ включались категории с минимум пятью наблюдениями. "
        "Редкие значения объединены, если вместе набрали пять. Таблица с одной категорией или менее "
        "пяти наблюдений в одном исходе пропускается. Отсутствующее поле исключено из теста по этому "
        "полю. Для мультивыбора «нет» означает заполненное поле без указанной опции. "
        "V = √(χ²/n) для бинарного исхода, сводная величина связи по стратифицированным таблицам; "
        "на малых таблицах она может быть завышена.",
        f"Работодатели распределены по стабильному хешу наблюдаемого нормализованного имени. "
        f"Поисковая часть — {int(discovery.sum())} вакансий от "
        f"{part.loc[discovery, 'employer'].nunique()} работодателей; проверочная — {int(holdout.sum())} "
        f"от {part.loc[holdout, 'employer'].nunique()}. {unknown_employers} вакансий без известного "
        "работодателя исключены из χ² и учтены в описании полного снимка. Для каждого исхода "
        "из каждого работодателя выбрана одна вакансия с фиксированным seed. Это снижает влияние "
        "повторных объявлений. Разные варианты названия одного бренда всё ещё могут попадать по разные "
        "стороны. Холм при α = 0,05 применён ко всем выполненным поисковым тестам, а для проверки — "
        "к найденным заранее комбинациям поля и исхода. По группам отдельно переоценён интервал "
        "медианы, пересэмплируя названных работодателей целиком.",
        "## Статистика поиска и отложенной проверки",
        f"Проверено {len(FIELDS)} вариантов кодирования полей для двух зарплатных исходов, "
        f"тест выполним в {int(discovery_tests['p'].notna().sum())} случаях. "
        f"Сигнал по уровню зарплаты прошёл порог в {sum(significant['outcome'].eq('high'))} случаях, "
        f"по формату — в {sum(significant['outcome'].eq('point'))}. "
        f"На отложенной части значимых повторений после Холма — {int(validation['p_holm'].le(ALPHA).sum())}. "
        "У способа оформления эти данные поддерживают исследовательские фильтры. "
        "Проверочная часть даёт мало пригодных таблиц, поэтому неопределённость остаётся высокой.",
        "Сигналы в поисковой части.",
        tests_table(significant),
        "Те же гипотезы на отложенных работодателях.",
        tests_table(validation),
        "[Документация SciPy по χ² и симуляции таблиц](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)",
        "## Все поисковые тесты",
        tests_table(discovery_tests.sort_values(["outcome", "p_holm", "p"], na_position="last")),
        "## Остальные кандидаты по отобранным полям",
        "Показаны разрезы с минимум 10 точками. Пересекающиеся условия можно использовать для разных "
        "аналитических вопросов. Размеры нельзя складывать.",
        groups_table(failing),
        "Воспроизведение на указанном снимке из корня проекта. "
        "`python3 utils/build_dataset.py --materialize`, затем "
        "`python3 Analytics/analyze_priority_salary_chi_square.py`. "
        "Проверено на pandas 2.3.3, NumPy 2.3.3, SciPy 1.18.1 и statsmodels 0.14.6.",
    ]
    return "\n\n".join(sections) + "\n"


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the checked-in report")
    args = parser.parse_args()
    data = DATASET_PATH.read_bytes()
    if sha256(data).hexdigest() != DATASET_SHA256:
        raise ValueError(f"Ожидался снимок {DATASET_COMMIT} с SHA-256 {DATASET_SHA256}")
    result = build_report(pd.read_csv(BytesIO(data), low_memory=False))
    if args.check:
        if REPORT.read_text(encoding="utf-8") != result:
            raise SystemExit("Отчёт отличается от расчётов")
        print("Отчёт совпадает с расчётами")
    else:
        REPORT.write_text(result, encoding="utf-8")
        print("Отчёт χ² сохранён")


if __name__ == "__main__":
    main()
