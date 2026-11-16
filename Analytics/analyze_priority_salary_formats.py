"""Reproduce the salary-format, workload and collection report for the 15 priority segments."""
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
from math import ceil
import re

import numpy as np
import pandas as pd

from build_priority_segment_passports import SALARY_RE, UNITS
from segment_normality import PRIORITY, ROOT, SEGMENT_KEYS, holm, lilliefors_row, priority_vacancies


DATASET_SHA256 = "7831e53ccd38c3c85e33c7e0bb3b8a521db58292e90349ebaae9cc4b025f873f"
DATASET_COMMIT = "7dc8e62cbc141972333c99fcc1ccb9f466139211"
DATASET_PATH = ROOT / "Data/dataset/vacancies.csv"
REPORT = ROOT / "Analytics/salary_segment_results/priority_salary_formats_and_collection.md"
KNOWN_TAX = ("net", "gross")
COMPARABLE_UNITS = ("month", "week", "day", "shift", "hour")
SCENARIO_FACTORS = {"month": 1, "week": 52 / 12, "day": 21, "shift": 21, "hour": 176}
FORMAT_KEYS = ["unit", "tax", "shape"]
UNIT_LABEL = {"month": "месяц", "week": "неделя", "day": "день", "shift": "смена", "hour": "час",
              "piecework": "сдельно", "unspecified": "период не указан"}
TAX_LABEL = {"net": "на руки", "gross": "до налога", "unknown": "база неизвестна"}
SHAPE_LABEL = {"exact": "сумма", "range": "закрытая вилка", "from": "от", "to": "до",
               "missing": "зарплата не указана", "unrecognized": "не распознано"}
MIN_POINTS = 30
TARGET_POINTS = 50
MIN_EMPLOYERS = 10
MAX_EMPLOYER_SHARE = .30
EMPLOYER_UNKNOWN = ("", "пользователь", "частное лицо", "не указан", "нет данных")
COMMON_FILTERS = {
    "Занятость": "Занятость",
    "График": "График",
    "Опыт": "Опыт работы",
    "Медкнижка": "Медкнижка",
}
SPLITS = [
    ("Занятость",), ("График",), ("Опыт",), ("Медкнижка",), ("Часы",), ("Ротация",),
    ("Права",), ("Тип техники",),
    ("Занятость", "График"), ("Занятость", "Опыт"), ("График", "Опыт"),
]


def clean(series):
    return series.astype("string").str.replace(r"[\ufeff\s]+", " ", regex=True).str.strip().replace("", pd.NA)


def salary_frame(raw):
    text = clean(raw)
    missing = text.isna() | text.str.casefold().eq("зарплата не указана")
    matched = text.where(text.str.fullmatch(SALARY_RE, na=False)).str.extract(SALARY_RE)
    bound, first, second, unit, tax = (matched[c] for c in matched)
    first = pd.to_numeric(first.str.replace(" ", ""), errors="coerce")
    second = pd.to_numeric(second.str.replace(" ", ""), errors="coerce")
    shape = bound.str.casefold().map({"от": "from", "до": "to"}).astype("string").fillna("exact")
    shape = shape.mask(second.notna(), "range").mask(first.isna(), "unrecognized").mask(missing, "missing")
    upper = first.where(shape.eq("to"), second.where(shape.eq("range"), first.where(shape.eq("exact"))))
    lower = first.where(shape.isin(["exact", "range", "from"]))
    swap = upper.lt(lower).fillna(False)
    upper, lower = upper.mask(swap, lower), lower.mask(swap, upper)
    return pd.DataFrame({
        "shape": shape,
        "unit": unit.str.casefold().map(UNITS).fillna("unspecified"),
        "tax": tax.str.casefold().map({"на руки": "net", "до вычета налогов": "gross"}).fillna("unknown"),
        "lower": lower, "upper": upper,
    })


def rotation(value):
    """Average workdays per week from a single stated working/rest cycle."""
    if pd.isna(value):
        return None
    ratios = set()
    for item in re.split(r"\s*[,;]\s*", value.casefold()):
        match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", item)
        if match:
            work, rest = map(int, match.groups())
            if not (1 <= work <= 7 and 1 <= rest <= 7):
                return None
            ratios.add(Fraction(7 * work, work + rest))
        elif item not in {"дневные", "ночные", "утренние", "вечерние", "работа по выходным"}:
            return None
    return float(ratios.pop()) if len(ratios) == 1 else None


def unique_rotations(part):
    shifts = clean(part["Смены"])
    mode = clean(part["Режим работы"])
    by_shifts, by_mode = shifts.map(rotation), mode.map(rotation)
    agreement = shifts.isna() | mode.isna() | (
        by_shifts.notna() & by_mode.notna() & by_shifts.eq(by_mode)
    )
    return pd.to_numeric(by_shifts.combine_first(by_mode).where(agreement), errors="coerce")


def workload(part):
    """Use exact workdays/hours or a compatible documented rotation for monthly equivalents."""
    days = clean(part["Количество рабочих дней в неделю"])
    exact_days = pd.to_numeric(days, errors="coerce")
    exact_days = exact_days.where(exact_days.between(1, 7) & exact_days.mod(1).eq(0))
    rotations = unique_rotations(part)
    brackets = days.str.extract(r"^(\d+)\s*[-–—]\s*(\d+)$").apply(pd.to_numeric)
    consistent_rotation = (
        days.isna() | exact_days.eq(rotations)
        | (rotations.ge(brackets[0]) & rotations.le(brackets[1]))
    )
    # A populated work-pattern field with no unique workload may specify alternatives.
    schedule_observed = clean(part["Смены"]).notna() | clean(part["Режим работы"]).notna()
    exact_days = exact_days.where(~schedule_observed | exact_days.eq(rotations))
    days = exact_days.combine_first(rotations.where(consistent_rotation))
    hours = pd.to_numeric(clean(part["Количество рабочих часов в день"]), errors="coerce")
    hours = hours.where(hours.between(1, 24))
    return days, hours, rotations


def prepared(raw):
    part = priority_vacancies(raw).copy()
    part = part.join(salary_frame(part["salary_raw"]))
    part["employer"] = clean(part["employer_name_observed"]).str.casefold()
    part["employer"] = part["employer"].mask(part["employer"].isin(EMPLOYER_UNKNOWN))
    days, hours, rotations = workload(part)
    factor = part["unit"].map({"month": 1, "week": 52 / 12})
    factor = factor.mask(part["unit"].isin(["day", "shift"]), days * 52 / 12)
    factor = factor.mask(part["unit"].eq("hour"), hours * days * 52 / 12)
    part["strict_factor"] = factor
    part["scenario_factor"] = part["unit"].map(SCENARIO_FACTORS)
    for field, column in COMMON_FILTERS.items():
        part[field] = clean(part[column])
    hours_label = clean(part["Количество рабочих часов в день"])
    part["Часы"] = hours_label.where(hours_label.str.fullmatch(r"\d+(?:\s*[-–—]\s*\d+)?", na=False))
    part["Ротация"] = rotations.map({3.5: "половина дней (например, 2/2)", 5.: "5 дней/нед.", 6.: "6 дней/нед."})
    part["Права"] = clean(part["Категория водительских прав"]).str.replace(r"\s*,\s*", ", ", regex=True)
    part["Тип техники"] = clean(part["Техника"])
    return part


def amounts(group, factor=None):
    lower, upper = group["lower"], group["upper"]
    points = ((lower + upper) / 2).where(group["shape"].isin(["exact", "range"]))
    points = points.where(points.gt(0))
    if factor:
        points = points * group[factor]
    return points


def pass_metrics(group, values=None):
    if values is None:
        values = amounts(group)
    x = values.dropna()
    n = len(x)
    parsed = len(group)
    employers = group.loc[x.index, "employer"].value_counts()
    largest_n = int(employers.max()) if len(employers) else 0
    employer_share = largest_n / n if n else np.nan
    employer_coverage = int(employers.sum()) / n if n else np.nan
    q1, median, q3 = x.quantile([.25, .5, .75])
    iqr = (q3 - q1) / median if n else np.nan
    ci_width, outliers = np.nan, np.nan
    if n:
        log_x = np.log1p(x)
        center = log_x.median()
        mad = (log_x - center).abs().median()
        outliers = float((.6745 * (log_x - center).abs() / mad > 3.5).mean()) if mad else 0.0
        rng = np.random.default_rng(42)
        lower_ci, upper_ci = np.quantile(np.median(rng.choice(x.to_numpy(), size=(300, n)), axis=1), [.025, .975])
        ci_width = (upper_ci - lower_ci) / median
    failures = []
    for failed, label in [
        (n < MIN_POINTS, f"точек < {MIN_POINTS}"),
        (len(employers) < MIN_EMPLOYERS, f"работодателей < {MIN_EMPLOYERS}"),
        (employer_share > MAX_EMPLOYER_SHARE, "крупнейший >30%"),
        (n < .5 * parsed, "точек <50%"),
        (iqr > .60, "IQR/медиана >60%"),
        (ci_width > .25, "ширина ДИ >25%"),
        (outliers > .10, "выбросы >10%"),
    ]:
        if failed:
            failures.append(label)
    return {
        "n": n, "parsed": parsed, "employers": len(employers), "largest_n": largest_n,
        "largest_share": employer_share, "employer_coverage": employer_coverage,
        "iqr_rel": iqr, "ci_rel": ci_width, "outliers": outliers, "q1": q1, "median": median,
        "q3": q3, "failures": failures, "class_a": not failures,
    }


def search_splits(parent):
    """Evaluate disjoint children of each predeclared partition within the same pay slice."""
    results = []
    if amounts(parent).count() < MIN_EMPLOYERS:
        return pd.DataFrame()
    for fields in SPLITS:
        names = list(fields)
        for keys, child in parent.groupby(names, dropna=True):
            if amounts(child).count() < MIN_EMPLOYERS:
                continue
            label = " · ".join(f"{field} = {value}" for field, value in zip(fields, keys))
            results.append({
                "filter": label, "split": " × ".join(fields), "depth": len(fields), "conditions": tuple(zip(fields, keys)),
                **pass_metrics(child),
            })
    return pd.DataFrame(results)


def additional_points(metrics, target):
    """Minimum additions for count, ≥50% points and employer screens under best-case sourcing."""
    return max(
        0, target - metrics["n"], metrics["parsed"] - 2 * metrics["n"],
        MIN_EMPLOYERS - metrics["employers"],
        ceil(metrics["largest_n"] / MAX_EMPLOYER_SHARE) - metrics["n"],
    )


def rub(value):
    return f"{value:,.0f}".replace(",", " ") if pd.notna(value) else "—"


def number(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",") if pd.notna(value) else "—"


def percent(value):
    return number(value * 100) + "%" if pd.notna(value) else "—"


def quartile_text(values):
    q1, median, q3 = values.quantile([.25, .5, .75])
    if pd.isna(median):
        return "—"
    return f"{rub(median)} ({rub(q1)}–{rub(q3)})"


def p_text(row):
    if pd.isna(row["p"]):
        return "—"
    raw_p, adjusted_p = row["p"], row["p_Holm"]
    p = "≤0,001" if np.isclose(raw_p, .001) else "≥0,990" if np.isclose(raw_p, .990) else number(raw_p, 3)
    return f"{p} / {number(adjusted_p, 3)}"


def table(headers, rows):
    def line(items):
        return "| " + " | ".join(str(item) for item in items) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def get_group(part, unit, tax):
    return part.loc[part["unit"].eq(unit) & part["tax"].eq(tax)]


def strict_groups(part, tax):
    by_tax = part.loc[part["tax"].eq(tax) & part["unit"].isin(COMPARABLE_UNITS)]
    return by_tax.loc[by_tax["strict_factor"].notna()]


def source_series(part, shape):
    if shape == "from":
        return part["lower"]
    if shape == "to":
        return part["upper"]
    return amounts(part)


def normality_test(scope, unit, tax, shape, values):
    reason = None
    if tax == "unknown" or unit in ("piecework", "unspecified"):
        reason = "Период, единица работы или налоговая база неизвестны"
    if shape in ("from", "to", "missing", "unrecognized"):
        reason = "Нет точечной зарплаты"
    return {"scope": scope, "unit": unit, "tax": tax, "shape": shape, **lilliefors_row(values, reason)}


def salary_distributions(groups):
    tests = []
    for scope, part in groups.items():
        for (unit, tax, shape), group in part.groupby(FORMAT_KEYS, sort=False):
            tests.append(normality_test(scope, unit, tax, shape, source_series(group, shape)))
        for (unit, tax), group in part.groupby(["unit", "tax"], sort=False):
            tests.append(normality_test(scope, unit, tax, "point", amounts(group)))
        for tax in (*KNOWN_TAX, "unknown"):
            for mode in ("strict_factor", "scenario_factor"):
                group = part.loc[part["tax"].eq(tax) & part[mode].notna()]
                tests.append(normality_test(scope, mode, tax, "point", amounts(group, mode)))
    names = ["unit", "tax", "shape"]
    out = pd.concat([holm(batch) for _, batch in pd.DataFrame(tests).groupby(names, sort=False)])
    return out.set_index(["scope", *names])


def raw_rows(part, distributions):
    rows = []
    for (unit, tax, shape), group in part.groupby(FORMAT_KEYS):
        if shape in ("missing", "unrecognized"):
            rows.append(["—", "—", SHAPE_LABEL[shape], len(group), "—", "—", "—"])
            continue
        values = source_series(group, shape)
        limits = "—"
        if shape == "range":
            limits = f"{rub(group['lower'].median())} / {rub(group['upper'].median())}"
        rows.append([
            UNIT_LABEL[unit], TAX_LABEL[tax], SHAPE_LABEL[shape], len(group),
            quartile_text(values), limits, p_text(distributions.loc[(unit, tax, shape)]),
        ])
    return rows


def monthly_rows(part, distributions):
    rows = []
    for tax in (*KNOWN_TAX, "unknown"):
        available = part.loc[part["tax"].eq(tax) & part["unit"].isin(COMPARABLE_UNITS)]
        strict = strict_groups(part, tax)
        strict_points = amounts(strict, "strict_factor")
        scenario_points = amounts(available, "scenario_factor")
        if not len(available):
            continue
        contributions = ", ".join(
            f"{UNIT_LABEL[unit]} {n}" for unit, n in strict.loc[strict_points.dropna().index]["unit"].value_counts().items()
        )
        rows.append([
            TAX_LABEL[tax], len(available), len(strict), strict_points.count(), contributions or "—",
            quartile_text(strict_points), p_text(distributions.loc[("strict_factor", tax, "point")]),
            scenario_points.count(), quartile_text(scenario_points), p_text(distributions.loc[("scenario_factor", tax, "point")]),
        ])
    return rows


def monthly_format_rows(part):
    rows = []
    for (unit, tax, shape), group in part.loc[part["strict_factor"].notna()].groupby(FORMAT_KEYS):
        monthly = source_series(group, shape) * group["strict_factor"]
        endpoints = "—"
        if shape == "range":
            lower = (group["lower"] * group["strict_factor"]).median()
            upper = (group["upper"] * group["strict_factor"]).median()
            endpoints = f"{rub(lower)} / {rub(upper)}"
        rows.append([UNIT_LABEL[unit], TAX_LABEL[tax], SHAPE_LABEL[shape], len(group),
                     quartile_text(monthly), endpoints])
    return rows


def employer_block_ci(group):
    values = amounts(group).dropna()
    data = group.loc[values.index].assign(point=values)
    clusters = [points["point"].to_numpy() for _, points in data.groupby("employer")]
    if not clusters:
        return np.nan
    rng = np.random.default_rng(42)
    estimates = [
        np.median(np.concatenate([clusters[i] for i in indices]))
        for indices in rng.integers(len(clusters), size=(1000, len(clusters)))
    ]
    left, right = np.quantile(estimates, [.025, .975])
    return (right - left) / data.loc[data["employer"].notna(), "point"].median()


def split_results(groups):
    tables = []
    for scope, part in groups.items():
        for unit in COMPARABLE_UNITS:
            for tax in KNOWN_TAX:
                data = get_group(part, unit, tax)
                candidates = search_splits(data)
                if not candidates.empty:
                    tables.append(candidates.assign(scope=scope, unit=unit, tax=tax))
    return pd.concat(tables, ignore_index=True)


def priority_report(groups):
    dists = salary_distributions(groups)
    shifts = {scope: get_group(part, "shift", "net") for scope, part in groups.items()}
    shift_metrics = {scope: pass_metrics(group) for scope, group in shifts.items()}
    candidates = split_results(groups)
    splits = candidates.loc[candidates["n"].ge(MIN_POINTS)]
    passed = splits.loc[splits["class_a"]].sort_values(["scope", "n", "depth"], ascending=[True, False, True])
    board, confirmation_board, split_board, detail = [], [], [], []
    strict_taxed_total, known_tax_points = 0, 0
    for scope, part in groups.items():
        raw = shift_metrics[scope]
        current = passed.loc[passed["scope"].eq(scope)]
        blocker = ", ".join(raw["failures"]) or "A"
        net_strict = strict_groups(part, "net")
        net_points = amounts(net_strict, "strict_factor")
        net_monthly = pass_metrics(net_strict, net_points)
        comparable = part.loc[part["tax"].isin(KNOWN_TAX) & part["unit"].isin(COMPARABLE_UNITS)]
        strict_taxed_total += amounts(comparable, "strict_factor").count()
        known_tax_points += amounts(comparable).count()

        target_message = "0 · класс A" if raw["class_a"] else additional_points(raw, TARGET_POINTS)
        board.append([
            scope, f"{raw['n']}/{raw['parsed']}", f"{raw['employers']} / {percent(raw['largest_share'])}", blocker,
            additional_points(raw, MIN_POINTS), target_message,
            f"{net_monthly['n']} · {quartile_text(net_points)}",
        ])
        net_unknown_unit = get_group(part, "unspecified", "net")
        unlabeled_values = amounts(net_unknown_unit)
        confirmation_board.append([
            scope, f"{unlabeled_values.count()}/{len(net_unknown_unit)}", quartile_text(unlabeled_values),
            net_monthly["n"], max(0, MIN_POINTS - net_monthly["n"]), max(0, TARGET_POINTS - net_monthly["n"]),
        ])

        tests = splits.loc[splits["scope"].eq(scope)]
        note = "Во всех разрезах меньше 30 точек" if tests.empty else "Проверенные фильтры сохраняют исходные ограничения."
        if not current.empty:
            note = ", ".join(
                f"{row['filter']} — {row['n']}/{row['parsed']}" for _, row in current.head(3).iterrows()
            )
        else:
            nearby = candidates.loc[
                candidates["scope"].eq(scope)
                & candidates["unit"].eq("shift")
                & candidates["tax"].eq("net")
                & (candidates["n"].ge(MIN_POINTS) | candidates["depth"].eq(1))
                & candidates["employers"].ge(5)
            ]
            cheaper = sorted(
                (
                    (additional_points(row, MIN_POINTS), additional_points(row, TARGET_POINTS), row)
                    for row in nearby.to_dict("records")
                    if additional_points(row, MIN_POINTS) < additional_points(raw, MIN_POINTS)
                ),
                key=lambda entry: (entry[0], entry[1], entry[2]["depth"], len(entry[2]["failures"]), -entry[2]["n"]),
            )
            if cheaper:
                minimum, target, nearest = cheaper[0]
                note = (
                    f"{nearest['filter']} — {nearest['n']}/{nearest['parsed']}. "
                    f"Добрать минимум {minimum}, до цели 50 — {target}. "
                    f"Остаётся: {', '.join(nearest['failures'])}."
                )
        split_board.append([scope, len(tests), len(current), note])

        profile = dists.loc[scope]
        detail.append(f"## {scope}\n\nПосле исключений {len(part)} вакансий. Сменная оплата на руки, "
                      f"{raw['n']}/{raw['parsed']} точек, медиана (Q1–Q3) "
                      f"**{quartile_text(amounts(shifts[scope]))} ₽**, p / p Холма "
                      f"**{p_text(profile.loc[('shift', 'net', 'point')])}**. "
                      f"Статус: {blocker}.")
        detail.append(table(
            ["Единица", "Налоговая база", "Формат", "n", "Медиана (Q1–Q3), ₽",
             "Медианы нижней / верхней границ", "p / p Холма"],
            raw_rows(part, profile),
        ))
        detail.append(table(
            ["База", "Распознано с известной единицей", "Обоснованный пересчёт, все форматы",
             "Точек", "Исходные единицы точек", "В месяц (Q1–Q3), ₽", "p / p Холма",
             "Точек по сценарию", "Сценарий (Q1–Q3), ₽", "p / p Холма сценария"],
            monthly_rows(part, profile),
        ))
        transformed = monthly_format_rows(part)
        if transformed:
            detail.append("Обоснованно пересчитанные суммы и границы в разрезе исходных форматов.")
            detail.append(table(
                ["Исходная единица", "База", "Формат", "n", "Медиана (Q1–Q3), ₽/мес.",
                 "Медианы нижней / верхней границ за месяц"],
                transformed,
            ))
        if net_monthly["n"]:
            detail.append(f"Месячный эквивалент на руки: {net_monthly['n']} точек. "
                          f"Статус: {', '.join(net_monthly['failures']) or 'A'}.")
        else:
            detail.append("Точек для обоснованного месячного эквивалента на руки нет.")

    selected_splits = []
    for _, row in passed.iterrows():
        child = get_group(groups[row.scope], row.unit, row.tax)
        for field, value in row.conditions:
            child = child.loc[child[field].eq(value)]
        selected_splits.append([
            row.scope, row["filter"], row.n, row.parsed, row.employers, percent(row.largest_share),
            f"{rub(row.q1)}–{rub(row.q3)}", percent(row.iqr_rel), percent(row.outliers),
            percent(row.ci_rel), percent(employer_block_ci(child)),
        ])

    distinct_passes = passed.groupby(["scope", "unit", "tax", "split"]).size()
    shift_tests = dists.xs(("shift", "net", "point"), level=["unit", "tax", "shape"])
    shift_tests_run = shift_tests["p"].notna().sum()
    shift_rejections = shift_tests["p_Holm"].le(.05).sum()
    largest_strict_n = max(amounts(strict_groups(group, "net"), "strict_factor").count() for group in groups.values())
    lines = [
        "# Добор, подсегменты и форматы зарплаты в 15 приоритетных ячейках",
        f"Снимок main `{DATASET_COMMIT}`, SHA-256 `{DATASET_SHA256}`. "
        f"В пуле {rub(sum(map(len, groups.values())))} вакансий после исключения вахты, удалёнки и military. "
        "Ключи — `profession_resolved × city_resolved`. Целевой формат добора — за смену на руки. "
        "Во всех 15 городско-профессиональных ячейках это крупнейший разрез с явно указанными единицей и налоговой базой.",
        "## Что добрать",
        "Засчитывайте новые уникальные вакансии с точной зарплатой или закрытой вилкой за смену на руки. "
        "Сохраняйте весь поток, включая границы «от/до», если он нужен для оценки охвата. Числа ниже — минимальные "
        "квоты **новых пригодных точек**. Они предполагают, что новых односторонних границ в оцениваемом разрезе "
        "пока нет, а новые точки приходят от разных работодателей вне крупнейшего. "
        "При добавлении каждой новой границы считайте баланс заново. Порог объёма — 30, рабочая цель добора — 50. "
        "Чтобы доля точек достигла 50%, к n точкам среди N распознанных требуется как минимум N − 2n новых точек. "
        "Для крупнейшего работодателя с k точками считаем также ⌈k/0,30⌉ − n, плюс проверяем минимум 10 работодателей. "
        "Разброс, выбросы и доверительный интервал проверяются на новом составе выборки.",
        table(
            ["Ячейка", "Точек / зарплат", "Работодателей / доля крупнейшего", "Что мешает A",
             "Минимум добрать", "Добор с целью 50", "Обоснованный месяц на руки · n и медиана (Q1–Q3)"],
            board,
        ),
        "Для кассира Санкт-Петербурга широкий разрез требует 64 новые точки при текущем знаменателе. "
        "При фильтре «Занятость = Постоянная» доступно 22 из 67, минимум добора 23 и цель 28. "
        "Для комплектовщика Москвы в узком разрезе с указанным интервалом 11–12 часов уже 35 точек. "
        "До 50 недостаёт 15. Добавляйте работодателей, потому что интервал по работодателям сейчас 29,1% "
        "при пороге 25%. Для пересчёта смен фиксируйте рабочие дни или ротацию, для часа дополнительно "
        "фиксируйте часы работы. Порог суммы в поиске меняет распределение.",
        "### Если нужен именно месячный эксперимент",
        "Ни одна из 15 ячеек не содержит явно опубликованной суммы за месяц. "
        f"Максимум в ячейке после обоснованного пересчёта — {largest_strict_n} точек. "
        "Добор прямых месячных зарплат с нуля означает минимум 30, целевой объём 50 при одной налоговой базе. "
        "Если допустим общий месячный эквивалент, ниже показан дефицит объёма на текущем снимке. "
        "Прежде чем собирать новые карточки, можно подтвердить период у уже собранных сумм «на руки». "
        "Их объёмы и квартили приведены как очередь проверки, все они пока относятся к неизвестному периоду. "
        "После подтверждения периода пересчитайте долю точечных зарплат, разнообразие работодателей и разброс.",
        table(
            ["Ячейка", "На руки, но период неизвестен · точек / зарплат", "Медиана (Q1–Q3), ₽ за неизвестный период",
             "Точек после пересчёта", "Дефицит до 30", "Дефицит до 50"],
            confirmation_board,
        ),
        "## Помогает ли деление на подсегменты",
        f"Проверено {len(splits)} непустых подсегментов с ≥30 точками, "
        f"из них {len(passed)} проходят исходные пороги внутри {passed['scope'].nunique()} ячеек. "
        "Деление выполнялось внутри единицы и налоговой базы. Известные значения графика, занятости, "
        "опыта, часов, ротации, медкнижки, прав и типа техники образуют разбиения. Три парных разбиения "
        "используют график, опыт и занятость. Пропуски остаются вне соответствующего разбиения. "
        "Строки из разных разбиений могут пересекаться. Краткая таблица показывает до трёх примеров на ячейку. "
        f"Проходящие условия выделяют ядро ячейки. Максимум проходящих частей в одном разбиении — {distinct_passes.max()}. "
        "Ниже также указаны кандидаты на добор. Для групп ниже 30 показываем только одинарные фильтры "
        "с минимум 10 точками и 5 известными работодателями, когда фильтр уменьшает минимум добора.",
        table(["Ячейка", "Групп с n ≥ 30", "Групп класса A", "Характеристики"], split_board),
        "Порог A — ≥30 точек, ≥10 названных работодателей, доля крупнейшего ≤30%, ≥50% точек среди "
        "распознанных зарплат, IQR/медиана ≤60%, относительная ширина 95% bootstrap ДИ медианы ≤25%, "
        "доля выбросов по прежнему log-MAD ≤10%. В численности работодателей используем наблюдаемое имя "
        "без регистра и лишних пробелов; общие заглушки исключаем. Имена могут объединять разных "
        "работодателей или дробить один бренд. При MAD = 0 сохранено поведение исходного рейтинга. "
        "Последняя колонка дополнительно пересэмплирует работодателей целиком среди записей с известным "
        "именем. Группы подобраны на этом снимке. Для нового эксперимента их фильтры нужно зафиксировать "
        "до добора и проверять на новых работодателях. Проверка класса A измеряет пригодность данных, "
        "точность ML оценивается отдельно.",
        table(["Ячейка", "Фильтр", "Точек", "Распознано", "Работодателей", "Доля крупнейшего",
               "Q1–Q3, ₽", "IQR/медиана", "Выбросы", "ДИ по вакансиям", "ДИ по работодателям"], selected_splits),
        "## Как читать профили зарплат",
        "Рубли за исходную единицу по форме, периоду и налоговой базе. Для закрытой вилки «медиана "
        "(Q1–Q3)» относится к её серединам. Дополнительно указаны медианы самих нижних и верхних границ. "
        "Для «от» и «до» описана соответствующая граница. Эти наблюдения участвуют в доле пригодных "
        "точек, а их числа и квантили сами по себе описывают только объявленные ограничения. "
        "Малые n показаны для прозрачности, их квартильный интервал годится для описания снимка.",
        f"Обоснованно пересчитываются **{strict_taxed_total} из {known_tax_points} положительных точечных "
        "значений с известными единицей и налоговой базой**. Месяц ×1, неделя ×52/12. День и смена "
        "пересчитываются по одному наблюдаемому числу рабочих дней в неделю либо по одной однозначной "
        "частоте из ротации, согласующейся со структурированным полем дней. Для часа нужны ещё точно "
        "заданные часы в день. Один день работы в ротации соответствует одной смене. Частоту выплат "
        "для определения периода не используем. При противоречии, диапазоне без уточнения или "
        "нескольких вариантах нагрузки пересчёт не выполняем. Недельную оплату считаем средней "
        "месячной при той же недельной нагрузке. Сценарный расчёт использует 21 смену/день и 176 "
        "часов в месяц, его суммы зависят от заданной нагрузки. Базы «на руки», «до налога» "
        "и «неизвестно» сохраняются отдельно. Водители Санкт-Петербурга имеют два опубликованных недельных "
        "диапазона с серединами 175 и 250 тыс. ₽; их необходимо проверить по карточке перед использованием. "
        "По общему снимку известная сопоставимая база представлена значениями «на руки».",
        "Для точных сумм, середин вилок, объединённой точечной цели и месячных эквивалентов применён "
        "критерий Лиллиефорса при α = 0,05. В таблицах p / p Холма; поправка считается поперёк "
        "15 ячеек отдельно для каждой комбинации периода, базы и формы, а для пересчётов — "
        "отдельно для каждой базы и метода. «—» означает отсутствие теста (n < 20, константа, "
        "граница или неизвестная единица/налоговая база). p ≤ 0,001 — нижняя граница реализации. "
        "Повторы сумм и работодателей делают p ориентировочными. Класс A и Q1–Q3 не зависят от нормальности. "
        "Месячный пересчёт описывает конкретную часть вакансий с указанной нагрузкой. "
        f"По точечным зарплатам за смену на руки нормальность отвергнута после Холма в "
        f"{shift_rejections} из {shift_tests_run} проверенных ячеек. "
        f"На обоснованном месячном эквиваленте максимум {largest_strict_n} точек на ячейку, "
        "поэтому ни одна из них не дошла до минимального объёма теста. "
        "По сценарным значениям решение о нормальности относится к заданному числу смен и часов.",
        "Пересчёт отчёта из корня репозитория: `python3 utils/build_dataset.py --materialize` и "
        "`python3 Analytics/analyze_priority_salary_formats.py`. Проверено на pandas 2.3.3, NumPy 2.3.3, "
        "SciPy 1.18.1, statsmodels 0.14.6. "
        "Для восстановления именно этого снимка используйте файлы `Data/dataset/manifest.json` и "
        "`Data/dataset/parts/` из commit в шапке.",
        *detail,
    ]
    return "\n\n".join(lines) + "\n"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the checked-in report")
    args = parser.parse_args()
    data = DATASET_PATH.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError(f"Ожидался снимок {DATASET_COMMIT} ({DATASET_SHA256}), получен {digest}")
    part = prepared(pd.read_csv(BytesIO(data), low_memory=False))
    by_segment = part.groupby(SEGMENT_KEYS)
    groups = {" × ".join(key): by_segment.get_group(key) for key in PRIORITY}
    report = priority_report(groups)
    if args.check:
        if REPORT.read_text() != report:
            raise SystemExit("Отчёт отличается от расчётов")
        print("Отчёт совпадает с расчётами")
    else:
        REPORT.write_text(report, encoding="utf-8")
        print(f"Сохранён отчёт для {len(groups)} ячеек")


if __name__ == "__main__":
    main()
