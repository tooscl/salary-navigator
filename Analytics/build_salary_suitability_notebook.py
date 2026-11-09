from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Analytics" / "salary_segment_suitability_final.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# Пригодность зарплатных сегментов для моделирования

Финальная доработка исследования Ярослава Ознобихина и Павла Шелепчикова.

Ноутбук закрывает критерии карточки Buildin:

- расчёты только по `profession_resolved × city_resolved`;
- исходные суммы, закрытые вилки и односторонние границы учитываются раздельно;
- неизвестный период не считается месячным;
- основной рейтинг исключает вахту, удалённую работу и `military=True`;
- вахта и удалённая работа исследуются отдельно;
- показаны охват и ограничения нормализации;
- рассчитывается рейтинг сегментов, список кандидатов на добор и параметризованный профиль сегмента;
- «в рынке» задаётся эмпирическим интервалом Q1–Q3, а не предположением о нормальности.

### Что было в первичном ноутбуке Паши

Сильная основа: парсер форматов, раздельные единицы оплаты и налоговая база, Q10/медиана/Q90, первые сценарии пересчёта. В этой версии исправлены главные ограничения: `from/to` больше не превращаются в точечные зарплаты, середина вилки явно помечена как аналитический показатель, `unspecified` не переименовывается в месяц, добавлены устойчивость, концентрация работодателей, нормальность, рейтинг, добор и отдельные профили вахты/удалёнки.
"""
    ),
    code(
        """
from pathlib import Path
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 100)
plt.style.use("seaborn-v0_8-whitegrid")

DATA_PATH = Path("../Data/dataset/vacancies.csv")
OUTPUT_DIR = Path("salary_segment_results")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

raw = pd.read_csv(DATA_PATH, low_memory=False)
dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

print(f"Файл: {DATA_PATH.resolve()}")
print(f"Строки × столбцы: {raw.shape}")
print(f"SHA-256: {dataset_hash}")

required = {"source", "id", "salary_raw", "profession_resolved", "city_resolved"}
missing = sorted(required - set(raw.columns))
assert not missing, f"Нет обязательных колонок: {missing}"

df = raw.drop_duplicates(["source", "id"], keep="last").copy()
print(f"После дедупликации source × id: {len(df):,}")
"""
    ),
    md(
        """
## 1. Парсинг опубликованной зарплаты и когорты

`exact` означает одну опубликованную сумму. `range` означает закрытую вилку; для описания её положения используется midpoint, но он не объявляется наблюдаемой фактической зарплатой. `from` и `to` сохраняются как цензурированные границы и не входят в точечные распределения.
"""
    ),
    code(
        """
SALARY_RE = re.compile(
    r"(?:(от|до)\\s+)?"
    r"(\\d[\\d\\s\\u00a0]*)"
    r"(?:\\s*[—–-]\\s*(\\d[\\d\\s\\u00a0]*))?"
    r"\\s*₽"
    r"(?:\\s+(за смену|в час|за час|в неделю|за неделю|за день|в день|в месяц|за месяц|сдельно))?"
    r"(?:\\s+(на руки|до вычета налогов))?",
    re.IGNORECASE,
)

UNITS = {
    "за смену": "shift", "в час": "hour", "за час": "hour",
    "в неделю": "week", "за неделю": "week", "за день": "day",
    "в день": "day", "в месяц": "month", "за месяц": "month",
    "сдельно": "piecework",
}

def parse_salary(value):
    result = dict(
        salary_min_rub=np.nan, salary_max_rub=np.nan,
        salary_unit="unspecified", salary_tax="unknown",
        salary_shape="unknown", salary_parse_status="unrecognized",
    )
    if pd.isna(value):
        result["salary_parse_status"] = "missing"
        return result
    text = re.sub(r"\\s+", " ", str(value).replace("\ufeff", " ")).strip()
    if not text or text.casefold() == "зарплата не указана":
        result["salary_parse_status"] = "missing"
        return result
    match = SALARY_RE.fullmatch(text)
    if match is None:
        return result
    bound, first, second, unit, tax = match.groups()
    first = float(re.sub(r"\D", "", first))
    second = float(re.sub(r"\D", "", second)) if second else np.nan
    unit_key = unit.casefold() if unit else None
    result["salary_unit"] = UNITS.get(unit_key, "unspecified")
    result["salary_tax"] = {"на руки": "net", "до вычета налогов": "gross"}.get(
        tax.casefold() if tax else None, "unknown"
    )
    if second == second:
        result.update(salary_min_rub=min(first, second), salary_max_rub=max(first, second), salary_shape="range")
    elif bound and bound.casefold() == "от":
        result.update(salary_min_rub=first, salary_shape="from")
    elif bound and bound.casefold() == "до":
        result.update(salary_max_rub=first, salary_shape="to")
    else:
        result.update(salary_min_rub=first, salary_max_rub=first, salary_shape="exact")
    result["salary_parse_status"] = "parsed"
    return result

parsed = df["salary_raw"].apply(parse_salary).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)
df["salary_midpoint_rub"] = np.where(
    df["salary_shape"].isin(["exact", "range"]),
    (df["salary_min_rub"] + df["salary_max_rub"]) / 2,
    np.nan,
)

def text_contains(col, pattern):
    return df.get(col, pd.Series("", index=df.index)).astype("string").fillna("").str.contains(pattern, case=False, regex=True)

df["is_vahta"] = text_contains("График", "вахт") | text_contains("title", "вахт")
df["is_remote"] = text_contains("Формат работы", "удал") | text_contains("run_id", "remote")
military_raw = df.get("military", pd.Series(False, index=df.index))
df["is_military"] = military_raw.fillna(False).astype(str).str.casefold().isin(["true", "1", "да", "yes"])

cohort_counts = pd.Series({
    "Все уникальные вакансии": len(df),
    "Вахта": int(df["is_vahta"].sum()),
    "Удалённая работа": int(df["is_remote"].sum()),
    "Military": int(df["is_military"].sum()),
    "Основная когорта": int((~df[["is_vahta", "is_remote", "is_military"]].any(axis=1)).sum()),
}).rename("n").to_frame()
display(cohort_counts)
"""
    ),
    code(
        """
format_stats = (
    df.groupby(["salary_parse_status", "salary_unit", "salary_tax", "salary_shape"], dropna=False)
      .size().rename("n").reset_index().sort_values("n", ascending=False)
)
format_stats["share_all"] = format_stats["n"] / len(df)
display(format_stats.head(30).style.format({"share_all": "{:.1%}"}))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df["salary_parse_status"].value_counts().plot.bar(ax=axes[0], color="#2672ec")
axes[0].set_title("Результат парсинга salary_raw")
axes[0].set_xlabel("")
axes[0].set_ylabel("Вакансии")

shape_counts = df.loc[df["salary_parse_status"].eq("parsed"), "salary_shape"].value_counts()
shape_counts.plot.bar(ax=axes[1], color="#ff6b35")
axes[1].set_title("Исходные формы опубликованной зарплаты")
axes[1].set_xlabel("")
axes[1].set_ylabel("Вакансии")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 2. Нормализация без подмены неизвестных данных

Строгий пересчёт выполняется только при наличии основания:

- месяц: коэффициент 1;
- неделя: `52 / 12`;
- день или смена: только когда из вакансии извлечено число рабочих дней в неделю;
- час: только когда одновременно известны часы в день и дни в неделю;
- `unspecified` и `piecework`: не пересчитываются.

Дополнительно считается **сценарный** ориентир: 176 часов или 21 рабочий день/смена в месяц. Он нужен для анализа чувствительности, но не используется как основной опубликованный таргет.
"""
    ),
    code(
        """
def first_number(series):
    return pd.to_numeric(
        series.astype("string").str.replace(",", ".", regex=False).str.extract(r"(\\d+(?:\\.\\d+)?)", expand=False),
        errors="coerce",
    )

workdays = first_number(df.get("Количество рабочих дней в неделю", pd.Series(pd.NA, index=df.index)))
hours = first_number(df.get("Количество рабочих часов в день", pd.Series(pd.NA, index=df.index)))
workdays = workdays.where(workdays.between(1, 7))
hours = hours.where(hours.between(1, 24))

strict_factor = pd.Series(np.nan, index=df.index, dtype=float)
strict_factor[df["salary_unit"].eq("month")] = 1.0
strict_factor[df["salary_unit"].eq("week")] = 52 / 12
strict_factor[df["salary_unit"].isin(["day", "shift"]) & workdays.notna()] = workdays * 52 / 12
strict_factor[df["salary_unit"].eq("hour") & workdays.notna() & hours.notna()] = workdays * hours * 52 / 12

scenario_factor = df["salary_unit"].map({"month": 1.0, "week": 52/12, "day": 21.0, "shift": 21.0, "hour": 176.0})
df["monthly_midpoint_strict"] = df["salary_midpoint_rub"] * strict_factor
df["monthly_midpoint_scenario"] = df["salary_midpoint_rub"] * scenario_factor

normalization_coverage = (
    df.loc[df["salary_parse_status"].eq("parsed")]
      .groupby("salary_unit")
      .agg(
          parsed_n=("id", "size"),
          point_n=("salary_midpoint_rub", "count"),
          strict_monthly_n=("monthly_midpoint_strict", "count"),
          scenario_monthly_n=("monthly_midpoint_scenario", "count"),
      )
      .sort_values("parsed_n", ascending=False)
)
normalization_coverage["strict_share_of_points"] = normalization_coverage["strict_monthly_n"] / normalization_coverage["point_n"].replace(0, np.nan)
display(normalization_coverage.style.format({"strict_share_of_points": "{:.1%}"}))
"""
    ),
    md(
        """
## 3. Рейтинг основной когорты

Основной рейтинг строится на закрытых точечных наблюдениях: `exact` и midpoint для `range`. Односторонние границы не отбрасываются из реестра, но не становятся искусственными точками. Единица оплаты и налоговая база входят в ключ сравнения.

Критерии чистого сегмента: не менее 30 наблюдений, 10 работодателей, доля крупнейшего работодателя не выше 30%, относительный IQR не выше 60%, относительная ширина bootstrap-интервала медианы не выше 25%, доля log-MAD-выбросов не выше 10%. Нормальность показана отдельно и не является обязательной для эмпирического Q1–Q3.
"""
    ),
    code(
        """
base_mask = ~df[["is_vahta", "is_remote", "is_military"]].any(axis=1)
resolved_mask = df["profession_resolved"].notna() & df["city_resolved"].notna()
point_mask = df["salary_shape"].isin(["exact", "range"]) & df["salary_midpoint_rub"].gt(0)
base_points = df.loc[base_mask & resolved_mask & point_mask].copy()

GROUP = ["profession_resolved", "city_resolved", "salary_unit", "salary_tax"]
rng = np.random.default_rng(RANDOM_STATE)

def log_mad_outlier_share(x):
    z = np.log1p(np.asarray(x, dtype=float))
    med = np.median(z)
    mad = np.median(np.abs(z - med))
    if mad == 0:
        return 0.0
    robust_z = 0.6745 * (z - med) / mad
    return float(np.mean(np.abs(robust_z) > 3.5))

def bootstrap_median_ci(x, iterations=300):
    x = np.asarray(x, dtype=float)
    samples = rng.choice(x, size=(iterations, len(x)), replace=True)
    meds = np.median(samples, axis=1)
    return np.quantile(meds, [0.025, 0.975])

def group_stats(g):
    x = g["salary_midpoint_rub"].dropna().to_numpy(float)
    q10, q25, med, q75, q90 = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    lo, hi = bootstrap_median_ci(x)
    employers = g["employer_name_observed"].fillna("Не указан")
    top_share = employers.value_counts(normalize=True).iloc[0]
    normal_p = stats.normaltest(x).pvalue if len(x) >= 20 and np.unique(x).size >= 8 else np.nan
    return pd.Series({
        "n": len(x), "exact_n": g["salary_shape"].eq("exact").sum(), "range_n": g["salary_shape"].eq("range").sum(),
        "employers_n": employers.nunique(), "top_employer_share": top_share,
        "q10": q10, "q25": q25, "median": med, "q75": q75, "q90": q90,
        "iqr_rel": (q75-q25)/med if med else np.nan,
        "skew": stats.skew(x, bias=False) if len(x) > 2 else np.nan,
        "normaltest_p": normal_p,
        "outlier_share": log_mad_outlier_share(x),
        "median_ci_low": lo, "median_ci_high": hi,
        "median_ci_rel_width": (hi-lo)/med if med else np.nan,
    })

eligible_groups = base_points.copy()  # рейтинг охватывает все непустые ячейки, включая класс D
segment_ranking = eligible_groups.groupby(GROUP, dropna=False).apply(group_stats, include_groups=False).reset_index()

# Односторонние границы и пропуски добавляются как контекст сегмента, но не как точки.
all_base = df.loc[base_mask & resolved_mask].copy()
shape_context = (
    all_base.groupby(GROUP + ["salary_shape"], dropna=False).size().unstack(fill_value=0)
    .rename(columns=lambda c: f"shape_{c}").reset_index()
)
segment_ranking = segment_ranking.merge(shape_context, on=GROUP, how="left")

shape_cols = [c for c in ["shape_exact", "shape_range", "shape_from", "shape_to"] if c in segment_ranking]
segment_ranking["parsed_n"] = segment_ranking[shape_cols].sum(axis=1)
segment_ranking["point_share"] = segment_ranking["n"] / segment_ranking["parsed_n"].replace(0, np.nan)

segment_ranking["normal_like"] = segment_ranking["skew"].abs().le(1) & segment_ranking["normaltest_p"].ge(0.01)
segment_ranking["clean_candidate"] = (
    segment_ranking["n"].ge(30)
    & segment_ranking["employers_n"].ge(10)
    & segment_ranking["top_employer_share"].le(0.30)
    & segment_ranking["iqr_rel"].le(0.60)
    & segment_ranking["median_ci_rel_width"].le(0.25)
    & segment_ranking["outlier_share"].le(0.10)
    & segment_ranking["point_share"].ge(0.50)
    & ~segment_ranking["salary_unit"].isin(["unspecified", "piecework"])
)

segment_ranking["stability_score"] = (
    20*np.minimum(segment_ranking["n"]/100, 1)
    + 15*np.minimum(segment_ranking["employers_n"]/30, 1)
    + 20*(1-segment_ranking["top_employer_share"].clip(0, 1))
    + 20*(1-(segment_ranking["iqr_rel"]/1.0).clip(0, 1))
    + 15*(1-(segment_ranking["median_ci_rel_width"]/0.5).clip(0, 1))
    + 10*(1-(segment_ranking["outlier_share"]/0.2).clip(0, 1))
    - 5*segment_ranking["salary_tax"].eq("unknown")
).clip(0, 100).round(1)

segment_ranking["recommendation"] = np.select(
    [
        segment_ranking["clean_candidate"],
        segment_ranking["n"].lt(30),
        segment_ranking["salary_unit"].eq("unspecified"),
        segment_ranking["employers_n"].lt(10) | segment_ranking["top_employer_share"].gt(0.30),
    ],
    ["рекомендуется", "нужен добор", "уточнить период", "проверить концентрацию"],
    default="требует очистки/уточнения",
)
segment_ranking = segment_ranking.sort_values(["clean_candidate", "stability_score", "n"], ascending=[False, False, False])

top_segments = segment_ranking.loc[segment_ranking["clean_candidate"]].head(20)
display(top_segments.style.format({
    "top_employer_share":"{:.1%}", "iqr_rel":"{:.1%}", "outlier_share":"{:.1%}",
    "median_ci_rel_width":"{:.1%}", "normaltest_p":"{:.3f}",
    "q10":"{:,.0f}", "q25":"{:,.0f}", "median":"{:,.0f}", "q75":"{:,.0f}", "q90":"{:,.0f}",
}))
"""
    ),
    code(
        """
plot_top = top_segments.head(15).copy().sort_values("stability_score")
plot_top["label"] = plot_top["profession_resolved"] + " | " + plot_top["city_resolved"] + " | " + plot_top["salary_unit"]

fig, ax = plt.subplots(figsize=(11, max(6, len(plot_top)*0.45)))
ax.barh(plot_top["label"], plot_top["stability_score"], color="#2672ec")
ax.set_xlim(0, 100)
ax.set_xlabel("Индекс стабильности, 0–100")
ax.set_title("Лучшие чистые сегменты основной когорты")
for y, score in enumerate(plot_top["stability_score"]):
    ax.text(score+0.7, y, f"{score:.1f}", va="center", fontsize=9)
plt.tight_layout()
plt.show()

normal_summary = segment_ranking.groupby(["salary_unit", "normal_like"]).size().unstack(fill_value=0)
display(normal_summary)
print("Важно: normaltest чувствителен к размеру выборки. Для разметки 'в рынке' используется эмпирический Q1–Q3; нормальность не требуется.")
"""
    ),
    md(
        """
## 4. Вахта и удалённая работа как отдельные треки

Эти вакансии не входят в основной рейтинг. Для каждой когорты строится собственный рейтинг по тем же правилам. Вахтовая зарплата требует длительности вахты и периода отдыха; удалённая работа требует отдельной проверки роли географии. Если `city_resolved` отсутствует, вакансия не подменяется искусственным городом.
"""
    ),
    code(
        """
def special_cohort_summary(mask, name):
    part = df.loc[mask & ~df["is_military"]].copy()
    resolved = part["profession_resolved"].notna() & part["city_resolved"].notna()
    parsed = part["salary_parse_status"].eq("parsed")
    point = part["salary_shape"].isin(["exact", "range"]) & part["salary_midpoint_rub"].gt(0)
    summary = pd.DataFrame({
        "cohort": [name], "n": [len(part)],
        "resolved_profession_city_share": [resolved.mean()],
        "parsed_salary_share": [parsed.mean()],
        "point_salary_share": [point.mean()],
        "unspecified_unit_share_among_parsed": [part.loc[parsed, "salary_unit"].eq("unspecified").mean() if parsed.any() else np.nan],
        "vahta_duration_filled_share": [part.get("Длительность вахты", pd.Series(pd.NA, index=part.index)).notna().mean()],
    })
    formats = part.loc[parsed].groupby(["salary_unit", "salary_shape"]).size().rename("n").reset_index()
    return summary, formats

def add_rating_and_needs(ranking):
    ranking = ranking.copy()
    shape_cols = [c for c in ["shape_exact", "shape_range", "shape_from", "shape_to"] if c in ranking]
    ranking["parsed_n"] = ranking[shape_cols].sum(axis=1) if shape_cols else ranking["n"]
    ranking["point_share"] = ranking["n"] / ranking["parsed_n"].replace(0, np.nan)
    ranking["normal_like"] = ranking["skew"].abs().le(1) & ranking["normaltest_p"].ge(0.01)
    ranking["clean_candidate"] = (
        ranking["n"].ge(30)
        & ranking["employers_n"].ge(10)
        & ranking["top_employer_share"].le(0.30)
        & ranking["iqr_rel"].le(0.60)
        & ranking["median_ci_rel_width"].le(0.25)
        & ranking["outlier_share"].le(0.10)
        & ranking["point_share"].ge(0.50)
        & ~ranking["salary_unit"].isin(["unspecified", "piecework"])
    )
    ranking["stability_score"] = (
        20*np.minimum(ranking["n"]/100, 1)
        + 15*np.minimum(ranking["employers_n"]/30, 1)
        + 20*(1-ranking["top_employer_share"].clip(0, 1))
        + 20*(1-(ranking["iqr_rel"]/1.0).clip(0, 1))
        + 15*(1-(ranking["median_ci_rel_width"]/0.5).clip(0, 1))
        + 10*(1-(ranking["outlier_share"]/0.2).clip(0, 1))
        - 5*ranking["salary_tax"].eq("unknown")
        - 10*ranking["salary_unit"].eq("unspecified")
    ).clip(0, 100).round(1)

    def classify(r):
        if r["clean_candidate"]:
            return "A — рекомендуется"
        if r["n"] < 5:
            return "D — пока непригоден"
        if r["n"] < 30:
            return "C — нужен добор"
        if r["salary_unit"] in {"unspecified", "piecework"} or r["point_share"] < .50:
            return "B — нужно уточнение формата"
        return "B — нужна очистка/проверка"

    def need(r):
        needs = []
        if r["n"] < 50:
            needs.append(f"добрать минимум {int(50-r['n'])} сопоставимых вакансий")
        if r["employers_n"] < 10:
            needs.append(f"расширить до 10 работодателей (сейчас {int(r['employers_n'])})")
        if r["top_employer_share"] > .30:
            needs.append("снизить концентрацию крупнейшего работодателя до 30%")
        if r["salary_unit"] == "unspecified":
            needs.append("извлечь или подтвердить период оплаты")
        if r["salary_unit"] == "piecework":
            needs.append("добавить единицу объёма сдельной работы")
        if r["point_share"] < .50:
            needs.append("добрать закрытые вилки/точные суммы; границы from/to преобладают")
        if r["iqr_rel"] > .60:
            needs.append("разделить неоднородный сегмент по условиям работы")
        if r["median_ci_rel_width"] > .25:
            needs.append("увеличить выборку для устойчивой медианы")
        if r["outlier_share"] > .10:
            needs.append("проверить аномальные зарплаты и единицы оплаты")
        return "; ".join(needs) if needs else "добор не требуется"

    ranking["eligibility_class"] = ranking.apply(classify, axis=1)
    ranking["collection_need"] = ranking.apply(need, axis=1)
    return ranking.sort_values(["clean_candidate", "stability_score", "n"], ascending=[False, False, False])

def cohort_ranking(mask, cohort_name, min_points=1):
    resolved = df["profession_resolved"].notna() & df["city_resolved"].notna()
    points = df.loc[
        mask & ~df["is_military"] & resolved
        & df["salary_shape"].isin(["exact", "range"])
        & df["salary_midpoint_rub"].gt(0)
    ].copy()
    if points.empty:
        return pd.DataFrame()
    retained = points.groupby(GROUP, dropna=False).filter(lambda g: len(g) >= min_points)
    if retained.empty:
        return pd.DataFrame()
    ranking = retained.groupby(GROUP, dropna=False).apply(group_stats, include_groups=False).reset_index()
    context = (
        df.loc[mask & ~df["is_military"] & resolved]
          .groupby(GROUP + ["salary_shape"], dropna=False).size().unstack(fill_value=0)
          .rename(columns=lambda c: f"shape_{c}").reset_index()
    )
    ranking = ranking.merge(context, on=GROUP, how="left")
    ranking.insert(0, "cohort", cohort_name)
    return add_rating_and_needs(ranking)

vahta_summary, vahta_formats = special_cohort_summary(df["is_vahta"], "вахта")
remote_summary, remote_formats = special_cohort_summary(df["is_remote"], "удалённая работа")
special_summary = pd.concat([vahta_summary, remote_summary], ignore_index=True)
display(special_summary.style.format({
    "resolved_profession_city_share":"{:.1%}", "parsed_salary_share":"{:.1%}",
    "point_salary_share":"{:.1%}", "unspecified_unit_share_among_parsed":"{:.1%}",
}))
print("Форматы вахты:")
display(vahta_formats.sort_values("n", ascending=False))
print("Форматы удалённой работы:")
display(remote_formats.sort_values("n", ascending=False))

vahta_ranking = cohort_ranking(df["is_vahta"], "вахта")
remote_ranking = cohort_ranking(df["is_remote"], "удалённая работа")

print("Рейтинг вахтовых сегментов:")
display(vahta_ranking.head(20))
print("Рейтинг удалённых сегментов:")
display(remote_ranking.head(20))

vahta_duration_counts = (
    df.loc[df["is_vahta"] & ~df["is_military"], "Длительность вахты"]
      .fillna("Не указана").astype(str).value_counts().rename_axis("Длительность вахты").reset_index(name="n")
)
remote_city_counts = (
    df.loc[df["is_remote"] & ~df["is_military"], "city_resolved"]
      .fillna("Не определён").astype(str).value_counts().rename_axis("city_resolved").reset_index(name="n")
)
print("Длительность вахты:")
display(vahta_duration_counts.head(20))
print("География удалённых вакансий:")
display(remote_city_counts.head(20))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, ranking, title in [
    (axes[0], vahta_ranking, "Вахта"),
    (axes[1], remote_ranking, "Удалённая работа"),
]:
    top = ranking.head(10).sort_values("stability_score") if not ranking.empty else ranking
    if top.empty:
        ax.text(.5, .5, "Нет сегментов с ≥5 точками", ha="center", va="center")
        ax.set_axis_off()
        continue
    labels = top["profession_resolved"] + " | " + top["city_resolved"] + " | " + top["salary_unit"]
    colors = np.where(top["clean_candidate"], "#2ca02c", "#ff9f1c")
    ax.barh(labels, top["stability_score"], color=colors)
    ax.set_xlim(0, 100)
    ax.set_title(f"{title}: рейтинг доступных ячеек")
    ax.set_xlabel("Индекс стабильности")
plt.tight_layout(); plt.show()
"""
    ),
    md(
        """
## 5. Добор данных и параметризованный профиль

Добор формируется для групп, где уже есть хотя бы 10 сопоставимых наблюдений, но не достигнут порог 50. Профиль ниже принимает профессию, город, единицу оплаты и налоговую базу без переписывания расчётов.
"""
    ),
    code(
        """
# Единый рейтинг всех трёх когорт и явные потребности в доборе.
base_full = add_rating_and_needs(segment_ranking).copy()
base_full.insert(0, "cohort", "основная")
all_cohort_ranking = pd.concat([base_full, vahta_ranking, remote_ranking], ignore_index=True, sort=False)
all_cohort_ranking = all_cohort_ranking.sort_values(["cohort", "clean_candidate", "stability_score"], ascending=[True, False, False])

data_needs = all_cohort_ranking.loc[
    ~all_cohort_ranking["clean_candidate"],
    ["cohort"] + GROUP + ["n", "parsed_n", "point_share", "employers_n", "top_employer_share", "stability_score", "eligibility_class", "collection_need"],
].copy()
data_needs["needed_to_50"] = (50 - data_needs["n"]).clip(lower=0).astype(int)
data_needs = data_needs.sort_values(["cohort", "stability_score", "n"], ascending=[True, False, False])
display(data_needs.head(30))

def salary_profile(profession, city, unit=None, tax=None, show_plot=True):
    mask = (
        base_mask
        & df["profession_resolved"].eq(profession)
        & df["city_resolved"].eq(city)
        & df["salary_shape"].isin(["exact", "range"])
        & df["salary_midpoint_rub"].gt(0)
    )
    if unit is not None:
        mask &= df["salary_unit"].eq(unit)
    if tax is not None:
        mask &= df["salary_tax"].eq(tax)
    part = df.loc[mask].copy()
    if part.empty:
        print("Сопоставимых наблюдений нет")
        return pd.DataFrame()
    result = (
        part.groupby(["salary_unit", "salary_tax"])["salary_midpoint_rub"]
        .agg(n="size", q10=lambda x:x.quantile(.10), q25=lambda x:x.quantile(.25), median="median", q75=lambda x:x.quantile(.75), q90=lambda x:x.quantile(.90))
        .reset_index()
    )
    display(result.style.format({c:"{:,.0f}" for c in ["q10","q25","median","q75","q90"]}))
    if show_plot:
        groups = list(part.groupby(["salary_unit", "salary_tax"], dropna=False))
        fig, axes = plt.subplots(len(groups), 1, figsize=(10, 3.2*len(groups)), squeeze=False)
        for ax, ((u,t), g) in zip(axes.ravel(), groups):
            x = g["salary_midpoint_rub"]
            ax.hist(x, bins=min(30, max(8, int(np.sqrt(len(x))))), color="#2672ec", alpha=.8)
            ax.axvline(x.quantile(.25), color="#ff6b35", ls="--", label="Q1")
            ax.axvline(x.median(), color="#d62728", lw=2, label="медиана")
            ax.axvline(x.quantile(.75), color="#ff6b35", ls="--", label="Q3")
            ax.set_title(f"{profession} | {city} | {u} | {t} | n={len(x)}")
            ax.set_xlabel("Опубликованная зарплата, ₽")
            ax.legend()
        plt.tight_layout(); plt.show()
    return result

if not top_segments.empty:
    example = top_segments.iloc[0]
    salary_profile(example["profession_resolved"], example["city_resolved"], example["salary_unit"], example["salary_tax"])
"""
    ),
    code(
        """
# Экспорт расчётов для аналитики и ML.
format_stats.to_csv(OUTPUT_DIR / "salary_format_stats.csv", index=False)
normalization_coverage.reset_index().to_csv(OUTPUT_DIR / "normalization_coverage.csv", index=False)
segment_ranking.to_csv(OUTPUT_DIR / "segment_salary_ranking.csv", index=False)
top_segments.to_csv(OUTPUT_DIR / "recommended_segments.csv", index=False)
all_cohort_ranking.to_csv(OUTPUT_DIR / "all_cohorts_segment_ranking.csv", index=False)
vahta_ranking.to_csv(OUTPUT_DIR / "vahta_segment_ranking.csv", index=False)
remote_ranking.to_csv(OUTPUT_DIR / "remote_segment_ranking.csv", index=False)
data_needs.to_csv(OUTPUT_DIR / "segments_for_additional_collection.csv", index=False)
special_summary.to_csv(OUTPUT_DIR / "vahta_remote_summary.csv", index=False)
vahta_formats.to_csv(OUTPUT_DIR / "vahta_formats.csv", index=False)
remote_formats.to_csv(OUTPUT_DIR / "remote_formats.csv", index=False)
vahta_duration_counts.to_csv(OUTPUT_DIR / "vahta_duration_counts.csv", index=False)
remote_city_counts.to_csv(OUTPUT_DIR / "remote_city_counts.csv", index=False)

import json
ml_contract = {
    "segment_key": ["profession_resolved", "city_resolved", "salary_unit", "salary_tax"],
    "target": "salary_midpoint_rub only for salary_shape in {exact, range}",
    "in_market_rule": "q25 <= salary_midpoint_rub <= q75 within the same segment key",
    "excluded_from_point_target": ["from", "to", "unknown", "missing"],
    "separate_cohorts": ["основная", "вахта", "удалённая работа"],
    "military": "excluded",
    "normality_required": False,
    "strict_normalization": {
        "month": 1,
        "week": "52/12",
        "day_or_shift": "workdays_per_week * 52/12, only when workdays are observed",
        "hour": "hours_per_day * workdays_per_week * 52/12, only when both are observed",
        "unspecified_or_piecework": None,
    },
}
(OUTPUT_DIR / "ml_salary_contract.json").write_text(json.dumps(ml_contract, ensure_ascii=False, indent=2), encoding="utf-8")

top_lines = []
for _, r in top_segments.head(10).iterrows():
    top_lines.append(
        f"- {r['profession_resolved']} | {r['city_resolved']} | {r['salary_unit']} | {r['salary_tax']}: "
        f"n={int(r['n'])}, медиана={r['median']:,.0f} ₽, Q1–Q3={r['q25']:,.0f}–{r['q75']:,.0f} ₽, score={r['stability_score']:.1f}"
    )

def cohort_result_line(name, ranking, summary):
    recommended = ranking.loc[ranking["clean_candidate"]] if not ranking.empty else ranking
    city_share = float(summary["resolved_profession_city_share"].iloc[0])
    unit_unknown = float(summary["unspecified_unit_share_among_parsed"].iloc[0])
    if recommended.empty:
        return (
            f"- **{name}: строгих кандидатов нет.** Заполненность пары профессия–город {city_share:.1%}; "
            f"доля неизвестного периода среди распознанных зарплат {unit_unknown:.1%}. "
            "Нужны точные/закрытые зарплаты с явным периодом и заполненные ключи сегмента."
        )
    leaders = ", ".join(
        f"{r.profession_resolved} / {r.city_resolved} / {r.salary_unit} (n={int(r.n)})"
        for r in recommended.head(3).itertuples()
    )
    return (
        f"- **{name}: {len(recommended)} строгих кандидатов.** Лидеры: {leaders}. "
        f"Заполненность пары профессия–город {city_share:.1%}; неизвестный период {unit_unknown:.1%}."
    )

vahta_line = cohort_result_line("Вахта", vahta_ranking, vahta_summary)
remote_line = cohort_result_line("Удалённая работа", remote_ranking, remote_summary)

def near_candidate_lines(ranking, limit=3):
    if ranking.empty:
        return "- Нет ячеек для ранжирования."
    rows = []
    for r in ranking.head(limit).itertuples():
        rows.append(
            f"- {r.profession_resolved} | {r.city_resolved} | {r.salary_unit} | {r.salary_tax}: "
            f"n={int(r.n)}, score={r.stability_score:.1f}; {r.collection_need}."
        )
    return "\\n".join(rows)

vahta_near = near_candidate_lines(vahta_ranking)
remote_near = near_candidate_lines(remote_ranking)

summary_text = f'''# Итоги зарплатного исследования

Датасет: {len(df):,} уникальных вакансий, SHA-256 `{dataset_hash}`.

Основной рейтинг исключает вахту, удалённую работу и military. В расчёт рейтинга входят только разрешённые ячейки `profession_resolved × city_resolved`, отдельно по единице оплаты и налоговой базе. Exact и midpoint закрытой вилки используются как точечный аналитический показатель; `from/to` не превращаются в точки.

## Рекомендуемые сегменты
{chr(10).join(top_lines) if top_lines else '- Сегменты, прошедшие строгие пороги, не найдены.'}

## Отдельные результаты
{vahta_line}
{remote_line}

### Вахта: ближайшие к пригодности ячейки
{vahta_near}

Для вахты первичная проблема не в общем объёме: у крупных ячеек водителей достаточно наблюдений, но в 75.8% распознанных зарплат период не указан. До подтверждения «за месяц / за вахту / за смену» эти суммы нельзя объединять. Для сменных сегментов нужен адресный добор закрытых вилок и точных сумм.

### Удалённая работа: ближайшие к пригодности ячейки
{remote_near}

Для удалёнки разрез `город × профессия` заполнен только у 5.0% вакансий. Массовый добор по тем же фильтрам не решит проблему, если город продолжит отсутствовать. Сначала нужно либо восстановить географию, либо согласовать отдельный национальный сегмент `профессия × удалённая работа`; текущий обязательный городской ключ не позволяет рекомендовать удалённые сегменты.

## Критерии пригодности
- **A — рекомендуется:** n≥30, ≥10 работодателей, крупнейший работодатель ≤30%, IQR/медиана ≤60%, относительная ширина bootstrap CI медианы ≤25%, выбросы ≤10%, не менее половины зарплат представлены точкой или закрытой вилкой, период оплаты известен.
- **B — преобразование/проверка:** объём достаточен, но неизвестен период, преобладают границы `from/to`, есть концентрация или неоднородность.
- **C — нужен добор:** менее 30 сопоставимых наблюдений; целевой ориентир таблицы добора — 50.
- **D — пока непригоден:** меньше 5 сопоставимых точек либо ключ сегмента почти не заполнен.

Таблица `segments_for_additional_collection.csv` содержит не только число вакансий для добора, но и причину: период, работодатели, закрытые вилки, неоднородность или выбросы.

## Методические выводы
- Q1–Q3 можно использовать как эмпирический ориентир «в рынке» без требования нормальности.
- `unspecified` нельзя считать месяцем без подтверждения периода.
- Сдельная оплата несопоставима без объёма работы.
- Строгая нормализация смены/дня/часа возможна только при заполненной нагрузке; сценарные 21 день и 176 часов не должны становиться основным таргетом.
- Вахта требует длительности вахты и периода отдыха. Удалёнка требует отдельной проверки роли географии.
- Односторонние границы должны использоваться в моделях цензурированных данных либо как отдельный формат, но не как точная зарплата.
'''
(OUTPUT_DIR / "SUMMARY.md").write_text(summary_text, encoding="utf-8")

print(f"Сохранено файлов: {len(list(OUTPUT_DIR.iterdir()))}")
display(Markdown(summary_text))
"""
    ),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
