"""Shared preparation, quantile fitting and inference for the priority-segment study."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Analytics"))

from analyze_priority_salary_formats import (  # noqa: E402
    COMPARABLE_UNITS, EMPLOYER_UNKNOWN, KNOWN_TAX, additional_points, clean,
    pass_metrics, salary_frame, unique_rotations,
)
from build_priority_segment_passports import CORE, ROLE_FIELDS  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from segment_normality import PRIORITY, priority_vacancies  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from utils.build_dataset import assemble  # noqa: E402

spec = importlib.util.spec_from_file_location("ssrtatarin", ROOT / "ML/experiments/expiriemnts.py")
ssr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssr)

ALPHAS = np.array([.25, .50, .75])
KEY = ["profession_resolved", "city_resolved", "unit", "tax"]
CASHIER = ("Кассир", "Санкт-Петербург")
SELLER = ("Продавец", "Санкт-Петербург")
PICKER = ("Комплектовщик", "Санкт-Петербург")
MOSCOW_PICKER = ("Комплектовщик", "Москва")
FIRST = [CASHIER, SELLER, PICKER]
UNKNOWN_EMPLOYER = set(EMPLOYER_UNKNOWN) | {"работодатель", "не указано", "неизвестно"}
CORE_FEATURES = KEY + list(CORE)
ADMIN = ["Способ оформления", "Оформление"]
WORKLOAD = ["Количество рабочих часов в день", "Количество рабочих дней в неделю", "Рабочих дней (строго)", "Часов (строго)"]
DETAIL = list(dict.fromkeys(CORE_FEATURES + WORKLOAD + [
    "Формат работы", "Тип предприятия", "Смены", "Регистрация", *ADMIN, *[
        field for fields in ROLE_FIELDS.values() for field in fields
    ],
]))
CHEWBY = KEY + [
    "График", "Занятость", "Опыт работы", "Количество рабочих часов в день",
    "Количество рабочих дней в неделю", "Формат работы", "Сфера деятельности компании",
    "Тип предприятия", "Тип работ", "Ассортимент", "Способ доставки", "Автомобиль",
]
TEXT = ["title", "description", "Что получают работники"]


def load_snapshot():
    """Use the canonical CSV or reconstruct its exact bytes from verified parts."""
    directory = ROOT / "Data/dataset"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    parts = {entry["path"]: entry["sha256"] for entry in manifest["parts"]}

    def read_part(name):
        data = (directory / name).read_bytes()
        return data if hashlib.sha256(data).hexdigest() == parts[name] else data.replace(b"\r\n", b"\n")

    location = directory / "vacancies.csv"
    data = location.read_bytes() if location.is_file() else b""
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        data = assemble(manifest, read_part)
    raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    assert not raw.duplicated(["source", "id"]).any()
    return raw, manifest["sha256"]


def make_groups(raw):
    """Reuse the grouping logic from PR #113 and resolve aliases before hashing offers."""
    notebook = json.loads((ROOT / "ML/korchagin_salary_research.ipynb").read_text(encoding="utf-8"))
    nodes = [node for cell in (3, 6) for node in ast.parse("".join(notebook["cells"][cell]["source"])).body
             if isinstance(node, ast.FunctionDef) and node.name in {"clean", "assign_groups"}]
    assert len(nodes) == 2
    ns = dict(pd=pd, re=re, hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "PR113 grouping", "exec"), ns)
    g = raw[["source", "id", "title", "description", "employer_name_observed", "employer_profile_url"]].copy()
    g["city"] = raw.city_resolved
    g["profession"] = raw.profession_resolved
    names = clean(g.employer_name_observed).str.casefold()
    g["employer_name_observed"] = g.employer_name_observed.mask(names.isin(UNKNOWN_EMPLOYER), "")
    return ns["assign_groups"](g, 42).split_group


def add_workload(frame):
    """Derive the analysts’ strict workload from canonical input columns."""
    frame = frame.copy()
    for name in ("Количество рабочих дней в неделю", "Количество рабочих часов в день", "Смены", "Режим работы"):
        if name not in frame:
            frame[name] = ""
    days = clean(frame["Количество рабочих дней в неделю"])
    exact = pd.to_numeric(days, errors="coerce").astype(float)
    exact = exact.where(exact.between(1, 7) & exact.mod(1).eq(0))
    rotation = unique_rotations(frame).astype(float)
    bounds = days.str.extract(r"^(\d+)\s*[-–—]\s*(\d+)$").apply(pd.to_numeric)
    consistent = days.isna() | exact.eq(rotation) | (rotation.ge(bounds[0]) & rotation.le(bounds[1]))
    schedule = clean(frame["Смены"]).notna() | clean(frame["Режим работы"]).notna()
    exact = exact.where(~schedule | exact.eq(rotation))
    days = exact.combine_first(rotation.where(consistent))
    hours = pd.to_numeric(clean(frame["Количество рабочих часов в день"]), errors="coerce").astype(float)
    hours = hours.where(hours.between(1, 24))
    frame["Рабочих дней (строго)"] = days
    frame["Часов (строго)"] = hours
    return frame


def native_salary(raw):
    """Parse explicit salary terms and derive observed-load conversion factors."""
    frame = add_workload(raw.join(salary_frame(raw.salary_raw)))
    days, hours = frame["Рабочих дней (строго)"], frame["Часов (строго)"]
    factor = frame.unit.map({"month": 1., "week": 52 / 12}).astype(float)
    factor = factor.mask(frame.unit.isin(["day", "shift"]), days * 52 / 12)
    factor = factor.mask(frame.unit.eq("hour"), hours * days * 52 / 12)
    frame["factor"] = factor
    frame["y"] = ((frame.lower + frame.upper) / 2).where(frame["shape"].isin(["exact", "range"]))
    frame["y"] = frame.y.where(frame.y.gt(0))
    return frame


def prepare(raw, *, assign_folds=True):
    gold_indices = priority_vacancies(raw).index
    raw = raw.assign(group=make_groups(raw))
    raw = raw[raw.source.eq("avito") & raw.analysis_status.eq("validated") & raw.export_eligibility.eq("included")]
    watch = (raw["График"] + " " + raw.title).str.contains("вахт", case=False)
    remote = (raw["Формат работы"] + " " + raw.run_id).str.contains("удал|remote", case=False)
    military = raw.military.str.casefold().isin(["true", "1", "yes", "да"])
    # Use exactly the analysts' cohort for gold. The remainder has the same
    # principal exclusions and resolved city/profession.
    primary = ~(watch | remote | military) & raw.city_resolved.ne("") & raw.profession_resolved.ne("")
    raw = raw.loc[primary | raw.index.isin(gold_indices)].copy()
    data = native_salary(raw)
    data["priority"] = data.index.isin(gold_indices)
    data["employer"] = clean(data.employer_name_observed).str.casefold()
    data["employer"] = data.employer.mask(data.employer.isin(UNKNOWN_EMPLOYER))
    data["date"] = pd.to_datetime(data.first_seen_at.where(data.first_seen_at.ne(""), data.observed_at),
                                  utc=True, errors="coerce").dt.floor("D")
    data["evaluation_fold"] = -1
    pool = data.y.notna() & data.tax.isin(KNOWN_TAX) & data.unit.isin(COMPARABLE_UNITS)
    # Freeze fold assignments across pay units and all experiment branches.
    rows = data.loc[pool].sort_values(["source", "id"])
    if assign_folds:
        for fold, (_, test) in enumerate(GroupKFold(n_splits=3).split(rows, groups=rows.group)):
            data.loc[rows.iloc[test].index, "evaluation_fold"] = fold
    return data


def cell(frame, key):
    role, city = key
    return frame.profession_resolved.eq(role) & frame.city_resolved.eq(city) & frame.priority


def is_flag(value, pattern):
    return value.astype("string").str.contains(pattern, case=False, regex=True, na=False)


def contract_listed(frame):
    return is_flag(frame.get("Способ оформления", pd.Series("", index=frame.index)), r"трудов")


def subgroup_masks(d):
    hours = clean(d["Количество рабочих часов в день"]).str.replace(r"\s+", "", regex=True)
    return {
        "cashier_spb_permanent": cell(d, CASHIER) & d["Занятость"].eq("Постоянная"),
        "cashier_spb_medical_help": cell(d, CASHIER) & d["Медкнижка"].eq("Поможем оформить"),
        "seller_spb_labor_contract": cell(d, SELLER) & contract_listed(d),
        "picker_spb_self_employed_only": cell(d, PICKER) & d["Способ оформления"].astype("string").str.strip().str.casefold().isin(["договор гпх с самозанятым", "гпх с самозанятым"]),
        "picker_moscow_hours_11_12": cell(d, MOSCOW_PICKER) & hours.str.fullmatch(r"11[-–—]12", na=False),
    }


def split(d, fold):
    test = d.evaluation_fold.eq(fold)
    bucket = d.group.map(lambda g: int(hashlib.sha256(f"{fold}:{g}".encode()).hexdigest()[:8], 16) % 100)
    val = (~test) & bucket.lt(15)
    return d.loc[~(test | val)], d.loc[val], d.loc[test]


def matrix(d, columns):
    unsafe = r"₽|\bруб\w*|зарплат|оклад|\d{4,}"
    result = {}
    for c in columns:
        s = d[c] if c in d else pd.Series("", index=d.index)
        s = s.astype("string").fillna("").str.strip().replace("", "__MISSING__")
        result[c] = s.mask(s.str.contains(unsafe, case=False, regex=True), "__REDACTED__").astype(str)
    return pd.DataFrame(result, index=d.index)


def texts(d):
    # The publication target can occur in any free-text field. Remove salary
    # fragments, then remove every remaining numeric expression.
    parts = d.reindex(columns=TEXT).fillna("").astype(str)
    joined = parts.agg("\n".join, axis=1).map(ssr.delete_salary)
    joined = joined.str.replace(r"\d+(?:[.,\u00a0\u202f ]\d+)*\s*[кk]?(?=\W|$)", " ", regex=True, flags=re.I)
    return joined.str.replace(r"\d+", " ", regex=True)


def make_frame(d, bundle):
    x = matrix(d, bundle["columns"])
    if bundle.get("text_encoder") is not None:
        vectorizer, svd = bundle["text_encoder"]
        projected = svd.transform(vectorizer.transform(texts(d)))
        for i in range(projected.shape[1]):
            x[f"svd_{i:02}"] = projected[:, i]
    return x


def reference_table(train, *, quantiles=ALPHAS):
    """Empirical quantiles in matching unit and tax; hierarchical fallback."""
    levels = [KEY, ["profession_resolved", "unit", "tax"], ["unit", "tax"]]
    tables = []
    for keys in levels:
        table = {}
        for key, block in train.groupby(keys, sort=True):
            if len(block) >= (5 if keys == KEY else 3) and block.group.nunique() >= 3:
                table[key] = block.label.quantile(quantiles).to_numpy(dtype=float)
        tables.append((keys, table))
    return tables


def reference(d, tables):
    values, sizes = [], []
    shape = next((len(v) for _, lookup in tables for v in lookup.values()), len(ALPHAS))
    for row in d[KEY].to_dict("records"):
        for i, (cols, lookup) in enumerate(tables):
            estimate = lookup.get(tuple(row[c] for c in cols))
            if estimate is not None:
                values.append(estimate)
                sizes.append(i)
                break
        else:
            values.append(np.full(shape, np.nan))
            sizes.append(-1)
    return np.asarray(values).reshape(len(d), shape), np.asarray(sizes)


def grouped_anchors(train):
    """Cross-fit the denominator; a salary does not contribute to its own scale."""
    result = pd.Series(index=train.index, dtype=float)
    for a, b in GroupKFold(n_splits=3).split(train, groups=train.group):
        inner = reference_table(train.iloc[a], quantiles=[.5])
        z, _ = reference(train.iloc[b], inner)
        result.loc[train.iloc[b].index] = z[:, 0]
    return result


def fit(train, val, *, method, columns, text=False, relative=False, iterations=250):
    bundle = {"method": method, "columns": columns, "text_encoder": None, "scale": None, "shifts": None}
    if text:
        bundle["text_encoder"] = ssr.fit_text_encoder(texts(train), 32)
    if relative:
        bundle["scale"] = reference_table(train, quantiles=[.5])
        denominator = grouped_anchors(train)
        train = train.loc[denominator.notna()]
        label = train.label / denominator.loc[train.index]
        val_scale, _ = reference(val, bundle["scale"])
        val_target = val.label / val_scale[:, 0]
    else:
        label, val_target = train.label, val.label
    x, v = make_frame(train, bundle), make_frame(val, bundle)
    if method == "gbr":
        bundle["models"] = []
        for alpha in ALPHAS:
            pipeline = make_pipeline(ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), columns)]),
                                     GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=80,
                                                               max_depth=2, min_samples_leaf=20, random_state=42))
            pipeline.fit(x, label)
            bundle["models"].append(pipeline)
    else:
        model = CatBoostRegressor(**{**ssr.CATBOOST_PARAMS, "iterations": iterations, "depth": 6, "od_wait": 40,
                                    "verbose": False, "thread_count": 5})
        good = val_target.notna()
        validation = dict(eval_set=(v.loc[good], np.log1p(val_target.loc[good])), use_best_model=True) if good.sum() >= 5 else {}
        model.fit(x, np.log1p(label.to_numpy(dtype=float)), cat_features=columns, **validation)
        bundle["models"] = model
        if good.sum() >= 5:
            raw = np.sort(model.predict(v.loc[good]), axis=1)
            error = np.log1p(val_target.loc[good].to_numpy(dtype=float))[:, None] - raw
            bundle["shifts"] = np.array([np.quantile(error[:, j], q) for j, q in enumerate(ALPHAS)])
    bundle["training_n"] = len(train)
    return bundle


def predict(d, bundle, *, calibrate=True, median_table=None):
    """Predict sorted Q1/P50/Q3, accepting a replacement median table for ratios."""
    if not len(d):
        return np.empty((0, 3))
    if bundle["method"] == "empirical":
        return reference(d, bundle["reference"])[0]
    x = make_frame(d, bundle)
    model = bundle["models"]
    if bundle["method"] == "gbr":
        values = np.sort(np.column_stack([est.predict(x) for est in model]), axis=1)
    else:
        raw = np.sort(model.predict(x), axis=1)
        if calibrate and bundle["shifts"] is not None:
            raw = np.sort(raw + bundle["shifts"], axis=1)
        values = np.expm1(raw)
    if bundle["scale"] is not None:
        scales, _ = reference(d, bundle["scale"] if median_table is None else median_table)
        values *= scales
    return np.maximum(values, 0)


def losses(y, p):
    residual = np.asarray(y, dtype=float)[:, None] - p
    return np.maximum(residual * ALPHAS, -residual * (1 - ALPHAS))


def score(rows, p, target="y"):
    y = rows[target].to_numpy(dtype=float)
    good = np.isfinite(p).all(axis=1) & np.isfinite(y)
    if not good.any():
        return {"n": 0, "available_n": len(rows), "groups": 0}
    y, p = y[good], p[good]
    loss = losses(y, p).mean(axis=0)
    return {
        "n": len(y), "available_n": len(rows), "groups": rows.loc[good, "group"].nunique(),
        "pinball": loss.mean(), "pinball_p25": loss[0], "pinball_p50": loss[1], "pinball_p75": loss[2],
        "mae": np.abs(y - p[:, 1]).mean(), "medae": np.median(np.abs(y - p[:, 1])),
        "coverage": ((p[:, 0] <= y) & (y <= p[:, 2])).mean(),
        "width": np.median(p[:, 2] - p[:, 0]),
        **{f"below_p{q}": (y <= p[:, j]).mean() for j, q in enumerate((25, 50, 75))},
    }


def refresh_medians(table, observed):
    """Replace exact cell medians on support; retain earlier fallbacks."""
    fresh = observed.groupby(KEY).filter(lambda block: len(block) >= 10 and block.group.nunique() >= 5)
    fresh_table = reference_table(fresh, quantiles=[.5])[0][1]
    updates = {key: value for key, value in fresh_table.items() if key in table[0][1]}
    return [(table[0][0], {**table[0][1], **updates}), *table[1:]], updates
