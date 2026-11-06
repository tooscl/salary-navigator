"""Salary quantile experiment: char TF-IDF text features + CatBoost.

Pipeline
    1. load ML/data/experiments/01_location_monthly.csv (plain amounts
       treated as monthly; see that folder's README for the rules); drop
       military contracts and implausible amounts that slipped through;
       redact salary amounts from every feature;
    2. grouped random 80/10/10 split stratified by profession x city; likely
       duplicates never cross a split boundary;
    3. title + description + benefits -> character TF-IDF -> TruncatedSVD,
       both fitted on train only;
    4. CatBoost MultiQuantile (P25/P50/P75) on log-salary, one model for the
       lower salary bound and one for the upper bound;
    5. pinball loss, MAE and calibration (share of actual salaries below each
       predicted quantile, P25-P75 coverage) on validation and test;
    6. models, metrics, split ids and an experiment passport are saved
       following ML/README.md.

Optional Optuna search (step 4) tunes CatBoost, the number of SVD components
and whether to use the employer name on the validation split only; the test
split is touched only by the final model.

Run from the repository root or directly:
    python ML/experiments/expiriemnts.py                     # Optuna, OPTUNA_TRIALS trials
    python ML/experiments/expiriemnts.py --trials 0          # default parameters, no search
    python ML/experiments/expiriemnts.py --trials 100 --timeout 3600
    python ML/experiments/expiriemnts.py --study-name <name> # resume a study
    python ML/experiments/expiriemnts.py --cv 5              # 5-fold CV, mean +- std, no Optuna
    python ML/experiments/expiriemnts.py --cv 5 --drop-employer
    python ML/experiments/expiriemnts.py --final             # the model for the navigator (see run_final)
    python ML/experiments/expiriemnts.py --data ML/data/experiments/02_remote_monthly.csv
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "experiments"
DEFAULT_DATA_PATH = DATA_DIR / "01_location_monthly.csv"

EXPERIMENTS_DIR = ROOT / "ML" / "experiments"
MODELS_DIR = ROOT / "ML" / "models"
EXPERIMENT_NAME = "tfidf_catboost_quantile"

# Make repository utilities available when this file is launched directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.delete_salary import delete_salary  # noqa: E402


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Known before a vacancy is published. All are strings, so every one is passed
# through delete_salary to keep salary amounts out of the features.
FEATURE_COLUMNS = [
    "city",
    "region",
    "profession",
    "title",
    "description",
    "employer_name_observed",
    "work_schedule",
    "employment_type",
    "work_format",
    "registration_method",
    "work_hours_per_day",
    "pay_frequency",
    "experience",
]
# Vacancy parameters stored in extra_attributes_json (column name -> Avito field).
EXTRA_ATTRIBUTE_FEATURES = {
    "company_sphere": "Сфера деятельности компании",
    "enterprise_type": "Тип предприятия",
    "medical_book": "Медкнижка",
    "self_employment": "Статус самозанятости",
    "internship": "Стажировка",
    "driving_licence": "Категория водительских прав",
    "work_regime": "Режим работы",
    "shifts": "Смены",
    "candidate_groups": "В том числе для кандидатов",
}
# Free-text list of perks ("Униформа, Питание, Проживание"): goes into TF-IDF.
BENEFITS_ATTRIBUTE = "Что получают работники"

# How the employer wrote the salary: range / lower_bound ("от") /
# upper_bound ("до") / point. It is visible in the published ad and changes the
# meaning of a bound: "от 50 000" is a teaser floor, the lower end of a range
# is not. Contains no amount.
SALARY_KIND_COLUMN = "salary_value_kind"

TEXT_COLUMNS = ["title", "description", "benefits"]
CATEGORICAL_COLUMNS = (
    [c for c in FEATURE_COLUMNS if c not in TEXT_COLUMNS] + list(EXTRA_ATTRIBUTE_FEATURES) + [SALARY_KIND_COLUMN]
)

# Keep both observed bounds instead of inventing a midpoint.
TARGET_COLUMNS = ["salary_lower_rub", "salary_upper_rub"]

# Military / SVO contract ads: amounts include one-off sign-up payments
# ("до 4 500 000"), so they are not a market monthly salary for the job.
# A row is dropped if the dataset flag `military` is true or this pattern
# matches title + employer + description, lower-cased.
MILITARY_PATTERN = (
    r"\bсво\b|минобороны|министерств\w* обороны|\bмо рф\b|спецопер|специальн\w* военн"
    r"|служб\w* по контракту|контрактник|военнослужащ|вооруж\w* сил|\bбпла\b|\bдрон"
    r"|\bпво\b|тылов|\bв тыл\b|\bтылу\b|защит\w* неба|охран\w* неба|добровольц"
    r"|огнев\w* групп|[(/]мог\b|\bваи\b|пункт\w* отбора|военкомат|штурмов|артиллер"
)
# Plausible monthly salary range. Below it the amount is almost surely a rate
# per shift written without "за смену" (median ~4 000 rub, 11-12 h shifts);
# above it a lump sum.
MIN_MONTHLY_RUB = 10_000
MAX_MONTHLY_RUB = 1_000_000

SPLIT_FRACTIONS = (0.80, 0.10, 0.10)
# Cross-validation (--cv K): each fold is test once (1/K of rows); of the rest,
# this share of ALL rows is validation (early stopping + calibration), the
# remainder is train (~70 % with K=5).
CV_VALIDATION_SHARE = 0.10
# profession x city cells with fewer rows than this go entirely to train.
MIN_STRATUM_ROWS = 10
SPLIT_RULE = (
    "grouped random 80/10/10 stratified by profession x city (seeded); likely "
    "duplicates (same employer|title|city, all known) are kept in one split; "
    f"cells with < {MIN_STRATUM_ROWS} rows go to train. A time split is not used: "
    "collection order is profession -> city -> page, so it would test unseen "
    "cities/professions rather than a new period."
)

# Character n-grams survive typos, word forms and glued words
# ("кассир-продавец", "2/2", "ЗП") better than words; in the baseline
# comparison they matched or beat multilingual-e5-base.
TFIDF_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 3,
    "max_features": 150_000,
    "sublinear_tf": True,
    "lowercase": True,
}
SVD_COMPONENTS = 64
QUANTILES = (0.25, 0.50, 0.75)
RANDOM_SEED = 42

# Salaries are right-skewed, so the models learn log1p(rub). Quantiles commute
# with monotonic transforms: expm1(P75 of log) == P75 of rub, so nothing is
# lost, and the loss now treats a 20 % miss equally for cheap and costly jobs.
TARGET_TRANSFORM = "log1p"

CATBOOST_PARAMS = {
    "loss_function": "MultiQuantile:alpha=" + ",".join(str(q) for q in QUANTILES),
    "eval_metric": "MultiQuantile:alpha=" + ",".join(str(q) for q in QUANTILES),
    "iterations": 1_500,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_SEED,
    "random_strength": 1.0,
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": 100,
    "thread_count": -1,
    "allow_writing_files": False,
}

# Optuna. 0 trials = train once with CATBOOST_PARAMS / SVD_COMPONENTS.
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT_SECONDS: int | None = None
# Studies are stored in SQLite so an interrupted search can be resumed with
# --study-name and inspected with `optuna-dashboard sqlite:///ML/.cache/optuna.db`.
OPTUNA_STORAGE_PATH = ROOT / "ML" / ".cache" / "optuna.db"
OPTUNA_MAX_ITERATIONS = 3_000
OPTUNA_EARLY_STOPPING = 150


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------

def read_raw_rows(data_path: Path) -> pd.DataFrame:
    """One experiment dataset (utils/build_experiment_datasets.py builds them)."""
    df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False).assign(data_file=data_path.name)
    assert not df.duplicated(["source", "vacancy_id"]).any(), "A vacancy appears twice"
    return df


def extra_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Selected Avito parameters from extra_attributes_json, plus the military flag."""
    parsed = [json.loads(value) if isinstance(value, str) and value else {} for value in df["extra_attributes_json"]]
    columns = {name: [row.get(field) for row in parsed] for name, field in EXTRA_ATTRIBUTE_FEATURES.items()}
    columns["benefits"] = [row.get(BENEFITS_ATTRIBUTE) for row in parsed]
    columns["military_flag"] = [str(row.get("military", "")).lower() == "true" for row in parsed]
    return pd.DataFrame(columns, index=df.index)


def exclusion_reasons(df: pd.DataFrame, military_flag: pd.Series) -> pd.Series:
    """Reason to drop a row, or empty string to keep it."""
    targets = df[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    text = (
        df["title"].fillna("") + " | " + df["employer_name_observed"].fillna("") + " | " + df["description"].fillna("")
    ).str.lower()

    reasons = pd.Series("", index=df.index)
    reasons[targets.isna().all(axis=1)] = "no_salary_amount"
    reasons[(targets > MAX_MONTHLY_RUB).any(axis=1)] = f"salary_above_{MAX_MONTHLY_RUB}"
    reasons[(targets < MIN_MONTHLY_RUB).any(axis=1)] = f"salary_below_{MIN_MONTHLY_RUB}"
    reasons[military_flag | text.str.contains(MILITARY_PATTERN, regex=True)] = "military_contract"
    return reasons


def build_features(df: pd.DataFrame, extras: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model inputs from rows in the experiment-dataset schema, salary amounts redacted.

    Used for training and for scoring new vacancies (predict_corridor)."""
    extras = extra_attributes(df) if extras is None else extras
    features = pd.concat([df[FEATURE_COLUMNS], extras[list(EXTRA_ATTRIBUTE_FEATURES) + ["benefits"]]], axis=1)
    for column in features.columns:
        # Only strings: delete_salary(NaN) would return the string 'nan'.
        features[column] = features[column].map(lambda v: delete_salary(v) if isinstance(v, str) else v)
        features[column] = features[column].replace("", pd.NA)
    features[SALARY_KIND_COLUMN] = df[SALARY_KIND_COLUMN]
    return features


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return kept rows, redacted features, salary bounds and excluded rows."""
    df = read_raw_rows(data_path)
    extras = extra_attributes(df)

    reasons = exclusion_reasons(df, extras["military_flag"])
    excluded = df.loc[reasons != "", ["source", "vacancy_id", "data_file", "title"]].assign(
        reason=reasons[reasons != ""]
    )
    keep = reasons == ""
    df, extras = df.loc[keep].reset_index(drop=True), extras.loc[keep].reset_index(drop=True)
    print("Rows by file:", df["data_file"].value_counts().to_dict())
    print(f"Excluded {len(excluded)} rows:", excluded["reason"].value_counts().to_dict())

    features = build_features(df, extras)
    targets = df[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    assert targets.notna().any(axis=1).all(), "A training row has no salary target"
    assert (targets.isna() | (targets > 0)).all().all(), "Salary bounds must be positive for log1p"
    return df, features, targets, excluded


# --------------------------------------------------------------------------
# 2. Split
# --------------------------------------------------------------------------

def duplicate_groups(df: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per row: likely-duplicate group and profession x city stratum.

    A group is a likely duplicate only when employer, title and city are all
    known and identical after salary redaction. Otherwise a vacancy is its own
    group, so missing employer names cannot merge unrelated ads.
    """
    duplicate_fields = features[["employer_name_observed", "title", "city"]].fillna("").astype(str)
    has_fingerprint = duplicate_fields.ne("").all(axis=1)
    fingerprint = duplicate_fields.agg("|".join, axis=1)
    fallback_group = "vacancy:" + df["source"].astype(str) + ":" + df["vacancy_id"].astype(str)
    duplicate_group = fingerprint.where(has_fingerprint, fallback_group)
    stratum = (
        features["profession"].fillna("__MISSING__").astype(str) + " | " + features["city"].fillna("__MISSING__").astype(str)
    )
    return duplicate_group, stratum


def group_positions(duplicate_group: pd.Series, stratum: pd.Series, seed: int) -> pd.DataFrame:
    """Groups shuffled (seeded) inside every stratum; `position` in [0, 1] is the
    share of the cell's rows up to the middle of the group, so a big group is
    placed by its centre and cannot push a whole cell past a boundary."""
    groups = (
        pd.DataFrame({"duplicate_group": duplicate_group, "stratum": stratum, "rows": 1})
        .groupby("duplicate_group", as_index=False)
        .agg(stratum=("stratum", "first"), rows=("rows", "sum"))
        .sort_values("duplicate_group", kind="stable")
    )
    groups["random_key"] = np.random.default_rng(seed).random(len(groups))
    groups = groups.sort_values(["stratum", "random_key"], kind="stable")
    groups["stratum_rows"] = groups.groupby("stratum")["rows"].transform("sum")
    groups["position"] = (groups.groupby("stratum")["rows"].cumsum() - groups["rows"] / 2) / groups["stratum_rows"]
    return groups


def assign_splits(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Grouped random 80/10/10 split stratified by profession x city.

    Inside every profession x city cell duplicate groups are shuffled with a
    fixed seed, so each cell is represented in train, validation and test in
    roughly 80/10/10 shares.
    """
    duplicate_group, stratum = duplicate_groups(df, features)
    groups = group_positions(duplicate_group, stratum, RANDOM_SEED)
    stratum_rows = groups["stratum_rows"]

    groups["split"] = "test"
    groups.loc[groups["position"] < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1], "split"] = "validation"
    groups.loc[groups["position"] < SPLIT_FRACTIONS[0], "split"] = "train"
    groups.loc[stratum_rows < MIN_STRATUM_ROWS, "split"] = "train"

    assignment = df[["source", "vacancy_id", "data_file"]].copy()
    assignment["duplicate_group"] = duplicate_group
    assignment["stratum"] = stratum
    assignment["split"] = assignment["duplicate_group"].map(groups.set_index("duplicate_group")["split"])

    # Every row is assigned once and a duplicate never crosses a boundary.
    assert assignment["split"].notna().all(), "A vacancy did not receive a split"
    assert assignment.groupby("duplicate_group")["split"].nunique().max() == 1
    assert set(assignment["split"]) == {"train", "validation", "test"}
    return assignment


def assign_cv_splits(df: pd.DataFrame, features: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Grouped K-fold stratified by profession x city, plus an inner split.

    `fold` (0..K-1): every row is in the test fold exactly once. For fold k the
    remaining rows are divided again (another seed, same grouping and
    stratification) so that CV_VALIDATION_SHARE of ALL rows become validation
    (early stopping + calibration) and the rest is train. Duplicates never
    cross a boundary.
    """
    duplicate_group, stratum = duplicate_groups(df, features)
    groups = group_positions(duplicate_group, stratum, RANDOM_SEED)
    groups["fold"] = np.minimum((groups["position"] * n_folds).astype(int), n_folds - 1)
    # Inner order: a second shuffle; validation = first share of the non-test part.
    inner = group_positions(duplicate_group, stratum, RANDOM_SEED + 1).set_index("duplicate_group")["position"]
    groups["inner_position"] = groups["duplicate_group"].map(inner)

    assignment = df[["source", "vacancy_id", "data_file"]].copy()
    assignment["duplicate_group"] = duplicate_group
    assignment["stratum"] = stratum
    by_group = groups.set_index("duplicate_group")
    assignment["fold"] = assignment["duplicate_group"].map(by_group["fold"])
    assignment["inner_position"] = assignment["duplicate_group"].map(by_group["inner_position"])
    assert assignment["fold"].notna().all() and assignment.groupby("duplicate_group")["fold"].nunique().max() == 1
    assert set(assignment["fold"]) == set(range(n_folds)), "A fold is empty"
    return assignment


def cv_fold_split(assignment: pd.DataFrame, fold: int, n_folds: int) -> pd.Series:
    """train / validation / test labels for one CV fold."""
    validation_share_of_rest = CV_VALIDATION_SHARE / (1 - 1 / n_folds)
    split = pd.Series("train", index=assignment.index)
    split[assignment["inner_position"] < validation_share_of_rest] = "validation"
    split[assignment["fold"] == fold] = "test"
    return split


# --------------------------------------------------------------------------
# 3. Text features
# --------------------------------------------------------------------------

def make_text(frame: pd.DataFrame) -> pd.Series:
    """Title, description and the list of perks, one per line."""
    parts = [frame[column].fillna("").astype(str).str.strip() for column in TEXT_COLUMNS]
    return parts[0].str.cat(parts[1:], sep="\n").str.strip()


def fit_text_encoder(train_texts: pd.Series, n_components: int) -> tuple[TfidfVectorizer, TruncatedSVD]:
    """TF-IDF vocabulary/IDF and SVD axes are learnt from train texts only."""
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    matrix = vectorizer.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=min(n_components, matrix.shape[1] - 1), random_state=RANDOM_SEED)
    svd.fit(matrix)
    return vectorizer, svd


def prepare_categories(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    """Explicit '__MISSING__' category; plain str dtype for CatBoost."""
    return frame[categorical_columns].fillna("__MISSING__").astype(str)


def build_model_frame(
    features: pd.DataFrame, text_components: np.ndarray, categorical_columns: list[str]
) -> pd.DataFrame:
    """Categorical columns + dense SVD components of the text."""
    result = prepare_categories(features, categorical_columns).reset_index(drop=True)
    components = pd.DataFrame(
        text_components,
        columns=[f"text_svd_{i:03d}" for i in range(text_components.shape[1])],
    )
    return pd.concat([result, components], axis=1)


# --------------------------------------------------------------------------
# 4. Quantile models
# --------------------------------------------------------------------------

def fit_quantile_model(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    catboost_params: dict,
    categorical_columns: list[str],
) -> CatBoostRegressor:
    """Fit P25/P50/P75 of log1p(salary) for one observed bound.

    Validation is used for early stopping (and by Optuna), so validation
    metrics are optimistic; the test split is the unbiased estimate.
    """
    train_mask = train_target.notna().to_numpy()
    validation_mask = validation_target.notna().to_numpy()
    assert train_mask.any() and validation_mask.any(), f"No data for {train_target.name}"

    model = CatBoostRegressor(**catboost_params)
    model.fit(
        train_features.loc[train_mask],
        np.log1p(train_target.to_numpy()[train_mask]),
        cat_features=categorical_columns,
        eval_set=(
            validation_features.loc[validation_mask],
            np.log1p(validation_target.to_numpy()[validation_mask]),
        ),
        use_best_model=True,
    )
    return model


def predict_log_quantiles(model: CatBoostRegressor, features: pd.DataFrame) -> np.ndarray:
    """Raw P25/P50/P75 of log1p(salary), shape (n, 3), sorted."""
    predictions = np.asarray(model.predict(features), dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, len(QUANTILES))
    if predictions.shape[0] == len(QUANTILES) and predictions.shape[1] == len(features):
        predictions = predictions.T
    return np.sort(predictions, axis=1)


def predict_quantiles(
    model: CatBoostRegressor, features: pd.DataFrame, shifts: np.ndarray | None = None
) -> np.ndarray:
    """P25/P50/P75 in rubles, shape (n, 3); calibrated when shifts are given."""
    log_predictions = predict_log_quantiles(model, features)
    if shifts is not None:
        log_predictions = np.sort(log_predictions + np.asarray(shifts), axis=1)
    return np.expm1(log_predictions)


# --------------------------------------------------------------------------
# 4b. Calibration
# --------------------------------------------------------------------------

def fit_calibration(model: CatBoostRegressor, features: pd.DataFrame, target: pd.Series) -> np.ndarray:
    """Per-quantile shifts in log space learnt on a held-out split.

    Boosted quantile models are over-confident: their P25-P75 covers far less
    than half of new vacancies. For each quantile q we add
        shift_q = q-th quantile of (log1p(y) - predicted log P_q)
    so that on the calibration split exactly a share q of salaries lies below
    the corrected P_q (hence P25-P75 covers ~50 %). A shift in log space is a
    multiplier in rubles, e.g. shift -0.08 means P25 x 0.92.
    """
    mask = target.notna().to_numpy()
    residuals = np.log1p(target.to_numpy()[mask])[:, None] - predict_log_quantiles(model, features.loc[mask])
    return np.array([np.quantile(residuals[:, i], q) for i, q in enumerate(QUANTILES)])


# --------------------------------------------------------------------------
# 5. Evaluation
# --------------------------------------------------------------------------

def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    residual = actual - predicted
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


def evaluate_bound(
    model: CatBoostRegressor,
    features: pd.DataFrame,
    target: pd.Series,
    split_name: str,
    shifts: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Per-quantile accuracy + calibration, and P25-P75 interval quality.

    share_below should be close to the quantile itself (0.25 / 0.50 / 0.75):
    otherwise the model is biased up or down. coverage_p25_p75 should be
    close to 0.50: lower means intervals are too narrow and normal vacancies
    will be flagged as unusual.
    """
    mask = target.notna().to_numpy()
    actual = target.to_numpy()[mask]
    predicted = predict_quantiles(model, features.loc[mask], shifts)
    calibration = "raw" if shifts is None else "calibrated"

    rows = []
    for index, quantile in enumerate(QUANTILES):
        rows.append(
            {
                "split": split_name,
                "calibration": calibration,
                "target": target.name,
                "quantile": quantile,
                "n": int(mask.sum()),
                "pinball_rub": pinball_loss(actual, predicted[:, index], quantile),
                "pinball_log": pinball_loss(np.log1p(actual), np.log1p(predicted[:, index]), quantile),
                "mae_rub": float(np.mean(np.abs(actual - predicted[:, index]))),
                "mae_log": float(np.mean(np.abs(np.log1p(actual) - np.log1p(predicted[:, index])))),
                "mape": float(np.mean(np.abs(actual - predicted[:, index]) / actual)),
                "median_ape": float(np.median(np.abs(actual - predicted[:, index]) / actual)),
                "share_below": float(np.mean(actual < predicted[:, index])),
            }
        )

    interval = {
        "split": split_name,
        "calibration": calibration,
        "target": target.name,
        "n": int(mask.sum()),
        "coverage_p25_p75": float(np.mean((actual >= predicted[:, 0]) & (actual <= predicted[:, 2]))),
        "median_width_rub": float(np.median(predicted[:, 2] - predicted[:, 0])),
        "median_relative_width": float(np.median((predicted[:, 2] - predicted[:, 0]) / predicted[:, 1])),
    }
    return rows, interval


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def prepare_data(data_path: Path) -> dict[str, object]:
    """Load, filter and split once; everything after this is cheap."""
    df, features, targets, excluded = load_dataset(data_path)
    split_assignment = assign_splits(df, features)
    profession = features["profession"].fillna("__MISSING__")
    print(pd.crosstab(profession, split_assignment["split"]))
    return {
        "data_path": data_path,
        "excluded": excluded,
        "split_assignment": split_assignment,
        "parts": make_parts(features, targets, split_assignment["split"]),
        "text_encoders": {},
    }


def make_parts(features: pd.DataFrame, targets: pd.DataFrame, split: pd.Series) -> dict[str, dict]:
    """Features, texts, targets and original row index for train / validation / test."""
    parts = {}
    for name in ("train", "validation", "test"):
        index = split.index[split == name]
        part_features = features.loc[index]
        parts[name] = {
            "index": index,
            "features": part_features,
            "texts": make_text(part_features),
            "targets": targets.loc[index].reset_index(drop=True),
        }
    return parts


def default_config() -> dict[str, object]:
    return {"catboost": dict(CATBOOST_PARAMS), "svd_components": SVD_COMPONENTS, "drop_columns": []}


def text_components(prepared: dict[str, object], n_components: int):
    """TF-IDF + SVD per number of components, cached for Optuna trials."""
    cache = prepared["text_encoders"]
    if n_components not in cache:
        parts = prepared["parts"]
        vectorizer, svd = fit_text_encoder(parts["train"]["texts"], n_components)
        cache[n_components] = {
            "vectorizer": vectorizer,
            "svd": svd,
            "components": {name: svd.transform(vectorizer.transform(part["texts"])) for name, part in parts.items()},
        }
    return cache[n_components]


def fit_models(prepared: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    """Text components, then one MultiQuantile model per salary bound."""
    parts = prepared["parts"]
    categorical_columns = [c for c in CATEGORICAL_COLUMNS if c not in config["drop_columns"]]
    text = text_components(prepared, config["svd_components"])
    frames = {
        name: build_model_frame(part["features"], text["components"][name], categorical_columns)
        for name, part in parts.items()
    }
    models = {
        target_name: fit_quantile_model(
            frames["train"],
            parts["train"]["targets"][target_name],
            frames["validation"],
            parts["validation"]["targets"][target_name],
            config["catboost"],
            categorical_columns,
        )
        for target_name in TARGET_COLUMNS
    }
    return {
        "models": models,
        "frames": frames,
        "vectorizer": text["vectorizer"],
        "svd": text["svd"],
        "categorical_columns": categorical_columns,
    }


def calibrate(fitted: dict[str, object], parts: dict) -> dict[str, np.ndarray]:
    """Shifts per salary bound, learnt on validation (test stays untouched).

    Validation is also used for early stopping and Optuna; that choice of the
    number of trees barely affects interval width, and the test metrics show
    whether the corrected intervals really hold on unseen vacancies.
    """
    return {
        target_name: fit_calibration(
            model, fitted["frames"]["validation"], parts["validation"]["targets"][target_name]
        )
        for target_name, model in fitted["models"].items()
    }


def evaluate(
    fitted: dict[str, object],
    parts: dict,
    split_names: tuple[str, ...],
    calibration: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw metrics, plus calibrated ones when calibration shifts are given."""
    metric_rows, interval_rows = [], []
    variants = [None] if calibration is None else [None, calibration]
    for split_name in split_names:
        for target_name, model in fitted["models"].items():
            for shifts in variants:
                rows, interval = evaluate_bound(
                    model,
                    fitted["frames"][split_name],
                    parts[split_name]["targets"][target_name],
                    split_name,
                    None if shifts is None else shifts[target_name],
                )
                metric_rows += rows
                interval_rows.append(interval)
    return pd.DataFrame(metric_rows), pd.DataFrame(interval_rows)


def suggest_config(trial) -> dict[str, object]:
    """Optuna search space."""
    catboost_params = dict(CATBOOST_PARAMS) | {
        "iterations": OPTUNA_MAX_ITERATIONS,
        "od_wait": OPTUNA_EARLY_STOPPING,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [2, 10, 30]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
    }
    if catboost_params["bootstrap_type"] == "Bayesian":
        catboost_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 2.0)
    else:
        catboost_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    use_employer = trial.suggest_categorical("use_employer", [True, False])
    return {
        "catboost": catboost_params,
        "svd_components": trial.suggest_categorical("svd_components", [16, 32, 64, 128]),
        "drop_columns": [] if use_employer else ["employer_name_observed"],
    }


def tune_hyperparameters(
    prepared: dict[str, object], n_trials: int, timeout: int | None, study_name: str | None
):
    """Minimise the mean validation MultiQuantile (pinball on log-salary) of both models."""
    import optuna  # imported here so --trials 0 works without optuna installed

    OPTUNA_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name or f"{EXPERIMENT_NAME}_{datetime.now():%Y-%m-%d_%H%M%S}",
        storage=f"sqlite:///{OPTUNA_STORAGE_PATH.as_posix()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        load_if_exists=True,
    )

    def objective(trial) -> float:
        config = suggest_config(trial)
        config["catboost"]["verbose"] = 0
        fitted = fit_models(prepared, config)
        losses = [next(iter(m.get_best_score()["validation"].values())) for m in fitted["models"].values()]
        _, intervals = evaluate(fitted, prepared["parts"], ("validation",))
        for row in intervals.itertuples():
            trial.set_user_attr(f"coverage_{row.target}", round(row.coverage_p25_p75, 4))
        for target, model in fitted["models"].items():
            trial.set_user_attr(f"best_iteration_{target}", model.get_best_iteration())
        return float(np.mean(losses))

    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    print(f"Optuna best validation loss {study.best_value:.5f}: {study.best_params}")
    print("Validation coverage of the best trial:", study.best_trial.user_attrs)
    return study


def config_from_params(params: dict[str, object]) -> dict[str, object]:
    """Rebuild a training config from Optuna best_params (final model is verbose)."""
    import optuna

    config = suggest_config(optuna.trial.FixedTrial(params))
    config["catboost"]["verbose"] = 100
    return config


def run_experiment(
    data_path: Path = DEFAULT_DATA_PATH,
    n_trials: int = OPTUNA_TRIALS,
    timeout: int | None = OPTUNA_TIMEOUT_SECONDS,
    study_name: str | None = None,
) -> dict[str, object]:
    prepared = prepare_data(data_path)

    study = None
    config = default_config()
    if n_trials > 0:
        study = tune_hyperparameters(prepared, n_trials, timeout, study_name)
        config = config_from_params(study.best_params)

    fitted = fit_models(prepared, config)
    calibration = calibrate(fitted, prepared["parts"])
    metrics, intervals = evaluate(fitted, prepared["parts"], ("validation", "test"), calibration)

    feature_importance = pd.DataFrame(
        {"feature": fitted["frames"]["train"].columns}
        | {f"{target}_importance": model.get_feature_importance() for target, model in fitted["models"].items()}
    )
    feature_importance["mean_importance"] = feature_importance.filter(like="_importance").mean(axis=1)
    feature_importance = feature_importance.sort_values("mean_importance", ascending=False, ignore_index=True)

    return {
        "data_path": data_path,
        "excluded": prepared["excluded"],
        "split_assignment": prepared["split_assignment"],
        "config": config,
        "categorical_columns": fitted["categorical_columns"],
        "study": study,
        "vectorizer": fitted["vectorizer"],
        "svd": fitted["svd"],
        "svd_explained_variance": float(fitted["svd"].explained_variance_ratio_.sum()),
        "models": fitted["models"],
        "calibration": calibration,
        "metrics": metrics,
        "intervals": intervals,
        "feature_importance": feature_importance,
    }


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def pooled_metrics(oof: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Metrics of calibrated out-of-fold P50 / P25-P75, pooled over all folds."""
    rows = []
    for target in TARGET_COLUMNS:
        part = oof[oof[f"{target}_actual"].notna()]
        for key, group in ([((), part)] if not by else part.groupby(by)):
            actual = group[f"{target}_actual"].to_numpy()
            p25, p50, p75 = (group[f"{target}_p{q}"].to_numpy() for q in (25, 50, 75))
            ape = np.abs(actual - p50) / actual
            pinball = np.mean([
                pinball_loss(np.log1p(actual), np.log1p(group[f"{target}_p{int(q * 100)}"].to_numpy()), q)
                for q in QUANTILES
            ])
            key = key if isinstance(key, tuple) else (key,)
            rows.append(
                dict(zip(by, key))
                | {
                    "target": target,
                    "n": len(group),
                    "median_salary": float(np.median(actual)),
                    "mae_rub": float(np.mean(np.abs(actual - p50))),
                    "median_abs_error_rub": float(np.median(np.abs(actual - p50))),
                    "mape": float(np.mean(ape)),
                    "median_ape": float(np.median(ape)),
                    "pinball_log": float(pinball),
                    "share_below_p50": float(np.mean(actual < p50)),
                    "coverage_p25_p75": float(np.mean((actual >= p25) & (actual <= p75))),
                }
            )
    return pd.DataFrame(rows)


def run_cv(
    data_path: Path = DEFAULT_DATA_PATH, n_folds: int = 5, config: dict[str, object] | None = None
) -> dict[str, object]:
    """Grouped stratified K-fold: every vacancy is predicted exactly once by a
    model that never saw it (out-of-fold). Text encoder, CatBoost and the
    calibration shifts are refitted inside every fold."""
    config = config or default_config()
    config["catboost"] = dict(config["catboost"], verbose=0)
    df, features, targets, excluded = load_dataset(data_path)
    assignment = assign_cv_splits(df, features, n_folds)

    fold_metrics, fold_intervals, oof_parts = [], [], []
    best_iterations = {target: [] for target in TARGET_COLUMNS}
    for fold in range(n_folds):
        split = cv_fold_split(assignment, fold, n_folds)
        prepared = {"parts": make_parts(features, targets, split), "text_encoders": {}}
        fitted = fit_models(prepared, config)
        for target, model in fitted["models"].items():
            best_iterations[target].append(model.get_best_iteration() + 1)
        calibration = calibrate(fitted, prepared["parts"])
        metrics, intervals = evaluate(fitted, prepared["parts"], ("test",), calibration)
        fold_metrics.append(metrics.assign(fold=fold))
        fold_intervals.append(intervals.assign(fold=fold))

        test = prepared["parts"]["test"]
        oof = pd.DataFrame(
            {
                "row": test["index"],
                "fold": fold,
                "profession": features.loc[test["index"], "profession"].fillna("Без профессии").to_numpy(),
                "city": features.loc[test["index"], "city"].fillna("__MISSING__").to_numpy(),
            }
        )
        for target, model in fitted["models"].items():
            predicted = predict_quantiles(model, fitted["frames"]["test"], calibration[target])
            raw_log = predict_log_quantiles(model, fitted["frames"]["test"])
            oof[f"{target}_actual"] = test["targets"][target].to_numpy()
            for i, q in enumerate((25, 50, 75)):
                oof[f"{target}_p{q}"] = predicted[:, i]  # calibrated on this fold's validation
                oof[f"{target}_rawlog_p{q}"] = raw_log[:, i]  # before any calibration
        oof_parts.append(oof)
        cal_test = intervals[intervals["calibration"] == "calibrated"]
        print(
            f"fold {fold + 1}/{n_folds}: train {len(split[split == 'train'])}, val {len(split[split == 'validation'])}, "
            f"test {len(split[split == 'test'])}; coverage "
            + " / ".join(f"{v:.3f}" for v in cal_test["coverage_p25_p75"]),
            flush=True,
        )

    fold_metrics, fold_intervals = pd.concat(fold_metrics), pd.concat(fold_intervals)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof = oof.join(df[["source", "vacancy_id"]], on="row")

    p50 = fold_metrics[fold_metrics["quantile"] == 0.5]
    per_fold = p50.merge(
        fold_intervals[["fold", "calibration", "target", "coverage_p25_p75", "median_relative_width"]],
        on=["fold", "calibration", "target"],
    )
    summary = (
        per_fold.groupby(["calibration", "target"])[
            ["pinball_log", "mae_rub", "mape", "median_ape", "share_below", "coverage_p25_p75", "median_relative_width"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return {
        "data_path": data_path,
        "n_folds": n_folds,
        "config": config,
        "excluded": excluded,
        "features": features,
        "targets": targets,
        "best_iterations": best_iterations,
        "assignment": assignment,
        "fold_metrics": fold_metrics,
        "fold_intervals": fold_intervals,
        "summary": summary.reset_index(),
        "oof": oof,
        "pooled": pooled_metrics(oof, []),
        "pooled_by_profession": pooled_metrics(oof, ["profession"]),
    }


def save_cv(results: dict[str, object]) -> Path:
    """CV artefacts: per-fold metrics, mean +- std, pooled out-of-fold metrics."""
    experiment_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_{EXPERIMENT_NAME}_cv{results['n_folds']}"
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    for name in ("fold_metrics", "fold_intervals", "summary", "oof", "pooled", "pooled_by_profession"):
        results[name].to_csv(experiment_dir / f"cv_{name}.csv", index=False)
    results["assignment"][["source", "vacancy_id", "stratum", "fold"]].to_csv(experiment_dir / "cv_folds.csv", index=False)

    data_path = Path(results["data_path"]).resolve()
    manifest = json.loads((DATA_DIR / "manifest.json").read_text("utf-8")) if (DATA_DIR / "manifest.json").exists() else {}
    script = Path(__file__).resolve().relative_to(ROOT).as_posix()
    git_status = _git("status", "--porcelain", "--", script)
    passport = {
        "experiment_id": experiment_id,
        "status": "executed",
        "evaluation": (
            f"grouped stratified {results['n_folds']}-fold CV (profession x city, duplicates together, seed "
            f"{RANDOM_SEED}); per fold: test = 1/{results['n_folds']}, validation = {CV_VALIDATION_SHARE:.0%} of all "
            "rows (early stopping + calibration), rest train. No model is saved: every fold refits."
        ),
        "dataset": {
            "file_path": data_path.relative_to(ROOT).as_posix(),
            "file_sha256": _sha256(data_path),
            "rows": int(len(results["assignment"])),
            "rules_version": manifest.get("rules_version"),
        },
        "code_git_commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": None if git_status is None else bool(git_status),
        "row_filters": {"excluded_by_reason": results["excluded"]["reason"].value_counts().to_dict()},
        "parameters": {
            "catboost": results["config"]["catboost"],
            "svd_components": results["config"]["svd_components"],
            "dropped_columns": results["config"]["drop_columns"],
            "tfidf_params": {k: list(v) if isinstance(v, tuple) else v for k, v in TFIDF_PARAMS.items()},
        },
        "metrics": {
            "fold_mean_std": results["summary"].to_dict(orient="records"),
            "pooled_out_of_fold_calibrated": results["pooled"].to_dict(orient="records"),
        },
        "conclusion": None,
    }
    (experiment_dir / "experiment.json").write_text(json.dumps(passport, ensure_ascii=False, indent=2), "utf-8")
    return experiment_dir


# --------------------------------------------------------------------------
# Final model (--final): CV estimate + calibration from out-of-fold
# predictions + one model trained on all rows
# --------------------------------------------------------------------------

FINAL_DROP_COLUMNS = ["employer_name_observed"]  # 5-fold CV: no gain from it (0.103 vs 0.104 pinball)
CALIBRATION_GROUP_COLUMN = "profession"
CALIBRATION_MISSING_GROUP = "Без профессии"
# A profession gets its own multipliers only with at least this many
# out-of-fold predictions for the bound; smaller and unseen ones use global.
CALIBRATION_MIN_ROWS = 200
GLOBAL_GROUP = "__global__"


def calibration_groups(features: pd.DataFrame) -> pd.Series:
    return features[CALIBRATION_GROUP_COLUMN].fillna(CALIBRATION_MISSING_GROUP).astype(str)


def fit_shift_table(raw_log: np.ndarray, actual: np.ndarray, groups: np.ndarray, min_rows: float) -> dict[str, list]:
    """Per-group log shifts (see fit_calibration) + a global fallback."""
    residuals = np.log1p(actual)[:, None] - raw_log
    table = {GLOBAL_GROUP: [float(np.quantile(residuals[:, i], q)) for i, q in enumerate(QUANTILES)]}
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() >= min_rows:
            table[str(group)] = [float(np.quantile(residuals[mask, i], q)) for i, q in enumerate(QUANTILES)]
    return table


def apply_shift_table(raw_log: np.ndarray, groups: np.ndarray, table: dict[str, list]) -> np.ndarray:
    """Calibrated P25/P50/P75 in rubles."""
    shifts = np.array([table.get(str(group), table[GLOBAL_GROUP]) for group in groups])
    return np.expm1(np.sort(raw_log + shifts, axis=1))


def oof_variant(oof: pd.DataFrame, how: str) -> pd.DataFrame:
    """Out-of-fold predictions under one calibration scheme.

    raw                  no calibration
    fold_validation      shifts from each fold's own validation split (what --cv reports)
    global_crossfit      one set of shifts, learnt on the OTHER folds' out-of-fold predictions
    profession_crossfit  per-profession shifts, learnt on the OTHER folds' out-of-fold predictions
    Cross-fitting keeps the check honest: a fold is never calibrated on itself.
    """
    out = oof.copy()
    if how == "fold_validation":
        return out
    for target in TARGET_COLUMNS:
        raw = out[[f"{target}_rawlog_p{q}" for q in (25, 50, 75)]].to_numpy()
        actual = out[f"{target}_actual"].to_numpy()
        has = ~np.isnan(actual)
        if how == "raw":
            predicted = np.expm1(raw)
        else:
            groups = out["profession"].to_numpy() if how == "profession_crossfit" else np.full(len(out), "all")
            min_rows = CALIBRATION_MIN_ROWS if how == "profession_crossfit" else np.inf
            predicted = np.full(raw.shape, np.nan)
            for fold in np.unique(out["fold"]):
                fit_mask = has & (out["fold"].to_numpy() != fold)
                apply_mask = out["fold"].to_numpy() == fold
                table = fit_shift_table(raw[fit_mask], actual[fit_mask], groups[fit_mask], min_rows)
                predicted[apply_mask] = apply_shift_table(raw[apply_mask], groups[apply_mask], table)
        for i, q in enumerate((25, 50, 75)):
            out[f"{target}_p{q}"] = predicted[:, i]
    return out


def fit_final_models(
    features: pd.DataFrame, targets: pd.DataFrame, config: dict[str, object], best_iterations: dict[str, list]
) -> dict[str, object]:
    """Text encoder + one CatBoost per bound on ALL rows. No rows are held out,
    so the number of trees is the mean best iteration of the CV folds."""
    categorical_columns = [c for c in CATEGORICAL_COLUMNS if c not in config["drop_columns"]]
    texts = make_text(features)
    vectorizer, svd = fit_text_encoder(texts, config["svd_components"])
    frame = build_model_frame(features, svd.transform(vectorizer.transform(texts)), categorical_columns)
    params = {k: v for k, v in config["catboost"].items() if k not in ("od_type", "od_wait")}
    models, iterations = {}, {}
    for target in TARGET_COLUMNS:
        mask = targets[target].notna().to_numpy()
        iterations[target] = int(round(np.mean(best_iterations[target])))
        model = CatBoostRegressor(**(params | {"iterations": iterations[target], "verbose": 0}))
        model.fit(frame.loc[mask], np.log1p(targets[target].to_numpy()[mask]), cat_features=categorical_columns)
        models[target] = model
    importance = pd.DataFrame(
        {"feature": frame.columns}
        | {f"{target}_importance": model.get_feature_importance() for target, model in models.items()}
    )
    importance["mean_importance"] = importance.filter(like="_importance").mean(axis=1)
    return {
        "models": models,
        "iterations": iterations,
        "vectorizer": vectorizer,
        "svd": svd,
        "categorical_columns": categorical_columns,
        "feature_importance": importance.sort_values("mean_importance", ascending=False, ignore_index=True),
    }


def run_final(data_path: Path = DEFAULT_DATA_PATH, n_folds: int = 5) -> dict[str, object]:
    config = default_config()
    config["drop_columns"] = list(FINAL_DROP_COLUMNS)
    cv = run_cv(data_path, n_folds, config)
    oof = cv["oof"]

    comparison, comparison_by_profession = [], []
    for how in ("raw", "fold_validation", "global_crossfit", "profession_crossfit"):
        variant = oof_variant(oof, how)
        comparison.append(pooled_metrics(variant, []).assign(calibration=how))
        comparison_by_profession.append(pooled_metrics(variant, ["profession"]).assign(calibration=how))

    calibration_tables = {}
    for target in TARGET_COLUMNS:
        has = oof[f"{target}_actual"].notna().to_numpy()
        calibration_tables[target] = fit_shift_table(
            oof.loc[has, [f"{target}_rawlog_p{q}" for q in (25, 50, 75)]].to_numpy(),
            oof.loc[has, f"{target}_actual"].to_numpy(),
            oof.loc[has, "profession"].to_numpy(),
            CALIBRATION_MIN_ROWS,
        )

    final = fit_final_models(cv["features"], cv["targets"], cv["config"], cv["best_iterations"])
    return {
        "cv": cv,
        "config": cv["config"],
        "comparison": pd.concat(comparison, ignore_index=True),
        "comparison_by_profession": pd.concat(comparison_by_profession, ignore_index=True),
        "calibration_tables": calibration_tables,
        "final": final,
    }


def save_final(results: dict[str, object]) -> Path:
    """Experiment folder (CV evidence) + model folder (what the navigator loads)."""
    cv, final = results["cv"], results["final"]
    experiment_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_{EXPERIMENT_NAME}_final"
    experiment_dir, model_dir = EXPERIMENTS_DIR / experiment_id, MODELS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)

    for name in ("fold_metrics", "fold_intervals", "summary", "oof"):
        cv[name].to_csv(experiment_dir / f"cv_{name}.csv", index=False)
    cv["assignment"][["source", "vacancy_id", "stratum", "fold"]].to_csv(experiment_dir / "cv_folds.csv", index=False)
    results["comparison"].to_csv(experiment_dir / "calibration_comparison.csv", index=False)
    results["comparison_by_profession"].to_csv(experiment_dir / "calibration_comparison_by_profession.csv", index=False)
    final["feature_importance"].to_csv(experiment_dir / "feature_importance.csv", index=False)
    cv["excluded"].to_csv(experiment_dir / "excluded_ids.csv", index=False)

    for target, model in final["models"].items():
        model.save_model(str(model_dir / f"{target}.cbm"))
    joblib.dump({"vectorizer": final["vectorizer"], "svd": final["svd"]}, model_dir / "text_encoder.joblib")
    calibration = {
        "method": (
            "per-profession additive shifts of predicted log1p(salary), learnt on out-of-fold "
            f"predictions of {cv['n_folds']}-fold CV; professions with < {CALIBRATION_MIN_ROWS} rows "
            f"and unseen ones use '{GLOBAL_GROUP}'"
        ),
        "group_column": CALIBRATION_GROUP_COLUMN,
        "missing_group": CALIBRATION_MISSING_GROUP,
        "quantiles": list(QUANTILES),
        "shifts": results["calibration_tables"],
        "rub_multipliers": {
            target: {group: [round(float(np.exp(s)), 4) for s in shifts] for group, shifts in table.items()}
            for target, table in results["calibration_tables"].items()
        },
    }
    (model_dir / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), "utf-8")
    input_schema = {
        "input": "rows in the experiment-dataset schema (ML/data/experiments/*.csv); see build_features",
        "text_columns": TEXT_COLUMNS,
        "extra_attribute_columns": EXTRA_ATTRIBUTE_FEATURES,
        "categorical_columns": final["categorical_columns"],
        "text_encoder": "text_encoder.joblib: TfidfVectorizer -> TruncatedSVD",
        "target_transform": TARGET_TRANSFORM,
        "apply": "raw = sort(model.predict(X)); rub = expm1(sort(raw + shifts[profession or __global__]))",
        "models": {target: f"{target}.cbm" for target in final["models"]},
        "trees": final["iterations"],
        "usage": "from expiriemnts import load_final_model, predict_corridor",
    }
    (model_dir / "input_schema.json").write_text(json.dumps(input_schema, ensure_ascii=False, indent=2), "utf-8")

    data_path = Path(cv["data_path"]).resolve()
    manifest = json.loads((DATA_DIR / "manifest.json").read_text("utf-8")) if (DATA_DIR / "manifest.json").exists() else {}
    script = Path(__file__).resolve().relative_to(ROOT).as_posix()
    git_status = _git("status", "--porcelain", "--", script)
    headline = results["comparison"][results["comparison"]["calibration"] == "profession_crossfit"]
    passport = {
        "experiment_id": experiment_id,
        "status": "executed",
        "evaluation": (
            f"model quality = grouped stratified {cv['n_folds']}-fold CV (out-of-fold predictions for every row); "
            "calibration checked by cross-fitting (each fold calibrated on the other folds). The final model is "
            "trained on all rows and has no held-out test of its own."
        ),
        "dataset": {
            "file_path": data_path.relative_to(ROOT).as_posix(),
            "file_sha256": _sha256(data_path),
            "rows": int(len(cv["assignment"])),
            "rules_version": manifest.get("rules_version"),
        },
        "code_git_commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": None if git_status is None else bool(git_status),
        "row_filters": {"excluded_by_reason": cv["excluded"]["reason"].value_counts().to_dict()},
        "features": TEXT_COLUMNS + final["categorical_columns"],
        "parameters": {
            "catboost": results["config"]["catboost"],
            "trees_final": final["iterations"],
            "trees_per_fold": cv["best_iterations"],
            "svd_components": results["config"]["svd_components"],
            "dropped_columns": results["config"]["drop_columns"],
        },
        "metrics": {
            "headline_pooled_out_of_fold_profession_calibrated": headline.to_dict(orient="records"),
            "fold_mean_std_fold_validation_calibrated": cv["summary"].to_dict(orient="records"),
            "calibration_comparison_file": "calibration_comparison.csv",
        },
        "model_path": model_dir.relative_to(ROOT).as_posix(),
        "input_schema_path": (model_dir / "input_schema.json").relative_to(ROOT).as_posix(),
        "conclusion": None,
    }
    (experiment_dir / "experiment.json").write_text(json.dumps(passport, ensure_ascii=False, indent=2), "utf-8")
    return experiment_dir


def load_final_model(model_dir: Path) -> dict[str, object]:
    """Everything predict_corridor needs, from an ML/models/<id>_final folder."""
    model_dir = Path(model_dir)
    schema = json.loads((model_dir / "input_schema.json").read_text("utf-8"))
    calibration = json.loads((model_dir / "calibration.json").read_text("utf-8"))
    encoder = joblib.load(model_dir / "text_encoder.joblib")
    models = {}
    for target, file_name in schema["models"].items():
        models[target] = CatBoostRegressor()
        models[target].load_model(str(model_dir / file_name))
    return {
        "models": models,
        "vectorizer": encoder["vectorizer"],
        "svd": encoder["svd"],
        "categorical_columns": schema["categorical_columns"],
        "shifts": calibration["shifts"],
    }


def predict_corridor(bundle: dict[str, object], rows: pd.DataFrame) -> pd.DataFrame:
    """Market P25/P50/P75 for the lower ("от") and upper ("до") bound of every
    row, and, where the row states a salary, a verdict below / in / above the
    market (P25-P75 corridor)."""
    rows = rows.reset_index(drop=True)
    features = build_features(rows)
    texts = make_text(features)
    frame = build_model_frame(
        features, bundle["svd"].transform(bundle["vectorizer"].transform(texts)), bundle["categorical_columns"]
    )
    groups = calibration_groups(features).to_numpy()
    out = rows[[c for c in ("source", "vacancy_id", "title", "city", "profession") if c in rows]].copy()
    for target, model in bundle["models"].items():
        predicted = apply_shift_table(predict_log_quantiles(model, frame), groups, bundle["shifts"][target])
        for i, q in enumerate((25, 50, 75)):
            out[f"{target}_p{q}"] = np.round(predicted[:, i], -2)
        if target in rows:
            stated = pd.to_numeric(rows[target], errors="coerce")
            out[f"{target}_stated"] = stated
            out[f"{target}_verdict"] = np.select(
                [stated < out[f"{target}_p25"], stated > out[f"{target}_p75"], stated.notna()],
                ["ниже рынка", "выше рынка", "в рынке"],
                default="",
            )
    return out


# --------------------------------------------------------------------------
# 6. Saving (layout and passport fields follow ML/README.md)
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_experiment(results: dict[str, object]) -> Path:
    """Write metrics, split ids, models and a passport; never overwrite."""
    experiment_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_{EXPERIMENT_NAME}"
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    model_dir = MODELS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)

    results["metrics"].to_csv(experiment_dir / "metrics.csv", index=False)
    results["intervals"].to_csv(experiment_dir / "intervals.csv", index=False)
    results["feature_importance"].to_csv(experiment_dir / "feature_importance.csv", index=False)
    results["split_assignment"][["source", "vacancy_id", "data_file", "stratum", "split"]].to_csv(
        experiment_dir / "split_ids.csv", index=False
    )
    results["excluded"].to_csv(experiment_dir / "excluded_ids.csv", index=False)
    study = results["study"]
    if study is not None:
        study.trials_dataframe().to_csv(experiment_dir / "optuna_trials.csv", index=False)

    for target_name, model in results["models"].items():
        model.save_model(str(model_dir / f"{target_name}.cbm"))
    joblib.dump({"vectorizer": results["vectorizer"], "svd": results["svd"]}, model_dir / "text_encoder.joblib")
    calibration = {
        "method": "per-quantile additive shift of predicted log1p(salary), fitted on validation",
        "apply": "log_pred = sort(sort(model.predict(X)) + shifts); rub = expm1(log_pred)",
        "quantiles": list(QUANTILES),
        "shifts": {target: [float(s) for s in shifts] for target, shifts in results["calibration"].items()},
        "rub_multipliers": {
            target: [float(np.exp(s)) for s in shifts] for target, shifts in results["calibration"].items()
        },
    }
    (model_dir / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), "utf-8")
    input_schema = {
        "text_columns": TEXT_COLUMNS,
        "text_template": "title, description, benefits joined by '\\n'; every field passed through delete_salary",
        "benefits_source": f"extra_attributes_json['{BENEFITS_ATTRIBUTE}']",
        "extra_attribute_columns": EXTRA_ATTRIBUTE_FEATURES,
        "categorical_columns": results["categorical_columns"],
        "missing_category": "__MISSING__",
        "city": "as in the experiment CSV (recovered from location_observed by the dataset builder when empty)",
        "text_encoder": "text_encoder.joblib: TfidfVectorizer -> TruncatedSVD",
        "tfidf_params": {k: list(v) if isinstance(v, tuple) else v for k, v in TFIDF_PARAMS.items()},
        "target_transform": TARGET_TRANSFORM,
        "inverse_transform": "sort, add calibration shifts (calibration.json), sort, expm1",
        "calibration": "calibration.json",
        "quantiles": list(QUANTILES),
        "models": {target: f"{target}.cbm" for target in results["models"]},
    }
    (model_dir / "input_schema.json").write_text(json.dumps(input_schema, ensure_ascii=False, indent=2), "utf-8")

    script = Path(__file__).resolve().relative_to(ROOT).as_posix()
    git_status = _git("status", "--porcelain", "--", script)
    data_files = [Path(results["data_path"]).resolve()]
    manifest = json.loads((DATA_DIR / "manifest.json").read_text("utf-8")) if (DATA_DIR / "manifest.json").exists() else {}
    passport = {
        "experiment_id": experiment_id,
        "status": "executed",
        "dataset": {
            "file_path": [path.relative_to(ROOT).as_posix() for path in data_files],
            "file_sha256": {path.name: _sha256(path) for path in data_files},
            "rows_by_file": results["split_assignment"]["data_file"].value_counts().to_dict(),
            "rules_version": manifest.get("rules_version"),
            "passport_path": None,
            "passport_sha256": None,
            "dataset_id": None,
        },
        "code_git_commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": None if git_status is None else bool(git_status),
        "environment_file": None,
        "features": TEXT_COLUMNS + results["categorical_columns"],
        "preprocessing_version": (
            f"delete_salary + char TF-IDF{TFIDF_PARAMS['ngram_range']} + SVD{results['config']['svd_components']}"
        ),
        "row_filters": {
            "military": "extra_attributes_json.military or MILITARY_PATTERN",
            "military_pattern": MILITARY_PATTERN,
            "monthly_salary_range_rub": [MIN_MONTHLY_RUB, MAX_MONTHLY_RUB],
            "excluded_ids_file": "excluded_ids.csv",
            "excluded_by_reason": results["excluded"]["reason"].value_counts().to_dict(),
        },
        "split": {"rule": SPLIT_RULE, "ids_file": "split_ids.csv", "seed": RANDOM_SEED},
        "parameters": {
            "target_transform": TARGET_TRANSFORM,
            "catboost": results["config"]["catboost"],
            "svd_components": results["config"]["svd_components"],
            "svd_explained_variance": results["svd_explained_variance"],
            "dropped_columns": results["config"]["drop_columns"],
            "best_iteration": {t: m.get_best_iteration() for t, m in results["models"].items()},
            "calibration": calibration,
            "optuna": None
            if study is None
            else {
                "study_name": study.study_name,
                "storage": OPTUNA_STORAGE_PATH.relative_to(ROOT).as_posix(),
                "n_trials": len(study.trials),
                "objective": "mean validation MultiQuantile loss (log-salary) of both bounds",
                "best_value": study.best_value,
                "best_params": study.best_params,
                "trials_file": "optuna_trials.csv",
            },
        },
        "metrics": {
            "note": (
                "validation is used for early stopping, Optuna and calibration (calibrated validation "
                "metrics are ~ideal by construction); test is the unbiased estimate"
            ),
            "per_quantile": results["metrics"].to_dict(orient="records"),
            "intervals": results["intervals"].to_dict(orient="records"),
        },
        "model_path": model_dir.relative_to(ROOT).as_posix(),
        "input_schema_path": (model_dir / "input_schema.json").relative_to(ROOT).as_posix(),
        "conclusion": None,
    }
    (experiment_dir / "experiment.json").write_text(json.dumps(passport, ensure_ascii=False, indent=2), "utf-8")
    return experiment_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="experiment CSV")
    parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS, help="Optuna trials, 0 = no search")
    parser.add_argument("--timeout", type=int, default=OPTUNA_TIMEOUT_SECONDS, help="Optuna time limit, seconds")
    parser.add_argument("--study-name", default=None, help="resume this Optuna study from the SQLite storage")
    parser.add_argument("--cv", type=int, default=0, help="K-fold cross-validation instead of one split (no Optuna)")
    parser.add_argument("--drop-employer", action="store_true", help="with --cv: do not use employer_name_observed")
    parser.add_argument(
        "--final", action="store_true",
        help="CV estimate + per-profession calibration from out-of-fold predictions + model on all rows",
    )
    arguments = parser.parse_args()

    pd.set_option("display.width", 200)
    if arguments.final:
        final_results = run_final(arguments.data, arguments.cv or 5)
        output_dir = save_final(final_results)
        shown = ["calibration", "target", "n", "mape", "median_ape", "pinball_log", "share_below_p50", "coverage_p25_p75"]
        print("\nOut-of-fold quality by calibration scheme (profession_crossfit = the saved model's scheme):")
        print(final_results["comparison"][shown].round(3).to_string(index=False))
        by_prof = final_results["comparison_by_profession"]
        print("\nCoverage P25-P75 by profession, global vs per-profession calibration (cross-fitted):")
        print(
            by_prof[by_prof["calibration"].isin(["global_crossfit", "profession_crossfit"])]
            .pivot_table(index=["target", "profession"], columns="calibration", values="coverage_p25_p75")
            .round(3)
            .to_string()
        )
        print("\nTrees in the final model:", final_results["final"]["iterations"])
        print("Saved to", output_dir.relative_to(ROOT).as_posix(), "and", (MODELS_DIR / output_dir.name).relative_to(ROOT).as_posix())
        raise SystemExit(0)
    if arguments.cv:
        cv_config = default_config()
        if arguments.drop_employer:
            cv_config["drop_columns"] = ["employer_name_observed"]
        cv_results = run_cv(arguments.data, arguments.cv, cv_config)
        output_dir = save_cv(cv_results)
        print("\nMean +- std over folds (test, P50 and P25-P75):")
        print(cv_results["summary"].to_string(index=False))
        print("\nPooled out-of-fold, calibrated:")
        print(cv_results["pooled"].round(3).to_string(index=False))
        print("\nPooled out-of-fold by profession, calibrated:")
        print(cv_results["pooled_by_profession"].round(3).to_string(index=False))
        print("\nSaved to", output_dir.relative_to(ROOT).as_posix())
        raise SystemExit(0)

    experiment_results = run_experiment(arguments.data, arguments.trials, arguments.timeout, arguments.study_name)
    output_dir = save_experiment(experiment_results)

    pd.set_option("display.width", 200)
    print("SVD explained variance:", f"{experiment_results['svd_explained_variance']:.2%}")
    print("\nCalibration multipliers for P25/P50/P75 (rubles):")
    for target, shifts in experiment_results["calibration"].items():
        print(f"  {target}: " + " / ".join(f"x{np.exp(s):.3f}" for s in shifts))
    print("\nPer quantile (share_below should be close to quantile):")
    print(experiment_results["metrics"].round(3).to_string(index=False))
    print("\nP25-P75 intervals (coverage should be close to 0.50):")
    print(experiment_results["intervals"].round(3).to_string(index=False))
    print("\nTop features:")
    print(experiment_results["feature_importance"].head(20).round(2).to_string(index=False))
    print("\nSaved to", output_dir.relative_to(ROOT).as_posix())
