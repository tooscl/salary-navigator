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

    features = pd.concat([df[FEATURE_COLUMNS], extras[list(EXTRA_ATTRIBUTE_FEATURES) + ["benefits"]]], axis=1)
    for column in features.columns:
        # Only strings: delete_salary(NaN) would return the string 'nan'.
        features[column] = features[column].map(lambda v: delete_salary(v) if isinstance(v, str) else v)
        features[column] = features[column].replace("", pd.NA)
    features[SALARY_KIND_COLUMN] = df[SALARY_KIND_COLUMN]

    targets = df[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    assert targets.notna().any(axis=1).all(), "A training row has no salary target"
    assert (targets.isna() | (targets > 0)).all().all(), "Salary bounds must be positive for log1p"
    return df, features, targets, excluded


# --------------------------------------------------------------------------
# 2. Split
# --------------------------------------------------------------------------

def assign_splits(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Grouped random 80/10/10 split stratified by profession x city.

    A group is a likely duplicate only when employer, title and city are all
    known and identical after salary redaction. Otherwise a vacancy is its own
    group, so missing employer names cannot merge unrelated ads. Inside every
    profession x city cell groups are shuffled with a fixed seed, so each cell
    is represented in train, validation and test in roughly 80/10/10 shares.
    """
    duplicate_fields = features[["employer_name_observed", "title", "city"]].fillna("").astype(str)
    has_fingerprint = duplicate_fields.ne("").all(axis=1)
    fingerprint = duplicate_fields.agg("|".join, axis=1)
    fallback_group = "vacancy:" + df["source"].astype(str) + ":" + df["vacancy_id"].astype(str)
    duplicate_group = fingerprint.where(has_fingerprint, fallback_group)

    stratum = (
        features["profession"].fillna("__MISSING__").astype(str) + " | " + features["city"].fillna("__MISSING__").astype(str)
    )
    groups = (
        pd.DataFrame({"duplicate_group": duplicate_group, "stratum": stratum, "rows": 1})
        .groupby("duplicate_group", as_index=False)
        .agg(stratum=("stratum", "first"), rows=("rows", "sum"))
        .sort_values("duplicate_group", kind="stable")
    )
    groups["random_key"] = np.random.default_rng(RANDOM_SEED).random(len(groups))
    groups = groups.sort_values(["stratum", "random_key"], kind="stable")
    stratum_rows = groups.groupby("stratum")["rows"].transform("sum")
    # Share of the cell's rows up to the middle of this group: a big group is
    # placed by its centre, so it cannot push a whole cell past a boundary.
    groups["position"] = (groups.groupby("stratum")["rows"].cumsum() - groups["rows"] / 2) / stratum_rows

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

    parts = {}
    for name in ("train", "validation", "test"):
        index = split_assignment.index[split_assignment["split"] == name]
        part_features = features.loc[index]
        parts[name] = {
            "features": part_features,
            "texts": make_text(part_features),
            "targets": targets.loc[index].reset_index(drop=True),
        }
    return {
        "data_path": data_path,
        "excluded": excluded,
        "split_assignment": split_assignment,
        "parts": parts,
        "text_encoders": {},
    }


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
    arguments = parser.parse_args()

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
