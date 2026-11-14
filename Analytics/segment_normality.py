"""Shared data loading, segment filters and Lilliefors diagnostics for the two notebook copies."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multitest import multipletests

from build_priority_segment_passports import CITIES, EXPECTED_SHA256, PROFESSIONS


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = "Data/dataset/vacancies.csv"
SNAPSHOTS = {
    "info": (
        "6d1e0cfe560a212b32579dc32ce8a100ba85040b",
        "24350fd63bceb2f549009ae9842aaaf3bbdca6f918a46873c692c0ce58c4c3c9",
    ),
    "decision": ("7450447f481931cb62cef837c52d21216e53b83c", EXPECTED_SHA256),
}
SEGMENT_KEYS = ["profession_resolved", "city_resolved"]
PRIORITY = pd.MultiIndex.from_product([PROFESSIONS, CITIES], names=SEGMENT_KEYS)
ALPHA = 0.05
MIN_TEST_N = 20  # Matches the original salary notebook's diagnostic sample-size threshold.
REJECT = "H₀ отвергнута"
NO_REJECT = "H₀ не отвергнута"


def load_snapshot(name):
    commit, expected_hash = SNAPSHOTS[name]
    source = f"{commit}:{DATASET_PATH}"
    try:
        blob = subprocess.check_output(["git", "show", source], cwd=ROOT, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Не найден {source}. Запустите ноутбук из полного Git-клона репозитория."
        ) from error
    digest = sha256(blob).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"{source}: ожидался SHA-256 {expected_hash}, получен {digest}")
    return pd.read_csv(BytesIO(blob), low_memory=False), source, digest


def priority_vacancies(df):
    """Use the profession, city and exclusions from the agreed segment passports."""
    df = df.drop_duplicates(["source", "id"], keep="last")
    scope = df["profession_resolved"].isin(PROFESSIONS) & df["city_resolved"].isin(CITIES)
    vahta = df["График"].str.contains("вахт", case=False, na=False) | df["title"].str.contains(
        "вахт", case=False, na=False
    )
    remote = df["Формат работы"].str.contains("удал", case=False, na=False) | df[
        "run_id"
    ].str.contains("remote", case=False, na=False)
    military = df["military"].astype("string").str.casefold().isin(["true", "1", "да", "yes"])
    return df.loc[scope & ~(vahta | remote | military)]


def lilliefors_row(values, skip_reason=None):
    x = values.dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    unique = np.unique(x).size
    result = {"n": len(x), "unique": unique, "D": np.nan, "p": np.nan}
    if skip_reason:
        return dict(result, status=skip_reason)
    if not len(x):
        return dict(result, status="Нет точечных наблюдений")
    if unique < 2:
        return dict(result, status="Константа, тест неприменим")
    if len(x) < MIN_TEST_N:
        return dict(result, status=f"n < {MIN_TEST_N}")
    statistic, p = lilliefors(x, dist="norm", pvalmethod="table")
    return dict(result, D=statistic, p=p, status=REJECT if p <= ALPHA else NO_REJECT)


def holm(table):
    """Adjust only the tests run in the supplied family, keeping skipped rows visible."""
    table = table.copy()
    tested = table["p"].notna()
    table["p_Holm"] = np.nan
    table["status_Holm"] = table["status"]
    if tested.any():
        reject, adjusted, _, _ = multipletests(table.loc[tested, "p"], alpha=ALPHA, method="holm")
        table.loc[tested, "p_Holm"] = adjusted
        table.loc[tested, "status_Holm"] = np.where(reject, REJECT, NO_REJECT)
    return table
