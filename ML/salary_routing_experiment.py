"""Evaluation and journal writing for the salary routing notebook."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

import run_salary_routing as routing
import salary_routing as r

ROOT = r.ROOT
OUTPUT = ROOT / "ML/experiments"
WEIGHTS_ROOT = ROOT / "ML/.cache/salary_routing_weights"
NOTEBOOK = "ML/salary_experiment_pipeline.ipynb"
SPLIT = r.EXPERIMENT_DIR / "split_ids.csv"
POLICY = r.EXPERIMENT_DIR / "routing_policy.json"
ARCHIVED_WEIGHTS = r.MODEL_DIR / "global.joblib"
ARCHIVED_WEIGHTS_SHA256 = "242d36a8fc63f855ba130156594eb1690c2c8320004375de155c4951eb08c71b"
KEY = ["source", "id"]
RULES = ("native_exact_or_range_v1", "native_exact_only_v1")
JOURNAL_VERSION = 3
JOURNAL_SOURCE = "ML/salary_routing_experiment.py"
PREVIOUS_JOURNAL_SHA256 = {
    2: "bc06a34d4313683aa8b4ade333a1f1bbdcafb65e4655417cbc8bbe06770768e6",
    3: "91489737487c36a39f9d0fb10a25a55ad4003894aa7d1a276e2ffadc8aa869c8",
}
SOURCES = [
    JOURNAL_SOURCE, "ML/run_salary_routing.py",
    "ML/salary_routing.py", "ML/salary_gold.py", "ML/experiments/expiriemnts.py",
    "ML/korchagin_salary_research.ipynb", "Analytics/analyze_priority_salary_formats.py",
    "Analytics/build_priority_segment_passports.py", "Analytics/segment_normality.py",
    "utils/build_dataset.py", "utils/delete_salary.py",
]
QUALITY_FIELDS = {"n", "available_n", "groups", "scaled_pinball", "scaled_p50_mae", "coverage"}
PAIRED_FIELDS = ("scaled_pinball", "scaled_p50_mae", "coverage")


def file_ref(path, *, relative_to=ROOT, text_file=True):
    data = path.read_bytes()
    if text_file:
        data = data.replace(b"\r\n", b"\n")
    return {"path": path.relative_to(relative_to).as_posix(), "sha256": hashlib.sha256(data).hexdigest()}


def code_version():
    files = {name: file_ref(ROOT / name)["sha256"] for name in SOURCES}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        head = git("rev-parse", "HEAD")
        commit = None if git("status", "--porcelain", "--", NOTEBOOK, *SOURCES) else head
    except (OSError, subprocess.CalledProcessError):
        head = commit = None
    return {"notebook": NOTEBOOK, "commit": commit, "checkout_head": head, "files_sha256_lf": files}


def compatible_replay_code(record, current):
    prior = record["code"]["files_sha256_lf"]
    if record["version"] == JOURNAL_VERSION and prior == current:
        return True
    # These journal revisions used the same model and evaluation code.
    previous = PREVIOUS_JOURNAL_SHA256.get(record["version"])
    return previous is not None and prior == {**current, JOURNAL_SOURCE: previous}


def machine_details():
    system = platform.system()
    cpu = platform.processor()
    if system == "Windows":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif system == "Linux":
        try:
            fields = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            cpu = next((line.partition(":")[2].strip() for line in fields if line.partition(":")[0].strip() == "model name"), cpu)
        except OSError:
            pass
    return {
        "os": system, "os_release": platform.release(), "os_version": platform.version(),
        "architecture": platform.machine(), "cpu_model": cpu or None, "logical_cpus": os.cpu_count(),
    }


def hypothesis(config, change):
    if not isinstance(change, dict) or len(change) != 1:
        raise ValueError("Set one parameter in CHANGE, or use None for the baseline only.")
    (parameter, value), = change.items()
    choices = {"scheme": routing.SCHEMES, "training_target_rule": RULES}
    limits = {"global_iterations": (1, None), "depth": (1, 16)}
    if parameter in limits and type(value) is int:
        lower, upper = limits[parameter]
        if value < lower or (upper is not None and value > upper):
            raise ValueError(f"Value out of range for {parameter}: {value}")
    elif value not in choices.get(parameter, ()) and not (parameter == "text_features" and type(value) is bool):
        raise ValueError(f"Unknown parameter or invalid value: {change!r}")
    if config[parameter] == value:
        raise ValueError(f"{parameter} must differ from the baseline.")
    if parameter == "global_iterations" and config["scheme"] not in {"global", "hybrid", "manual"}:
        raise ValueError("The selected scheme does not train a global model.")
    if parameter in {"depth", "text_features"} and config["scheme"] == "empirical":
        raise ValueError(f"The empirical scheme does not use {parameter}.")
    delta = {"parameter": parameter, "before": config[parameter], "after": value}
    return {**config, parameter: value}, delta


def training_rows(rows, rule):
    """Apply a training/validation target rule in the original evaluation units."""
    if rule == "native_exact_or_range_v1":
        return rows
    if rule == "native_exact_only_v1":
        return rows.loc[rows["shape"].eq("exact")]
    raise ValueError(f"Unknown target rule: {rule}")


def check_split(archived, fold, train, val, test):
    fields = KEY + ["group"]
    for part, rows in (("train", train), ("validation", val), ("test", test.loc[test.eligible])):
        expected = archived.loc[archived.fold.eq(fold) & archived.part.eq(part), fields]
        pd.testing.assert_frame_equal(
            rows[fields].sort_values(KEY).reset_index(drop=True),
            expected.sort_values(KEY).reset_index(drop=True), check_dtype=False,
        )


def load_pinned_snapshot(config, dataset):
    expected = config["dataset_sha256"]
    revision = config["snapshot_revision"]
    dataset_dir = ROOT / "Data/dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if dataset is not None or manifest["sha256"] == expected:
        raw, sha = r.gold.load_snapshot(dataset)
        if sha != expected:
            raise ValueError(f"Snapshot SHA-256 mismatch: {sha} != {expected}")
        return raw

    def read_git(path):
        return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:Data/dataset/{path}"])

    manifest = json.loads(read_git("manifest.json"))
    if manifest["sha256"] != expected:
        raise ValueError(f"Revision {revision} does not contain the pinned snapshot {expected}.")
    return r.gold.read_snapshot(r.gold.assemble(manifest, read_git))


def score(rows):
    values = rows[r.QUANTILES].to_numpy(dtype=float)
    metrics = r.gold.score(rows, values)
    good = np.isfinite(values).all(axis=1)
    if good.any():
        actual, scale = rows.loc[good, ["y", "scale"]].to_numpy(dtype=float).T
        metrics["scaled_pinball"] = (r.gold.losses(actual, values[good]).mean(axis=1) / scale).mean()
        metrics["scaled_p50_mae"] = (np.abs(actual - values[good, 1]) / scale).mean()
    return metrics


def metrics(rows, counts):
    out = {}
    for scope, sample in (("all", rows), ("gold", rows.loc[rows.priority])):
        coverage = counts.loc[scope].to_dict()
        out[scope] = {
            "coverage": {**coverage, "prediction_share": coverage["scored"] / coverage["total"],
                         "test_prediction_share": coverage["scored_eligible"] / coverage["eligible"]},
            "quality": {key: value for key, value in score(sample).items() if key in QUALITY_FIELDS},
            "by_unit_and_tax": [dict(unit=unit, tax=tax, **score(group))
                                for (unit, tax), group in sample.groupby(["unit", "tax"])],
        }
    return out


def paired(rows, old, current_metrics, baseline_metrics):
    index = [*KEY, "fold", "group", "priority", "unit", "tax"]
    fixed = [*index, "y", "scale"]
    pd.testing.assert_frame_equal(rows[fixed], old[fixed], check_dtype=False, rtol=1e-12, atol=1e-12)
    shared = rows[r.QUANTILES].notna().all(axis=1) & old[r.QUANTILES].notna().all(axis=1)
    out = {}
    for scope, select in (("all", shared), ("gold", shared & rows.priority)):
        candidate, baseline = rows.loc[select], old.loc[select]
        a, b = score(candidate), score(baseline)
        delta = {key: a[key] - b[key] for key in PAIRED_FIELDS if key in a}
        actual = candidate.y.to_numpy()
        loss_candidate = r.gold.losses(actual, candidate[r.QUANTILES].to_numpy()).mean(axis=1)
        loss_baseline = r.gold.losses(actual, baseline[r.QUANTILES].to_numpy()).mean(axis=1)
        interval = routing.paired_group_interval((loss_candidate - loss_baseline) / candidate.scale, candidate.group)
        out[scope] = {
            "n": len(candidate), "groups": a["groups"],
            "baseline": {key: a_value for key, a_value in b.items() if key in PAIRED_FIELDS},
            "candidate": {key: a_value for key, a_value in a.items() if key in PAIRED_FIELDS},
            "candidate_minus_baseline": delta,
            "scaled_pinball_delta_group_bootstrap_95pct": interval,
            "prediction_share_delta": current_metrics[scope]["coverage"]["prediction_share"]
            - baseline_metrics[scope]["coverage"]["prediction_share"],
            "test_prediction_share_delta": current_metrics[scope]["coverage"]["test_prediction_share"]
            - baseline_metrics[scope]["coverage"]["test_prediction_share"],
        }
    return out


def inventory(bundles, train, val, weights):
    records = []
    for name, bundle in bundles.items():
        is_cat = bundle["kind"] == "catboost"
        model = bundle["model"]["models"] if is_cat else None
        valid = int(r.match(val, bundle["filters"]).sum())
        text_columns = [column for column in model.feature_names_ if column not in r.FEATURES] if is_cat else []
        records.append({
            "name": name, "model": bundle["kind"], "filters": bundle["filters"],
            "contexts": bundle["contexts"], "training_rows": bundle["fit_points"],
            "training_groups": train.loc[r.match(train, bundle["filters"]), "group"].nunique(),
            "validation_rows": valid, "calibration_rows": valid if is_cat and bundle["model"]["shifts"] is not None else 0,
            "hyperparameters": model.get_all_params() if is_cat else None,
            "fit_parameters": model.get_params() if is_cat else None,
            "iteration_limit": model.get_params()["iterations"] if is_cat else None,
            "tree_count": model.tree_count_ if is_cat else None,
            "description_derived_columns": text_columns,
            "weights": weights.get(name),
        })
    return records


def check_archived_predictions(predictions):
    path = r.EXPERIMENT_DIR / "predictions_oof.csv"
    prior = pd.read_csv(path, dtype={"id": str}).query("scheme == 'global' and fold == 0")
    prior = prior.sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions[KEY], prior[KEY])
    actual, expected = (frame[r.QUANTILES].to_numpy() for frame in (predictions, prior))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9, equal_nan=True)
    return {"predictions": file_ref(path), "max_absolute_difference": np.nanmax(np.abs(actual - expected))}


def load_weights(reference, weights_root):
    if not reference:
        raise ValueError("The experiment has no weights for this model. Retrain from the recorded configuration.")
    root = {"repository": ROOT, "local_weights": weights_root}[reference["storage"]]
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Weight path must be inside {root}")
    if not path.is_file():
        raise FileNotFoundError(f"Weights are missing: {path}. Restore the bundle or retrain from the recorded configuration.")
    if file_ref(path, relative_to=root, text_file=False)["sha256"] != reference["sha256"]:
        raise ValueError(f"Weight checksum mismatch: {path}")
    return joblib.load(path)


def run_case(name, config, folds, policy, weight_stage, weight_destination, *, replay, weights_root):
    archived = name == "archived_global"
    scheme = config["scheme"]
    route = {"scheme": scheme, "implementation": "ML/run_salary_routing.py::predict_routes",
             "manual_policy": file_ref(POLICY) if scheme == "manual" else None}
    if replay is not None and route != replay["routing"]:
        raise ValueError(f"The routing policy differs from the one recorded for {name}.")
    if archived:
        folds = {0: folds[0]}
    predictions, coverage, training_ids, fit = [], [], {}, []
    for fold, (train, val, test) in folds.items():
        train, val = (training_rows(part, config["training_target_rule"]) for part in (train, val))
        if config["training_target_rule"] != RULES[0]:
            training_ids[str(fold)] = {"train": train.id.tolist(), "validation": val.id.tolist()}
        weights = {}
        training_time = None
        if archived:
            weights["global"] = {"format": "joblib", "storage": "repository",
                                 "path": ARCHIVED_WEIGHTS.relative_to(ROOT).as_posix(), "sha256": ARCHIVED_WEIGHTS_SHA256}
        elif replay is not None:
            prior_fit = next(entry for entry in replay["fit"] if entry["fold"] == fold)
            weights = {model["name"]: model["weights"] for model in prior_fit["models"]}
        if archived or replay is not None:
            bundles = {model: load_weights(ref, weights_root) for model, ref in weights.items()}
        else:
            start_wall, start_cpu = perf_counter(), process_time()
            bundles, _ = routing.fit_routes(train, val, fold, (scheme,), global_iterations=config["global_iterations"],
                                            depth=config["depth"], text=config["text_features"])
            training_time = {"process_cpu_seconds": process_time() - start_cpu, "wall_seconds": perf_counter() - start_wall}
            if weight_stage is not None:
                for model, bundle in bundles.items():
                    path = weight_stage / f"{name}_fold_{fold}_{model}.joblib"
                    joblib.dump(bundle, path, compress=3)
                    ref = file_ref(path, relative_to=weight_stage, text_file=False)
                    ref["path"] = (weight_destination.relative_to(weights_root) / ref["path"]).as_posix()
                    weights[model] = {"format": "joblib", "storage": "local_weights", **ref}
        _, (values, names, _) = next(routing.predict_routes(test, bundles, policy, (scheme,)))
        coverage.extend(routing.full_coverage(test, scheme, names))
        eligible = test.eligible.to_numpy(dtype=bool)
        rows = test.loc[eligible, [*KEY, "group", "priority", "unit", "tax", "y", "scale"]].copy()
        rows[r.QUANTILES] = values[eligible]
        predictions.append(rows.assign(fold=fold))
        fit.append({"fold": fold, "training_time": training_time, "models": inventory(bundles, train, val, weights)})
        print(f"{name}, fold {fold}: {len(train)} train, {len(val)} validation, {len(rows)} test", flush=True)

    rows = pd.concat(predictions).sort_values(KEY).reset_index(drop=True)
    counts = pd.DataFrame(coverage).drop(columns="scheme").groupby("scope").sum()
    text_columns = sorted({column for fold in fit for model in fold["models"]
                           for column in model["description_derived_columns"]})
    sample = {"split": "data.splits", "folds": list(folds), "overrides": training_ids}
    if replay is not None and sample != replay["sample"]:
        raise ValueError(f"Fit IDs differ from the ones recorded for {name}.")
    measured = metrics(rows, counts)
    if replay is not None:
        for scope, result in measured.items():
            prior = replay["metrics"][scope]
            if result["coverage"] != prior["coverage"] or result["quality"].keys() != prior["quality"].keys():
                raise ValueError(f"Replayed coverage differs for {name}: {scope}.")
            keys = list(result["quality"])
            np.testing.assert_allclose([result["quality"][key] for key in keys],
                                       [prior["quality"][key] for key in keys], rtol=1e-9, atol=1e-10)
    times = [fold["training_time"] for fold in fit if fold["training_time"] is not None]
    result = {
        "config": config, "sample": sample, "routing": route, "fit": fit, "metrics": measured,
        "training_time": {metric: sum(fold[metric] for fold in times) for metric in times[0]} if times else None,
        "features": {
            "catboost_columns": r.FEATURES if scheme != "empirical" else [],
            "description_derived_columns": text_columns, "description_used_for_duplicate_grouping": True,
            "text_encoder": {"method": "TF-IDF + TruncatedSVD", "inputs": r.gold.TEXT,
                             "fit_scope": "train only", "tfidf": r.gold.ssr.TFIDF_PARAMS,
                             "svd": {"requested_components": r.gold.TEXT_COMPONENTS,
                                     "random_state": r.gold.ssr.RANDOM_SEED},
                             "cleaning": "ML/salary_gold.py::texts"} if text_columns else None,
        },
    }
    if archived:
        result["archived_prediction_check"] = check_archived_predictions(rows)
    return result, rows


def run_experiment(name, baseline, *, change=None, replay_path=None, dataset=None, save_weights=True, table_row_url=None,
                   output_root=OUTPUT, weights_root=WEIGHTS_ROOT):
    """Execute the notebook cases on one pinned snapshot and publish one JSON record."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Use an experiment ID consisting of letters, digits, dashes and underscores.")
    if table_row_url is not None:
        url = urlsplit(table_row_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("Use the full URL of the experiment's row in the shared table.")
    output_root = Path(output_root).resolve()
    weights_root = Path(weights_root).resolve()
    destination, weights_destination = output_root / name, weights_root / name
    if destination.exists() or weights_destination.exists():
        raise FileExistsError(f"Choose a new experiment ID: {name}")
    code = code_version()
    machine = machine_details()
    libraries = {p: version(p) for p in ("catboost", "joblib", "numpy", "pandas", "scikit-learn", "statsmodels")}
    replay = None
    if replay_path is not None:
        replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
        if (not compatible_replay_code(replay, code["files_sha256_lf"])
                or libraries != {key: replay["libraries"][key] for key in libraries}):
            raise ValueError("Use the code and library versions recorded in the source JSON to restore weights.")
        baseline = replay["runs"]["baseline"]["config"]
        recorded_change = replay["change"]
        change = {recorded_change["parameter"]: recorded_change["after"]} if recorded_change else None

    configs = {"baseline": baseline}
    delta = None
    if change is not None:
        configs["hypothesis"], delta = hypothesis(baseline, change)
    raw = load_pinned_snapshot(baseline, dataset)
    prepared = r.prepare(raw)
    archived = pd.read_csv(SPLIT, dtype={"id": str}, keep_default_na=False)
    folds = {fold: routing.split_fold(prepared, fold) for fold in range(3)}
    split_ids = {}
    for fold, (train, val, test) in folds.items():
        check_split(archived, fold, train, val, test)
        split_ids[str(fold)] = {"train": train.id.tolist(), "validation": val.id.tolist(),
                                "test": test.loc[test.eligible, "id"].tolist()}
    if replay is not None and split_ids != replay["data"]["splits"]:
        raise ValueError("The training, validation or test IDs differ from the source record.")

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    write_weights = save_weights and replay is None
    staging_root = weights_root if write_weights else output_root
    staging_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{name}-", dir=staging_root) as staging, TemporaryDirectory(
        prefix=f".{name}-", dir=output_root
    ) as journal:
        runs, predictions = {}, {}
        for case, config in configs.items():
            runs[case], predictions[case] = run_case(
                case, config, folds, policy, Path(staging) if write_weights else None,
                weights_destination, replay=replay["runs"][case] if replay else None, weights_root=weights_root,
            )
        # The archived model has its own training specification and fold.
        archive_config = {**baseline, "scheme": "global", "global_iterations": 240, "depth": 6,
                          "training_target_rule": RULES[0], "text_features": False}
        runs["archived_global"], _ = run_case("archived_global", archive_config, folds, policy, None,
                                               weights_destination, replay=None, weights_root=weights_root)
        comparison = paired(predictions["hypothesis"], predictions["baseline"], runs["hypothesis"]["metrics"],
                            runs["baseline"]["metrics"]) if delta is not None else None
        report = {
            "version": JOURNAL_VERSION, "experiment_id": name, "change": delta, "code": code,
            "libraries": {**libraries, "python": platform.python_version()},
            "machine": machine,
            "training_time_scope": "Feature encoding, model fitting and calibration, measured around fit_routes. "
                                   "Excludes loading, snapshot preparation, evaluation and saving packages.",
            "experiment_table": {"experiment_id": name, "row_url": table_row_url},
            "reference_baseline": {
                "experiment_id": r.RUN, "fold": 0,
                "package": runs["archived_global"]["fit"][0]["models"][0]["weights"],
            },
            "replayed_from": {"experiment_id": replay["experiment_id"],
                              "sha256": file_ref(Path(replay_path), relative_to=Path(replay_path).parent)["sha256"]}
            if replay else None,
            "data": {"path": "Data/dataset/vacancies.csv", "revision": baseline["snapshot_revision"],
                     "sha256": baseline["dataset_sha256"], "rows": len(raw), "id_key": KEY, "source": "avito",
                     "archived_split": file_ref(SPLIT), "splits": split_ids},
            "rules": {
                "segmentation": {"version": "work_mode_unit_tax_role_city_v1",
                                 "implementation": "ML/salary_routing.py::work_modes", "keys": r.CELL},
                "target": {"implementation": "ML/salary_gold.py::native_salary",
                           "evaluation_version": RULES[0], "catboost_fit_transform": "log1p",
                           "catboost_prediction_transform": "expm1"},
                "minimum_context_rows_and_groups": r.MIN_CONTEXT,
                "minimum_segment_rows_and_groups": r.MIN_SEGMENT,
                "minimum_segment_validation_rows": r.MIN_SEGMENT_VAL,
            },
            "evaluation": {"grouping": "employer/duplicate groups with a separate validation selection",
                           "gold": "priority_vacancies within each held-out fold",
                           "normalization": "training median per unit and tax before the change",
                           "interval_nominal_coverage": 0.50},
            "runs": runs, "comparison": comparison,
        }
        folder = Path(journal)
        journal_path = folder / "experiment.json"
        journal_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=lambda value: value.item()) + "\n",
            encoding="utf-8",
        )
        if write_weights:
            copyfile(journal_path, Path(staging) / journal_path.name)
            Path(staging).rename(weights_destination)
        folder.rename(destination)
    return destination / "experiment.json"
