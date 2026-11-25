"""Generate the experiment table from the checked-in JSON passports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re


EXPERIMENTS = Path(__file__).resolve().parent / "experiments"


def repository_link(path):
    return (PurePosixPath("..") / PurePosixPath(path).relative_to("ML")).as_posix()


def package_refs(run):
    for fold in run["fit"]:
        for model in fold["models"]:
            if model["weights"]:
                yield model["weights"]


def paired_pinball(record, scope):
    comparison = record["comparison"]
    if comparison is None:
        score = record["runs"]["baseline"]["metrics"][scope]["quality"].get("scaled_pinball")
        return f"{score:.4f}" if score is not None else "—"
    paired = comparison[scope]
    base = paired["baseline"].get("scaled_pinball")
    candidate = paired["candidate"].get("scaled_pinball")
    return f"{base:.4f} → {candidate:.4f}" if base is not None and candidate is not None else "—"


def training_time(runs, metric):
    cells = [f"{run['training_time'][metric]:.1f}" if run.get("training_time") is not None else "—" for run in runs]
    return " / ".join(cells) if any(cell != "—" for cell in cells) else "—"


def row(record):
    name = record["experiment_id"]
    runs = record["runs"]
    measured = [runs[key] for key in ("baseline", "hypothesis") if key in runs]
    change = record["change"]
    description = f"`{change['parameter']}` → `{change['after']}`" if change else "База"
    if record["replayed_from"]:
        description = f"Проверка весов · {description}"
    coverage = []
    for run in measured:
        score = run["metrics"]["all"]["quality"]
        coverage.append(f"{score['n']}/{score['available_n']}")

    packages = {str(PurePosixPath(ref["path"]).parent)
                for run in measured for ref in package_refs(run) if ref["storage"] == "repository"}
    if packages:
        weights = " · ".join(f"[Пакеты]({repository_link(path)})" for path in sorted(packages))
    else:
        weights = "Локально" if any(ref for run in measured for ref in package_refs(run)) else "—"
    baseline = record.get("reference_baseline", {}).get("package")
    if not baseline:
        baseline = next(package_refs(runs["archived_global"]))
    anchor = f'<a name="run-{name}"></a>'
    return "| " + " | ".join([
        f"{anchor}[`{name}`]({name}/experiment.json)",
        description, paired_pinball(record, "all"), paired_pinball(record, "gold"), " → ".join(coverage),
        training_time(measured, "wall_seconds"), training_time(measured, "process_cpu_seconds"), weights,
        f"[Архивная база]({repository_link(baseline['path'])})",
    ]) + " |"


def render_index(directory):
    rows, historical = [], []
    for path in sorted(directory.glob("*/experiment.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        name = path.parent.name
        if record.get("version") in {2, 3} and "runs" in record:
            if record["experiment_id"] != name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError(f"Invalid experiment ID in {path}")
            rows.append(row(record))
        else:
            historical.append(f"- [{name}]({name}/experiment.json)")

    lines = [
        "# Журнал ML-экспериментов", "",
        "Одна строка соответствует одному запуску ноутбука. ID ведёт к паспорту прогона. Pinball указан на общих ID при сравнении двух вариантов, охват — по всем тестовым строкам. Время обучения суммировано по фолдам в порядке база / вариант.", "",
        "[Пайплайн экспериментов](../salary_experiment_pipeline.ipynb) · [Хранение пакетов](../models/README.md)", "",
        "Пересобрать таблицу из JSON можно из корня проекта командой `python ML/update_experiment_index.py`.", "",
        "## Сопоставимые прогоны", "",
        "| ID и паспорт | Изменение | Pinball, общий тест ↓ | Pinball, gold ↓ | Прогнозы на общем тесте | Время по часам, с | Время CPU, с | Пакеты прогона | Архивный бейзлайн |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        *rows, "", "## Архивные исследования", "",
        *historical, "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that the Git table matches the passports.")
    args = parser.parse_args()
    index = EXPERIMENTS / "README.md"
    expected = render_index(EXPERIMENTS)
    if args.check:
        if not index.is_file() or index.read_text(encoding="utf-8") != expected:
            parser.exit(1, "Refresh with python ML/update_experiment_index.py\n")
    else:
        index.write_text(expected, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
