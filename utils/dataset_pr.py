"""Render a short, factual dataset PR from actual CSVs; reject staged browser dumps."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    columns = reader.fieldnames or []
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Missing or repeated CSV columns")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV rows")
    return columns, rows


def browser_evidence(path):
    path = Path(path)
    if path.parts[:2] != ("Data", "avito_runs"):
        return False
    if any(part in {"snapshots", "screenshots"} for part in path.parts):
        return True
    return path.suffix.lower() in {
        ".txt", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".html", ".mhtml"}


def validate_automation_changes(changes, code_paths):
    """Require each changed source plus a regression and updated skill, not a token note."""
    if not code_paths:
        return
    noted = {change.get('file') for change in changes
             if change.get('reason', '').strip() and change.get('validation', '').strip()}
    missing = set(code_paths) - noted
    if missing:
        raise ValueError('Record collector changes for every source/test/skill file: ' + ', '.join(sorted(missing)))
    if not any(Path(path).suffix.lower() in {'.py', '.js'} for path in code_paths):
        return
    if not any(path and (Path(path).name.startswith('test_') or '/tests/' in path) for path in noted):
        raise ValueError('Collector fix requires a recorded regression test')
    if 'Scraper/skills/collect-avito/SKILL.md' not in noted:
        raise ValueError('Collector fix requires an updated and recorded SKILL.md')


def render(summary, before, after, batches):
    old_columns, old_rows = before
    new_columns, new_rows = after
    key = lambda row: (row["source"], row["id"])
    old_keys, new_keys = {key(r) for r in old_rows}, {key(r) for r in new_rows}
    if len(old_keys) != len(old_rows) or len(new_keys) != len(new_rows):
        raise ValueError("Duplicate dataset keys")
    if not old_keys <= new_keys:
        raise ValueError("Publication removes existing vacancy keys")
    observations = [(source, r["id"]) for source, rows in batches for r in rows]
    if not observations or any(not identity for _, identity in observations):
        raise ValueError("No accepted observations")
    if not set(observations) <= new_keys:
        raise ValueError("Batch vacancies are absent from shared CSV")
    added = [c for c in new_columns if c not in old_columns]
    accepted = "; ".join(f"{source}: {identity}" for source, identity in observations)
    new = len(set(observations) - old_keys)
    repeated = len(observations) - new
    return (summary.strip() + "\n\n"
            f"- Приняты ID: {accepted}.\n"
            f"- Новые ключи: {new}; повторные наблюдения: {repeated}.\n"
            f"- Общая таблица: {len(old_rows)} → {len(new_rows)} строк; "
            f"{len(old_columns)} → {len(new_columns)} колонок.\n"
            f"- Новые характеристики: {', '.join(added) if added else 'нет'}.\n\n"
            "Файлы: плоский CSV с полными описаниями, raw JSON, план и паспорт. "
            "Проверка общей таблицы: --check пройден.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--summary", required=True, help="One factual paragraph about selection and geography")
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    staged = subprocess.check_output([
        "git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"
    ], cwd=ROOT).decode().split("\0")
    forbidden = [path for path in staged if path and browser_evidence(path)]
    if forbidden:
        raise SystemExit("Keep browser evidence local; unstage: " + ", ".join(forbidden))
    base = subprocess.check_output(["git", "show", f"{args.base}:Data/dataset/vacancies.csv"], cwd=ROOT)
    current = (ROOT / "Data/dataset/vacancies.csv").read_bytes()
    # Validate source hashes and repository-relative paths before opening batch CSVs.
    subprocess.run(["python3", "utils/build_dataset.py", "--check"], cwd=ROOT, check=True)
    registry = json.loads((ROOT / "Data/dataset/sources.json").read_text())
    entries = {entry["run_id"]: entry for entry in registry["exports"]}
    batches = []
    automation_changes = []
    for run_id in args.run_id:
        entry = entries[run_id]
        _, rows = read_csv((ROOT / entry["path"]).read_bytes())
        batches.append((entry["source"], rows))
        manifest_path = ROOT / 'Scraper/runs' / run_id / 'manifest.json'
        if manifest_path.exists():
            automation_changes.extend(json.loads(manifest_path.read_text()).get('automation_changes', []))
    code = [p for p in staged if p.startswith(('Scraper/', 'utils/')) and
            (p.endswith(('.py', '.js', 'SKILL.md')) and '/runs/' not in p)]
    validate_automation_changes(automation_changes, code)
    body = render(args.summary, read_csv(base), read_csv(current), batches)
    if automation_changes:
        body += '\nИзменения сборщика:\n' + ''.join(
            f'- `{c["file"]}`: {c["reason"]}. Проверка: {c["validation"]}.\n' for c in automation_changes)
    args.output.write_text(body, encoding="utf-8")
    print(f"PR body saved: {args.output}")


if __name__ == "__main__":
    main()
