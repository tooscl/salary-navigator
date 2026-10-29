"""Build the shared CSV offline from explicitly registered, immutable exports."""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

try:
    from . import title_flags
except ImportError:  # Direct `python3 utils/build_dataset.py` entry point.
    import title_flags

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1"
METADATA = ["source", "region", "first_seen_at", "last_seen_at", "run_id",
            "run_ids", "enrichment_status", "enriched_fields", "enrichment_version"]
GENERATED = {"source", "region", "first_seen_at", "last_seen_at", "run_id", "run_ids"}
REQUIRED = {"id", "source_url", "observed_at", "city", "profession", "title",
            "description", "salary_raw", "analysis_status", "export_eligibility"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp requires timezone: {value}")
    return parsed


def parts(value):
    return [item.strip() for item in value.split(";") if item.strip()]


def build(root, registry):
    """Return deterministic output bytes; reject ambiguous observations."""
    if registry.get("schema_version") != VERSION or not registry.get("exports"):
        raise ValueError("Expected schema_version=1 and nonempty exports")
    records, columns, inputs = {}, [], []
    runs = set()
    for entry in sorted(registry["exports"], key=lambda item: item["run_id"]):
        run_id, source = entry["run_id"], entry["source"]
        if not run_id or not source or run_id in runs:
            raise ValueError("Expected unique nonempty run_id and nonempty source")
        runs.add(run_id)
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source must be inside repository: {path}")
        data = (root / path).read_bytes()
        if digest(data) != entry["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
        correction = entry.get('profession_correction')
        if correction:
            correction_path = Path(correction['path'])
            if correction_path.is_absolute() or '..' in correction_path.parts or not (root / correction_path).resolve().is_relative_to(root.resolve()):
                raise ValueError('Profession correction must be inside repository')
            if digest((root / correction_path).read_bytes()) != correction['sha256']:
                raise ValueError('Profession correction SHA-256 mismatch')
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        names = reader.fieldnames or []
        if len(names) != len(set(names)) or not REQUIRED.issubset(names) or GENERATED.intersection(names):
            raise ValueError(f"Invalid or reserved columns: {path}")
        for name in names:
            if not name or name != " ".join(name.split()):
                raise ValueError(f"Noncanonical column: {name!r}")
            if name not in columns:
                columns.append(name)
        seen = set()
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row: {path}")
            identity = row["id"]
            if not identity or identity != identity.strip() or identity in seen:
                raise ValueError(f"Empty/duplicate ID in {path}: {identity}")
            seen.add(identity)
            if row["analysis_status"] != "validated" or row["export_eligibility"] not in {"included", "fit_review_required"}:
                raise ValueError(f"Unreviewed record: {path}:{identity}")
            if not row["source_url"]:
                raise ValueError(f"Missing URL: {identity}")
            observed = timestamp(row["observed_at"])
            cities = parts(row["city"])
            regions = entry["placement_regions"]
            unknown = entry.get("unknown_city_placements", {}).get(identity)
            city_source = row.get('city__source', '')
            if city_source not in ('', 'card', 'search_filter', 'missing'):
                raise ValueError(f'Unknown city source: {identity}')
            if city_source == 'search_filter' and (
                    row['city'] != entry.get('search_city') or not entry.get('search_city_evidence')
                    or row.get('city__evidence') != entry['search_city_evidence'] or not row.get('city_anomaly')):
                raise ValueError(f'Missing search-filter provenance: {identity}')
            if city_source == 'missing' and cities:
                raise ValueError(f'Missing city source has a value: {identity}')
            if not cities:
                missing_location = (isinstance(unknown, dict)
                    and unknown.get('location_missing') is True
                    and row.get('city__source') == 'missing'
                    and row.get('read_status_location') == 'missing'
                    and row.get('location_observed') == ''
                    and row.get('city__evidence') == ''
                    and unknown.get('evidence') == '' and unknown.get('region') == ''
                    and unknown.get('region_unknown') is True)
                if (row["city"] or not isinstance(unknown, dict)
                        or (not unknown.get('region') and unknown.get('region_unknown') is not True)
                        or (not missing_location and (not unknown.get("evidence")
                            or unknown["evidence"] not in row.get("location_observed", "")))):
                    raise ValueError(f"Missing reviewed unknown-city placement: {identity}")
                region = unknown["region"]
            else:
                if unknown or any(not regions.get(city) for city in cities):
                    raise ValueError(f"Missing placement region mapping: {row['city']}")
                region = "; ".join(dict.fromkeys(regions[c] for c in cities))
            status = row.get("enrichment_status", entry.get("enrichment_status", "not_run"))
            fields = parts(row.get("enriched_fields", ""))
            version = row.get("enrichment_version", "")
            if status not in {"not_run", "completed", "failed", "unknown"}:
                raise ValueError(f"Invalid enrichment status: {identity}")
            if status != "completed" and (fields or version):
                raise ValueError(f"Enrichment metadata requires completed status: {identity}")
            if status == "completed" and not version:
                raise ValueError(f"Missing enrichment version: {identity}")
            for field in fields:
                if (field in REQUIRED or field in METADATA or field in title_flags.COLUMNS or field.endswith(("__source", "__evidence"))
                        or not row.get(field) or row.get(field + "__source") != "description"
                        or not row.get(field + "__evidence")
                        or row[field + "__evidence"] not in row["description"]):
                    raise ValueError(f"Missing enrichment value/source/quote: {identity}:{field}")
            row.update(source=source, region=region,
                       run_id=run_id, enrichment_status=status,
                       enriched_fields="; ".join(fields), enrichment_version=version)
            records.setdefault((source, identity), []).append((observed, run_id, row))
        if not seen:
            raise ValueError(f"Empty export: {path}")
        inputs.append(dict(entry, rows=len(seen)))
    columns = ["source", "id", "city", "region", "profession"] + [
        name for name in columns if name not in {"id", "city", "profession"} and name not in METADATA and name not in title_flags.COLUMNS
    ] + list(title_flags.COLUMNS) + [name for name in METADATA if name not in {"source", "region"}]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for key in sorted(records):
        observations = sorted(records[key], key=lambda item: (item[0], item[1]))
        # Identical-time changes have no defensible 'latest' winner.
        for previous, current in zip(observations, observations[1:]):
            if previous[0] == current[0]:
                comparable = lambda row: {k: v for k, v in row.items() if k != "run_id" and k not in title_flags.COLUMNS and v != ""}
                if comparable(previous[2]) != comparable(current[2]):
                    raise ValueError(f"Conflicting observations at same timestamp: {key}")
        latest = dict(observations[-1][2])
        latest.update(first_seen_at=observations[0][2]["observed_at"],
                      last_seen_at=latest["observed_at"],
                      run_ids="; ".join(sorted({item[1] for item in observations})))
        latest.update(title_flags.military_flags(latest["title"]))
        writer.writerow(latest)
    csv_bytes = output.getvalue().encode("utf-8-sig")
    manifest = {"schema_version": VERSION, "builder_version": VERSION,
                "builder_sha256": digest(Path(__file__).read_bytes()),
                "title_flags": {"version": title_flags.VERSION,
                                "sha256": digest(Path(title_flags.__file__).read_bytes()),
                                "source": "title", "policy": "keyword_signal_not_exclusion"},
                "rows": len(records), "columns": columns, "sha256": digest(csv_bytes),
                "inputs": inputs, "update_policy": "latest_complete_observation",
                "geography": "placement_on_site", "empty_value": "unknown_not_zero"}
    return csv_bytes, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed outputs differ")
    args = parser.parse_args()
    directory = ROOT / "Data/dataset"
    registry = json.loads((directory / "sources.json").read_text())
    csv_bytes, manifest_bytes = build(ROOT, registry)
    for name, data in [("vacancies.csv", csv_bytes), ("manifest.json", manifest_bytes)]:
        path = directory / name
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Outdated {path.relative_to(ROOT)}; run python3 utils/build_dataset.py")
        else:
            path.write_bytes(data)
    manifest = json.loads(manifest_bytes)
    print(f"{manifest['rows']} vacancies, {len(manifest['columns'])} columns; {len(registry['exports'])} exports")


if __name__ == "__main__":
    main()
