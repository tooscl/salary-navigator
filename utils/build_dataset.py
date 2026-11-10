"""Build the shared CSV offline from explicitly registered, immutable exports."""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path, PureWindowsPath

try:
    from . import title_flags, vacancy_labels, placement_transform, placement_transform_v2, city_scope_subset, city_scope_multi
except ImportError:  # Direct `python3 utils/build_dataset.py` entry point.
    import title_flags
    import vacancy_labels
    import placement_transform
    import placement_transform_v2
    import city_scope_subset
    import city_scope_multi

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1"
BUILDER_VERSION = "3"
METADATA = ["source", "region", "first_seen_at", "last_seen_at", "run_id",
            "run_ids", "search_geography", "search_profession", "search_filter_observations", "search_geography_context_observations",
            "enrichment_status", "enriched_fields", "enrichment_version"]
GENERATED = {"source", "region", "first_seen_at", "last_seen_at", "run_id", "run_ids", "search_geography", "search_profession", "search_filter_observations", "search_geography_context_observations"}
REQUIRED = {"id", "source_url", "observed_at", "city", "profession", "title",
            "description", "salary_raw", "analysis_status", "export_eligibility"}
# Recomputed from the latest observation on every build; never taken from exports.
DERIVED = tuple(title_flags.COLUMNS) + tuple(vacancy_labels.COLUMNS)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp requires timezone: {value}")
    return parsed


def parts(value):
    return [item.strip() for item in value.split(";") if item.strip()]


def repository_path(value):
    """Read historical Windows separators without accepting absolute or escaping paths."""
    if not isinstance(value, str) or not value or PureWindowsPath(value).drive:
        raise ValueError('Expected a relative repository path')
    path = Path(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected a relative repository path')
    return path


def build(root, registry):
    """Return deterministic output bytes; reject ambiguous observations."""
    if registry.get("schema_version") != VERSION or not registry.get("exports"):
        raise ValueError("Expected schema_version=1 and nonempty exports")
    records, columns, inputs = {}, [], []
    runs, transformations = set(), set()
    for entry in sorted(registry["exports"], key=lambda item: item["run_id"]):
        run_id, source = entry["run_id"], entry["source"]
        if not run_id or not source or run_id in runs:
            raise ValueError("Expected unique nonempty run_id and nonempty source")
        runs.add(run_id)
        path = repository_path(entry["path"])
        if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source must be inside repository: {path}")
        data = (root / path).read_bytes()
        if digest(data) != entry["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
        transform = entry.get('transformation')
        if transform:
            spec_path = repository_path(transform['path'])
            if not (root / spec_path).resolve().is_relative_to(root.resolve()):
                raise ValueError('Transformation must be inside repository')
            spec_bytes = (root / spec_path).read_bytes()
            if digest(spec_bytes) != transform['sha256']:
                raise ValueError('Transformation SHA-256 mismatch')
            spec = json.loads(spec_bytes)
            ident = spec['transformation_id']
            directory = Path('Data/transformations') / ident
            if spec_path != directory / 'transformation.json' or path != directory / 'vacancies_flat.csv':
                raise ValueError('Unexpected transformation artifact paths')
            if ident in transformations or spec['source_run_id'] != run_id:
                raise ValueError('Duplicate transformation or mismatched source run')
            transformations.add(ident)
            if spec.get('version') == placement_transform.VERSION:
                transformed = placement_transform.render(root, spec)
                for identity, decision in spec['placements'].items():
                    if (entry['placement_regions'].get(decision['city']) != decision['region'] or
                            identity in entry.get('unknown_city_placements', {})):
                        raise ValueError('Transformed placement registry mismatch')
            elif spec.get('version') == placement_transform_v2.VERSION:
                transformed = placement_transform_v2.render(root, spec)
                for identity, decision in spec['placements'].items():
                    if (entry['placement_regions'].get(decision['city']) != decision['region'] or
                            identity in entry.get('unknown_city_placements', {})):
                        raise ValueError('Transformed placement registry mismatch')
            elif spec.get('version') == city_scope_subset.VERSION:
                transformed = city_scope_subset.render(root, spec)
            elif spec.get('version') == city_scope_multi.VERSION:
                transformed = city_scope_multi.render(root, spec)
            else:
                raise ValueError('Unknown transformation version')
            if transformed != data:
                raise ValueError('Transformed CSV does not match source and decisions')
        correction = entry.get('profession_correction')
        if correction:
            correction_path = repository_path(correction['path'])
            if correction_path.is_absolute() or '..' in correction_path.parts or not (root / correction_path).resolve().is_relative_to(root.resolve()):
                raise ValueError('Profession correction must be inside repository')
            if digest((root / correction_path).read_bytes()) != correction['sha256']:
                raise ValueError('Profession correction SHA-256 mismatch')
        # A run plan is evidence of search scope, independent of placement fields.
        # Never infer filter values from a vacancy, a filename, or placement mappings.
        plan_path = Path("Scraper/runs") / run_id / "plan.json"
        if not (root / plan_path).resolve().is_relative_to(root.resolve()):
            raise ValueError("Plan must be inside repository")
        plan_bytes = (root / plan_path).read_bytes() if (root / plan_path).exists() else None
        plan = json.loads(plan_bytes) if plan_bytes is not None else {}
        if plan.get("run_id", run_id) != run_id:
            raise ValueError(f"Mismatched plan run_id: {run_id}")
        geography = plan.get("geography", plan.get("city"))
        profession = plan.get("profession", plan.get("profession_filter"))
        if geography is not None and not isinstance(geography, str) or profession is not None and not isinstance(profession, str):
            raise ValueError(f"Invalid search filter in plan: {run_id}")
        context = geography
        controls = plan.get("selected_controls", [])
        modes = {x.get("value") for x in controls if isinstance(x, dict) and x.get("marker") == "params[127818]/toggle" and x.get("value") in {"3054249", "3054250"}}
        if len(modes) > 1:
            raise ValueError(f"Conflicting geographic modes in plan: {run_id}")
        mode = {"3054249": "По всей России", "3054250": "Только в моём регионе"}.get(next(iter(modes), None))
        if mode == "По всей России":
            geography = "По всей России"
        search = {"run_id": run_id, "geography": geography, "profession": profession, "context": context, "mode": mode}
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
                if (field in REQUIRED or field in METADATA or field in DERIVED or field.endswith(("__source", "__evidence"))
                        or not row.get(field) or row.get(field + "__source") != "description"
                        or not row.get(field + "__evidence")
                        or row[field + "__evidence"] not in row["description"]):
                    raise ValueError(f"Missing enrichment value/source/quote: {identity}:{field}")
            row.update(source=source, region=region,
                       run_id=run_id, enrichment_status=status,
                       enriched_fields="; ".join(fields), enrichment_version=version)
            records.setdefault((source, identity), []).append((observed, run_id, row, search))
        if not seen:
            raise ValueError(f"Empty export: {path}")
        inputs.append(dict(entry, rows=len(seen), search_plan={"path": plan_path.as_posix(), "sha256": digest(plan_bytes)} if plan_bytes is not None else None))
    columns = ["source", "id", "city", "region", "profession"] + [
        name for name in columns if name not in {"id", "city", "profession"} and name not in METADATA and name not in DERIVED
    ] + list(DERIVED) + [name for name in METADATA if name not in {"source", "region"}]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for key in sorted(records):
        observations = sorted(records[key], key=lambda item: (item[0], item[1]))
        # Identical-time changes have no defensible 'latest' winner.
        for previous, current in zip(observations, observations[1:]):
            if previous[0] == current[0]:
                comparable = lambda row: {k: v for k, v in row.items() if k != "run_id" and k not in DERIVED and v != ""}
                if comparable(previous[2]) != comparable(current[2]):
                    raise ValueError(f"Conflicting observations at same timestamp: {key}")
        latest = dict(observations[-1][2])
        latest.update(first_seen_at=observations[0][2]["observed_at"],
                      last_seen_at=latest["observed_at"],
                      run_ids="; ".join(sorted({item[1] for item in observations})),
                      search_geography=observations[-1][3]["geography"] or "",
                      search_profession=observations[-1][3]["profession"] or "",
                      search_filter_observations=json.dumps([[item[3]["run_id"], item[3]["geography"], item[3]["profession"]] for item in observations], ensure_ascii=False, separators=(",", ":")),
                      search_geography_context_observations=json.dumps([[item[3]["run_id"], item[3]["context"], item[3]["mode"]] for item in observations], ensure_ascii=False, separators=(",", ":")))
        latest.update(title_flags.military_flags(latest["title"]))
        latest.update(vacancy_labels.vacancy_labels(latest))
        writer.writerow(latest)
    csv_bytes = output.getvalue().encode("utf-8-sig")
    manifest = {"schema_version": VERSION, "builder_version": BUILDER_VERSION,
                "builder_sha256": digest(Path(__file__).read_bytes()),
                "placement_transform_sha256": digest(Path(placement_transform.__file__).read_bytes()),
                "placement_transform_v2_sha256": digest(Path(placement_transform_v2.__file__).read_bytes()),
                "city_scope_audit_sha256": digest(Path(city_scope_subset.city_scope_audit.__file__).read_bytes()),
                "city_scope_audit_v1_sha256": digest(Path(city_scope_subset.city_scope_audit_v1.__file__).read_bytes()),
                "city_scope_subset_sha256": digest(Path(city_scope_subset.__file__).read_bytes()),
                "city_scope_multi_sha256": digest(Path(city_scope_multi.__file__).read_bytes()),
                "title_flags": {"version": title_flags.VERSION,
                                "sha256": digest(Path(title_flags.__file__).read_bytes()),
                                "source": "title", "policy": "keyword_signal_not_exclusion"},
                "vacancy_labels": {"version": vacancy_labels.VERSION,
                                   "sha256": {path.name: digest(path.read_bytes()) for path in vacancy_labels.DEPENDENCIES},
                                   "source": "title, profession, city, location_observed",
                                   "policy": "added_next_to_observed_columns_not_replacing"},
                "rows": len(records), "columns": columns, "sha256": digest(csv_bytes),
                "inputs": inputs, "update_policy": "latest_complete_observation",
                "geography": "placement_on_site", "empty_value": "unknown_not_zero"}
    return csv_bytes, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()



SHARD_COUNT = 64

def split_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    columns = reader.fieldnames
    outputs = [io.StringIO(newline="") for _ in range(SHARD_COUNT)]
    writers = [csv.DictWriter(out, fieldnames=columns, lineterminator="\n") for out in outputs]
    for writer in writers:
        writer.writeheader()
    counts = [0] * SHARD_COUNT
    for row in reader:
        key = json.dumps([row["source"], row["id"]], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % SHARD_COUNT
        writers[index].writerow(row)
        counts[index] += 1
    files = {f"parts/part-{i:02d}.csv": out.getvalue().encode("utf-8-sig") for i, out in enumerate(outputs)}
    metadata = [{"path": name, "rows": counts[i], "bytes": len(content), "sha256": digest(content)}
                for i, (name, content) in enumerate(files.items())]
    return files, metadata


def assemble(manifest, read_part):
    rows = []
    columns = manifest["columns"]
    seen = set()
    for part in manifest["parts"]:
        data = read_part(part["path"])
        if len(data) != part["bytes"] or digest(data) != part["sha256"]:
            raise ValueError(f"Invalid part: {part['path']}")
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
        if reader.fieldnames != columns:
            raise ValueError(f"Invalid header: {part['path']}")
        count = 0
        for row in reader:
            key = (row["source"], row["id"])
            if key in seen:
                raise ValueError(f"Duplicate key: {key}")
            seen.add(key)
            rows.append(row)
            count += 1
        if count != part["rows"]:
            raise ValueError(f"Invalid row count: {part['path']}")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: (row["source"], row["id"])))
    data = output.getvalue().encode("utf-8-sig")
    if len(rows) != manifest["rows"] or digest(data) != manifest["sha256"]:
        raise ValueError("Assembled CSV does not match manifest")
    return data


def write_outputs(directory, csv_bytes, manifest_bytes, check=False):
    manifest = json.loads(manifest_bytes)
    files, metadata = split_csv(csv_bytes)
    manifest["parts"] = metadata
    manifest["partition"] = "sha256-json-source-id-first64-mod-64-v1"
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    files["manifest.json"] = manifest_bytes
    existing = {p.relative_to(directory).as_posix() for p in (directory / "parts").glob("*.csv")}
    expected = set(files) - {"manifest.json"}
    if check and existing != expected:
        raise SystemExit("Dataset parts differ; run python3 utils/build_dataset.py")
    for name, data in files.items():
        path = directory / name
        if check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Outdated {path}; run python3 utils/build_dataset.py")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if not check:
        for name in existing - expected:
            (directory / name).unlink()
        (directory / "vacancies.csv").write_bytes(csv_bytes)
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed outputs differ")
    parser.add_argument("--materialize", action="store_true", help="Reconstruct CSV from committed parts")
    args = parser.parse_args()
    directory = ROOT / "Data/dataset"
    if args.materialize:
        if args.check:
            parser.error("--materialize and --check are mutually exclusive")
        manifest = json.loads((directory / "manifest.json").read_text())
        (directory / "vacancies.csv").write_bytes(assemble(manifest, lambda path: (directory / path).read_bytes()))
        print(f"Materialized {manifest['rows']} vacancies; SHA-256 {manifest['sha256']}")
        return
    registry = json.loads((directory / "sources.json").read_text())
    csv_bytes, manifest_bytes = build(ROOT, registry)
    manifest = write_outputs(directory, csv_bytes, manifest_bytes, check=args.check)
    print(f"{manifest['rows']} vacancies, {len(manifest['columns'])} columns; {len(registry['exports'])} exports")


if __name__ == "__main__":
    main()
