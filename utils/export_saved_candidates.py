"""Export valid, previously captured candidates absent from the current dataset."""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'collector_runner', ROOT / 'Scraper/skills/collect-avito/scripts/run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

PRIVATE_LOCATOR = re.compile(rb'/Users/|/home/|\.aside[/\\]|TL_ASIDE_|profileId|target_id', re.I)


def archive_candidates(source_root, run_ids, output_id, *, root=ROOT):
    """Preserve saved observations and their selection decisions outside the dataset."""
    out, _, _ = runner.paths(root, output_id)
    if out.exists():
        raise ValueError('Output run already exists')
    rows, copies = [], {}
    for run_id in sorted(set(run_ids)):
        parent, _, _ = runner.paths(source_root, run_id)
        decisions = {}
        for name in ('selection.json', 'reviewed_selection.json'):
            path = parent / 'exports' / name
            if path.is_file():
                for item in runner.read(path)['decisions']:
                    decisions[item['id']] = item
        raw_paths = sorted((parent / 'raw').glob('[0-9]*.json'))
        if not raw_paths:
            raise ValueError(f'No raw observations for {run_id}')
        for path in raw_paths:
            raw = path.read_bytes()
            if PRIVATE_LOCATOR.search(raw):
                raise ValueError(f'Private locator in observation {run_id}/{path.name}')
            data = json.loads(raw)
            identity = path.stem
            digest = hashlib.sha256(raw).hexdigest()
            raw_path = path.relative_to(source_root)
            published_path = root / raw_path
            if not published_path.is_file() or published_path.read_bytes() != raw:
                raw_path = out.relative_to(root) / 'raw' / run_id / path.name
                copies[root / raw_path] = raw
            decision = decisions.get(identity, {})
            recorded_hash = decision.get('raw_sha256')
            if recorded_hash and recorded_hash != digest:
                raise ValueError(f'Selection hash differs for {run_id}/{identity}')
            accepted = decision.get('accepted')
            if data.get('state') != 'DETAIL' or str(data.get('id', '')) != identity:
                status = 'incomplete_observation'
            elif accepted is not None:
                status = 'accepted' if accepted else 'excluded'
            else:
                status = 'decision_not_recorded'
            row = {
                'source_run_id': run_id, 'id': identity, 'source_url': data.get('source_url', ''),
                'observed_at': data.get('observed_at', ''), 'capture_state': data.get('state', ''),
                'selection_status': status, 'selection_reason': decision.get('reason', ''),
                'raw_path': raw_path.as_posix(), 'raw_sha256': digest,
                'title': data.get('title', ''), 'salary_raw': data.get('salary_raw', ''),
                'description': data.get('description', ''),
                'employer_name_observed': '; '.join(dict.fromkeys(data.get('employer_raw', []))),
                'location_observed': '; '.join(dict.fromkeys(data.get('location_raw', []))),
            }
            for field in data.get('fields', []):
                if field.get('name') and field.get('value'):
                    row['Параметр — ' + runner.norm(field['name'])] = runner.norm(field['value'])
            rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    for path, content in copies.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'observations.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        run_id=output_id, source_runs=sorted(set(run_ids)),
        observations=len(rows), unique_ids=len({row['id'] for row in rows}),
        raw_files_copied=len(copies), csv_sha256=runner.sha(csv_path),
    )
    runner.write(out / 'manifest.json', summary)
    return summary


def verify_parent_file(root, manifest, path):
    info = manifest['files'].get(str(path.relative_to(root)))
    if not info or info['sha256'] != runner.sha(path):
        raise ValueError(f'Parent provenance mismatch: {path}')


def export_candidates(parent_id, output_id):
    parent, service, _ = runner.paths(ROOT, parent_id)
    out, target_service, _ = runner.paths(ROOT, output_id)
    if out.exists() or target_service.exists():
        raise ValueError('Output run already exists')
    plan = runner.read(service / 'plan.json')
    parent_manifest = runner.read(service / 'manifest.json')
    if parent_manifest.get('status') != 'complete':
        raise ValueError('Parent run is not finalized')
    verify_parent_file(ROOT, parent_manifest, service / 'plan.json')
    verify_parent_file(ROOT, parent_manifest, parent / 'queue.json')
    queue = runner.read(parent / 'queue.json')
    selection_path = parent / 'exports/selection.json'
    verify_parent_file(ROOT, parent_manifest, selection_path)
    selection = runner.read(selection_path)
    decisions = {item['id']: item for item in selection['decisions']}
    if len(decisions) != len(selection['decisions']):
        raise ValueError('Duplicate selection decision')
    registry_path = ROOT / 'Data/dataset/sources.json'
    registry = runner.read(registry_path)
    if any(entry['run_id'] == output_id for entry in registry['exports']):
        raise ValueError('Output run already registered')
    regions = {}
    for entry in registry['exports']:
        for city, region in entry['placement_regions'].items():
            if city in regions and regions[city] != region:
                raise ValueError('Conflicting region mappings')
            regions[city] = region
    dataset = ROOT / 'Data/dataset/vacancies.csv'
    dataset_manifest = runner.read(ROOT / 'Data/dataset/manifest.json')
    if runner.sha(dataset) != dataset_manifest['sha256']:
        raise ValueError('Materialized dataset differs from manifest')
    with dataset.open(encoding='utf-8-sig', newline='') as stream:
        existing_ids = {row['id'] for row in csv.DictReader(stream) if row['source'] == 'avito'}
    plan = dict(plan, city_anomaly_policy='blank')
    rows, reviews, provenance, mapping, unknown = [], [], [], {}, {}
    for item in queue:
        decision = decisions.get(item['id'])
        if not decision or decision['accepted']:
            continue
        raw_path = parent / 'raw' / (item['id'] + '.json')
        verify_parent_file(ROOT, parent_manifest, raw_path)
        data = runner.reviewed_capture(parent, item)
        row, review, region = runner.reviewed_row(data, item, regions, plan)
        included = item['id'] not in existing_ids
        provenance.append(dict(id=item['id'], source_decision=decision,
            raw_path=str(raw_path.relative_to(ROOT)), raw_sha256=runner.sha(raw_path),
            included=included, reason='new_valid_observation' if included else 'existing_dataset_id'))
        if included:
            if row['city']:
                mapping[row['city']] = region
            else:
                unknown[item['id']] = dict(region=region, evidence=row['location_observed'],
                    region_unknown=not bool(region), **({'location_missing': True} if not row['location_observed'] else {}))
            rows.append(row)
            reviews.append(review)
    if not rows:
        raise ValueError('No new valid candidates')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    exports = out / 'exports'
    exports.mkdir(parents=True)
    target_service.mkdir(parents=True)
    with (exports / 'vacancies_flat.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    runner.write(exports / 'vacancies.json', rows)
    for review in reviews:
        runner.write(out / 'reviewed' / (review['id'] + '.json'), review)
    runner.write(out / 'selection.json', dict(method='reassess_complete_captured_candidates_v1',
        parent_run_id=parent_id, parent_manifest_sha256=runner.sha(service / 'manifest.json'),
        parent_selection_sha256=runner.sha(selection_path), decisions=provenance))
    source_plan = dict(run_id=output_id, source_run_id=parent_id,
        source_plan_sha256=runner.sha(service / 'plan.json'),
        source_manifest_sha256=runner.sha(service / 'manifest.json'),
        selection_method='reassess_complete_captured_candidates_v1',
        geography=plan['geography'], profession=plan.get('profession', ''),
        listing_url=plan['listing_url'], observed_at='original raw timestamps')
    runner.write(target_service / 'plan.json', source_plan)
    entry = dict(run_id=output_id, source='avito',
        path=str((exports / 'vacancies_flat.csv').relative_to(ROOT)),
        sha256=runner.sha(exports / 'vacancies_flat.csv'),
        placement_regions=mapping,
        enrichment_status='not_run')
    if unknown:
        entry['unknown_city_placements'] = unknown
    if any(row['city__source'] == 'search_filter' for row in rows):
        entry.update(search_city=plan['geography'], search_city_evidence=plan['listing_url'])
    registry['exports'].append(entry)
    csv_bytes, manifest_bytes = runner.build(ROOT, registry)
    runner.write(registry_path, registry)
    from utils.build_dataset import write_outputs
    write_outputs(ROOT / 'Data/dataset', csv_bytes, manifest_bytes)
    files = {}
    for path in sorted(out.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=runner.sha(path))
    runner.write(target_service / 'manifest.json', dict(run_id=output_id, status='complete',
        source=source_plan, accepted=len(rows), files=files))
    print(json.dumps(dict(accepted=len(rows), reviewed=len(provenance),
        already_present=len(provenance)-len(rows)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run-id', required=True, action='append')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--archive-from', type=Path, help='Preserve raw observations from this checkout')
    args = parser.parse_args()
    if args.archive_from:
        print(json.dumps(archive_candidates(args.archive_from.resolve(), args.parent_run_id, args.run_id)))
    elif len(args.parent_run_id) == 1:
        export_candidates(args.parent_run_id[0], args.run_id)
    else:
        parser.error('Only archive mode accepts multiple parent runs')
