"""Export valid, previously captured candidates absent from the current dataset."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'collector_runner', ROOT / 'Scraper/skills/collect-avito/scripts/run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


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
    parser.add_argument('--parent-run-id', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    export_candidates(args.parent_run_id, args.run_id)
