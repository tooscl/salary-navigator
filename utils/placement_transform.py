"""Deterministic, offline corrections of missing placement from immutable Avito raw."""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path, PureWindowsPath

VERSION = 'placement_from_raw_v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(root, value):
    if not isinstance(value, str) or not value or PureWindowsPath(value).drive:
        raise ValueError('Invalid lineage path')
    path = Path(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError('Invalid lineage path')
    return root / path


def normalize_explicit_region(value):
    """Normalize only spellings explicitly present in the address."""
    value = re.sub(r'\bобл\.', 'область', value.strip(), flags=re.I)
    aliases = {'Республика Татарстан (Татарстан)': 'Республика Татарстан',
               'Чувашская Республика — Чувашия': 'Чувашская Республика'}
    return aliases.get(value, value)


def render(root, spec, *, explicit_region=False):
    """Verify source passport and reproduce only explicitly reviewed blank-city corrections."""
    root = Path(root)
    accepted_version = 'placement_from_explicit_region_v2' if explicit_region else VERSION
    if spec.get('version') != accepted_version or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', spec.get('transformation_id', '')):
        raise ValueError('Invalid transformation identity or version')
    run = spec['source_run_id']
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', run):
        raise ValueError('Invalid source run ID')
    manifest_path = checked_path(root, spec['source_manifest']['path'])
    if manifest_path != root / 'Scraper' / 'runs' / run / 'manifest.json':
        raise ValueError('Source manifest must belong to the source run')
    manifest_bytes = manifest_path.read_bytes()
    if sha(manifest_bytes) != spec['source_manifest']['sha256']:
        raise ValueError('Source manifest SHA-256 mismatch')
    manifest = json.loads(manifest_bytes)
    if manifest.get('run_id') != run or manifest.get('status') != 'complete':
        raise ValueError('Expected completed source run')
    files = manifest['files']
    for name, info in files.items():
        data = checked_path(root, name).read_bytes()
        if len(data) != info['bytes'] or sha(data) != info['sha256']:
            raise ValueError(f'Source file changed: {name}')
    csv_name = f'Data/avito_runs/{run}/exports/vacancies_flat.csv'
    if csv_name not in files:
        raise ValueError('Source CSV absent from source manifest')
    data = checked_path(root, csv_name).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)) or not {'id','city','city__source','city__evidence','city_anomaly','location_observed','observed_at'}.issubset(fields):
        raise ValueError('Invalid source CSV')
    rows = list(reader)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source ID')
    listing = {}
    if explicit_region:
        queue_name = f'Data/avito_runs/{run}/queue.json'
        if queue_name not in files: raise ValueError('Source queue absent from manifest')
        queue = json.loads(checked_path(root, queue_name).read_bytes())
        listing = {item['id']: item for item in queue}
        if len(listing) != len(queue): raise ValueError('Duplicate listing ID')
    corrections = spec['placements']
    if not corrections or not isinstance(corrections, dict):
        raise ValueError('Expected nonempty placements')
    seen = set()
    for row in rows:
        ident = row['id']
        if ident not in corrections:
            continue
        change = corrections[ident]
        raw_path = f'Data/avito_runs/{run}/raw/{ident}.json'
        review_path = f'Data/avito_runs/{run}/reviewed/{ident}.json'
        if raw_path not in files or review_path not in files:
            raise ValueError('Raw/review missing from manifest')
        raw = json.loads(checked_path(root, raw_path).read_bytes())
        review = json.loads(checked_path(root, review_path).read_bytes())
        city, region, evidence = change['city'], change['region'], change['evidence']
        address = '\n'.join(raw.get('location_raw', [])).split('\n', 1)[0].strip()
        pattern = r'(?:^|[,;])\s*(?:г\.\s*|город\s+|с\.\s*|село\s+|пос\.\s*(?:учхоза\s+)?|посёлок\s+|станица\s+|ст\.\s*|д\.\s*)?' + re.escape(city) + r'\s*(?:[,;]|$)'
        match = re.search(pattern, address, re.I) if city else None
        remainder = address[match.end():] if match else ''
        # A more specific locality after the selected one makes the choice ambiguous.
        tail = re.split(r',\s*(?:ул\.|улица|пр\.|проспект|ш\.|шоссе|пер\.|переулок)', remainder, maxsplit=1, flags=re.I)[0]
        nested_locality = re.search(r'(?:^|[,;])\s*(?:с\.|село\s|д\.|деревня\s|пос\.|пос[её]лок\s|станица\s|ст\.\s*|хутор\s|х\.\s*|пгт\.?\s)', tail, re.I)
        region_valid = review.get('region') == region
        if explicit_region and not review.get('region'):
            first = address.split(',', 1)[0].strip()
            explicit = normalize_explicit_region(first)
            listing_location = listing.get(ident, {}).get('location_raw', '')
            components = [part.strip() for part in address.split(',')]
            region_valid = (bool(re.search(r'область|край|республика', explicit, re.I))
                and explicit == region and len(components) > 1 and components[1] == city
                and bool(re.fullmatch(r'[А-Яа-яЁёA-Za-z\- ]+', city))
                and not re.search(r'край|район|р-н|округ|область|республика|городской|шоссе|площадь|улица|территория', city, re.I)
                and bool(re.search(r'(?<!\w)' + re.escape(city) + r'(?!\w)', listing_location, re.I)))
        if (not city or not region or not evidence or evidence != address or
                not match or nested_locality or raw.get('id') != ident or
                row['city'] or row['city__source'] != 'missing' or
                review.get('city') or not region_valid or
                evidence not in row['location_observed']):
            raise ValueError(f'Unsupported placement correction: {ident}')
        row.update(city=city, city__source='card', city__evidence=row['location_observed'], city_anomaly='')
        seen.add(ident)
    if seen != set(corrections):
        raise ValueError('Placement ID not present in source')
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue().encode('utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--transformation-id', required=True)
    parser.add_argument('--placements', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', args.run_id) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', args.transformation_id):
        parser.error('IDs must contain only letters, digits, underscores and hyphens')
    manifest = root / 'Scraper/runs' / args.run_id / 'manifest.json'
    spec = {'version': VERSION, 'transformation_id': args.transformation_id,
            'source_run_id': args.run_id,
            'source_manifest': {'path': manifest.relative_to(root).as_posix(), 'sha256': sha(manifest.read_bytes())},
            'placements': json.loads(args.placements.read_text(encoding='utf-8'))}
    result = render(root, spec)
    out = root / 'Data/transformations' / args.transformation_id
    out.mkdir(parents=True, exist_ok=False)
    (out / 'transformation.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'vacancies_flat.csv').write_bytes(result)
    print(f'Created {out.relative_to(root)}; register it for the existing collection run ID.')


if __name__ == '__main__':
    main()
