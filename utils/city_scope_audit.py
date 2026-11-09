"""Audit visible city placement of every item in a completed Avito run, offline."""
import argparse
import csv
import io
import json
import re
from pathlib import Path

try:
    from .placement_transform import checked_path, sha
except ImportError:
    from placement_transform import checked_path, sha

VERSION = 'city_scope_audit_v1'


def visible_component(address, city):
    pattern = r'(?:^|[,;])\s*(?:г\.\s*|город\s+|с\.\s*|село\s+|пос\.\s*(?:учхоза\s+)?|пос[её]лок\s+|д\.\s*)?' + re.escape(city) + r'\s*(?:[,;]|$)'
    match = re.search(pattern, address, re.I) if city else None
    if not match:
        return False
    tail = re.split(r',\s*(?:ул\.|улица|пр\.|проспект|ш\.|шоссе|пер\.|переулок)', address[match.end():], maxsplit=1, flags=re.I)[0]
    return not re.search(r'(?:^|[,;])\s*(?:с\.|село\s|д\.|деревня\s|пос\.|пос[её]лок\s|пгт\.?\s)', tail, re.I)


def audit(root, run, target_city, target_region_quote):
    root = Path(root).resolve()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', run) or not target_city.strip() or not target_region_quote.strip():
        raise ValueError('Invalid run, city or required region quote')
    manifest_name = f'Scraper/runs/{run}/manifest.json'
    manifest_bytes = checked_path(root, manifest_name).read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('run_id') != run or manifest.get('status') != 'complete':
        raise ValueError('Completed source manifest required')
    files = manifest['files']
    for name, info in files.items():
        data = checked_path(root, name).read_bytes()
        if len(data) != info['bytes'] or sha(data) != info['sha256']:
            raise ValueError(f'Source file changed: {name}')
    base = f'Data/avito_runs/{run}'
    def source_file(name):
        if name not in files: raise ValueError(f'File missing from manifest: {name}')
        return checked_path(root, name)
    queue = json.loads(source_file(f'{base}/queue.json').read_bytes())
    ids = [str(x['id']) for x in queue]
    if len(ids) != len(set(ids)) or len(ids) != manifest['coverage']['planned']:
        raise ValueError('Queue coverage or identity mismatch')
    with source_file(f'{base}/exports/vacancies_flat.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row['id']:row for row in rows}
    if len(by_id) != len(rows) or len(rows) != manifest['coverage']['accepted']:
        raise ValueError('Export coverage or identity mismatch')
    coverage = manifest['coverage']
    excluded = {}
    for key in ('unavailable','unresolved_observations','user_skipped_observations'):
        for record in coverage.get(key, []):
            ident = str(record['id'])
            if ident in excluded: raise ValueError('Conflicting excluded ID')
            excluded[ident] = {'category':key,'record':record}
    if set(ids) != set(by_id) | set(excluded) or set(by_id) & set(excluded):
        raise ValueError('Every queued ID must be accounted for exactly once')
    decisions = []
    for item in queue:
        ident = str(item['id'])
        if ident in excluded:
            evidence = excluded[ident]
            raw_name = f'{base}/raw/{ident}.json'
            raw_info = {'path':raw_name,'sha256':files[raw_name]['sha256']} if raw_name in files else None
            decisions.append({'id':ident,'decision':'unavailable_or_unresolved','reason':evidence['category'],
                              'coverage_record':evidence['record'],'raw':raw_info})
            continue
        row = by_id[ident]
        raw = json.loads(source_file(f'{base}/raw/{ident}.json').read_bytes())
        review = json.loads(source_file(f'{base}/reviewed/{ident}.json').read_bytes())
        if raw.get('id') != ident or raw.get('source_url') != item['url'] or row.get('source_url') != item['url'] or review.get('id') != ident:
            raise ValueError(f'Identity or URL mismatch: {ident}')
        address = '\n'.join(raw.get('location_raw', [])).split('\n',1)[0].strip()
        city = row.get('city','')
        confirmed = (row.get('city__source') == 'card' and review.get('city_source') == 'card' and
                     raw.get('read_status', {}).get('location') == 'observed' and review.get('city') == city and
                     review.get('analysis_status') == 'validated' and address and
                     address in row.get('location_observed','') and visible_component(address,city))
        if not confirmed:
            decision, reason = 'unknown', 'visible_card_city_not_unambiguous'
        elif city == target_city and target_region_quote in address:
            decision, reason = 'in_city', 'visible_card_city_and_region'
        elif city != target_city and not re.search(r'(?<!\w)' + re.escape(target_city) + r'(?!\w)', address, re.I):
            decision, reason = 'outside_city', 'different_visible_card_city'
        elif city != target_city:
            decision, reason = 'unknown', 'target_and_other_locality_both_visible'
        else:
            decision, reason = 'unknown', 'target_city_region_not_visible'
        decisions.append({'id':ident,'decision':decision,'reason':reason,'visible_address':address,
                          'reviewed_city':city,'reviewed_region':review.get('region','')})
    counts = {name:sum(x['decision']==name for x in decisions) for name in ('in_city','outside_city','unknown','unavailable_or_unresolved')}
    return {'version':VERSION,'source_run_id':run,'source_manifest':{'path':manifest_name,'sha256':sha(manifest_bytes)},
            'audit_code_sha256':sha(Path(__file__).read_bytes()),'target_city':target_city,'target_region_quote':target_region_quote,
            'scope':'saved queue only; completeness of city coverage is not established by this audit',
            'filter_validation':'not performed; verify remote format, profession scope and regional filter evidence separately',
            'counts':counts,'decisions':decisions}


def render_included(root, run, report):
    path = Path(root) / 'Data/avito_runs' / run / 'exports/vacancies_flat.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
    include={d['id'] for d in report['decisions'] if d['decision']=='in_city'}
    if len(include) != report['counts']['in_city'] or not include.issubset({r['id'] for r in rows}):
        raise ValueError('Included ID set differs from audit')
    output=io.StringIO(newline=''); writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n')
    writer.writeheader(); writer.writerows(row for row in rows if row['id'] in include)
    return output.getvalue().encode('utf-8-sig')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True);parser.add_argument('--city',required=True)
    parser.add_argument('--region-quote',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--included-csv',type=Path,help='Optional provisional CSV of in-city decisions; never auto-registers it')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    result=audit(args.root,args.run_id,args.city,args.region_quote)
    if args.included_csv:
        data=render_included(args.root,args.run_id,result)
        result['included_csv']={'path':str(args.included_csv),'sha256':sha(data),'status':'provisional_not_registered'}
        if args.output.exists() or args.included_csv.exists() or args.output.resolve() == args.included_csv.resolve():
            raise FileExistsError('Output already exists or paths are identical')
        with args.included_csv.open('xb') as stream: stream.write(data)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result['counts'],ensure_ascii=False))

if __name__ == '__main__': main()
