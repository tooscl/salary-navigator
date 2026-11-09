"""Reproduce a provisional city subset of a completed, remote-filtered Avito run."""
import json
import re
from pathlib import Path

try:
    from . import city_scope_audit
except ImportError:
    import city_scope_audit

VERSION = 'city_remote_subset_v1'


def make_spec(root, run, transformation_id, city, region_quote):
    root=Path(root).resolve()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', transformation_id):
        raise ValueError('Invalid transformation ID')
    report=city_scope_audit.audit(root,run,city,region_quote)
    manifest=json.loads((root / report['source_manifest']['path']).read_bytes())
    files=manifest['files']; base=f'Data/avito_runs/{run}'; plan_name=f'Scraper/runs/{run}/plan.json'; listing_name=f'{base}/listing.json'
    if plan_name not in files or listing_name not in files:
        raise ValueError('Filter evidence absent from source manifest')
    plan=json.loads((root/plan_name).read_bytes());listing=json.loads((root/listing_name).read_bytes())
    controls=listing.get('selected_controls',[])
    selected={(x.get('marker'),x.get('label')) for x in controls}
    profession=listing.get('profession_filter') or {}
    if (plan.get('profession') != '' or plan.get('profession_scope') != 'all' or
            profession.get('observed') is not True or profession.get('selected_count') != 0 or
            ('params[172815]/3286367','Удалённо') not in selected):
        raise ValueError('Unfiltered profession and remote search evidence required')
    for item in report['decisions']:
        if item['decision'] != 'in_city': continue
        raw=json.loads((root/base/'raw'/f"{item['id']}.json").read_bytes())
        values=[x['value'] for x in raw.get('fields',[]) if x.get('name')=='Формат работы']
        if len(values)>1: raise ValueError('Conflicting format fields')
        modes=[x.strip() for x in values[0].split(',')] if values else []
        if not values: evidence={'basis':'remote_search_filter','listing_path':listing_name,'listing_sha256':files[listing_name]['sha256']}
        elif 'Удалённо' not in modes: raise ValueError(f"Card contradicts remote search: {item['id']}")
        else: evidence={'basis':'card_format','observed_value':values[0]}
        item['remote_evidence']=evidence
    return {'version':VERSION,'transformation_id':transformation_id,'source_run_id':run,
            'description':'Provisional observed city subset; full city coverage is not established.',
            'audit':report}


def render(root, spec):
    if spec.get('version') != VERSION: raise ValueError('Unknown city subset version')
    audit=spec['audit']
    expected=make_spec(root,spec['source_run_id'],spec['transformation_id'],audit['target_city'],audit['target_region_quote'])
    if spec != expected: raise ValueError('City subset audit differs from source')
    return city_scope_audit.render_included(root,spec['source_run_id'],audit)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True);parser.add_argument('--transformation-id',required=True)
    parser.add_argument('--city',required=True);parser.add_argument('--region-quote',required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.root.resolve();spec=make_spec(root,args.run_id,args.transformation_id,args.city,args.region_quote)
    data=render(root,spec);out=root/'Data/transformations'/args.transformation_id
    out.mkdir(parents=True,exist_ok=False)
    (out/'transformation.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'vacancies_flat.csv').write_bytes(data)
    print(f"Created {out.relative_to(root)}; provisional scope: {spec['audit']['counts']}")


if __name__ == '__main__': main()
