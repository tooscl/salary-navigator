"""Reproduce separate city audits and their union for a single captured run."""
import csv
import io
from pathlib import Path

try:
    from . import city_scope_subset
except ImportError:
    import city_scope_subset

VERSION='city_remote_multi_v1'


def make_spec(root, run, transformation_id, targets):
    if not targets or len({(x['city'],x['region_quote']) for x in targets}) != len(targets):
        raise ValueError('Expected distinct city targets')
    audits=[]; outputs=[]; seen=set()
    for target in targets:
        child=city_scope_subset.make_spec(root,run,transformation_id,target['city'],target['region_quote'])
        audit=child['audit']; matched={x['id'] for x in audit['decisions'] if x['decision']=='in_city'}
        if seen & matched: raise ValueError('City subsets overlap')
        seen.update(matched);audits.append(audit)
        data=city_scope_subset.city_scope_audit.render_included(root,run,audit)
        outputs.append({'file':f'city-{len(audits):02d}.csv','sha256':city_scope_subset.city_scope_audit.sha(data),'rows':len(matched)})
    return {'version':VERSION,'transformation_id':transformation_id,'source_run_id':run,
            'description':'Provisional observed city subsets; full city coverage is not established.',
            'targets':targets,'audits':audits,'city_outputs':outputs}


def render(root,spec):
    if spec.get('version') != VERSION: raise ValueError('Unknown multi-city version')
    expected=make_spec(root,spec['source_run_id'],spec['transformation_id'],spec['targets'])
    if spec != expected: raise ValueError('Multi-city audit differs from source')
    for audit,info in zip(spec['audits'],spec['city_outputs']):
        data=city_scope_subset.city_scope_audit.render_included(root,spec['source_run_id'],audit)
        path=Path(root)/'Data/transformations'/spec['transformation_id']/info['file']
        if path.read_bytes() != data: raise ValueError('City output differs from source')
    ids={d['id'] for audit in spec['audits'] for d in audit['decisions'] if d['decision']=='in_city'}
    path=Path(root)/'Data/avito_runs'/spec['source_run_id']/'exports/vacancies_flat.csv'
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(row for row in rows if row['id'] in ids)
    return output.getvalue().encode('utf-8-sig')


def main():
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True);parser.add_argument('--transformation-id',required=True)
    parser.add_argument('--target',nargs=2,action='append',required=True,metavar=('CITY','REGION_QUOTE'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.root.resolve();targets=[{'city':city,'region_quote':region} for city,region in args.target]
    spec=make_spec(root,args.run_id,args.transformation_id,targets)
    out=root/'Data/transformations'/args.transformation_id;out.mkdir(parents=True,exist_ok=False)
    for audit,info in zip(spec['audits'],spec['city_outputs']):
        (out/info['file']).write_bytes(city_scope_subset.city_scope_audit.render_included(root,args.run_id,audit))
    (out/'transformation.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'vacancies_flat.csv').write_bytes(render(root,spec))
    print(f"Created {out.relative_to(root)}; counts: {[a['counts'] for a in spec['audits']]}")


if __name__=='__main__':main()
