import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from utils import placement_transform as t
from utils import placement_transform_v2 as t2
from utils.build_dataset import build


class PlacementTransformTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = 'example-run'
        self.base = f'Data/avito_runs/{self.run}'
        self.csvpath = f'{self.base}/exports/vacancies_flat.csv'
        self.rawpath = f'{self.base}/raw/123.json'
        self.reviewpath = f'{self.base}/reviewed/123.json'
        self.row = {'id':'123', 'city':'', 'city__source':'missing', 'city__evidence':'',
                    'city_anomaly':'unknown', 'location_observed':'Рязанская обл., Рязань, ул. Ленина',
                    'observed_at':'2026-11-10T00:00:00+00:00', 'source_url':'https://example.test/123',
                    'profession':'Продавец','title':'Продавец','description':'', 'salary_raw':'',
                    'analysis_status':'validated','export_eligibility':'included'}
        out = io.StringIO(newline=''); writer = csv.DictWriter(out, fieldnames=self.row, lineterminator='\n')
        writer.writeheader(); writer.writerow(self.row)
        self.put(self.csvpath, out.getvalue().encode('utf-8-sig'))
        self.put(self.rawpath, json.dumps({'id':'123','location_raw':['Рязанская обл., Рязань, ул. Ленина']}, ensure_ascii=False).encode())
        self.put(self.reviewpath, json.dumps({'city':'','region':'Рязанская область'}, ensure_ascii=False).encode())
        files = {}
        for name in (self.csvpath, self.rawpath, self.reviewpath):
            data=(self.root/name).read_bytes(); files[name]={'bytes':len(data),'sha256':t.sha(data)}
        self.manifestpath=f'Scraper/runs/{self.run}/manifest.json'
        self.put(self.manifestpath,json.dumps({'run_id':self.run,'status':'complete','files':files}).encode())
        self.spec={'version':t.VERSION,'transformation_id':'example-placement-v1','source_run_id':self.run,
                   'source_manifest':{'path':self.manifestpath,'sha256':t.sha((self.root/self.manifestpath).read_bytes())},
                   'placements':{'123':{'city':'Рязань','region':'Рязанская область','evidence':'Рязанская обл., Рязань, ул. Ленина'}}}

    def put(self, name, data):
        p=self.root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)

    def test_correction_is_deterministic_and_does_not_mutate_source(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result=t.render(self.root,self.spec)
        self.assertEqual(result,t.render(self.root,self.spec))
        result_row=next(csv.DictReader(io.StringIO(result.decode('utf-8-sig'))))
        self.assertEqual(result_row['city'],'Рязань')
        self.assertEqual(result_row['observed_at'],self.row['observed_at'])
        self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_changed_raw_or_manifest_rejected(self):
        self.put(self.rawpath,b'{}')
        with self.assertRaisesRegex(ValueError,'Source file changed'): t.render(self.root,self.spec)
        self.spec['source_manifest']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'manifest SHA-256'): t.render(self.root,self.spec)

    def test_unrelated_address_or_city_and_unknown_id_rejected(self):
        self.spec['placements']['123']['city']='Москва'
        with self.assertRaisesRegex(ValueError,'Unsupported placement'): t.render(self.root,self.spec)
        self.spec['placements']={'999':{'city':'Рязань','region':'Рязанская область','evidence':'Рязань'}}
        with self.assertRaisesRegex(ValueError,'not present'): t.render(self.root,self.spec)

    def test_region_conflict_and_second_line_city_rejected(self):
        self.spec['placements']['123']['region']='Московская область'
        with self.assertRaisesRegex(ValueError,'Unsupported placement'): t.render(self.root,self.spec)
        self.spec['placements']['123']['region']='Рязанская область'
        self.spec['placements']['123']['evidence']='Рязань'
        with self.assertRaisesRegex(ValueError,'Unsupported placement'): t.render(self.root,self.spec)

    def test_dataset_checks_reproduction_and_preserves_collection_id(self):
        result=t.render(self.root,self.spec)
        self.put('Data/transformations/example-placement-v1/vacancies_flat.csv',result)
        self.put('Data/transformations/example-placement-v1/transformation.json',json.dumps(self.spec,ensure_ascii=False).encode())
        entry={'run_id':self.run,'source':'avito','path':'Data/transformations/example-placement-v1/vacancies_flat.csv','sha256':t.sha(result),
               'placement_regions':{'Рязань':'Рязанская область'},
               'transformation':{'path':'Data/transformations/example-placement-v1/transformation.json','sha256':t.sha((self.root/'Data/transformations/example-placement-v1/transformation.json').read_bytes())}}
        registry={'schema_version':'1','exports':[entry]}
        data,_=build(self.root,registry)
        row=next(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        self.assertEqual((row['run_id'],row['run_ids'],row['observed_at']),
                         (self.run,self.run,self.row['observed_at']))
        modified=result.replace('Рязань'.encode(), 'Москва'.encode())
        self.put('Data/transformations/example-placement-v1/vacancies_flat.csv',modified); entry['sha256']=t.sha(modified)
        with self.assertRaisesRegex(ValueError,'does not match'): build(self.root,registry)
        self.put('Data/transformations/example-placement-v1/vacancies_flat.csv',result); entry['sha256']=t.sha(result)
        registry['exports'].append(dict(entry))
        with self.assertRaisesRegex(ValueError,'unique'): build(self.root,registry)

    def test_more_specific_locality_is_not_replaced_by_city(self):
        address='Рязанская обл., Рязань, посёлок Канищево, ул. Чапаева'
        self.put(self.rawpath,json.dumps({'id':'123','location_raw':[address]},ensure_ascii=False).encode())
        row=dict(self.row, location_observed=address)
        output=io.StringIO(newline=''); writer=csv.DictWriter(output, fieldnames=row, lineterminator='\n')
        writer.writeheader(); writer.writerow(row); self.put(self.csvpath,output.getvalue().encode('utf-8-sig'))
        manifest=json.loads((self.root/self.manifestpath).read_bytes())
        data=(self.root/self.csvpath).read_bytes()
        manifest['files'][self.csvpath]={'bytes':len(data),'sha256':t.sha(data)}
        raw=(self.root/self.rawpath).read_bytes()
        manifest['files'][self.rawpath]={'bytes':len(raw),'sha256':t.sha(raw)}
        self.put(self.manifestpath,json.dumps(manifest).encode())
        self.spec['source_manifest']['sha256']=t.sha((self.root/self.manifestpath).read_bytes())
        self.spec['placements']['123']['evidence']=address
        with self.assertRaisesRegex(ValueError,'Unsupported placement'): t.render(self.root,self.spec)

    def test_stanitsa_component_is_supported_without_inference(self):
        address = 'Краснодарский край, г.о. Краснодар, станица Елизаветинская, Степная ул., 279'
        row = dict(self.row, location_observed=address)
        output = io.StringIO(newline=''); writer = csv.DictWriter(output, fieldnames=row, lineterminator='\n')
        writer.writeheader(); writer.writerow(row); self.put(self.csvpath, output.getvalue().encode('utf-8-sig'))
        self.put(self.rawpath, json.dumps({'id': '123', 'location_raw': [address]}, ensure_ascii=False).encode())
        self.put(self.reviewpath, json.dumps({'city': '', 'region': 'Краснодарский край'}, ensure_ascii=False).encode())
        manifest = json.loads((self.root/self.manifestpath).read_bytes())
        for name in (self.csvpath, self.rawpath, self.reviewpath):
            data = (self.root/name).read_bytes(); manifest['files'][name] = {'bytes': len(data), 'sha256': t.sha(data)}
        self.put(self.manifestpath, json.dumps(manifest).encode())
        self.spec['source_manifest']['sha256'] = t.sha((self.root/self.manifestpath).read_bytes())
        self.spec['placements']['123'] = {'city': 'Елизаветинская', 'region': 'Краснодарский край', 'evidence': address}
        result = next(csv.DictReader(io.StringIO(t.render(self.root, self.spec).decode('utf-8-sig'))))
        self.assertEqual(result['city'], 'Елизаветинская')
        self.spec['placements']['123']['city'] = 'Краснодар'
        with self.assertRaisesRegex(ValueError, 'Unsupported placement'):
            t.render(self.root, self.spec)

    def test_explicit_region_and_listing_corroboration_for_missing_region(self):
        queuepath=f'{self.base}/queue.json'
        self.put(queuepath,json.dumps([{'id':'123','location_raw':'Рязань'}],ensure_ascii=False).encode())
        self.put(self.reviewpath,json.dumps({'city':'','region':''}).encode())
        manifest=json.loads((self.root/self.manifestpath).read_bytes())
        for name in (queuepath,self.reviewpath):
            data=(self.root/name).read_bytes();manifest['files'][name]={'bytes':len(data),'sha256':t.sha(data)}
        self.put(self.manifestpath,json.dumps(manifest).encode())
        self.spec['source_manifest']['sha256']=t.sha((self.root/self.manifestpath).read_bytes())
        with self.assertRaisesRegex(ValueError,'Unsupported placement'): t.render(self.root,self.spec)
        self.spec['version']=t2.VERSION
        row=next(csv.DictReader(io.StringIO(t2.render(self.root,self.spec).decode('utf-8-sig'))))
        self.assertEqual(row['city'],'Рязань')
        transformed=t2.render(self.root,self.spec)
        ident=self.spec['transformation_id']; specpath=f'Data/transformations/{ident}/transformation.json';csvpath=f'Data/transformations/{ident}/vacancies_flat.csv'
        self.put(specpath,json.dumps(self.spec,ensure_ascii=False).encode());self.put(csvpath,transformed)
        entry={'run_id':self.run,'source':'avito','path':csvpath,'sha256':t.sha(transformed),'placement_regions':{'Рязань':'Рязанская область'},
               'transformation':{'path':specpath,'sha256':t.sha((self.root/specpath).read_bytes())}}
        built,_=build(self.root,{'schema_version':'1','exports':[entry]})
        result=next(csv.DictReader(io.StringIO(built.decode('utf-8-sig'))))
        self.assertEqual((result['city'],result['region']),('Рязань','Рязанская область'))
        self.assertEqual(t.normalize_explicit_region('Республика Татарстан (Татарстан)'), 'Республика Татарстан')
        self.spec['placements']['123']['region']='Московская область'
        with self.assertRaisesRegex(ValueError,'Unsupported placement'): t2.render(self.root,self.spec)
        self.spec['placements']['123']['region']='Рязанская область'
        self.put(queuepath,json.dumps([{'id':'123','location_raw':'Москва'}]).encode())
        data=(self.root/queuepath).read_bytes();manifest['files'][queuepath]={'bytes':len(data),'sha256':t.sha(data)}
        self.put(self.manifestpath,json.dumps(manifest).encode());self.spec['source_manifest']['sha256']=t.sha((self.root/self.manifestpath).read_bytes())
        with self.assertRaisesRegex(ValueError,'Unsupported placement'): t2.render(self.root,self.spec)

    def test_escaping_lineage_path_rejected(self):
        self.spec['source_manifest']['path']='../manifest.json'
        with self.assertRaisesRegex(ValueError,'path'): t.render(self.root,self.spec)


if __name__ == '__main__': unittest.main()
