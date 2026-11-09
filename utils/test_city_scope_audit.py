import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from utils import city_scope_audit as c
from utils.placement_transform import sha


class CityScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.run='example';self.base=f'Data/avito_runs/{self.run}'
        self.files={}; self.rows=[]; self.queue=[]
        for ident,city,address in [('1','Краснодар','Краснодарский край, Краснодар, ул. Мира'),
                                   ('2','Москва','г. Москва, ул. Мира'),
                                   ('3','','Краснодарский край'),
                                   ('4','Карцево','Рязанская обл., Рязань, посёлок Карцево, 6')]:
            url=f'https://example.test/{ident}';self.queue.append({'id':ident,'url':url})
            self.put(f'{self.base}/raw/{ident}.json',{'id':ident,'source_url':url,'location_raw':[address],'read_status':{'location':'observed'}})
            self.put(f'{self.base}/reviewed/{ident}.json',{'id':ident,'city':city,'city_source':'card' if city else 'missing','region':'region','analysis_status':'validated'})
            self.rows.append({'id':ident,'city':city,'city__source':'card' if city else 'missing','source_url':url,'location_observed':address})
        self.queue.append({'id':'5','url':'https://example.test/5'})
        self.put(f'{self.base}/queue.json',self.queue)
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=self.rows[0],lineterminator='\n');writer.writeheader();writer.writerows(self.rows)
        self.putbytes(f'{self.base}/exports/vacancies_flat.csv',stream.getvalue().encode('utf-8-sig'))
        self.manifest={'run_id':self.run,'status':'complete','files':self.files,'coverage':{'planned':5,'accepted':4,'unavailable':[{'id':'5'}],'unresolved_observations':[],'user_skipped_observations':[]}}
        self.write_manifest()

    def putbytes(self,name,data):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        self.files[name]={'bytes':len(data),'sha256':sha(data)}

    def put(self,name,obj):self.putbytes(name,json.dumps(obj,ensure_ascii=False).encode())
    def write_manifest(self):
        p=self.root/f'Scraper/runs/{self.run}/manifest.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(self.manifest),encoding='utf-8')

    def test_complete_accounting_and_no_source_mutation(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result=c.audit(self.root,self.run,'Краснодар','Краснодарский край')
        self.assertEqual(result['counts'],{'in_city':1,'outside_city':2,'unknown':1,'unavailable_or_unresolved':1})
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        self.assertIn('not established',result['scope'])

    def test_provisional_csv_has_only_confirmed_in_city(self):
        report=c.audit(self.root,self.run,'Краснодар','Краснодарский край')
        first=c.render_included(self.root,self.run,report)
        rows=list(csv.DictReader(io.StringIO(first.decode('utf-8-sig'))))
        self.assertEqual([x['id'] for x in rows],['1'])
        self.assertEqual(first,c.render_included(self.root,self.run,report))

    def test_nested_locality_with_target_city_is_unknown(self):
        result=c.audit(self.root,self.run,'Рязань','Рязанская обл.')
        self.assertEqual(next(x for x in result['decisions'] if x['id']=='4')['decision'],'unknown')

    def test_unknown_target_region_is_not_in_city(self):
        result=c.audit(self.root,self.run,'Краснодар','some other region')
        self.assertEqual(next(x for x in result['decisions'] if x['id']=='1')['decision'],'unknown')

    def test_changed_raw_rejected(self):
        (self.root/f'{self.base}/raw/1.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Source file changed'): c.audit(self.root,self.run,'Краснодар','Краснодарский край')

    def test_missing_accounting_rejected(self):
        self.manifest['coverage']['unavailable']=[];self.write_manifest()
        with self.assertRaisesRegex(ValueError,'accounted'): c.audit(self.root,self.run,'Краснодар','Краснодарский край')

    def test_search_filter_city_is_unknown(self):
        p=self.root/f'{self.base}/exports/vacancies_flat.csv';rows=list(csv.DictReader(io.StringIO(p.read_bytes().decode('utf-8-sig'))));rows[0]['city__source']='search_filter'
        s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows);self.putbytes(f'{self.base}/exports/vacancies_flat.csv',s.getvalue().encode('utf-8-sig'));self.write_manifest()
        result=c.audit(self.root,self.run,'Краснодар','Краснодарский край')
        self.assertEqual(result['decisions'][0]['decision'],'unknown')


if __name__ == '__main__': unittest.main()
