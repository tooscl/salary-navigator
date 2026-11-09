import json
import unittest
from copy import deepcopy
from utils import city_scope_subset as s
from utils.test_city_scope_audit import CityScopeTests


class SubsetTests(CityScopeTests):
    def setUp(self):
        super().setUp()
        self.plan=f'Scraper/runs/{self.run}/plan.json';self.listing=f'{self.base}/listing.json'
        self.put(self.plan,{'profession':'','profession_scope':'all'})
        self.put(self.listing,{'profession_filter':{'observed':True,'selected_count':0},
            'selected_controls':[{'marker':'params[172815]/3286367','label':'Удалённо'}]})
        self.write_manifest()

    def spec(self):return s.make_spec(self.root,self.run,'subset-v1','Краснодар','Краснодарский край')

    def test_reproducible_subset_and_search_evidence(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        spec=self.spec(); self.assertEqual(spec['audit']['counts']['in_city'],1)
        self.assertEqual(spec['audit']['decisions'][0]['remote_evidence']['basis'],'remote_search_filter')
        self.assertEqual(s.render(self.root,spec),s.render(self.root,spec))
        self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_wrong_profession_or_missing_filter_rejected(self):
        self.put(self.plan,{'profession':'Продавец','profession_scope':'all'});self.write_manifest()
        with self.assertRaisesRegex(ValueError,'profession and remote'): self.spec()

    def test_changed_decision_or_timestamp_not_accepted(self):
        spec=self.spec();altered=deepcopy(spec);altered['audit']['decisions'][0]['decision']='unknown'
        with self.assertRaisesRegex(ValueError,'differs'): s.render(self.root,altered)
        path=self.root/f'{self.base}/raw/1.json';raw=json.loads(path.read_bytes());raw['observed_at']='new';path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,'Source file changed'):s.render(self.root,spec)

    def test_multiple_formats_and_contradictory_card(self):
        path=f'{self.base}/raw/1.json';raw=json.loads((self.root/path).read_bytes())
        raw['fields']=[{'name':'Формат работы','value':'Удалённо, Гибрид'}];self.put(path,raw);self.write_manifest()
        self.assertEqual(self.spec()['audit']['decisions'][0]['remote_evidence']['observed_value'],'Удалённо, Гибрид')
        raw['fields'][0]['value']='В офисе';self.put(path,raw);self.write_manifest()
        with self.assertRaisesRegex(ValueError,'contradicts'):self.spec()

if __name__ == '__main__': unittest.main()
