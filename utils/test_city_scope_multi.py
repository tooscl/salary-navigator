import unittest
from copy import deepcopy
from utils import city_scope_multi as m
from utils.test_city_scope_subset import SubsetTests


class MultiTests(SubsetTests):
    def test_two_independent_audits_are_reproducible(self):
        targets=[{'city':'Краснодар','region_quote':'Краснодарский край'},{'city':'Рязань','region_quote':'Рязанская обл.'}]
        spec=m.make_spec(self.root,self.run,'multi-v1',targets)
        self.assertEqual([a['counts']['in_city'] for a in spec['audits']],[1,0])
        for audit,info in zip(spec['audits'],spec['city_outputs']):
            self.putbytes('Data/transformations/multi-v1/'+info['file'],m.city_scope_subset.city_scope_audit.render_included(self.root,self.run,audit))
        self.assertEqual(m.render(self.root,spec),m.render(self.root,spec))
        altered=deepcopy(spec);altered['audits'][0]['decisions'][0]['decision']='unknown'
        with self.assertRaisesRegex(ValueError,'differs'):m.render(self.root,altered)
        with self.assertRaisesRegex(ValueError,'distinct'):m.make_spec(self.root,self.run,'multi-v1',[targets[0],targets[0]])

if __name__=='__main__':unittest.main()
