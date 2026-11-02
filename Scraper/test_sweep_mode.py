"""Offline regressions for branch sweep orchestration."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).parent / 'skills/collect-avito/scripts/sweep_mode.py'
SPEC = importlib.util.spec_from_file_location('sweep_mode', MODULE)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def control(label, marker, value='1'):
    return {'label': label, 'marker': marker, 'value': value}


def item(identity, promoted=False):
    return {'id': str(identity), 'url': f'https://www.avito.ru/moskva/vakansii/job_{identity}',
        'title': f'Job {identity}', 'promoted': promoted}


class SweepModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base_controls = [control('Москва', 'region', 'moskva'), control('Курьер', 'profession', 'courier')]
        self.shift = control('Дневная смена', 'shift', 'day')
        self.spec = {
            'schema_version': 1, 'entry_mode': 'city_profession', 'max_pages': 30,
            'base': {'geography': 'Москва', 'profession': 'Курьер', 'constraints': []},
            'routing': {},
            'filter_catalog': [{'name': 'Смена', 'values': [
                {'value': 'Дневная', 'status': 'planned', 'branch_id': 'shift_day'},
                {'value': 'Ночная', 'status': 'skipped', 'reason': 'not available in the base slice'}]}],
            'branches': [
                {'id': 'base', 'kind': 'base', 'expected_selected_controls': self.base_controls,
                 'expected_slice': {'geography': 'Москва', 'profession': 'Курьер'}},
                {'id': 'shift_day', 'kind': 'additional', 'expected_selected_controls': self.base_controls + [self.shift],
                 'expected_slice': {'geography': 'Москва', 'profession': 'Курьер'},
                 'additional_filter': {'name': 'Смена', 'value': 'Дневная'}, 'parent_context': []}
            ]}
        self.spec_path = self.root / 'spec.json'
        self.spec_path.write_text(json.dumps(self.spec, ensure_ascii=False), encoding='utf-8')
        sweep.init_sweep(self.root, 'demo', self.spec_path, {'instruction_version': 'test'})

    def tearDown(self):
        self.temp.cleanup()

    def freeze(self, run_id, controls, pages, end='confirmed_end'):
        out, service = sweep._run_paths(self.root, run_id)
        queue, seen, records = [], set(), []
        for number, identities in enumerate(pages, 1):
            listing = {'state': 'LISTING', 'source_url': f'https://www.avito.ru/moskva/vakansii?p={number}',
                'observed_at': f'2026-10-30T00:00:{number:02d}Z', 'selected_controls': controls,
                'items': [item(identity, identity % 2 == 0) for identity in identities]}
            path = out / f'listing_pages/{number:04d}.json'
            sweep._write(path, listing)
            records.append({'file': f'listing_pages/{number:04d}.json', 'sha256': sweep._sha(path)})
            for candidate in listing['items']:
                if candidate['id'] not in seen:
                    seen.add(candidate['id']); queue.append(dict(candidate, listing_rank=len(queue) + 1))
        sweep._write(out/'listing.json', sweep._read(out/'listing_pages/0001.json'))
        sweep._write(out/'queue.json', queue)
        sweep._write(service/'plan.json', {'run_id': run_id, 'geography': 'Москва', 'profession': 'Курьер',
            'profession_scope': None, 'segments': '', 'selected_controls': controls,
            'listing_pages': records, 'listing_end_reason': end, 'setup_seconds': 1,
            'listing_url': 'https://www.avito.ru/moskva/vakansii', 'branch_only': True})
        return out, service

    def reset_evidence(self):
        path = self.root / 'reset.json'
        sweep._write(path, {'state': 'LISTING', 'selected_controls': self.base_controls})
        return path

    def test_overlap_memberships_materialize_resume_and_demo_csv(self):
        self.freeze('base_run', self.base_controls, [[1, 2], [2]])
        self.freeze('shift_run', self.base_controls + [self.shift], [[2, 3]])
        sweep.register_branch(self.root, 'demo', 'base', 'base_run', None)
        sweep.register_branch(self.root, 'demo', 'shift_day', 'shift_run', self.reset_evidence())
        data, service = sweep._sweep_paths(self.root, 'demo')
        queue = sweep._read(data/'unique_queue.json')
        self.assertEqual([row['id'] for row in queue], ['1','2','3'])
        self.assertEqual([row['branch_id'] for row in queue[1]['sweep_memberships']], ['base','shift_day'])
        coverage = sweep._read(data/'coverage.json')['branches']
        self.assertEqual((coverage[0]['new_unique'],coverage[1]['new_unique'],coverage[1]['duplicates']), (2,1,1))
        result = sweep.materialize(self.root, 'demo', 'capture', {'instruction_version': 'test'}, False)
        self.assertEqual(result['unique_ids'], 3)
        out, run_service = sweep._run_paths(self.root, 'capture')
        self.assertEqual(sweep.validate_materialized(self.root, out, run_service)['unique_ids'], 3)
        checkpoint = sweep._read(run_service/'diagnostics/checkpoint.json')
        checkpoint['completed_ids'] = ['1']; checkpoint['next_id'] = '2'
        sweep._write(run_service/'diagnostics/checkpoint.json', checkpoint)
        self.assertEqual(sweep._read(out/'queue.json')[1]['id'], '2')
        exports = self.root/'Data/avito_runs/base_run/exports'; exports.mkdir(parents=True)
        with (exports/'vacancies_flat.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['id','title'],lineterminator='\n');writer.writeheader();writer.writerows([{'id':'1','title':'A'},{'id':'2','title':'B'}])
        demo=sweep.demo_export(self.root,'demo','base_run')
        self.assertEqual(demo['rows'],2)
        with (self.root/demo['path']).open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(rows[1]['discovery_branch_count'],'2')
        limited = sweep.demo_export(self.root, 'demo', 'base_run', 1)
        self.assertEqual(limited['rows'], 1)
        self.assertEqual(sweep._read(data/'demo_validation.json')['sample_limit'], 1)
        with self.assertRaisesRegex(ValueError, 'positive'):
            sweep.demo_export(self.root, 'demo', 'base_run', 0)
        original=(data/'memberships.json').read_bytes();(data/'memberships.json').write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'provenance'): sweep.validate_materialized(self.root,out,run_service)

    def test_accumulated_filter_and_error_are_not_zero_results(self):
        extra = control('Подработка', 'schedule', 'part')
        self.freeze('base_run', self.base_controls, [[9]])
        sweep.register_branch(self.root,'demo','base','base_run',None)
        self.freeze('bad', self.base_controls + [self.shift, extra], [[1]])
        with self.assertRaisesRegex(ValueError,'controls differ'):
            sweep.register_branch(self.root,'demo','shift_day','bad',self.reset_evidence())
        result=sweep.set_branch_status(self.root,'demo','shift_day','error','Aside timeout')
        self.assertEqual(result['status'],'error')
        data,_=sweep._sweep_paths(self.root,'demo')
        row=[r for r in sweep._read(data/'coverage.json')['branches'] if r['branch_id']=='shift_day'][0]
        self.assertEqual((row['status'],row['observed_ids']),('error',0))
        with self.assertRaisesRegex(ValueError,'--allow-partial'):
            sweep.materialize(self.root,'demo','capture',{'instruction_version':'test'},False)

    def test_reset_and_source_hashes_are_enforced(self):
        self.freeze('base_run', self.base_controls, [[1]])
        self.freeze('shift_run', self.base_controls + [self.shift], [[2]])
        sweep.register_branch(self.root,'demo','base','base_run',None)
        wrong=self.root/'wrong.json';sweep._write(wrong,{'state':'LISTING','selected_controls':self.base_controls+[self.shift]})
        with self.assertRaisesRegex(ValueError,'not removed'):
            sweep.register_branch(self.root,'demo','shift_day','shift_run',wrong)
        out,_=sweep._run_paths(self.root,'shift_run');page=out/'listing_pages/0001.json';page.write_bytes(page.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'page hash'):
            sweep.register_branch(self.root,'demo','shift_day','shift_run',self.reset_evidence())

    def test_wide_modes_require_supplied_lists_and_ranges_are_finite(self):
        bad=dict(self.spec,entry_mode='profession_regions',base={'geography':'Все регионы','profession':'Курьер'},routing={})
        path=self.root/'bad.json';path.write_text(json.dumps(bad,ensure_ascii=False),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'supplied region list'):
            sweep.init_sweep(self.root,'bad',path,{})
        ranged=json.loads(json.dumps(self.spec));ranged['branches'][1]['numeric_range']={'minimum':10,'maximum':5,'unit':'час'}
        path.write_text(json.dumps(ranged,ensure_ascii=False),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'minimum exceeds'):
            sweep.init_sweep(self.root,'bad2',path,{})


if __name__ == '__main__':
    unittest.main()
