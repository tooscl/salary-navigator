"""Offline collector regressions; no network, Aside or GitHub required."""
import argparse
import csv
import sys
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).parent / 'skills/collect-avito/scripts'
spec = importlib.util.spec_from_file_location('aside_runner', SCRIPTS / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def item(identity='123'):
    return dict(id=identity, url=f'https://www.avito.ru/moskva/vakansii/example_{identity}',
                listing_rank=1, promoted=True, location_raw='Москва')


def courier_controls():
    # Actual duplicated label/input controls saved by the first-run smoke.
    return [dict(label='Курьер', marker='params[144170]/checkbox/3201765', value=None),
            dict(label='Курьер', marker='params[144170]/checkbox/toggle', value='3201765'),
            dict(label='Любой', marker='params[196929]/toggle', value='all')]


def raw(identity='123'):
    return dict(id=identity, state='DETAIL', title='Работа', source_url=item(identity)['url'],
        observed_at='2026-10-27T14:00:00Z', description='Полное описание\nВторая строка',
        fields=[dict(name='Новый параметр', value='Да')], employer_raw=['Компания'],
        location_raw=['Москва, пос. Коммунарка'], parameter_audit={'rendered_complete': True})


class RunnerTests(unittest.TestCase):
    def test_staged_diff_allows_text_separator_only_in_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            def staged(path, content):
                subprocess.run(['git', 'reset', '-q'], cwd=root, check=True)
                target = root/path; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                subprocess.run(['git', 'add', '--', path], cwd=root, check=True)
            staged('Data/sample.csv', 'description\n"first\u2028unicode separator\n======= АКЦИЯ ОТ КУПЕР=========\nlast"\n')
            runner.check_staged_diff(root)
            for path, marker in [('Data/sample.csv', '<<<<<<< ours'), ('Data/sample.csv', '======='),
                                 ('Data/sample.csv', '>>>>>>> theirs'), ('notes.md', '======= АКЦИЯ')]:
                staged(path, marker + '\n')
                with self.assertRaises(ValueError):
                    runner.check_staged_diff(root)

    def test_post_finalize_review_pins_code_and_appends_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, service, _ = runner.paths(root, 'review')
            for name in runner.POST_FINALIZE_TESTED:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('current')
            runner.write(service/'manifest.json', {'status': 'complete'})
            name = runner.POST_FINALIZE_TESTED[0]
            manifest = {'collection': {'automation_sha256': {name: 'old-hash'}},
                        'automation_changes': [{'file': f, 'reason': 'regression', 'validation': 'passed'} for f in runner.POST_FINALIZE_TESTED]}
            args = argparse.Namespace(run_id='review', reason='restore page validation')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'validate_completed_export', return_value=manifest), patch.object(runner.subprocess, 'run') as run:
                runner.post_finalize_review(args)
                audit = runner.read(out/'exports/post_finalize_review.json')
                self.assertEqual(audit['original_manifest_sha256'], runner.sha(service/'manifest.json'))
                self.assertEqual(audit['current_automation_sha256'][name], runner.sha(root/name))
                self.assertEqual(run.call_count, 2)
                first = out/'exports/post_finalize_review.json'; original = first.read_bytes()
                runner.post_finalize_review(args)
                self.assertEqual(first.read_bytes(), original)
                self.assertEqual(runner.read(out/'exports/post_finalize_review_002.json')['previous_review_sha256'], runner.sha(first))

    def test_audited_profession_correction_requires_unchanged_consistent_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, service, _ = runner.paths(root, 'child')
            urls = ['https://www.avito.ru/moskva/vakansii?p=1', 'https://www.avito.ru/moskva/vakansii?p=2']
            pages = []
            for i, url in enumerate(urls):
                page = dict(source_url=url, selected_controls=courier_controls(), pagination={'next_url': urls[1] if i == 0 else None})
                path = out/f'listing_pages/000{i+1}.json'; runner.write(path, page)
                pages.append(dict(file=f'listing_pages/000{i+1}.json', sha256=runner.sha(path), source_url=url))
            runner.write(out/'listing.json', runner.read(out/'listing_pages/0001.json'))
            parent = dict(run_id='parent', profession='', profession_scope='all', listing_request_url=urls[0],
                listing_pages=pages, filter_evidence='original evidence', limit=2, requested_limit=None, available_unique=2)
            parent_path = out/'provenance/parent-plan.json'; runner.write(parent_path, parent)
            info = dict(parent_plan_sha256=runner.sha(parent_path), parent_queue_sha256='queue-hash', start=0, end=2)
            plan = dict(parent, run_id='child', limit=2, requested_limit=2, available_unique=2, partition=info)
            runner.write(service/'plan.json', plan)
            before = (service/'plan.json').read_bytes()
            result = runner.observed_profession_correction(out, service, plan, 'Wrong scope and evidence URL')
            self.assertEqual(result['corrected_profession'], 'Курьер')
            self.assertEqual(result['observed_source_url'], urls[0])
            self.assertEqual(before, (service/'plan.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'reason'):
                runner.observed_profession_correction(out, service, plan, '')
            with self.assertRaisesRegex(ValueError, 'legacy all-scope'):
                runner.observed_profession_correction(out, service, dict(plan, profession_scope='top20'), 'reason')
            second = out/'listing_pages/0002.json'
            changed = runner.read(second); changed['selected_controls'] = [dict(label='Водитель', marker='params[144170]/checkbox/1')]
            runner.write(second, changed)
            with self.assertRaisesRegex(ValueError, 'hash'):
                runner.observed_profession_correction(out, service, plan, 'reason')
            pages[1]['sha256'] = runner.sha(second)
            runner.write(parent_path, parent); info['parent_plan_sha256'] = runner.sha(parent_path)
            runner.write(service/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'controls differ'):
                runner.observed_profession_correction(out, service, plan, 'reason')
            changed['selected_controls'] = []
            changed['profession_filter'] = {'observed': True, 'selected_count': 0}
            runner.write(second, changed); pages[1]['sha256'] = runner.sha(second)
            runner.write(parent_path, parent); info['parent_plan_sha256'] = runner.sha(parent_path); runner.write(service/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'empty'):
                runner.observed_profession_correction(out, service, plan, 'reason')
            changed['selected_controls'] = courier_controls(); changed['source_url'] = urls[1] + '&unexpected=1'
            runner.write(second, changed); pages[1]['sha256'] = runner.sha(second)
            runner.write(parent_path, parent); info['parent_plan_sha256'] = runner.sha(parent_path); runner.write(service/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'URL chain'):
                runner.observed_profession_correction(out, service, plan, 'reason')

    def test_validate_all_pages_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); url = 'https://www.avito.ru/moskva/vakansii'
            first = dict(source_url=url, selected_controls=courier_controls())
            second = dict(source_url=url+'?p=2', selected_controls=courier_controls())
            runner.write(out/'listing.json', first); runner.write(out/'listing_pages/0001.json', first)
            runner.write(out/'listing_pages/0002.json', second)
            pages = [dict(file=f'listing_pages/000{i}.json', sha256=runner.sha(out/f'listing_pages/000{i}.json')) for i in (1, 2)]
            plan = dict(profession='Курьер', listing_sha256=runner.sha(out/'listing.json'), listing_pages=pages)
            runner.validate_plan_profession(plan, out/'listing.json')
            second['selected_controls'] = [dict(label='Водитель', marker='params[144170]/checkbox/1')]
            runner.write(out/'listing_pages/0002.json', second)
            with self.assertRaisesRegex(ValueError, 'page changed'):
                runner.validate_plan_profession(plan, out/'listing.json')

    def test_closed_policy_records_question_without_changing_authorization_budget(self):
        question = 'Разрешаете пропускать это и все последующие явно закрытые объявления, сохраняя подтверждение причины, и продолжить очередь?'
        answer = 'да, закрытые вакансии отмечай как недоступные, исключи из csv и продолжай поостальным вакансиям'
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'context')
            runner.record_failure(diag, 'capture', 'closed', 'capture', immediate=True)
            runner.authorize_browser(diag, answer, 'same-id')
            before = runner.read(diag/'failures.json')
            args = argparse.Namespace(run_id='context', authorization_id='same-id', authorization_question=question)
            runner.closed_policy(args)
            self.assertEqual(runner.read(diag/'failures.json'), before)
            policy = runner.read(out/'closed_policy.json')
            self.assertEqual(policy['authorization_question'], question)
            self.assertEqual(policy['human_instruction'], answer)
            self.assertEqual(policy['authorization_id'], 'same-id')
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'negative')
            runner.record_failure(diag, 'capture', 'closed', 'capture', immediate=True)
            runner.authorize_browser(diag, 'да, но не последующие', 'same-id')
            with self.assertRaises(ValueError):
                runner.closed_policy(argparse.Namespace(run_id='negative', authorization_id='same-id', authorization_question=question))

    def test_partition_preserves_order_provenance_and_shared_failure_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'parent')
            page = dict(state='LISTING', items=[item(str(i)) for i in range(1, 6)], source_url='https://www.avito.ru/moskva/vakansii')
            runner.write(out/'listing.json', page); runner.write(out/'listing_pages/0001.json', page)
            queue = runner.queue_from_pages([page], 5)
            runner.write(out/'queue.json', queue)
            plan = dict(run_id='parent', selection_mode='first_N_unique_on_frozen_listing', profession='',
                listing_pages=[dict(file='listing_pages/0001.json',sha256=runner.sha(out/'listing_pages/0001.json'))])
            runner.write(service/'plan.json', plan)
            with patch.object(runner, 'validate_plan_profession'):
                runner.partition(argparse.Namespace(run_id='parent', batch_size=2, prefix='piece'))
                children = [runner.paths(runner.ROOT, f'piece_{i:02d}') for i in range(1, 4)]
                self.assertEqual([x['id'] for child,_,_ in children for x in runner.read(child/'queue.json')], [str(i) for i in range(1,6)])
                self.assertEqual([x['listing_rank'] for child,_,_ in children for x in runner.read(child/'queue.json')], list(range(1,6)))
                for child, child_service, _ in children: runner.validate_partition(child, child_service)
                changed = runner.read(children[1][0]/'queue.json'); changed[0]['id']='999'
                runner.write(children[1][0]/'queue.json', changed)
                with self.assertRaisesRegex(ValueError, 'contiguous'): runner.validate_partition(children[1][0], children[1][1])
                with self.assertRaisesRegex(ValueError, 'Parent queue is partitioned'): runner.validate_partition(out,service)
            first, second = children[0][2], children[1][2]
            runner.record_failure(first, 'capture', 'first error', 'test')
            runner.ensure_allowed(second, 'capture', 'one correction', action=True)
            runner.record_failure(second, 'capture', 'second error', 'test')
            self.assertEqual(runner.load_failures(diag)['state'], 'NEEDS_HUMAN')
            self.assertEqual(runner.load_failures(diag)['retry_limit'], 1)
            with self.assertRaises(runner.HumanRequired): runner.ensure_allowed(first, 'capture')

    def test_standing_policy_shares_audit_and_skips_only_proven_unavailable(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            _, _, parent = runner.paths(runner.ROOT, 'parent')
            runner.record_failure(parent,'capture','earlier error','test',immediate=True)
            runner.authorize_browser(parent,'Continue','prior-auth')
            before = runner.load_failures(parent)
            children = [runner.paths(runner.ROOT, f'child{i}') for i in range(1,7)]
            for out, service, diag in children:
                runner.write(diag/'shared-ledger.json', {'parent_run_id':'parent'})
                runner.standing_unavailable_policy(argparse.Namespace(run_id=service.name,authorization_id=runner.STANDING_UNAVAILABLE_ID,human_instruction=runner.STANDING_UNAVAILABLE_INSTRUCTION))
            ledger = runner.load_failures(parent)
            events = [e for e in ledger['events'] if e.get('event') == 'project_unavailable_policy']
            self.assertEqual(len(events),1)
            self.assertEqual(events[0]['source'],runner.STANDING_UNAVAILABLE_SOURCE)
            self.assertEqual((ledger['failures'],ledger['retries_started']),(before['failures'],before['retries_started']))
            for (out,service,diag), marker in zip(children,(runner.CLOSED_MARKERS[0],runner.REVIEW_MARKER,runner.BLOCKED_LISTING_MARKER,*runner.OTHER_UNAVAILABLE_MARKERS)):
                candidate=item('123'); runner.write(out/'queue.json',[candidate]); runner.write(service/'plan.json',{})
                runner.write(diag/'checkpoint.json',{'state':'REVIEW_REQUIRED','stop_reason':'identity_or_template_review','next_id':'123','completed_ids':[]})
                data={'source_url':candidate['url'],'state':'UNKNOWN_PAGE','heading':'Объявление не посмотреть','observed_at':'2026-10-28T00:00:00Z'}
                runner.write(out/'raw/123.json',data); raw_before=(out/'raw/123.json').read_bytes()
                (diag/'problem.snapshot.txt').write_text(f'# title [url={candidate["url"]}]\n- text: "{marker}"\n')
                runner.mark_unavailable(argparse.Namespace(run_id=service.name,id='123',authorization_id=runner.STANDING_UNAVAILABLE_ID,automatic=True))
                self.assertEqual((out/'raw/123.json').read_bytes(),raw_before)
                self.assertEqual(runner.read(out/'unavailable.json')[0]['evidence'],marker)
                self.assertFalse((diag/'failures.json').exists())
            self.assertFalse(runner.closed_raw_matches({'state':'NEEDS_HUMAN','source_url':item()['url']},'123',item()['url'],runner.CLOSED_MARKERS[0]))
            self.assertFalse(runner.closed_raw_matches({'state':'UNKNOWN_PAGE','heading':'Объявление не посмотреть','source_url':'https://wrong.example'},'123',item()['url'],runner.CLOSED_MARKERS[0]))
            self.assertFalse(runner.closed_raw_matches({'state':'UNKNOWN_PAGE','heading':'Объявление не посмотреть','source_url':item()['url']},'123',item()['url'],'CAPTCHA'))

    def test_capture_auto_registers_standing_policy_and_skips_proven_review(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            _,_,parent=runner.paths(runner.ROOT,'parent')
            runner.record_failure(parent,'filters','prior stop','test',immediate=True)
            runner.authorize_browser(parent,'Continue','prior')
            out,service,diag=runner.paths(runner.ROOT,'child')
            runner.write(diag/'shared-ledger.json',{'parent_run_id':'parent'})
            candidate=item('123');runner.write(out/'queue.json',[candidate]);runner.write(service/'plan.json',{'automation_sha256':{},'selection_mode':'first_N_unique_on_frozen_listing'})
            runner.write(diag/'checkpoint.json',{'state':'READY','completed_ids':[],'next_id':'123'})
            def fake_browser(job,out,diag):
                runner.write(out/'raw/123.json',{'source_url':candidate['url'],'state':'UNKNOWN_PAGE','heading':'Объявление не посмотреть'})
                (diag/'problem.snapshot.txt').write_text(f'# title [url={candidate["url"]}]\n- text: "{runner.REVIEW_MARKER}"\n')
                return {'state':'REVIEW_REQUIRED','completed_ids':[],'next_id':'123','stop_reason':'identity_or_template_review'},1
            args=argparse.Namespace(run_id='child',batch_size=1,config=None,retry_reason=None)
            before=runner.load_failures(parent)
            with patch.object(runner,'validate_partition'),patch.object(runner,'validate_plan_profession'),patch.object(runner,'preflight'),patch.object(runner,'browser',side_effect=fake_browser):
                runner.capture(args)
            after=runner.load_failures(parent)
            self.assertEqual((after['failures'],after['retries_started']),(before['failures'],before['retries_started']))
            self.assertEqual(runner.read(out/'unavailable.json')[0]['status'],'unavailable_under_review')
            self.assertEqual(runner.read(diag/'checkpoint.json')['state'],'READY')

    def test_partition_closed_policy_uses_parent_authorization_and_keeps_raw(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            _, _, parent = runner.paths(runner.ROOT, 'parent')
            children = [runner.paths(runner.ROOT, f'child{i}') for i in (1, 2)]
            for _, _, diag in children:
                runner.write(diag/'shared-ledger.json', {'parent_run_id':'parent'})
            runner.record_failure(parent, 'capture', 'closed', 'read', immediate=True)
            args = lambda run: argparse.Namespace(run_id=run, authorization_id='human-closed')
            with self.assertRaises(runner.HumanRequired): runner.closed_policy(args('child1'))
            runner.authorize_browser(parent, 'Исключай закрытые и все последующие закрытые вакансии', 'human-closed')
            ledger_before = (parent/'failures.json').read_bytes()
            for run in ('child1','child2'): runner.closed_policy(args(run))
            out, service, diag = children[0]
            candidate = item('123'); runner.write(out/'queue.json', [candidate])
            runner.write(service/'plan.json', {'selection_mode':'first_N_unique_on_frozen_listing'})
            runner.write(diag/'checkpoint.json', {'state':'REVIEW_REQUIRED','stop_reason':'identity_or_template_review','next_id':'123','completed_ids':[]})
            data = {'source_url':candidate['url'],'state':'UNKNOWN_PAGE','heading':'Объявление не посмотреть','observed_at':'2026-10-28T00:00:00Z'}
            runner.write(out/'raw/123.json',data); before = (out/'raw/123.json').read_bytes()
            (diag/'problem.snapshot.txt').write_text(f'# title [url={candidate["url"]}]\n- text: "Объявление закрыто."\n')
            runner.mark_unavailable(argparse.Namespace(run_id='child1',id='123',authorization_id='human-closed'))
            self.assertEqual((out/'raw/123.json').read_bytes(),before)
            self.assertEqual((parent/'failures.json').read_bytes(),ledger_before)
            self.assertFalse(any((d/'failures.json').exists() for _,_,d in children))
            self.assertEqual(runner.read(children[1][0]/'closed_policy.json')['authorization_id'],'human-closed')
            runner.record_failure(children[1][2], 'capture', 'another stop', 'read', immediate=True)
            with self.assertRaises(runner.HumanRequired): runner.closed_policy(args('child2'))

    def test_archive_rejected_partial_listing_preserves_failure_history(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'parent')
            runner.write(diag/'freeze_request.json', {'url':'original'})
            runner.write(diag/'pending_listing.json', {'url':'original'})
            runner.write(diag/'listing.json', {'selected_controls':[]})
            runner.write(diag/'checkpoint.json', {'state':'READY'})
            runner.record_failure(diag, 'freeze', 'mismatched selection', 'freeze')
            before = (diag/'failures.json').read_bytes()
            runner.archive_partial_listing(argparse.Namespace(run_id='parent', reason='Preserve rejected observation'))
            self.assertEqual((diag/'failures.json').read_bytes(), before)
            self.assertFalse((diag/'listing.json').exists())
            archive = service/'rejected-partial-listing'
            self.assertEqual(runner.read(archive/'listing.json'), {'selected_controls':[]})
            self.assertEqual(runner.read(archive/'provenance.json')['files']['listing.json'], runner.sha(archive/'listing.json'))
            with self.assertRaises(ValueError): runner.archive_partial_listing(argparse.Namespace(run_id='parent', reason='again'))

    def test_group_authorization_records_one_retry_without_erasing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)/'diagnostics'
            runner.write(diag/'retry-limit.json', {'retry_limit':1})
            runner.record_failure(diag, 'filters', 'blocked', 'open', immediate=True)
            runner.authorize_browser(diag, 'Explicitly continue', 'auth-one')
            ledger = runner.load_failures(diag)
            self.assertEqual(ledger['retry_limit'], 1)
            self.assertEqual(ledger['events'][-1]['retry_limit'], 1)
            self.assertEqual(ledger['failures'], 1)
            self.assertEqual(ledger['budget_start_failures'], 1)

    def test_browser_batch_preserves_stopped_tab_and_external_ownership(self):
        script = (SCRIPTS / 'browser_batch.js').read_text()
        for external, state in [(False, 'NEEDS_HUMAN'), (True, 'NEEDS_HUMAN'), (True, 'LISTING'), (False, 'LISTING')]:
            with self.subTest(external=external, state=state), tempfile.TemporaryDirectory() as directory:
                job = dict(mode='freeze', url='https://www.avito.ru/moskva/vakansii', completed_ids=[], close_tab=True, tab_id='external-id' if external else None)
                body = script.replace('__JOB__', json.dumps(job), 1).replace('__EXTRACTOR__', '"extract"', 1)
                harness = """const fs = require('fs').promises;
const pwd = process.cwd(); let closed=0, attached=0, navigated=0;
const pg = {url:()=> 'https://www.avito.ru/moskva/vakansii?token=secret#x', goto:async()=>{navigated++}, evaluate:async()=>({state:STATE,selected_controls:[],items:[]})};
let opened=false;
const listBrowserTabs=async()=>opened?[{targetId:'new-id',url:pg.url()}]:[];
const openTab=async()=>{opened=true;return pg};
const attachBrowserTab=async id=>{if(id!=='external-id')throw Error('wrong ID');attached++;return pg};
const snapshot=async()=>({tree:'stopped'});const closeTab=async()=>{closed++};
(async()=>{BODY; const result=JSON.parse(await fs.readFile('browser-tab.json')); console.log('RESULT='+JSON.stringify({closed,attached,navigated,result}));})().catch(e=>{console.error(e);process.exit(1)});
""".replace('STATE', json.dumps(state)).replace('BODY', body)
                result = subprocess.run(['node', '-e', harness], cwd=directory, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout.split('RESULT=')[-1])
                self.assertEqual(data['closed'], int(not external and state == 'LISTING'))
                self.assertEqual(data['attached'], int(external))
                self.assertEqual(data['navigated'], 0)
                self.assertNotIn('secret', data['result']['url'])
                self.assertEqual(data['result']['target_id'], 'external-id' if external else 'new-id')

    def test_parent_exits_with_live_repl_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            script = """import importlib.util, pathlib, subprocess, sys
spec=importlib.util.spec_from_file_location('runner',sys.argv[1]); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
p=subprocess.Popen([sys.executable,'-u','-c','import sys; print("READY",flush=True); sys.stdin.read()'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
q, reader=r.pipe_chunks(p.stdout); assert b'READY' in q.get(timeout=5)
r.retain_repl(p,pathlib.Path(sys.argv[2]),None,{})
p.stdin.close()
"""
            result = subprocess.run([sys.executable, '-c', script, str(SCRIPTS/'run.py'), directory],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            info = runner.read(Path(directory)/'browser-session.local.json')
            try:
                os.kill(info['repl_pid'], 0)
            finally:
                Path(info['release_marker']).touch()

    def test_retained_repl_stays_alive_until_release(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            proc = subprocess.Popen([sys.executable, '-u', '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                runner.retain_repl(proc, diag, None, {})
                info = runner.read(diag/'browser-session.local.json')
                proc.stdin.close(); proc.stdout.close()
                self.assertIsNone(proc.poll())
                Path(info['release_marker']).touch()
                proc.wait(timeout=5)
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait()

    def test_remote_selection_stops_at_exact_matches_and_keeps_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            queue = [dict(item(str(i)), listing_rank=i) for i in range(1, 5)]
            def save(i, fields):
                data = raw(str(i)); data['fields'] = fields
                runner.write(out / 'raw' / f'{i}.json', data)
            with patch.object(runner, 'check', return_value={'structural_check_passed': True}):
                save(1, [dict(name='Формат работы', value='Офис')])
                save(2, [dict(name='Формат работы', value='Удалённо')])
                save(3, [dict(name='Формат работы', value='Удалённо')])
                decisions = runner.remote_decisions(queue, out, 2)
                self.assertEqual([d['accepted'] for d in decisions], [False, True, True])
                self.assertEqual(decisions[0]['observed_value'], 'Офис')
                save(4, [dict(name='Формат работы', value='Удалённо')])
                with self.assertRaisesRegex(ValueError, 'contiguous prefix'):
                    runner.remote_decisions(queue, out, 2)
                (out / 'raw/4.json').unlink()
                (out / 'raw/2.json').unlink()
                with self.assertRaisesRegex(ValueError, 'contiguous prefix'):
                    runner.remote_decisions(queue, out, 2)

    def test_remote_capture_reads_only_until_target_and_resumes_prefix(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'selection')
            queue = [dict(item(str(i)), listing_rank=i) for i in range(1, 5)]
            runner.write(out / 'queue.json', queue)
            runner.write(service / 'plan.json', dict(selection_mode='first_N_remote_parameter_matches',
                limit=2, automation_sha256={}))
            runner.write(diag / 'checkpoint.json', {'capture_seconds': 0})
            seen = []
            def capture_one(job, out, diag):
                done = list(job['completed_ids'])
                for candidate in job['queue']:
                    identity = candidate['id']; seen.append(identity)
                    data = raw(identity)
                    data['fields'] = [dict(name='Формат работы', value='Офис' if identity == '1' else 'Удалённо')]
                    runner.write(out / 'raw' / (identity + '.json'), data)
                    done.append(identity)
                    if sum(d['accepted'] for d in runner.remote_decisions(job['selection_queue'], out, job['selection_target'])) == job['selection_target']:
                        break
                return {'state':'CAPTURED', 'completed_ids':done}, 0.01
            args = argparse.Namespace(run_id='selection', batch_size=2, config=None, retry_reason=None)
            with patch.object(runner, 'validate_plan_profession'), patch.object(runner, 'preflight'), \
                    patch.object(runner, 'browser', side_effect=capture_one), \
                    patch.object(runner, 'check', return_value={'structural_check_passed': True}):
                runner.capture(args)
                self.assertEqual(seen, ['1', '2'])
                runner.capture(args)
                self.assertEqual(seen, ['1', '2', '3'])
                runner.capture(args)
                self.assertEqual(seen, ['1', '2', '3'])
                self.assertFalse((out / 'raw/4.json').exists())

    def test_remote_selection_rejects_missing_and_invalid_card(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); queue = [item('123')]
            runner.write(out / 'raw/123.json', raw())
            with patch.object(runner, 'check', return_value={'structural_check_passed': True}):
                result = runner.remote_decisions(queue, out, 1)
                self.assertFalse(result[0]['accepted'])
                self.assertIsNone(result[0]['observed_value'])
            with patch.object(runner, 'check', return_value={'structural_check_passed': False}):
                with self.assertRaisesRegex(ValueError, 'needs review'):
                    runner.remote_decisions(queue, out, 1)

    def test_reselect_preserves_freeze_and_failure_budget_and_rejects_capture(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'example')
            runner.write(out/'listing.json', {'source': 'original'})
            runner.write(out/'queue.json', [{'id': '123'}])
            (out/'raw').mkdir()
            runner.write(service/'plan.json', {'listing': 'original'})
            runner.write(diag/'failures.json', {'events': [], 'stages': {}, 'failures': 4,
                'state': 'RECOVERED', 'retry_limit': 5, 'retries_started': 4,
                'budget_start_failures': 0, 'retry_started': False})
            before = (diag/'failures.json').read_bytes()
            args = argparse.Namespace(run_id='example', reason='Incorrect listing')
            runner.reselect(args)
            self.assertEqual((diag/'failures.json').read_bytes(), before)
            self.assertEqual(runner.read(service/'superseded-freeze/data/listing.json'), {'source': 'original'})
            self.assertEqual(runner.read(service/'superseded-freeze/plan.json'), {'listing': 'original'})
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError):
                runner.reselect(args)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'example')
            runner.write(out/'raw/123.json', {'id': '123'})
            runner.write(out/'queue.json', [])
            runner.write(service/'plan.json', {})
            with self.assertRaisesRegex(ValueError, 'before card capture'):
                runner.reselect(args)
            self.assertTrue((out/'raw/123.json').exists())

    def test_city_anomaly_can_be_missing_or_explicit_search_fallback(self):
        data = dict(raw(), location_raw=['Свердловская обл., Белоярский муниципальный округ, промышленный район Прохладный'])
        regions = {'Екатеринбург':'Свердловская область'}
        plan = dict(city_anomaly_policy='blank', geography='Екатеринбург',
                    listing_url='https://www.avito.ru/ekaterinburg/vakansii', filter_evidence='Екатеринбург подтверждён')
        row, _, region = runner.reviewed_row(data,item(),regions,plan)
        self.assertEqual((row['city'],row['city__source'],region),('','missing','Свердловская область'))
        row, _, region = runner.reviewed_row(data,item(),regions,dict(plan,city_anomaly_policy='search_filter'))
        self.assertEqual((row['city'],row['city__source']),('Екатеринбург','search_filter'))
        self.assertEqual(row['city__evidence'],plan['listing_url'])
        self.assertEqual(row['location_observed'],data['location_raw'][0])
        self.assertTrue(row['city_anomaly'])
        for changes in [dict(geography='Все регионы'),dict(filter_evidence='')]:
            with self.assertRaises(ValueError):
                runner.reviewed_row(data,item(),regions,dict(plan,city_anomaly_policy='search_filter',**changes))
        with self.assertRaises(ValueError):
            runner.reviewed_row(dict(data,id='wrong'),item(),regions,dict(plan,city_anomaly_policy='search_filter'))

    def test_aside_target_is_pinned_across_capture_and_profile_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(runner.os.environ, {}, clear=True):
            diag = Path(directory) / 'diagnostics'
            with patch.object(runner, 'find_aside', return_value='aside'), \
                    patch.object(runner, 'aside_command', return_value=['aside', '--host', 'local', '--account', 'u0', '--profile', 'first']) as choose:
                runner.pin_aside_target(diag)
                self.assertEqual(runner.os.environ['TL_ASIDE_PROFILE'], 'first')
                self.assertEqual(runner.os.environ['TL_ASIDE_ACCOUNT'], 'u0')
                runner.os.environ.pop('TL_ASIDE_ACCOUNT')
                runner.os.environ.pop('TL_ASIDE_PROFILE')
                runner.pin_aside_target(diag)
                self.assertEqual(runner.os.environ['TL_ASIDE_PROFILE'], 'first')
                choose.assert_called_once()
                self.assertEqual(runner.os.environ['TL_ASIDE_ACCOUNT'], 'u0')
            runner.os.environ['TL_ASIDE_PROFILE'] = 'different'
            with self.assertRaisesRegex(RuntimeError, 'pinned'):
                runner.pin_aside_target(diag)
            self.assertEqual(runner.read(diag/'aside-target.local.json')['TL_ASIDE_PROFILE'], 'first')

    def test_publish_retry_reuses_own_commit_and_existing_pr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run_id = 'publish_retry'
            out, service, diag = runner.paths(root, run_id)
            runner.write(service/'plan.json', {})
            runner.write(service/'manifest.json', dict(status='complete', files={}, collection={}, coverage={'accepted':1}))
            state = {'commits': 0, 'pushes': 0}
            def output(command, **kwargs):
                if command[:2] == ['git','branch']:
                    return 'codex/publish_retry'
                if command[:5] == ['git','diff','--cached','--name-only']:
                    return str((out/'raw/123.json').relative_to(root)) if not state['commits'] else ''
                if command[:2] == ['git','commit']:
                    state['commits'] += 1
                if command[:2] == ['git','push']:
                    state['pushes'] += 1
                    if state['pushes'] == 1:
                        raise subprocess.CalledProcessError(1,command)
                if command[:3] == ['gh','pr','create']:
                    self.fail('Existing PR must be reused')
                return ''
            def run(command, **kwargs):
                return subprocess.CompletedProcess(command,0,stdout=json.dumps({'state':'OPEN','url':'https://github.com/example/repo/pull/1'}))
            args = argparse.Namespace(run_id=run_id, retry_reason=None, summary='Sample', title='Data', prepare_only=False)
            with patch.object(runner,'ROOT',root), patch.object(runner.subprocess,'check_output',side_effect=output), patch.object(runner.subprocess,'run',side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.publish(args)
                runner.record_failure(diag,'publish','push failed','push')
                args.retry_reason = 'Connection restored'
                runner.publish(args)
            self.assertEqual(state, {'commits':1,'pushes':2})

    def test_regionless_vacancy_entry_is_rejected_before_browser(self):
        for url in ['https://www.avito.ru/vakansii', 'https://www.avito.ru/vakansii?context=x',
                    'https://www.avito.ru/vakansii/kurer', 'https://www.avito.ru/moskva']:
            with tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(run_id='bad_entry', resume=False, url=url, retry_reason=None)
                with patch.object(runner, 'ROOT', Path(directory)), patch.object(runner, 'preflight') as preflight, \
                        patch.object(runner, 'browser') as browser:
                    with self.assertRaisesRegex(ValueError, '/<region>/vakansii'):
                        runner.freeze(args)
                    preflight.assert_not_called(); browser.assert_not_called()
        for url in ['https://www.avito.ru/all/vakansii',
                    'https://www.avito.ru/all/vakansii?context=observed',
                    'https://www.avito.ru/moskva/vakansii',
                    'https://www.avito.ru/sankt-peterburg/vakansii?context=observed',
                    'https://www.avito.ru/ekaterinburg/vakansii/kurer-ASgObserved', item()['url']]:
            self.assertEqual(runner.avito_url(url), url)

    def test_queue_preserves_unique_order_and_requires_enough_real_links(self):
        listing = dict(state='LISTING', items=[item('123'), item('123'), item('456')])
        queue = runner.queue_from(listing, 2)
        self.assertEqual([q['id'] for q in queue], ['123', '456'])
        self.assertEqual([q['listing_rank'] for q in queue], [1, 2])
        with self.assertRaisesRegex(ValueError, 'Only 2'):
            runner.queue_from(listing, 3)
        for url in ['https://avito.ru.evil.test/vakansii/a_123',
                    'https://user:pass@www.avito.ru/moskva/vakansii/a_123',
                    'https://www.avito.ru/moskva/vakansii/a_999']:
            with self.assertRaises(ValueError):
                runner.queue_from(dict(state='LISTING', items=[dict(item(), url=url)]), 1)

    def test_checks_catch_real_selector_gap_and_keep_description_out_of_geography(self):
        data = raw()
        data['description'] += '\nРаботать в другом регионе, Екатеринбург'
        row, review, region = runner.reviewed_row(data, item(),
            {'Москва': 'Москва', 'Коммунарка': 'Москва'}, {})
        self.assertEqual(row['city'], 'Коммунарка')
        self.assertEqual(row['description'], data['description'])
        self.assertEqual(row['profession'], '')
        self.assertEqual(row['Новый параметр'], 'Да')
        self.assertFalse(review['semantic_review_performed'])
        for change in [dict(location_raw=[]), dict(parameter_audit={}), dict(id='999'),
                       dict(fields=[{'name': 'id', 'value': 'oops'}])]:
            with self.assertRaises(ValueError):
                runner.reviewed_row(dict(data, **change), item(), {'Москва': 'Москва'}, {})

    def test_missing_optional_description_address_and_format_do_not_reject_card(self):
        for missing in [dict(description=''), dict(description=None), dict(location_raw=[]),
                        dict(description=None, location_raw=[])]:
            data = dict(raw(), **missing)
            row, review, region = runner.reviewed_row(data, item(),
                {'Москва': 'Москва', 'Коммунарка': 'Москва'}, {'city_anomaly_policy':'blank'})
            self.assertEqual(row['id'], '123')
            self.assertEqual(row['Новый параметр'], 'Да')
            self.assertNotIn('Формат работы', row)
            if not data.get('description'):
                self.assertEqual(row['description'], '')
                self.assertEqual(row['read_status_description'], 'missing')
            if not data.get('location_raw'):
                self.assertEqual((row['city'], row['location_observed'], region), ('', '', ''))
                self.assertEqual(row['read_status_location'], 'missing')
        for invalid in [dict(id='999'), dict(title=''), dict(parameter_audit={}), dict(description=123)]:
            with self.assertRaises(ValueError):
                runner.reviewed_row(dict(raw(), **invalid), item(), {}, {'city_anomaly_policy':'blank'})

    def test_unknown_city_requires_evidence_and_does_not_trust_url_slug(self):
        data = dict(raw(), location_raw=['Москва, д. Сосенки'])
        city, region, _ = runner.placement(data, item(), {'Сосенки': 'Москва'})
        self.assertEqual(city, 'Сосенки')
        with self.assertRaises(ValueError):
            runner.placement(data, item(), {})
        with self.assertRaises(ValueError):
            runner.placement(data, item(), {}, {'city': 'Москва', 'region': 'Москва', 'evidence': 'not in address'})
        data = dict(raw(), location_raw=['Москва, улица Тверь, дом 3'])
        self.assertEqual(runner.placement(data, item(), {'Москва': 'Москва', 'Тверь': 'Тверская область'})[0], 'Москва')
        with self.assertRaisesRegex(ValueError, 'Unknown locality'):
            runner.placement(dict(raw(), location_raw=['Москва, пос. Неизвестный']), item(), {'Москва': 'Москва'})

    def test_empty_city_requires_explicit_confirmation_and_visible_evidence(self):
        data = dict(raw(), location_raw=['Свердловская обл., Белоярский муниципальный округ'])
        override = {'city': '', 'region': 'Свердловская область', 'evidence': 'Свердловская обл.', 'city_unknown_confirmed': True}
        self.assertEqual(runner.placement(data, item(), {}, override)[:2], ('', 'Свердловская область'))
        for invalid in [dict(override, city_unknown_confirmed=False), dict(override, evidence='Москва'), dict(override, region='')]:
            with self.assertRaises(ValueError):
                runner.placement(data, item(), {}, invalid)

    def test_location_address_excludes_metro_and_stops_at_streets(self):
        regions = {'Екатеринбург': 'Свердловская область', 'Тверь': 'Тверская область'}
        for address in ['Свердловская обл., Екатеринбург\nПлощадь 1905 года\nдо 5 мин.',
                        'Свердловская обл., Екатеринбург, пл. 1905 года\nПлощадь 1905 года',
                        'Свердловская обл., Екатеринбург, Первая линия\nУралмаш',
                        'Свердловская обл., Екатеринбург, Сибирский тракт, 16-й километр, 2']:
            with self.subTest(address=address):
                self.assertEqual(runner.placement(dict(raw(), location_raw=[address]), item(), regions)[0], 'Екатеринбург')
        with self.assertRaisesRegex(ValueError, 'Unknown locality'):
            runner.placement(dict(raw(), location_raw=['Свердловская обл., Екатеринбург, пос. Неизвестный\nТверь']), item(), regions)

    def test_interrupted_transfer_preserves_raw_and_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sandbox = root / 'sandbox'; out = root / 'out'; diag = root / 'diag'
            diag.mkdir(); runner.write(sandbox / 'raw/123.json', raw())
            runner.write(sandbox / 'checkpoint.json', {'state': 'CAPTURED', 'completed_ids': ['123']})
            runner.import_sandbox(sandbox, out, diag)
            self.assertEqual(runner.read(out / 'raw/123.json'), raw())
            runner.write(sandbox / 'raw/123.json', dict(raw(), title='Changed'))
            (sandbox/'raw/456.json.tmp').write_text('{unfinished')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                runner.import_sandbox(sandbox, out, diag)
            self.assertEqual(runner.read(out / 'raw/123.json')['title'], 'Работа')

    def test_cli_human_resume_and_duplicate_checks_preserve_demo_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _, _, diag = runner.paths(root, 'demo')
            runner.record_failure(diag, 'filters', 'CDP timeout', 'attach tab', immediate=True)
            with patch.object(runner, 'ROOT', root):
                for arguments in [
                    ['resume-browser', '--human-instruction', 'Continue the same five vacancies', '--authorization-id', 'human-demo'],
                    ['check-stage', '--stage', 'filters'],
                ]:
                    with patch.object(sys, 'argv', ['run.py', arguments[0], '--run-id', 'demo', *arguments[1:]]):
                        runner.main()
                runner.record_failure(diag, 'filters', 'Transient read', 'read')
                for _ in range(2):
                    with patch.object(sys, 'argv', ['run.py','check-stage','--run-id','demo',
                            '--stage','filters','--retry-reason','Open own tab']):
                        runner.main()
                with patch.object(sys, 'argv', ['run.py','stage-ok','--run-id','demo',
                        '--stage','filters','--evidence','Snapshot confirms listing']):
                    runner.main()
                runner.ensure_allowed(diag, 'freeze')
            ledger = runner.read(diag/'failures.json')
            self.assertEqual(ledger['failures'], 2)
            self.assertEqual(ledger['retries_started'], 1)
            self.assertEqual(ledger['state'], 'RECOVERED')

    def test_profession_is_observed_deduplicated_and_not_inferred_from_other_fields(self):
        listing = dict(selected_controls=courier_controls(), heading='Курьер')
        self.assertEqual(runner.profession_from_listing(listing), 'Курьер')
        self.assertEqual(runner.profession_from_listing(listing, '  курьер '), 'Курьер')
        # Whitespace from the DOM is normalized, including nonbreaking spaces.
        truck = dict(label='Водитель\u00a0 грузового транспорта', marker='params[144170]/checkbox/22283175', value=None)
        controls = courier_controls() + [truck]
        self.assertEqual(runner.profession_from_listing({'selected_controls': controls}),
                         'Водитель грузового транспорта; Курьер')
        other = dict(label='Курьер', marker='params[999]/checkbox/1', value='1')
        for controls in [[], [other], [dict(label='Все', marker='params[144170]/all', value='all')]]:
            self.assertEqual(runner.profession_from_listing(dict(selected_controls=controls,
                profession_filter={'observed': True, 'selected_count': 0},
                source_url='https://www.avito.ru/moskva/vakansii/kurer', heading='Работа и вакансии')), '')

    def test_profession_mismatch_and_unreadable_filter_are_rejected(self):
        for listing, requested in [({'selected_controls': courier_controls()}, 'Водитель'),
                ({'selected_controls': []}, 'Курьер'),
                ({'selected_controls': [], 'heading': 'Вакансии по профессии Курьер'}, ''),
                ({'selected_controls': [], 'heading': 'Курьер: работа и вакансии в Москве'}, ''),
                ({'selected_controls': [], 'profession_filter': {'observed': False, 'selected_count': 0}}, ''),
                ({'selected_controls': [], 'profession_filter': {'observed': True, 'selected_count': 1}}, ''),
                ({'selected_controls': [dict(label='', marker='params[144170]/checkbox/1')]}, ''),
                ({}, '')]:
            with self.subTest(listing=listing, requested=requested), self.assertRaises(ValueError):
                runner.profession_from_listing(listing, requested)

    def test_inconsistent_plan_stops_capture_and_finalize_before_browser_or_export(self):
        for profession in ['', 'Водитель']:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); out, service, diag = runner.paths(root, 'bad_profession')
                runner.write(service/'plan.json', {'profession': profession})
                runner.write(out/'listing.json', {'selected_controls': courier_controls()})
                runner.write(out/'queue.json', [item()])
                args = argparse.Namespace(run_id='bad_profession', retry_reason=None)
                with patch.object(runner, 'ROOT', root), patch.object(runner, 'preflight') as probe, \
                        patch.object(runner, 'browser') as browser:
                    for stage in [runner.capture, runner.finalize]:
                        with self.assertRaisesRegex(ValueError, 'Plan profession'):
                            stage(args)
                    probe.assert_not_called(); browser.assert_not_called()
                self.assertFalse((out/'exports').exists())
                self.assertFalse((service/'manifest.json').exists())

    def test_profession_validation_rejects_missing_key_and_changed_listing(self):
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory)/'listing.json'
            runner.write(listing, {'selected_controls': [],
                'profession_filter': {'observed': True, 'selected_count': 0}})
            with self.assertRaisesRegex(ValueError, 'missing'):
                runner.validate_plan_profession({}, listing)
            plan = {'profession': '', 'listing_sha256': runner.sha(listing)}
            runner.validate_plan_profession(plan, listing)
            runner.write(listing, {'selected_controls': [], 'heading': 'Changed',
                'profession_filter': {'observed': True, 'selected_count': 0}})
            with self.assertRaisesRegex(ValueError, 'Frozen listing changed'):
                runner.validate_plan_profession(plan, listing)

    def test_offline_salary_review_preserves_raw_and_rejects_other_lines(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'salary')
            data = raw(); data.update(params_blocks=['Условия\nГрафик: Сменный\nОклад105\u00a0000\u00a0₽', 'Требования\nОпыт: Без опыта'], fields=[{'name':'График','value':'Сменный'}, {'name':'Опыт','value':'Без опыта'}], parameter_audit={'rendered_complete':False,'expansion_controls':[]})
            runner.write(out/'raw/123.json', data); before = (out/'raw/123.json').read_bytes()
            runner.write(out/'queue.json', [item('123'), item('456')]); runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[]))
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            runner.authorize_browser(diag, 'Исправь ошибку вакансии 123 и продолжай', 'auth')
            args = argparse.Namespace(run_id='salary', id='123', authorization_id='auth')
            runner.review_salary_line(args)
            runner.salary_review_policy(args)
            runner.salary_review_policy(args)
            reviewed = runner.reviewed_capture(out, item())
            self.assertEqual(reviewed['fields'][-1]['name'], 'Оклад')
            self.assertEqual(reviewed['fields'][-1]['evidence_raw'], 'Оклад105\u00a0000\u00a0₽')
            self.assertTrue(reviewed['parameter_audit']['rendered_complete'])
            self.assertEqual((out/'raw/123.json').read_bytes(), before)
            self.assertEqual(runner.read(diag/'checkpoint.json')['completed_ids'], ['123'])
            next_data = dict(data, id='456', source_url=item('456')['url'])
            runner.write(out/'raw/456.json', next_data)
            runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='456', completed_ids=['123']))
            runner.review_salary_line(argparse.Namespace(run_id='salary', id='456', authorization_id='auth', automatic=True))
            self.assertEqual(runner.reviewed_capture(out, item('456'))['fields'][-1]['name'], 'Оклад')
            with self.assertRaisesRegex(ValueError, 'Unrecognised'):
                runner.salary_line_review(dict(data, params_blocks=['Условия\nГрафик: Сменный\nОклад105 000 ₽\nНеизвестный элемент', data['params_blocks'][1]]), '123', item()['url'])
            ranged = dict(data, params_blocks=[data['params_blocks'][0].replace('105\u00a0000\u00a0₽', '2\u00a0360 - 2\u00a0950\u00a0₽'), data['params_blocks'][1]])
            self.assertEqual(runner.salary_line_review(ranged, '123', item()['url'])['fields'][-1]['value'], '2\u00a0360 - 2\u00a0950\u00a0₽')
            invalid = dict(data, params_blocks=[data['params_blocks'][0].replace('105\u00a0000\u00a0₽', '2 360 - текст ₽'), data['params_blocks'][1]])
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                runner.salary_line_review(invalid, '123', item()['url'])

    def test_salary_policy_accepts_affirmative_question_context_without_budget_change(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'salary-context')
            data = raw(); data.update(params_blocks=['Условия\nГрафик: Сменный\nОклад2 360 - 2 950 ₽', 'Требования\nОпыт: Без опыта'], fields=[{'name':'График','value':'Сменный'}, {'name':'Опыт','value':'Без опыта'}], parameter_audit={'rendered_complete':False,'expansion_controls':[]})
            runner.write(out/'raw/123.json', data)
            runner.write(out/'queue.json', [item('123'), item('456')])
            runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[]))
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            instruction = 'Контекст: карточка ID 123. Вопрос пользователю: «Разрешаете обработать эту карточку, применять такую же проверку к последующим аналогичным случаям и продолжить?» Ответ пользователя: «да, разрешаю для всех таких случаев».'
            runner.authorize_browser(diag, instruction, 'auth')
            args = argparse.Namespace(run_id='salary-context', id='123', authorization_id='auth')
            runner.review_salary_line(args)
            before = runner.read(diag/'failures.json')
            runner.salary_review_policy(args)
            self.assertEqual(runner.read(diag/'failures.json'), before)
            self.assertEqual(runner.read(out/'salary_review_policy.json')['human_instruction'], instruction)
            self.assertEqual(runner.read(out/'salary_review_policy.json')['method'], 'joined_salary_line_v1')
            (out/'salary_review_policy.json').unlink()
            ledger = runner.read(diag/'failures.json')
            ledger['events'][-1]['instruction'] = instruction.replace('да, разрешаю для всех таких случаев', 'нет, не для всех таких случаев')
            runner.write(diag/'failures.json', ledger)
            with self.assertRaisesRegex(ValueError, 'Authorization'):
                runner.salary_review_policy(args)

    def test_blocked_listing_policy_requires_exact_listing_evidence(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'blocked_listing')
            url = item()['url']; data = dict(state='UNKNOWN_PAGE', heading='Объявление не посмотреть', source_url=url)
            runner.write(out/'raw/123.json', data); before = (out/'raw/123.json').read_bytes()
            runner.write(out/'queue.json', [item('123'), item('456')]); runner.write(service/'plan.json', dict(selection_mode='saved_incomplete_listing'))
            runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[]))
            snapshot = diag/'problem.snapshot.txt'; snapshot.write_text(f'- title: "404" [url={url}]\n- text: "Доступ ограничен."\n')
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            runner.authorize_browser(diag, 'Отмечай последующие вакансии заблокировано после проверки', 'auth')
            runner.blocked_listing_policy(argparse.Namespace(run_id='blocked_listing', authorization_id='auth'))
            args = argparse.Namespace(run_id='blocked_listing', id='123', authorization_id='auth')
            with self.assertRaisesRegex(ValueError, 'evidence'): runner.mark_unavailable(args)
            snapshot.write_text(f'- title: "404" [url={url}]\n- text: "Оно заблокировано после проверки."\n')
            runner.mark_unavailable(args)
            record = runner.unavailable_records(out, runner.read(out/'queue.json'))[0]
            self.assertEqual(record['status'], 'unavailable_blocked_after_review')
            self.assertEqual((out/'raw/123.json').read_bytes(), before)
            self.assertFalse(runner.closed_raw_matches(dict(data, state='NEEDS_HUMAN'), '123', url, runner.BLOCKED_LISTING_MARKER))
            self.assertFalse(runner.closed_raw_matches(dict(data, source_url=item('456')['url']), '123', url, runner.BLOCKED_LISTING_MARKER))

    def test_review_policy_matches_nbsp_and_preserves_closed_policy(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'review')
            url = item()['url']; data = dict(state='UNKNOWN_PAGE', heading='Объявление не\u00a0посмотреть', source_url=url)
            runner.write(out/'raw/123.json', data); before = (out/'raw/123.json').read_bytes()
            runner.write(out/'queue.json', [item('123'), item('456')]); runner.write(service/'plan.json', dict(selection_mode='saved_incomplete_listing'))
            runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[]))
            (diag/'problem.snapshot.txt').write_text(f'- title: "404" [url={url}]\n- text: "Оно ещё на\u00a0проверке."\n')
            runner.record_failure(diag, 'capture', 'closed', 'capture', immediate=True)
            runner.authorize_browser(diag, 'Отмечай все последующие явно закрытые вакансии', 'closed-auth')
            runner.closed_policy(argparse.Namespace(run_id='review', authorization_id='closed-auth'))
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            runner.authorize_browser(diag, 'Отмечай последующие карточки на проверке', 'review-auth')
            runner.review_policy(argparse.Namespace(run_id='review', authorization_id='review-auth'))
            with self.assertRaisesRegex(ValueError, 'outside'):
                runner.mark_unavailable(argparse.Namespace(run_id='review', id='123', authorization_id='closed-auth'))
            runner.mark_unavailable(argparse.Namespace(run_id='review', id='123', authorization_id='review-auth'))
            record = runner.unavailable_records(out, runner.read(out/'queue.json'))[0]
            self.assertEqual(record['status'], 'unavailable_under_review')
            self.assertEqual(record['evidence'], runner.REVIEW_MARKER)
            self.assertEqual((out/'raw/123.json').read_bytes(), before)
            self.assertTrue((out/'closed_policy.json').exists())
            self.assertFalse(runner.closed_raw_matches(dict(data, state='NEEDS_HUMAN'), '123', url, runner.REVIEW_MARKER))

    def test_policy_accepts_explicit_unpublished_detail_only_with_matching_id(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'policy')
            url = item()['url']; data = dict(state='DETAIL', id='123', title='Работа', source_url=url)
            runner.write(out/'raw/123.json', data); runner.write(out/'queue.json', [item('123'), item('456')])
            runner.write(service/'plan.json', dict(selection_mode='saved_incomplete_listing'))
            runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[]))
            (diag/'problem.snapshot.txt').write_text(f'- title: "Вакансия" [url={url}]\n- text: "Объявление снято с публикации."\n')
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            runner.authorize_browser(diag, 'Отмечай все последующие явно закрытые вакансии', 'policy-auth')
            runner.closed_policy(argparse.Namespace(run_id='policy', authorization_id='policy-auth'))
            args = argparse.Namespace(run_id='policy', id='123', authorization_id='policy-auth')
            runner.write(out/'raw/123.json', dict(data, id='999'))
            with self.assertRaisesRegex(ValueError, 'evidence'): runner.mark_unavailable(args)
            runner.write(out/'raw/123.json', data)
            runner.mark_unavailable(args)
            self.assertEqual(runner.unavailable_records(out, runner.read(out/'queue.json'))[0]['evidence'], 'Объявление снято с публикации.')
            self.assertFalse(runner.closed_raw_matches(dict(data, state='NEEDS_HUMAN'), '123', url, 'Объявление снято с публикации.'))

    def test_authorized_closed_item_is_recorded_without_replacing_raw(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'closed')
            url = item()['url']
            data = dict(state='UNKNOWN_PAGE', heading='Объявление не посмотреть', source_url=url, observed_at='2026-10-28T00:00:00Z')
            runner.write(out/'raw/123.json', data); before = (out/'raw/123.json').read_bytes()
            runner.write(out/'queue.json', [item('123'), item('456')])
            runner.write(service/'plan.json', dict(selection_mode='saved_incomplete_listing'))
            runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[]))
            (diag/'problem.snapshot.txt').write_text(f'- title: "Ошибка 404" [url={url}]\n- text: "Объявление закрыто."\n')
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            args = argparse.Namespace(run_id='closed', id='123', authorization_id='auth')
            with self.assertRaises(runner.HumanRequired): runner.mark_unavailable(args)
            runner.authorize_browser(diag, 'Продолжить, исключив закрытое объявление 123', 'auth')
            runner.mark_unavailable(args)
            self.assertEqual((out/'raw/123.json').read_bytes(), before)
            records = runner.unavailable_records(out, runner.read(out/'queue.json'))
            self.assertEqual(records[0]['id'], '123')
            self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'], '456')
            self.assertEqual(runner.read(diag/'checkpoint.json')['completed_ids'], [])
            self.assertEqual(runner.read(diag/'failures.json')['retries_started'], 0)
            runner.write(out/'raw/123.json', dict(data, heading='Changed'))
            with self.assertRaisesRegex(ValueError, 'raw'): runner.unavailable_records(out, runner.read(out/'queue.json'))

    def test_preserve_captcha_observation_requires_authorization_and_retains_raw(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'captcha')
            blocked = dict(state='NEEDS_HUMAN', reason='access_restriction_or_captcha', source_url=item()['url'], observed_at='2026-10-28T03:00:00Z')
            runner.write(out/'raw/123.json', blocked); original = (out/'raw/123.json').read_bytes()
            runner.write(out/'queue.json', [item()])
            runner.write(diag/'checkpoint.json', dict(state='NEEDS_HUMAN', next_id='123', completed_ids=[], stop_reason='access_restriction_or_captcha'))
            runner.record_failure(diag, 'capture', 'captcha', 'capture', immediate=True)
            args = argparse.Namespace(run_id='captcha', authorization_id='human-1')
            with self.assertRaises(runner.HumanRequired): runner.preserve_stopped_observation(args)
            runner.authorize_browser(diag, 'Captcha passed, continue', 'human-1')
            runner.preserve_stopped_observation(args)
            self.assertFalse((out/'raw/123.json').exists())
            archived = list((out/'observations/123').glob('*.json'))
            raw_archive = next(p for p in archived if not p.name.endswith('.provenance.json'))
            self.assertEqual(raw_archive.read_bytes(), original)
            provenance = runner.read(next(p for p in archived if p.name.endswith('.provenance.json')))
            self.assertEqual(provenance['authorization_id'], 'human-1')
            self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'], '123')
            self.assertEqual(runner.read(diag/'failures.json')['retries_started'], 0)

    def test_accept_saved_pages_preserves_request_and_failure_history_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, service, diag = runner.paths(root, 'saved')
            for file in runner.AUTOMATION_FILES:
                path = root/file; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((runner.ROOT/file).read_bytes())
            url = 'https://www.avito.ru/moskva/vakansii'
            next_url = url + '?p=2'
            first = dict(state='LISTING', source_url=url, selected_controls=[], items=[item('123'), item('456')], pagination={'next_url': next_url})
            second = dict(state='LISTING', source_url=next_url, selected_controls=[], items=[item('456'), item('789')], pagination={'next_url': None})
            runner.write(out/'listing_pages/0001.json', first); runner.write(out/'listing_pages/0002.json', second)
            request = dict(url=url, limit=10, candidate_limit=None, geography='Москва', profession='', segments='', filter_evidence='Москва', all_pages=False, profession_scope='all')
            runner.write(diag/'freeze_request.json', request)
            original = (diag/'freeze_request.json').read_bytes()
            runner.record_failure(diag, 'freeze', 'no Next', 'freeze')
            args = argparse.Namespace(run_id='saved', resume=True, url=url, limit=10, geography='Москва', profession='', segments='', filter_evidence='Москва', profession_scope='all', agent_model='gpt-6-luna', retry_reason='Human accepted saved pages', accept_saved=True, human_instruction='Собери эту очередь')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'preflight') as preflight, patch.object(runner, 'browser') as browser, patch.object(runner, 'dirty_automation', return_value=[]), patch.object(runner.subprocess, 'check_output', return_value='abc'):
                runner.freeze(args)
                preflight.assert_not_called(); browser.assert_not_called()
            self.assertEqual([x['id'] for x in runner.read(out/'queue.json')], ['123', '456', '789'])
            plan = runner.read(service/'plan.json')
            self.assertEqual(plan['limit'], 3)
            self.assertEqual(plan['saved_selection_authorization']['original_request'], request)
            self.assertEqual(plan['saved_selection_authorization']['instruction'], 'Собери эту очередь')
            self.assertEqual((diag/'freeze_request.json').read_bytes(), original)
            self.assertEqual(runner.read(diag/'failures.json')['failures'], 1)
            self.assertEqual(runner.read(diag/'failures.json')['retries_started'], 1)
            self.assertEqual(runner.read(out/'listing_pages/0002.json'), second)

    def test_freeze_resumes_even_if_interrupted_before_request_was_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, service, diag = runner.paths(root, 'resume_test')
            diag.mkdir(parents=True); out.mkdir(parents=True)
            for file in runner.AUTOMATION_FILES:
                path = root/file; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((runner.ROOT/file).read_bytes())
            def fake_browser(job, output, diagnostics):
                self.assertTrue((diagnostics/'freeze_request.json').exists())
                runner.write(diagnostics/'listing.json', dict(state='LISTING', source_url=item()['url'],
                    heading='Вакансии', selected_controls=courier_controls(), items=[item()]))
                cp = dict(state='READY', completed_ids=[], next_id=None)
                runner.write(diagnostics/'checkpoint.json', cp)
                return cp, 0
            args = argparse.Namespace(run_id='resume_test', resume=True, url=item()['url'], limit=1,
                geography='Москва', profession='', segments='', filter_evidence='Москва, без фильтра', agent_model='gpt-6-luna')
            with patch.object(runner,'ROOT',root), patch.object(runner,'preflight'), \
                 patch.object(runner,'browser',side_effect=fake_browser), patch.object(runner,'dirty_automation',return_value=[]), \
                 patch.object(runner.subprocess,'check_output',return_value='abc'):
                runner.freeze(args)
            self.assertEqual(runner.read(out/'queue.json')[0]['id'], '123')
            self.assertEqual(runner.read(service/'plan.json')['execution_model'], 'gpt-6-luna')
            self.assertEqual(runner.read(service/'plan.json')['profession'], 'Курьер')

    def test_freeze_rejects_all_scope_with_checked_profession(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, service, diag = runner.paths(root, 'bad_scope')
            def browser(job, output, diagnostics):
                runner.write(diagnostics/'listing.json', dict(state='LISTING', source_url=item()['url'], selected_controls=courier_controls(), items=[item()]))
                return {'state':'READY', 'completed_ids':[]}, 0
            args = argparse.Namespace(run_id='bad_scope', resume=False, url=item()['url'], limit=1, geography='Москва', profession='', segments='', filter_evidence='filter', profession_scope='all', retry_reason=None)
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'preflight'), patch.object(runner, 'browser', side_effect=browser):
                with self.assertRaisesRegex(ValueError, 'All-scope conflicts'):
                    runner.freeze(args)
            self.assertFalse((service/'plan.json').exists())

    def test_finalize_registers_and_builds_once_without_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run_id = 'test_sample'
            out, service, diag = runner.paths(root, run_id)
            runner.write(root / 'Data/dataset/sources.json', {'schema_version': '1', 'exports': []})
            (root / 'Data/README.md').write_text('# Data\n')
            (root / 'Data/dataset/README.md').write_text('Текущий состав: старый\n')
            runner.write(out / 'queue.json', [item()]); runner.write(out / 'raw/123.json', raw())
            runner.write(out / 'listing.json', {'selected_controls': courier_controls()})
            runner.write(service / 'plan.json', {'setup_seconds': 1, 'profession': 'Курьер', 'segments': 'all'})
            runner.write(diag / 'checkpoint.json', {'capture_seconds': 0.5})
            placements = root / 'placements.json'
            runner.write(placements, {'123': {'city': 'Коммунарка', 'region': 'Москва', 'evidence': 'пос. Коммунарка'}})
            args = argparse.Namespace(run_id=run_id, placements=placements, review_model='gpt-6.1-terra')
            with patch.object(runner, 'ROOT', root):
                original_write = runner.write
                def fail_completion(path, data):
                    if path == service/'manifest.json':
                        raise OSError('simulated interruption after registry write')
                    return original_write(path, data)
                with patch.object(runner,'write',side_effect=fail_completion):
                    with self.assertRaisesRegex(OSError, 'simulated interruption'):
                        runner.finalize(args)
                self.assertEqual(len(runner.read(root/'Data/dataset/sources.json')['exports']), 1)
                self.assertFalse((service/'manifest.json').exists())
                runner.finalize(args)
                before = (out / 'exports/vacancies_flat.csv').read_bytes()
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    runner.finalize(args)
            registry = runner.read(root / 'Data/dataset/sources.json')
            self.assertEqual(len(registry['exports']), 1)
            shared, _ = runner.build(root, registry)
            self.assertEqual(shared, (root / 'Data/dataset/vacancies.csv').read_bytes())
            self.assertEqual(before, (out / 'exports/vacancies_flat.csv').read_bytes())
            manifest = runner.read(service / 'manifest.json')
            self.assertEqual(manifest['coverage']['accepted'], 1)
            with (out/'exports/vacancies_flat.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(next(csv.DictReader(stream))['profession'], 'Курьер')
            self.assertEqual(manifest['collection']['profession'], 'Курьер')
            self.assertTrue(all('diagnostics' not in name for name in manifest['files']))

    def test_finalize_applies_audited_profession_without_rewriting_plan_or_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, service, diag = runner.paths(root, 'correction')
            runner.write(root/'Data/dataset/sources.json', {'schema_version': '1', 'exports': []})
            (root/'Data/README.md').write_text('# Data\n'); (root/'Data/dataset/README.md').write_text('Текущий состав: старый\n')
            runner.write(out/'queue.json', [item()]); runner.write(out/'raw/123.json', raw())
            runner.write(out/'listing.json', {'selected_controls': courier_controls()})
            runner.write(service/'plan.json', {'setup_seconds': 1, 'profession': '', 'profession_scope': 'all', 'segments': ''})
            runner.write(diag/'checkpoint.json', {'capture_seconds': 1})
            plan_before = (service/'plan.json').read_bytes(); raw_before = (out/'raw/123.json').read_bytes()
            correction = {'corrected_profession': 'Курьер', 'reason': 'wrong scope'}
            args = argparse.Namespace(run_id='correction', placements=None, review_model='test', profession_from_observed_controls=True, profession_correction_reason='wrong scope')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'validate_partition'), patch.object(runner, 'observed_profession_correction', return_value=correction):
                runner.finalize(args)
            with (out/'exports/vacancies_flat.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(next(csv.DictReader(stream))['profession'], 'Курьер')
            self.assertEqual(runner.read(service/'manifest.json')['profession_correction'], correction)
            self.assertEqual(plan_before, (service/'plan.json').read_bytes())
            self.assertEqual(raw_before, (out/'raw/123.json').read_bytes())
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'validate_partition'), patch.object(runner, 'observed_profession_correction', return_value=correction):
                runner.validate_completed_export(out, service)
                (out/'raw/123.json').write_bytes(raw_before + b' ')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    runner.validate_completed_export(out, service)


if __name__ == '__main__':
    unittest.main()
