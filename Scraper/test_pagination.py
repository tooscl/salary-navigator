"""Pagination contract with browser doubles; never accesses Avito."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from Scraper.test_aside_runner import runner, item, courier_controls


class PaginationTests(unittest.TestCase):
    def test_external_threshold_authorization_preserves_budget_and_rejects_hard_stop(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            diag = runner.paths(runner.ROOT, 'run')[2]
            runner.record_failure(diag, 'freeze', 'Missing tab ID', 'freeze')
            runner.ensure_allowed(diag, 'freeze', 'Use tab ID', action=True)
            runner.record_failure(diag, 'freeze', 'Missing resume', 'freeze')
            before = runner.load_failures(diag)
            args = argparse.Namespace(run_id='run', human_instruction='Continue technical correction', authorization_id='human-1')
            runner.external_threshold_authorization(args)
            runner.external_threshold_authorization(args)
            after = runner.load_failures(diag)
            for field in ('failures', 'retries_started', 'retry_limit', 'budget_start_failures', 'state'):
                self.assertEqual(before[field], after[field])
            self.assertEqual(sum(e['event'] == 'external_threshold_authorization' for e in after['events']), 1)
            runner.ensure_allowed(diag, 'freeze', 'Use --resume and tab ID', action=True)
            runner.record_failure(diag, 'freeze', 'CAPTCHA', 'freeze', immediate=True)
            args.authorization_id = 'human-2'
            with self.assertRaises(runner.HumanRequired):
                runner.external_threshold_authorization(args)

    def test_pending_diagnostic_is_single_use_and_preserves_ledger(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(runner, 'ROOT', root))
            for name in ('validate_partition', 'validate_plan_profession', 'preflight', 'verify_partition_url_guard', 'validate_automation_changes'):
                stack.enter_context(patch.object(runner, name))
            out, service, diag = runner.paths(root, 'run')
            diag.mkdir(parents=True); (out/'raw').mkdir(parents=True)
            runner.write(service/'plan.json', {'automation_sha256': {}})
            runner.write(diag/'checkpoint.json', {'state': 'BROWSER_ERROR', 'stop_reason': 'Error: CDP command timeout: Page.navigate', 'completed_ids': [], 'next_id': '123'})
            runner.write(out/'queue.json', [{'id': '123', 'url': 'https://www.avito.ru/moskva/vakansii/x_123'}])
            runner.write(out/'listing.json', {})
            runner.write(diag/'sandbox.json', {'path': '/old/sandbox'})
            runner.record_failure(diag, 'capture', 'timeout', 'capture', immediate=True)
            original = runner.load_failures(diag)
            args = argparse.Namespace(run_id='run', id='123', human_instruction='Diagnostic 123', authorization_id='human-1', connection_evidence='Probe succeeded', tab_id='new-tab', config=None)
            def fake_browser(job, out, diag):
                self.assertEqual([x['id'] for x in job['queue']], ['123'])
                self.assertEqual(job['tab_id'], 'new-tab')
                result = {'state': 'CAPTURED', 'completed_ids': ['123']}
                runner.write(diag/'checkpoint.json', result)
                return result, 1
            with patch.object(runner, 'reviewed_capture', return_value={'parameter_audit': {'rendered_complete': True}}), patch.object(runner, 'check', return_value={'structural_check_passed': True}), patch.object(runner, 'browser', side_effect=fake_browser) as browser:
                runner.diagnostic_pending_capture(args)
                with self.assertRaises((ValueError, runner.HumanRequired)):
                    runner.diagnostic_pending_capture(args)
                self.assertEqual(browser.call_count, 1)
            ledger = runner.load_failures(diag)
            for field in ('state', 'failures', 'retries_started', 'budget_start_failures'):
                self.assertEqual(ledger[field], original[field])
            self.assertEqual(runner.read(diag/'diagnostic-original'/'sandbox.json')['path'], '/old/sandbox')
            with self.assertRaises(runner.HumanRequired):
                runner.ensure_allowed(diag, 'capture', 'No authorization')

    def simulate(self, count=5000, all_pages=False, interrupt=None, scope=None, cycle=False, blocked=False, requested=None, repeat_items=False, confirmed_end=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for file in runner.AUTOMATION_FILES:
                target = root/file; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((runner.ROOT/file).read_bytes())
            first = 'https://www.avito.ru/moskva/vakansii?f=observed&p=1&localPriority=0'
            calls = []; failed = False
            def browser(job, out, diag):
                nonlocal failed
                page = int(job['url'].split('p=')[1].split('&')[0]); calls.append(page)
                if page == interrupt and not failed:
                    failed = True
                    raise RuntimeError('interrupted')
                url = job['url'].replace('&localPriority=0', '')
                effective_page = page - 1 if repeat_items and page > 2 else page
                start = (effective_page-1)*100+1
                # Ads repeated across pages must keep their first listing rank.
                items = [item(str(i)) for i in range(max(1, start-5), min(start+100, count+1))]
                next_url = f'https://www.avito.ru/moskva/vakansii?f=observed&p={page+1}' if start+100 <= count else None
                if cycle: next_url = first
                if repeat_items and page == 2:
                    items = [item(str(i)) for i in range(1, 101)]
                listing = dict(state='LISTING', source_url=url, heading='Вакансии',
                    selected_controls=courier_controls() if page == 1 and scope != 'all' else [],
                    profession_filter=dict(observed=True, selected_count=0), items=items, pagination=dict(next_url=next_url, end_confirmed=confirmed_end and next_url is None))
                if blocked and page == 2:
                    cp = dict(state='NEEDS_HUMAN', completed_ids=[])
                else:
                    cp = dict(state='READY', completed_ids=[])
                runner.write(diag/'listing.json', listing); runner.write(diag/'checkpoint.json', cp)
                return cp, 0
            args = argparse.Namespace(run_id='pagination', resume=False, url=first,
                limit=None if all_pages else (requested or count), all_pages=all_pages, geography='Москва',
                profession='', profession_scope=scope, segments='', filter_evidence='Applied UI selection',
                agent_model='gpt-6-luna', candidate_limit=None)
            stack.enter_context(patch.object(runner, 'ROOT', root))
            stack.enter_context(patch.object(runner, 'preflight'))
            stack.enter_context(patch.object(runner, 'browser', side_effect=browser))
            stack.enter_context(patch.object(runner, 'dirty_automation', return_value=[]))
            stack.enter_context(patch.object(runner.subprocess, 'check_output', return_value='abc'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if blocked:
                with self.assertRaisesRegex(ValueError, 'NEEDS_HUMAN'):
                    runner.freeze(args)
                self.assertEqual(calls, [1, 2])
                out, service, _ = runner.paths(root, args.run_id)
                self.assertFalse((service/'plan.json').exists())
                self.assertTrue((out/'listing_pages/0001.json').exists())
                return
            if interrupt:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    runner.freeze(args)
                args.resume = True
            runner.freeze(args)
            out, service, _ = runner.paths(root, args.run_id)
            plan = runner.read(service/'plan.json'); queue = runner.read(out/'queue.json')
            available = 100 if cycle else count
            self.assertEqual([q['id'] for q in queue], [str(i) for i in range(1, available+1)])
            self.assertEqual([q['listing_rank'] for q in queue], list(range(1, available+1)))
            self.assertEqual(plan['limit'], available)
            self.assertEqual(plan['requested_limit'], None if all_pages else (requested or count))
            self.assertEqual(plan['available_unique'], available)
            if requested or all_pages or cycle or repeat_items:
                self.assertEqual(plan['available_listing_exhausted'], confirmed_end and not cycle)
                self.assertEqual(plan['listing_end_reason'], 'repeated_next_link' if cycle else 'confirmed_end' if confirmed_end else 'no_next_link')
            self.assertFalse((runner.paths(root, args.run_id)[2]/'failures.json').exists())
            self.assertFalse((runner.paths(root, args.run_id)[2]/'pending_listing.json').exists())
            if repeat_items:
                self.assertEqual(len(plan['listing_pages']), 5)
                self.assertTrue((out/plan['listing_pages'][-1]['file']).is_file())
            runner.validate_plan_profession(plan, out/'listing.json')
            if scope == 'top20':
                self.assertEqual(len(plan['profession'].split('; ')), 20)
            self.assertEqual(calls.count(1), 1)
            self.assertEqual(len(calls), 1 if cycle else (count+99)//100 + bool(interrupt) + bool(repeat_items))
            if len(plan['listing_pages']) > 1:
                path = out / plan['listing_pages'][1]['file']
                runner.write(path, dict(runner.read(path), heading='tampered'))
                with self.assertRaisesRegex(ValueError, 'page changed'):
                    runner.validate_plan_profession(plan, out/'listing.json')

    def test_five_thousand_without_per_page_filter_checks(self):
        self.simulate(scope='top20')

    def test_all_pages_stops_at_end(self):
        self.simulate(count=251, all_pages=True, scope='all', confirmed_end=True)

    def test_resume_uses_saved_pages(self):
        self.simulate(count=301, interrupt=3)

    def test_next_cycle_and_access_block_preserve_progress(self):
        self.simulate(count=301, cycle=True)
        self.simulate(count=301, blocked=True)

    def test_requested_ten_thousand_ends_normally_on_page_thirty(self):
        self.simulate(count=2920, requested=10000, scope='all')

    def test_requested_ten_thousand_ends_normally_before_page_thirty(self):
        self.simulate(count=251, requested=10000, scope='all')

    def test_page_without_new_ids_continues_to_unvisited_next(self):
        self.simulate(count=301, all_pages=True, repeat_items=True)

    def test_continuation_preserves_parent_and_excludes_prior_ids(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for file in runner.AUTOMATION_FILES:
                target = root/file; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((runner.ROOT/file).read_bytes())
            stack.enter_context(patch.object(runner, 'ROOT', root))
            stack.enter_context(patch.object(runner, 'preflight'))
            stack.enter_context(patch.object(runner, 'dirty_automation', return_value=[]))
            stack.enter_context(patch.object(runner.subprocess, 'check_output', return_value='abc'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            url = 'https://www.avito.ru/moskva/vakansii?p=1'
            next_url = 'https://www.avito.ru/moskva/vakansii?p=2'
            page = dict(state='LISTING', source_url=url, heading='Вакансии', selected_controls=[],
                items=[item('101')], pagination=dict(next_url=next_url, end_confirmed=False))
            out, service, diag = runner.paths(root, 'parent')
            runner.write(out/'listing_pages/0001.json', page)
            runner.write(out/'listing.json', page)
            queue = runner.queue_from_pages([page], None)
            runner.write(out/'queue.json', queue)
            plan = dict(run_id='parent', profession='', profession_scope='all', requested_all_pages=True,
                listing_sha256=runner.sha(out/'listing.json'), listing_pages=[dict(file='listing_pages/0001.json',
                sha256=runner.sha(out/'listing_pages/0001.json'), source_url=url)])
            runner.write(service/'plan.json', plan)
            request = dict(url=url, limit=None, candidate_limit=None, geography='Москва', profession='', segments='',
                filter_evidence='Москва', all_pages=True, profession_scope='all')
            runner.write(diag/'freeze_request.json', request)
            runner.write(diag/'aside-target.local.json', {'TL_ASIDE_PROFILE': 'pinned'})
            before = (service/'plan.json').read_bytes(), (out/'queue.json').read_bytes()
            page_file = out/'listing_pages/0001.json'
            original_page = page_file.read_bytes()
            runner.write(page_file, dict(page, heading='tampered'))
            with self.assertRaisesRegex(ValueError, 'page changed'):
                runner.continuation(argparse.Namespace(run_id='rejected', parent_run_id='parent'))
            page_file.write_bytes(original_page)
            with patch('sys.argv', ['run.py', 'continuation', '--run-id', 'tail', '--parent-run-id', 'parent']):
                runner.main()
            self.assertEqual(runner.read(runner.paths(root, 'tail')[2]/'aside-target.local.json'), {'TL_ASIDE_PROFILE': 'pinned'})
            def browser(job, out, diag):
                self.assertEqual(job['url'], next_url)
                new_page = dict(page, source_url=next_url, items=[item('101'), item('102')],
                    pagination=dict(next_url=None, end_confirmed=True))
                runner.write(diag/'listing.json', new_page)
                runner.write(diag/'checkpoint.json', dict(state='READY', completed_ids=[]))
                return dict(state='READY', completed_ids=[]), 0
            stack.enter_context(patch.object(runner, 'browser', side_effect=browser))
            args = argparse.Namespace(run_id='tail', resume=True, agent_model='gpt-6-luna', **request)
            runner.freeze(args)
            tail_out, tail_service, _ = runner.paths(root, 'tail')
            self.assertEqual([q['id'] for q in runner.read(tail_out/'queue.json')], ['102'])
            self.assertEqual(runner.read(tail_out/'queue.json')[0]['listing_rank'], 2)
            runner.validate_partition(tail_out, tail_service)
            self.assertEqual(before, ((service/'plan.json').read_bytes(), (out/'queue.json').read_bytes()))
            tail_diag = runner.paths(root, 'tail')[2]
            runner.write(tail_diag/'shared-ledger.json', {'parent_run_id': 'other'})
            with self.assertRaisesRegex(ValueError, 'ledger pointer changed'):
                runner.validate_partition(tail_out, tail_service)
            runner.write(tail_diag/'shared-ledger.json', {'parent_run_id': 'parent'})
            runner.write(tail_out/'continuation-provenance/parent-queue.json', [])
            with self.assertRaisesRegex(ValueError, 'provenance changed'):
                runner.validate_partition(tail_out, tail_service)

    def test_ui_transition_requires_landed_page_and_filters(self):
        first_url = 'https://www.avito.ru/moskva/vakansii/prodavec?context=one'
        href = 'https://www.avito.ru/moskva/vakansii?context=one&p=2'
        landed = 'https://www.avito.ru/moskva/vakansii/prodavec?context=two&p=2'
        first = dict(source_url=first_url, heading='Продавец', selected_controls=courier_controls(),
                     pagination=dict(next_url=href, current_pages=['1']))
        page = dict(source_url=landed, heading='Продавец', selected_controls=courier_controls(),
                    pagination=dict(current_pages=['2']), ui_transition=dict(method='observed_ui_next', recovered=False, next_snapshot='Next', from_page=1, landed_page=2, from_url=first_url, next_href=href, landed_url=landed))
        runner.validate_ui_transition(first, first, page)
        for change in (dict(source_url=href), dict(heading='Все вакансии'), dict(selected_controls=[]),
                       dict(pagination=dict(current_pages=['1'])), dict(ui_transition={})):
            with self.assertRaises(ValueError): runner.validate_ui_transition(first, first, dict(page, **change))

    def test_ui_next_click_and_recovery_with_browser_double(self):
        script = r"""
const nodefs = require('fs');
const source = nodefs.readFileSync(process.argv[1], 'utf8');
const url1='https://www.avito.ru/moskva/vakansii/prodavec';
const href='https://www.avito.ru/moskva/vakansii?p=2';
const url2=url1+'?p=2';
const controls=[{marker:'params[144170]/checkbox/1',value:'1',label:'Продавец'}];
const make=(url,n)=>({state:'LISTING',source_url:url,observed_at:String(Math.random()),heading:'Продавец',selected_controls:controls,items:[{id:'1'}],listing_item_audit:{malformed:0},pagination:{current_pages:[String(n)],next_url:href,evidence:{href:'/moskva/vakansii?p=2'}}});
const first=make(url1,1);
(async()=>{for(const scenario of ['click','recovery','delayed','broad','href_changed']) {
 const recovering=scenario==='recovery'; let current=recovering ? make(url2,2) : first, clicks=0, polls=0; const writes={};
 if(scenario==='href_changed') current={...first,pagination:{...first.pagination,next_url:href+'&changed=1',evidence:{href:'/moskva/vakansii?p=2&changed=1'}}};
 global.pwd='sandbox'; global.fs={mkdir:async()=>{},writeFile:async(n,v)=>writes[n]=v,rename:async(a,b)=>writes[b]=writes[a],readFile:async(n)=>writes[n]};
 global.attachBrowserTab=async()=>({url:()=>current.source_url,evaluate:async()=>{ if(clicks && scenario==='delayed' && ++polls===5) current=make(url2,2); return current; },locator:()=>({click:async()=>{clicks++; if(scenario==='broad') current=make(href,2); else if(scenario!=='delayed') current=make(url2,2)}})});
 global.snapshot=async(_page, options)=>({tree:options.selector==='[aria-label="Следующая страница"]' ? '- link "Следующая страница" [ref=e7]' : '- truncated listing'});
 global.sleep=async()=>{}; global.tlCollectorPage=null;
 const job={mode:'freeze',url:href,completed_ids:[],tab_id:'tab',ui_next:true,previous:first,first};
 await eval('(async()=>{'+source.replace('__JOB__',JSON.stringify(job)).replace('__EXTRACTOR__',JSON.stringify('extract'))+'})()');
 const cp=JSON.parse(writes['checkpoint.json']);
 if(scenario==='broad') { if(cp.state!=='BROWSER_ERROR' || writes['listing.json']) process.exit(3); continue; }
 if(scenario==='href_changed') { const d=JSON.parse(writes['next-comparison.json']); if(cp.state!=='BROWSER_ERROR'||clicks||!d.ref_found||d.current_next===d.expected_next) process.exit(4); continue; }
 const result=JSON.parse(writes['listing.json']);
 if(clicks!==Number(!recovering)||result.source_url!==url2||result.ui_transition.next_href!==href||result.ui_transition.recovered!==recovering||cp.state!=='READY'||(scenario==='delayed' && polls<5)) process.exit(1);
}})().catch(()=>process.exit(2));
"""
        subprocess.run(['node', '-e', script, str(runner.SCRIPTS/'browser_batch.js')], check=True, capture_output=True)

    def test_archive_uncertain_pages_keeps_bytes_and_ledger(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), patch.object(runner, 'AUTOMATION_FILES', []), patch.object(runner.subprocess, 'check_output', return_value='abc'):
            out, service, diag = runner.paths(runner.ROOT, 'run')
            page = out/'listing_pages/0001.json'; page.parent.mkdir(parents=True)
            page.write_bytes(b'{"uncertain": true}\n')
            runner.write(diag/'failures.json', {'state':'RECOVERED', 'stages':{'freeze':{'state':'RETRY_AVAILABLE'}}, 'events':[]})
            original = (diag/'failures.json').read_bytes()
            runner.archive_uncertain_freeze(argparse.Namespace(run_id='run', reason='mixed observation'))
            archive = service/'uncertain-freeze'
            self.assertEqual((archive/'data/listing_pages/0001.json').read_bytes(), b'{"uncertain": true}\n')
            self.assertEqual((diag/'failures.json').read_bytes(), original)
            self.assertEqual(runner.read(archive/'provenance.json')['files']['data/listing_pages/0001.json'], runner.sha(archive/'data/listing_pages/0001.json'))
            self.assertFalse(page.exists())

    def test_extractor_observes_next_and_end(self):
        script = r'''
global.location={href:'https://www.avito.ru/moskva/vakansii?f=actual',origin:'https://www.avito.ru'};
global.getComputedStyle=()=>({visibility:'visible'});
let disabled=false;
const next={innerText:'',offsetWidth:1,getClientRects:()=>[1],hasAttribute:(key)=>key==='disabled'&&disabled,
 getAttribute:(key)=>({rel:'next',href:'?f=actual&p=2','aria-label':'Следующая страница'}[key]||null)};
const a={href:'https://www.avito.ru/moskva/vakansii/test_123',pathname:'/moskva/vakansii/test_123',origin:'https://www.avito.ru',innerText:'Курьер'};
const card={innerText:'Курьер',querySelector:(s)=>s==='[data-marker="item-title"]'?a:null};
let showNext=true;
global.document={readyState:'complete',body:{innerText:'Вакансии'},querySelector:()=>null,
 querySelectorAll:(s)=>s==='[data-marker="item"]'?[card]:s==='a,[role="link"],button'&&showNext?[next]:[]};
const src=require('fs').readFileSync(process.argv[1],'utf8');
let result=eval(src);
if(result.pagination.next_url!=='https://www.avito.ru/moskva/vakansii?f=actual&p=2'||result.pagination.end_confirmed)process.exit(1);
showNext=false;result=eval(src);
if(result.pagination.next_url||result.pagination.end_confirmed)process.exit(2);
showNext=true;disabled=true;result=eval(src);
if(result.pagination.next_url||!result.pagination.end_confirmed)process.exit(3);
card.querySelector=()=>null;result=eval(src);
if(result.items.length||result.listing_item_audit.malformed!==1)process.exit(4);
'''
        subprocess.run(['node','-e',script,str(runner.ROOT/'Scraper/extract.js')],check=True)
