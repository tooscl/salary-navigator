"""Offline collector regressions; no network, Aside or GitHub required."""
import argparse
import csv
import hashlib
import sys
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import shutil
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
    def test_finalized_provenance_redaction_preserves_raw_and_audits(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            root=runner.ROOT;out,service,diag=runner.paths(root,'run');data=b'{"state":"NEEDS_HUMAN"}\n';digest=hashlib.sha256(data).hexdigest()
            raw=out/'observations/123'/f'{digest}.json';raw.parent.mkdir(parents=True);raw.write_bytes(data)
            provenance=raw.with_name(digest+'.provenance.json');runner.write(provenance,{'raw_sha256':digest,'sandbox_path':'/Users/private/.aside/session','reason':'restriction'})
            csvfile=out/'exports/vacancies_flat.csv';csvfile.parent.mkdir();csvfile.write_text('id\n123\n');queue=out/'queue.json';runner.write(queue,[{'id':'123'}])
            entries={p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':runner.sha(p)} for p in (raw,provenance,csvfile,queue)}
            manifest={'status':'complete','files':entries,'coverage':{'accepted':1}};runner.write(service/'manifest.json',manifest)
            before={p:p.read_bytes() for p in (raw,provenance,csvfile,queue,service/'manifest.json')}
            original_write=runner.write
            def fail_audit(path,value):
                if path == out/'exports/provenance_redaction_audit.json': raise OSError('simulated audit failure')
                return original_write(path,value)
            with patch.object(runner,'validate_completed_export',return_value=manifest),patch.object(runner,'write',side_effect=fail_audit):
                with self.assertRaises(OSError):runner.redact_finalized_provenance(argparse.Namespace(run_id='run'))
            for p in before:self.assertEqual(p.read_bytes(),before[p])
            with patch.object(runner,'validate_completed_export',return_value=manifest):
                runner.redact_finalized_provenance(argparse.Namespace(run_id='run'))
            after=runner.read(service/'manifest.json');audit=runner.read(out/'exports/provenance_redaction_audit.json')
            self.assertEqual(audit['original_manifest_sha256'],hashlib.sha256(before[service/'manifest.json']).hexdigest())
            private=diag/'private-provenance/123'/provenance.name;self.assertEqual(private.read_bytes(),before[provenance])
            self.assertEqual((diag/'private-provenance/original-manifest.json').read_bytes(),before[service/'manifest.json'])
            self.assertNotIn('sandbox_path',runner.read(provenance));self.assertEqual(runner.read(provenance)['private_provenance_sha256'],hashlib.sha256(before[provenance]).hexdigest())
            for p in (raw,csvfile,queue):self.assertEqual(p.read_bytes(),before[p])
            runner.validate_provenance_redaction(out,service,after)
            with patch.object(runner,'validate_completed_export',return_value=after):runner.redact_finalized_provenance(argparse.Namespace(run_id='run'))
            private.write_text('{}')
            with self.assertRaisesRegex(ValueError,'bytes differ'):runner.validate_provenance_redaction(out,service,after)

    def test_finalized_provenance_redaction_cli_dispatch(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            with patch('sys.argv',['run.py','redact-finalized-provenance','--run-id','run']),patch.object(runner,'redact_finalized_provenance') as action:
                runner.main();self.assertEqual(action.call_args.args[0].run_id,'run')

    def test_staged_privacy_guard_rejects_private_run_locator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);subprocess.run(['git','init','-q'],cwd=root,check=True)
            out,service,_=runner.paths(root,'run');path=out/'exports/info.json';runner.write(path,{'public':'ok'})
            subprocess.run(['git','add','--',str(out.relative_to(root))],cwd=root,check=True)
            runner.verify_staged_run_privacy(root,out,service)
            runner.write(path,{'sandbox_path':'/Users/private/.aside/session'});subprocess.run(['git','add','--',str(out.relative_to(root))],cwd=root,check=True)
            with self.assertRaisesRegex(ValueError,'private locator'):runner.verify_staged_run_privacy(root,out,service)

    def test_address_expansion_label_does_not_match_blagoveshchensk(self):
        source=(runner.ROOT/'Scraper/extract.js').read_text(encoding='utf8')
        import re
        expression=re.search(r'filter\(s => (/.+?/iu)\.test\(s\)\)',source)
        self.assertIsNotNone(expression)
        code="const f="+expression.group(1)+"; if(f.test('Амурская обл., Благовещенск, пл. Ленина') || !f.test('Показать ещё') || !f.test('Ещё') || !f.test('Раскрыть параметры')) process.exit(1);"
        subprocess.run(['node','-e',code],check=True)
        before=source.replace(expression.group(1),'/ещ[её]|показать|раскрыть/i')
        self.assertTrue(runner.audited_address_label_fix({'extractor_sha256':hashlib.sha256(before.encode()).hexdigest()}))
        self.assertFalse(runner.audited_address_label_fix({'extractor_sha256':'unrelated'}))

    def test_direct_address_button_review_preserves_raw_and_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            root=runner.ROOT;out,service,diag=runner.paths(root,'case');sandbox=root/'sandbox'; queue=[item('123'),item('456')]; url=queue[0]['url']
            runner.write(out/'queue.json',queue);runner.write(out/'listing.json',{});runner.write(out/'listing_pages/0001.json',{})
            runner.write(service/'plan.json',{'run_id':'case','listing_pages':[{'file':'listing_pages/0001.json'}]})
            cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='123',completed_ids=[])
            runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)});runner.write(sandbox/'browser-tab.json',{'url':url})
            address='Амурская обл., Благовещенск, пл. Ленина';blocks=[f'Условия\nГде предстоит работать: {address}\nГрафик: Вахта','Требования\nОпыт работы: Без опыта']
            data=raw('123');data.update(params_blocks=blocks,fields=[dict(name='Где предстоит работать',value=address),dict(name='График',value='Вахта'),dict(name='Опыт работы',value='Без опыта')],parameter_audit=dict(rendered_complete=False,expansion_controls=[address]),source_sections=[dict(marker='item-view/item-params',text=x) for x in blocks])
            runner.write(out/'raw/123.json',data);runner.write(sandbox/'raw/123.json',data)
            snap=f'- title: "Работа" [url={url}]\n- heading "Работа" [level=1]\n- text: "Где предстоит работать:"\n- button "{address}" [ref=e1]\n';(diag/'problem.snapshot.txt').write_text(snap);(sandbox/'problem.snapshot.txt').write_text(snap)
            ledger=runner.load_failures(diag);ledger.update(unbounded_retries=True);ledger['events'].append(dict(event='human_browser_authorization',authorization_id='auth',instruction='Continue technical fixes',unbounded_retries=True));runner.save_failures(diag,ledger)
            runner.record_failure(diag,'capture','stopped','capture',immediate=True);before=runner.load_failures(diag);original=(out/'raw/123.json').read_bytes()
            with patch.object(runner,'validate_partition'),patch.object(runner,'validate_plan_profession'),patch.object(runner,'queue_from_pages',return_value=queue):
                runner.review_address_button(argparse.Namespace(run_id='case',id='123',source='auth'))
            after=runner.load_failures(diag);self.assertEqual(after['state'],'RECOVERED')
            self.assertEqual((after['failures'],after['retries_started']),(before['failures'],before['retries_started']))
            self.assertEqual((out/'raw/123.json').read_bytes(),original);self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'],'456')
    def test_public_provenance_retains_private_bytes_and_raw_hash(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            out,service,diag=runner.paths(runner.ROOT,'private');runner.write(diag/'checkpoint.json',dict(state='CAPTURED',next_id=None));raw=b'{"state":"UNKNOWN_PAGE"}\n';digest=hashlib.sha256(raw).hexdigest();path=out/'observations/123'/f'{digest}.provenance.json';path.parent.mkdir(parents=True);(path.parent/f'{digest}.json').write_bytes(raw)
            runner.write(path,dict(raw_sha256=digest,sandbox_path='/Users/person/.aside/session',snapshot_sha256='abc'));original=path.read_bytes();args=argparse.Namespace(run_id='private');runner.public_provenance(args)
            private=diag/'private-provenance/123'/path.name;self.assertEqual(private.read_bytes(),original);public=runner.read(path);self.assertNotIn('sandbox_path',public);self.assertEqual(public['private_provenance_sha256'],hashlib.sha256(original).hexdigest());self.assertEqual((path.parent/f'{digest}.json').read_bytes(),raw)
            before=path.read_bytes();runner.public_provenance(args);self.assertEqual(path.read_bytes(),before)
            other=out/'observations/456'/f'{digest}.provenance.json';other.parent.mkdir(parents=True);(other.parent/f'{digest}.json').write_bytes(raw);runner.write(other,dict(raw_sha256=digest,sandbox_path='/private',other='/Users/private'))
            with self.assertRaises(ValueError): runner.public_provenance(args)
            self.assertIn('sandbox_path',runner.read(other))

    def test_joined_salary_recurring_review_and_continuation_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();scripts=root/'Scraper/skills/collect-avito/scripts';shutil.copytree(SCRIPTS,scripts);shutil.copytree(runner.ROOT/'utils',root/'utils',ignore=shutil.ignore_patterns('__pycache__'))
            out,service,diag=runner.paths(root,'salary');sandbox=root/'sandbox';queue=[item('123'),item('456')];listing=dict(state='LISTING',selected_controls=courier_controls(),items=queue)
            runner.write(out/'listing.json',listing);runner.write(out/'listing_pages/0001.json',listing);runner.write(out/'queue.json',runner.queue_from_pages([listing],2));runner.write(service/'plan.json',dict(run_id='salary',profession='Курьер',listing_sha256=runner.sha(out/'listing.json'),listing_pages=[dict(file='listing_pages/0001.json',sha256=runner.sha(out/'listing_pages/0001.json'))]))
            cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='123',completed_ids=[]);runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)});runner.write(sandbox/'browser-tab.json',{'url':queue[0]['url']})
            blocks=['Условия\nГрафик: Сменный\nОклад83 000 - 100 000 ₽','Требования\nОпыт работы: Без опыта'];data=raw('123');data.update(params_blocks=blocks,fields=[dict(name='График',value='Сменный'),dict(name='Опыт работы',value='Без опыта')],parameter_audit=dict(rendered_complete=False,expansion_controls=[]),source_sections=[dict(marker='item-view/item-params',text=x) for x in blocks]);runner.write(out/'raw/123.json',data);runner.write(sandbox/'raw/123.json',data)
            snap=f'- title: "Работа" [url={queue[0]["url"]}]\n- heading "Работа" [level=1]\n- text: "Оклад"\n- text: "83 000 - 100 000 ₽"\n';(diag/'problem.snapshot.txt').write_text(snap);(sandbox/'problem.snapshot.txt').write_text(snap)
            runner.record_failure(diag,'capture','stopped','capture',immediate=True);ledger=runner.load_failures(diag);ledger['retries_started']=5;ledger['events'].extend([dict(event='recurring_ip_policy',source='s',interval_seconds=1800),dict(event='recurring_incomplete_policy',source='s',instruction='разрешаю запросы')]);runner.save_failures(diag,ledger)
            base=[sys.executable,str(scripts/'run.py')];review=base+['review-salary-line','--run-id','salary','--id','123','--authorization-id','s','--source','s']
            (sandbox/'problem.snapshot.txt').write_text('tampered');self.assertNotEqual(subprocess.run(review,cwd=root,capture_output=True).returncode,0);(sandbox/'problem.snapshot.txt').write_text(snap)
            bad=dict(data);bad['params_blocks']=[blocks[0]+'\nНеизвестная строка',blocks[1]];runner.write(out/'raw/123.json',bad);self.assertNotEqual(subprocess.run(review,cwd=root,capture_output=True).returncode,0);runner.write(out/'raw/123.json',data)
            result=subprocess.run(review,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(runner.read(out/'raw/123.json'),data);self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'],'456')
            cont=base+['recurring-reviewed-continuation','--run-id','salary','--source','s','--id','456','--authorization-id','once'];self.assertNotEqual(subprocess.run(cont,cwd=root,capture_output=True).returncode,0)
            ledger=runner.load_failures(diag);[e.update(at='2020-01-01T00:00:00+00:00') for e in ledger['events'] if e.get('event')=='failure'];runner.save_failures(diag,ledger);result=subprocess.run(cont,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr)

    def test_visible_502_actual_cli_and_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();scripts=root/'Scraper/skills/collect-avito/scripts';shutil.copytree(SCRIPTS,scripts);shutil.copytree(runner.ROOT/'utils',root/'utils',ignore=shutil.ignore_patterns('__pycache__'))
            out,service,diag=runner.paths(root,'err');sandbox=root/'sandbox';candidate=item('456');url=candidate['url'];listing=dict(state='LISTING',selected_controls=courier_controls(),items=[candidate])
            runner.write(out/'listing.json',listing);runner.write(out/'listing_pages/0001.json',listing);runner.write(out/'queue.json',runner.queue_from_pages([listing],1));runner.write(service/'plan.json',dict(run_id='err',profession='Курьер',listing_sha256=runner.sha(out/'listing.json'),listing_pages=[dict(file='listing_pages/0001.json',sha256=runner.sha(out/'listing_pages/0001.json'))]))
            cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='456',completed_ids=[]);runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)});runner.write(sandbox/'browser-tab.json',{'url':url})
            rawdata=dict(state='UNKNOWN_PAGE',source_url=url,heading=None,items=[],unresolved_evidence=dict(complete=True,restriction=False,known_unavailable=False));runner.write(out/'raw/456.json',rawdata);runner.write(sandbox/'raw/456.json',rawdata)
            snapshot=f'- title: "HTTP 502" [url={url}]\n- text: "502 Error\\n Bad gateway\\n Сайт www.avito.ru не отвечает."\n';(diag/'problem.snapshot.txt').write_text(snapshot);(sandbox/'problem.snapshot.txt').write_text(snapshot)
            runner.record_failure(diag,'capture','stopped','capture',immediate=True);ledger=runner.load_failures(diag);ledger['retries_started']=5;ledger['events'][-1]['at']='2020-01-01T00:00:00+00:00';ledger['events'].extend([dict(event='recurring_ip_policy',source='s',interval_seconds=1800),dict(event='recurring_incomplete_policy',source='s',instruction='разрешаю запросы')]);runner.save_failures(diag,ledger)
            base=[sys.executable,str(scripts/'run.py')];cmd=base+['recurring-incomplete-attempt','--run-id','err','--source','s','--id','456','--authorization-id','once','--visible-http-502']
            for bad in (snapshot+'CAPTCHA',snapshot+'HTTP 429',snapshot.replace('www.avito.ru не отвечает','wrong.example не отвечает'),snapshot.replace('HTTP 502','HTTP 503')):
                (diag/'problem.snapshot.txt').write_text(bad);(sandbox/'problem.snapshot.txt').write_text(bad);self.assertNotEqual(subprocess.run(cmd,cwd=root,capture_output=True).returncode,0)
            (diag/'problem.snapshot.txt').write_text(snapshot);(sandbox/'problem.snapshot.txt').write_text(snapshot)
            wrong=cmd.copy();wrong[wrong.index('456')]='999';self.assertNotEqual(subprocess.run(wrong,cwd=root,capture_output=True).returncode,0)
            recent=runner.load_failures(diag);recent['events'][0]['at']=runner.datetime.now(runner.timezone.utc).isoformat();runner.save_failures(diag,recent);self.assertNotEqual(subprocess.run(cmd,cwd=root,capture_output=True).returncode,0);recent['events'][0]['at']='2020-01-01T00:00:00+00:00';runner.save_failures(diag,recent)
            result=subprocess.run(cmd,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr)
            preserve=base+['retry-incomplete-observation','--run-id','err','--id','456','--authorization-id','once'];result=subprocess.run(preserve,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(runner.read(sandbox/'raw/456.json'),rawdata);self.assertFalse((out/'raw/456.json').exists());self.assertTrue(next((out/'observations/456').glob('*.provenance.json')).read_text().find('visible_http_502')>=0)

    def test_address_button_offline_review_and_explicit_target_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();scripts=root/'Scraper/skills/collect-avito/scripts';shutil.copytree(SCRIPTS,scripts);shutil.copytree(runner.ROOT/'utils',root/'utils',ignore=shutil.ignore_patterns('__pycache__'))
            out,service,diag=runner.paths(root,'addr');sandbox=root/'sandbox';queue=[item('123'),item('456')];listing=dict(state='LISTING',selected_controls=courier_controls(),items=queue)
            runner.write(out/'listing.json',listing);runner.write(out/'listing_pages/0001.json',listing);runner.write(out/'queue.json',runner.queue_from_pages([listing],2));runner.write(service/'plan.json',dict(run_id='addr',profession='Курьер',listing_sha256=runner.sha(out/'listing.json'),listing_pages=[dict(file='listing_pages/0001.json',sha256=runner.sha(out/'listing_pages/0001.json'))]))
            cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='123',completed_ids=[]);runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)});runner.write(sandbox/'browser-tab.json',{'url':queue[0]['url']})
            address='Республика, Благовещенск, улица 1';blocks=[f'Условия\nГде предстоит работать: {address}\nГрафик: Вахта','Требования\nОпыт работы: Без опыта']
            rawdata=raw('123');rawdata.update(params_blocks=blocks,fields=[dict(name='Где предстоит работать',value=address),dict(name='График',value='Вахта'),dict(name='Опыт работы',value='Без опыта')],parameter_audit=dict(rendered_complete=False,expansion_controls=[address]),source_sections=[dict(marker='item-view/item-params',text=x) for x in blocks]);runner.write(out/'raw/123.json',rawdata);runner.write(sandbox/'raw/123.json',rawdata)
            snap=f'- title: "Работа" [url={queue[0]["url"]}]\n- heading "Работа" [level=1]\n- text: "Где предстоит работать:"\n- button "{address}" [ref=e1]\n';(diag/'problem.snapshot.txt').write_text(snap);(sandbox/'problem.snapshot.txt').write_text(snap)
            runner.record_failure(diag,'capture','stopped','capture',immediate=True);ledger=runner.load_failures(diag);ledger['retries_started']=5;ledger['events'].extend([dict(event='recurring_ip_policy',source='s',interval_seconds=1800),dict(event='recurring_incomplete_policy',source='s',instruction='разрешаю запросы')]);runner.save_failures(diag,ledger)
            runner.write(diag/'aside-target.local.json',dict(TL_ASIDE_HOST='local',TL_ASIDE_ACCOUNT='u0',TL_ASIDE_PROFILE='pinned'))
            base=[sys.executable,str(scripts/'run.py')];review=base+['review-address-button','--run-id','addr','--id','123','--source','s']
            bad=dict(rawdata);bad['parameter_audit']=dict(rendered_complete=False,expansion_controls=[address,'Показать ещё']);runner.write(out/'raw/123.json',bad);self.assertNotEqual(subprocess.run(review,cwd=root,capture_output=True).returncode,0);runner.write(out/'raw/123.json',rawdata)
            (sandbox/'problem.snapshot.txt').write_text('tampered');self.assertNotEqual(subprocess.run(review,cwd=root,capture_output=True).returncode,0);(sandbox/'problem.snapshot.txt').write_text(snap)
            result=subprocess.run(review,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(runner.read(out/'raw/123.json'),rawdata);self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'],'456')
            continuation=base+['recurring-reviewed-continuation','--run-id','addr','--source','s','--id','456','--authorization-id','one'];self.assertNotEqual(subprocess.run(continuation,cwd=root,capture_output=True).returncode,0)
            ledger=runner.load_failures(diag);[e.update(at='2020-01-01T00:00:00+00:00') for e in ledger['events'] if e.get('event')=='failure'];runner.save_failures(diag,ledger)
            result=subprocess.run(continuation,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr)
            before=(diag/'failures.json').read_bytes();wrong=base+['capture','--run-id','addr','--host','local','--account','u0','--profile','other'];self.assertNotEqual(subprocess.run(wrong,cwd=root,capture_output=True).returncode,0);self.assertEqual((diag/'failures.json').read_bytes(),before)
            env=dict(os.environ,TL_ASIDE_PROFILE='other');conflict=base+['capture','--run-id','addr','--host','local','--account','u0','--profile','pinned'];self.assertNotEqual(subprocess.run(conflict,cwd=root,env=env,capture_output=True).returncode,0);self.assertEqual((diag/'failures.json').read_bytes(),before)
            help_result=subprocess.run(base+['capture','--help'],cwd=root,text=True,capture_output=True);self.assertIn('--profile',help_result.stdout)

    def test_offline_standing_unavailable_and_scheduled_continuation_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();scripts=root/'Scraper/skills/collect-avito/scripts';shutil.copytree(SCRIPTS,scripts);shutil.copytree(runner.ROOT/'utils',root/'utils',ignore=shutil.ignore_patterns('__pycache__'))
            out,service,diag=runner.paths(root,'closed');sandbox=root/'sandbox';items=[item('123'),item('456')];listing=dict(state='LISTING',selected_controls=courier_controls(),items=items)
            runner.write(out/'listing.json',listing);runner.write(out/'listing_pages/0001.json',listing);runner.write(out/'queue.json',runner.queue_from_pages([listing],2))
            runner.write(service/'plan.json',dict(run_id='closed',profession='Курьер',listing_sha256=runner.sha(out/'listing.json'),listing_pages=[dict(file='listing_pages/0001.json',sha256=runner.sha(out/'listing_pages/0001.json'))]))
            cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='123',completed_ids=[]);runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)})
            raw=dict(state='DETAIL',id='123',title='Работа',source_url=items[0]['url']);runner.write(out/'raw/123.json',raw);runner.write(sandbox/'raw/123.json',raw);runner.write(sandbox/'browser-tab.json',{'url':items[0]['url']})
            snap=f'- title: "Работа" [url={items[0]["url"]}]\n- heading "Работа" [level=1]\n- text: "Объявление снято с публикации."\n';(diag/'problem.snapshot.txt').write_text(snap);(sandbox/'problem.snapshot.txt').write_text(snap)
            runner.write(out/'standing_unavailable_policy.json',dict(authorization_id=runner.STANDING_UNAVAILABLE_ID,human_instruction=runner.STANDING_UNAVAILABLE_INSTRUCTION,markers=list(runner.ALL_UNAVAILABLE_MARKERS)))
            ledger=runner.load_failures(diag);ledger['events'].append(dict(event='project_unavailable_policy',authorization_id=runner.STANDING_UNAVAILABLE_ID,instruction=runner.STANDING_UNAVAILABLE_INSTRUCTION,source=runner.STANDING_UNAVAILABLE_SOURCE));runner.save_failures(diag,ledger)
            runner.record_failure(diag,'capture','stopped','capture',immediate=True);ledger=runner.load_failures(diag);ledger['retries_started']=ledger['retry_limit'];ledger['events'].append(dict(event='recurring_ip_policy',source='s',interval_seconds=1800));runner.save_failures(diag,ledger)
            base=[sys.executable,str(scripts/'run.py')];mark=base+['mark-unavailable','--run-id','closed','--id','123','--authorization-id',runner.STANDING_UNAVAILABLE_ID,'--offline-standing']
            (sandbox/'problem.snapshot.txt').write_text('tampered');self.assertNotEqual(subprocess.run(mark,cwd=root,capture_output=True).returncode,0);(sandbox/'problem.snapshot.txt').write_text(snap)
            result=subprocess.run(mark,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(runner.read(out/'raw/123.json'),raw);self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'],'456')
            recover=base+['recurring-unavailable-continuation','--run-id','closed','--source','s','--id','456','--authorization-id','once']
            self.assertNotEqual(subprocess.run(recover,cwd=root,capture_output=True).returncode,0)
            ledger=runner.load_failures(diag);[e.update(at='2020-01-01T00:00:00+00:00') for e in ledger['events'] if e.get('event')=='failure'];runner.save_failures(diag,ledger)
            result=subprocess.run(recover,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(runner.load_failures(diag)['bounded_pending']['id'],'456')

    def test_recurring_incomplete_actual_cli_and_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); scripts=root/'Scraper/skills/collect-avito/scripts'; shutil.copytree(SCRIPTS,scripts); shutil.copytree(runner.ROOT/'utils',root/'utils',ignore=shutil.ignore_patterns('__pycache__'))
            out,service,diag=runner.paths(root,'inc'); sandbox=root/'sandbox'; candidate=item('456'); url=candidate['url']; listing=dict(state='LISTING',selected_controls=courier_controls(),items=[candidate])
            runner.write(out/'listing.json',listing);runner.write(out/'listing_pages/0001.json',listing);runner.write(out/'queue.json',runner.queue_from_pages([listing],1))
            runner.write(service/'plan.json',dict(run_id='inc',profession='Курьер',listing_sha256=runner.sha(out/'listing.json'),listing_pages=[dict(file='listing_pages/0001.json',sha256=runner.sha(out/'listing_pages/0001.json'))]))
            cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='456',completed_ids=[]);runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)})
            raw=dict(state='UNKNOWN_PAGE',source_url=url,heading=None,items=[],unresolved_evidence=dict(complete=False,restriction=False,known_unavailable=False));runner.write(out/'raw/456.json',raw);runner.write(sandbox/'raw/456.json',raw);runner.write(sandbox/'browser-tab.json',{'url':url})
            snap=f'- title: "Работа" [url={url}]\n- link "Помощь" [ref=e1]\n';(diag/'problem.snapshot.txt').write_text(snap);(sandbox/'problem.snapshot.txt').write_text(snap)
            runner.record_failure(diag,'capture','stopped','capture',immediate=True);ledger=runner.load_failures(diag);ledger['retries_started']=ledger['retry_limit'];ledger['events'][-1]['at']='2020-01-01T00:00:00+00:00';runner.save_failures(diag,ledger)
            base=[sys.executable,str(scripts/'run.py')];cmds=[['recurring-ip-policy','--run-id','inc','--source','s','--human-instruction','каждые 30 минут','--additional-instruction','все разрешения'],['recurring-incomplete-policy','--run-id','inc','--source','s','--human-instruction','все запросы к авито']]
            for cmd in cmds:
                result=subprocess.run(base+cmd,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr)
            cmd=base+['recurring-incomplete-attempt','--run-id','inc','--source','s','--id','456','--authorization-id','once']
            wrong=cmd.copy();wrong[wrong.index('456')]='999';self.assertNotEqual(subprocess.run(wrong,cwd=root,capture_output=True).returncode,0)
            (sandbox/'problem.snapshot.txt').write_text('tampered');self.assertNotEqual(subprocess.run(cmd,cwd=root,capture_output=True).returncode,0);(sandbox/'problem.snapshot.txt').write_text(snap)
            protected=snap+'CAPTCHA 429';(sandbox/'problem.snapshot.txt').write_text(protected);(diag/'problem.snapshot.txt').write_text(protected)
            self.assertNotEqual(subprocess.run(cmd,cwd=root,capture_output=True).returncode,0);(sandbox/'problem.snapshot.txt').write_text(snap);(diag/'problem.snapshot.txt').write_text(snap)
            data=runner.read(out/'raw/456.json');data['unresolved_evidence']['restriction']=True;runner.write(out/'raw/456.json',data)
            self.assertNotEqual(subprocess.run(cmd,cwd=root,capture_output=True).returncode,0);runner.write(out/'raw/456.json',raw)
            ledger=runner.load_failures(diag);ledger['events'][0]['at']=runner.datetime.now(runner.timezone.utc).isoformat();runner.save_failures(diag,ledger)
            self.assertNotEqual(subprocess.run(cmd,cwd=root,capture_output=True).returncode,0);ledger['events'][0]['at']='2020-01-01T00:00:00+00:00';runner.save_failures(diag,ledger)
            result=subprocess.run(cmd,cwd=root,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr)
            saved=subprocess.run(base+['retry-incomplete-observation','--run-id','inc','--id','456','--authorization-id','once'],cwd=root,text=True,capture_output=True);self.assertEqual(saved.returncode,0,saved.stderr)
            self.assertEqual(runner.read(sandbox/'raw/456.json'),raw);self.assertFalse((out/'raw/456.json').exists());self.assertEqual(runner.load_failures(diag)['retries_started'],5)

    def test_capture_waits_only_for_unrestricted_incomplete_shell(self):
        script=(SCRIPTS/'browser_batch.js').read_text(); candidate=item('123')
        job=dict(mode='capture',queue=[candidate],completed_ids=[],close_tab=False)
        body=script.replace('__JOB__',json.dumps(job),1).replace('__EXTRACTOR__','"extract"',1)
        for scenario,expected_calls,expected_state in [('ready',1,'CAPTURED'),('delayed',3,'CAPTURED'),('detail_delayed',3,'CAPTURED'),('detail_timeout',11,'REVIEW_REQUIRED'),('detail_blocked',2,'NEEDS_HUMAN'),('detail_wrong_id',1,'REVIEW_REQUIRED'),('timeout',11,'REVIEW_REQUIRED'),('blocked',2,'NEEDS_HUMAN'),('unavailable',1,'REVIEW_REQUIRED'),('wrongurl',1,'REVIEW_REQUIRED')]:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                harness="""const fs=require('fs').promises,pwd=process.cwd(),url=URL;let calls=0,sleeps=0;const scenario=SCENARIO;
const pg={url:()=>scenario==='wrongurl'?'https://www.avito.ru/wrong':url,evaluate:async()=>{calls++;if((scenario==='blocked'||scenario==='detail_blocked')&&calls===2)return {state:'NEEDS_HUMAN',source_url:url};if(scenario==='ready'||(scenario==='delayed'||scenario==='detail_delayed')&&calls===3)return {state:'DETAIL',id:'123',title:'Job',source_url:url,parameter_audit:{rendered_complete:true},fields:[{name:'x',value:'y'}]};if(scenario.startsWith('detail_'))return {state:'DETAIL',id:scenario==='detail_wrong_id'?'999':'123',title:'Job',source_url:url,parameter_audit:{rendered_complete:false},fields:[],params_blocks:[]};return {state:'UNKNOWN_PAGE',source_url:url,unresolved_evidence:{complete:false,restriction:false,known_unavailable:scenario==='unavailable'}}}};
const listBrowserTabs=async()=>[],openTab=async()=>pg,snapshot=async()=>({tree:'snapshot'}),sleep=async()=>{sleeps++};
(async()=>{BODY;console.log('CHECK='+JSON.stringify({calls,sleeps,cp:JSON.parse(await fs.readFile('checkpoint.json','utf8'))}))})().catch(e=>{console.error(e);process.exit(1)});
""".replace('URL',json.dumps(candidate['url'])).replace('SCENARIO',json.dumps(scenario)).replace('BODY',body)
                result=subprocess.run(['node','-e',harness],cwd=directory,text=True,capture_output=True,check=True)
                info=json.loads(result.stdout.split('CHECK=')[-1]);self.assertEqual(info['calls'],expected_calls);self.assertEqual(info['cp']['state'],expected_state);self.assertLessEqual(info['sleeps'],10)
                if scenario=='detail_timeout':
                    observed=json.loads((Path(directory)/'raw/123.json').read_text());self.assertEqual(observed['readiness_wait'],{'attempts':10,'interval_ms':500,'same_url_and_id':True})

    def test_recurring_ip_commands_parse_and_dispatch_in_real_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); scripts=root/'Scraper/skills/collect-avito/scripts'
            shutil.copytree(SCRIPTS,scripts); shutil.copytree(runner.ROOT/'utils',root/'utils',ignore=shutil.ignore_patterns('__pycache__'))
            out,service,diag=runner.paths(root,'cli'); sandbox=root/'sandbox'; candidate=item('456')
            listing=dict(state='LISTING',selected_controls=courier_controls(),items=[candidate])
            runner.write(out/'listing.json',listing); runner.write(out/'listing_pages/0001.json',listing)
            runner.write(out/'queue.json',runner.queue_from_pages([listing],1))
            runner.write(service/'plan.json',dict(run_id='cli',profession='Курьер',listing_sha256=runner.sha(out/'listing.json'),listing_pages=[dict(file='listing_pages/0001.json',sha256=runner.sha(out/'listing_pages/0001.json'))]))
            cp=dict(state='NEEDS_HUMAN',stop_reason='access_restriction_or_captcha',next_id='456',completed_ids=[])
            runner.write(diag/'checkpoint.json',cp); runner.write(sandbox/'checkpoint.json',cp); runner.write(diag/'sandbox.json',{'path':str(sandbox)})
            raw=dict(state='NEEDS_HUMAN',reason='access_restriction_or_captcha',source_url=candidate['url'])
            runner.write(out/'raw/456.json',raw); runner.write(sandbox/'raw/456.json',raw); runner.write(sandbox/'browser-tab.json',{'url':candidate['url']})
            snap=f'- title: "Доступ ограничен: проблема с IP" [url={candidate["url"]}]\n- heading "Страница не загрузилась из-за проблем с IP" [level=1]\n'
            (diag/'problem.snapshot.txt').write_text(snap,encoding='utf8'); (sandbox/'problem.snapshot.txt').write_text(snap,encoding='utf8')
            runner.record_failure(diag,'capture','stopped','capture',immediate=True)
            ledger=runner.load_failures(diag); ledger['retries_started']=ledger['retry_limit']; ledger['events'][-1]['at']='2020-01-01T00:00:00+00:00'; runner.save_failures(diag,ledger)
            base=[sys.executable,str(scripts/'run.py')]
            commands=[['recurring-ip-policy','--run-id','cli','--source','source','--human-instruction','каждые 30 минут','--additional-instruction','все разрешения'],
                      ['recurring-ip-attempt','--run-id','cli','--source','source','--id','456','--authorization-id','attempt']]
            for args in commands:
                result=subprocess.run(base+args,cwd=root,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(runner.load_failures(diag)['bounded_pending']['id'],'456')
            self.assertTrue((out/'raw/456.json').exists())
            preserved=subprocess.run(base+['preserve-stopped-observation','--run-id','cli','--authorization-id','attempt'],cwd=root,text=True,capture_output=True)
            self.assertEqual(preserved.returncode,0,preserved.stderr)
            runner.ensure_allowed(diag,'capture',action=True)
            failed_command=['/Library/Developer/CommandLineTools/usr/bin/python3',str(scripts/'preflight.py'),'--probe-aside','--config',str(root/'Scraper/collect-avito.local.json')]
            runner.record_failure(diag,'capture',str(subprocess.CalledProcessError(2,failed_command)),'capture')
            ledger=runner.load_failures(diag); ledger['events'][-1]['at']='2020-01-01T00:00:00+00:00'
            for e in ledger['events']:
                if e.get('event')=='recurring_ip_authorization': e['at']='2020-01-01T00:00:00+00:00'
            runner.save_failures(diag,ledger)
            recover=base+['recurring-ip-after-preflight','--run-id','cli','--source','source','--id','456','--authorization-id','attempt-2']
            oldbytes=(diag/'failures.json').read_bytes(); saved=next((diag/'observations/456').glob('*.problem.snapshot.txt')); snapbytes=saved.read_bytes()
            saved.write_text('tampered',encoding='utf8'); self.assertNotEqual(subprocess.run(recover,cwd=root,capture_output=True).returncode,0); saved.write_bytes(snapbytes)
            runner.write(out/'raw/456.json',raw); self.assertNotEqual(subprocess.run(recover,cwd=root,capture_output=True).returncode,0); (out/'raw/456.json').unlink()
            bad=recover.copy(); bad[bad.index('456')]='999'; self.assertNotEqual(subprocess.run(bad,cwd=root,capture_output=True).returncode,0)
            recent=runner.load_failures(diag); recent['events'][-1]['at']=runner.datetime.now(runner.timezone.utc).isoformat(); runner.save_failures(diag,recent)
            self.assertNotEqual(subprocess.run(recover,cwd=root,capture_output=True).returncode,0); (diag/'failures.json').write_bytes(oldbytes)
            result=subprocess.run(recover,cwd=root,text=True,capture_output=True); self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(runner.load_failures(diag)['bounded_pending']['state'],'reserved'); self.assertFalse((out/'raw/456.json').exists())

    def test_standing_run_registration_is_scoped_and_cannot_clear_stops(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            args=argparse.Namespace(run_id='fresh',goal_source='goal-123',authorization_id='source-1',
                technical_instruction='точная техническая инструкция',ip_instruction='точная инструкция IP')
            runner.register_standing_run(args); runner.register_standing_run(args)
            out,service,diag=runner.paths(runner.ROOT,'fresh'); ledger=runner.load_failures(diag)
            self.assertTrue(ledger['unbounded_retries']); self.assertEqual(ledger['state'],'READY')
            self.assertEqual(ledger['events'][0]['technical_instruction'],args.technical_instruction)
            self.assertEqual(ledger['events'][1]['instruction'],args.ip_instruction)
            self.assertEqual(ledger['events'][1]['interval_seconds'],1800)
            altered=argparse.Namespace(**{**vars(args),'goal_source':'other-goal'})
            with self.assertRaises(runner.HumanRequired): runner.register_standing_run(altered)
            for _ in range(6):
                runner.record_failure(diag,'capture','technical','capture',immediate=False)
                runner.ensure_allowed(diag,'capture',retry_reason='corrected',action=True)
            runner.record_failure(diag,'capture','CAPTCHA','capture',immediate=True)
            with self.assertRaises(runner.HumanRequired): runner.ensure_allowed(diag,'capture',action=True)
            with self.assertRaises(runner.HumanRequired): runner.register_standing_run(altered)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            _,service,_=runner.paths(runner.ROOT,'fresh');runner.write(service/'plan.json',{})
            with self.assertRaises(runner.HumanRequired): runner.register_standing_run(args)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            argv=['run.py','register-standing-run','--run-id','cli','--goal-source','goal',
                  '--authorization-id','source','--technical-instruction','exact technical',
                  '--ip-instruction','exact IP']
            with patch.object(sys,'argv',argv): runner.main()
            self.assertTrue(runner.load_failures(runner.paths(runner.ROOT,'cli')[2])['unbounded_retries'])

    def test_recurring_ip_policy_claim_and_preservation(self):
        for scenario in ('ok', 'unexhausted', 'unbounded', 'missing', 'missing_no_log', 'missing_tamper', 'missing_late', 'early', 'captcha_only', 'http429', 'wrong_id', 'tampered', 'prefix'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
                root=runner.ROOT; out,service,diag=runner.paths(root,'case'); sandbox=root/'sandbox'
                queue=[item('123'), item('456')]; queue[1]['listing_rank']=2; url=queue[1]['url']
                runner.write(out/'queue.json',queue); runner.write(out/'listing.json',{}); runner.write(out/'listing_pages/0001.json',{})
                runner.write(service/'plan.json',dict(run_id='case',listing_sha256='pinned',listing_pages=[dict(file='listing_pages/0001.json')]))
                cp=dict(state='NEEDS_HUMAN',stop_reason='access_restriction_or_captcha',next_id='456',completed_ids=[] if scenario=='prefix' else ['123'])
                runner.write(diag/'checkpoint.json',cp); runner.write(sandbox/'checkpoint.json',cp); runner.write(diag/'sandbox.json',{'path':str(sandbox)})
                raw=dict(state='NEEDS_HUMAN',reason='access_restriction_or_captcha',source_url=url)
                runner.write(out/'raw/456.json',raw); runner.write(sandbox/'raw/456.json',raw); runner.write(sandbox/'browser-tab.json',{'url':url})
                snap=f'- title: "Доступ ограничен: проблема с IP" [url={url}]\n- heading "Страница не загрузилась из-за проблем с IP" [level=1]\n'
                if scenario=='captcha_only': snap='- title: "CAPTCHA" [url='+url+']\n'
                if scenario=='http429': snap+='- text: "HTTP 429"\n'
                (diag/'problem.snapshot.txt').write_text(snap,encoding='utf8'); (sandbox/'problem.snapshot.txt').write_text(snap+('x' if scenario=='tampered' else ''),encoding='utf8')
                if scenario.startswith('missing'):
                    runner.write(diag/'browser-tab.json',{'url':url}); runner.write(diag/'browser-session.local.json',{'sandbox':str(sandbox)})
                    log=diag/'aside-123.local.log'; log.write_text(f'TL_SANDBOX={sandbox}\n'+snap+('TL_FINISHED\n[ok | 1ms]\n' if scenario!='missing_tamper' else ''),encoding='utf8')
                    shutil.rmtree(sandbox)
                runner.record_failure(diag,'capture','stopped','capture',immediate=True)
                ledger=runner.load_failures(diag); ledger['retries_started']=2 if scenario in ('unexhausted', 'unbounded') else ledger['retry_limit']
                if scenario == 'unbounded': ledger['unbounded_retries'] = True
                if scenario!='early': ledger['events'][-1]['at']='2020-01-01T00:00:00+00:00'
                runner.save_failures(diag,ledger)
                policy=argparse.Namespace(run_id='case',source='source',human_instruction='каждые 30 минут',additional_instruction='все разрешения')
                runner.recurring_ip_policy(policy); runner.recurring_ip_policy(policy)
                attempt=argparse.Namespace(run_id='case',source='source',id='999' if scenario=='wrong_id' else '456',authorization_id='attempt-1',missing_sandbox_log=log if scenario.startswith('missing') and scenario!='missing_no_log' else None)
                with patch.object(runner,'validate_partition'), patch.object(runner,'validate_plan_profession'), patch.object(runner,'queue_from_pages',return_value=queue), patch.object(runner,'unavailable_records',return_value=[]), patch.object(runner,'unresolved_records',return_value=[]):
                    if scenario in ('ok', 'unexhausted', 'unbounded', 'missing', 'missing_late'):
                        before=runner.load_failures(diag); runner.recurring_ip_attempt(attempt)
                        with self.assertRaises((ValueError,runner.HumanRequired)): runner.recurring_ip_attempt(attempt)
                        ledger=runner.load_failures(diag); self.assertEqual(ledger['retries_started'],before['retries_started']); self.assertEqual(ledger['failures'],before['failures'])
                        if scenario=='missing_late':
                            log.write_text('changed',encoding='utf8')
                            with self.assertRaises(ValueError): runner.preserve_stopped_observation(argparse.Namespace(run_id='case',authorization_id='attempt-1'))
                            continue
                        runner.preserve_stopped_observation(argparse.Namespace(run_id='case',authorization_id='attempt-1'))
                        self.assertEqual(runner.read(diag/'checkpoint.json')['completed_ids'],['123']); self.assertEqual(runner.read(diag/'checkpoint.json')['state'],'READY')
                        if scenario!='missing': self.assertEqual(runner.read(sandbox/'raw/456.json'),raw)
                        else: self.assertFalse(sandbox.exists())
                        self.assertFalse((out/'raw/456.json').exists())
                    else:
                        with self.assertRaises((ValueError,runner.HumanRequired)): runner.recurring_ip_attempt(attempt)
                        self.assertEqual(runner.load_failures(diag)['state'],'NEEDS_HUMAN'); self.assertTrue((out/'raw/456.json').exists())

    def test_bounded_pending_authorization_never_replenishes_budget(self):
        path = 'Scraper/skills/collect-avito/scripts/failure_log.py'
        self.assertNotIn(path, runner.allowed_capture_changes({}, []))
        self.assertIn(path, runner.allowed_capture_changes({}, [dict(file=path, reason='Guard incomplete shell preservation and authorize one pending capture without new budget', validation='passed')]))
        for outcome in ('success', 'failure', 'interrupted'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
                _, _, diag = runner.paths(runner.ROOT, 'case')
                runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', next_id='456'))
                runner.record_failure(diag, 'capture', 'stop', 'capture', immediate=True)
                ledger = runner.load_failures(diag); ledger['retries_started'] = ledger['retry_limit']; runner.save_failures(diag, ledger)
                original = runner.load_failures(diag); counts = (original['failures'], original['retries_started'], original['retry_limit'], original['budget_start_failures'])
                args = argparse.Namespace(run_id='case', human_instruction='Повторить 456 и продолжить очередь', authorization_id='one', pending_id_no_budget='456')
                runner.resume_browser(args); ledger = runner.load_failures(diag)
                self.assertEqual((ledger['failures'], ledger['retries_started'], ledger['retry_limit'], ledger['budget_start_failures']), counts)
                with self.assertRaises(runner.HumanRequired): runner.ensure_allowed(diag, 'freeze', action=True)
                runner.ensure_allowed(diag, 'capture', action=True)
                with self.assertRaises(runner.HumanRequired): runner.ensure_allowed(diag, 'capture', action=True)
                if outcome == 'success':
                    runner.succeeded(diag, 'capture', 'one ID complete')
                    self.assertEqual(runner.load_failures(diag)['bounded_pending']['state'], 'completed')
                    runner.record_failure(diag, 'capture', 'next error', 'capture')
                elif outcome == 'failure': runner.record_failure(diag, 'capture', 'pending failed', 'capture')
                else:
                    self.assertEqual(runner.load_failures(diag)['state'], 'RETRY_STARTED')
                    continue
                self.assertEqual(runner.load_failures(diag)['state'], 'NEEDS_HUMAN')
                with self.assertRaises(runner.HumanRequired): runner.resume_browser(args)

    def test_retry_incomplete_requires_authorized_matching_forensic_evidence(self):
        for scenario in ('ok', 'standing', 'missing_id', 'old_auth', 'wrong_id', 'restriction', 'snapshot', 'prefix'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
                root = runner.ROOT; run_id = 'case'; out, service, diag = runner.paths(root, run_id); sandbox = root/'sandbox'
                queue = [item('123'), item('456')]; queue[1]['listing_rank'] = 2
                runner.write(out/'queue.json', queue); runner.write(out/'listing.json', {})
                runner.write(out/'listing_pages/0001.json', {})
                runner.write(service/'plan.json', dict(run_id=run_id, listing_pages=[dict(file='listing_pages/0001.json')], listing_sha256='pinned'))
                cp = dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='456', completed_ids=['123'])
                if scenario == 'prefix': cp['completed_ids'] = []
                runner.write(diag/'checkpoint.json', cp); runner.write(sandbox/'checkpoint.json', cp)
                runner.write(diag/'sandbox.json', {'path':str(sandbox)}); runner.write(sandbox/'browser-tab.json', {'url':queue[1]['url']})
                data = dict(state='UNKNOWN_PAGE', source_url=queue[1]['url'], heading=None, items=[], unresolved_evidence=dict(complete=False, restriction=scenario == 'restriction', known_unavailable=False))
                runner.write(out/'raw/456.json', data); runner.write(sandbox/'raw/456.json', data)
                snap = '# note: interactive (clickable / focusable) elements only.\n- title: "Авито" [url='+queue[1]['url']+']\n- link "Помощь" [ref=e1]\n'
                (diag/'problem.snapshot.txt').write_text(snap, encoding='utf8'); (sandbox/'problem.snapshot.txt').write_text(snap + ('tamper' if scenario == 'snapshot' else ''), encoding='utf8')
                runner.record_failure(diag, 'capture', 'stopped', 'capture', immediate=True)
                runner.authorize_browser(diag, 'Самостоятельно повторять технические ошибки' if scenario in ('standing', 'missing_id') else 'Повторить карточку 456', 'auth')
                if scenario == 'old_auth': runner.record_failure(diag, 'capture', 'new stop', 'capture', immediate=True)
                args = argparse.Namespace(run_id=run_id, id='999' if scenario == 'wrong_id' else '456', authorization_id='auth', standing_incomplete=scenario == 'standing')
                original = (out/'raw/456.json').read_bytes(); plan_bytes = (service/'plan.json').read_bytes(); queue_bytes = (out/'queue.json').read_bytes()
                with patch.object(runner, 'validate_partition'), patch.object(runner, 'validate_plan_profession'), patch.object(runner, 'queue_from_pages', return_value=queue), patch.object(runner, 'unavailable_records', return_value=[]), patch.object(runner, 'unresolved_records', return_value=[]):
                    if scenario in ('ok', 'standing'): runner.retry_incomplete_observation(args)
                    else:
                        with self.assertRaises((ValueError, runner.HumanRequired)): runner.retry_incomplete_observation(args)
                self.assertEqual((service/'plan.json').read_bytes(), plan_bytes); self.assertEqual((out/'queue.json').read_bytes(), queue_bytes)
                if scenario in ('ok', 'standing'):
                    digest = hashlib.sha256(original).hexdigest(); self.assertEqual((out/'observations/456'/f'{digest}.json').read_bytes(), original)
                    self.assertFalse((out/'raw/456.json').exists()); self.assertEqual((sandbox/'raw/456.json').read_bytes(), original)
                    self.assertEqual(runner.read(diag/'checkpoint.json')['completed_ids'], ['123'])
                    self.assertEqual(runner.read(out/'observations/456'/f'{digest}.provenance.json')['snapshot_sha256'], runner.sha(diag/'problem.snapshot.txt'))
                else: self.assertEqual((out/'raw/456.json').read_bytes(), original)

    def test_detail_hydration_one_authorized_retry_preserves_evidence(self):
        for scenario in ('ok', 'restriction', 'wrong_id', 'snapshot'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
                root=runner.ROOT; out,service,diag=runner.paths(root,'case'); sandbox=root/'sandbox'
                queue=[item('123')]; url=queue[0]['url']
                runner.write(out/'queue.json',queue); runner.write(out/'listing.json',{}); runner.write(out/'listing_pages/0001.json',{})
                runner.write(service/'plan.json',dict(run_id='case',listing_pages=[dict(file='listing_pages/0001.json')],listing_sha256='pinned'))
                cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='123',completed_ids=[])
                runner.write(diag/'checkpoint.json',cp); runner.write(sandbox/'checkpoint.json',cp)
                runner.write(diag/'sandbox.json',{'path':str(sandbox)}); runner.write(sandbox/'browser-tab.json',{'url':url})
                data=dict(state='DETAIL',id='999' if scenario=='wrong_id' else '123',source_url=url,title='Job',observed_at='2026-11-06T07:56:29Z',fields=[],params_blocks=[],parameter_audit={'rendered_complete':False})
                runner.write(out/'raw/123.json',data); runner.write(sandbox/'raw/123.json',data)
                snap=f'- title: "Авито" [url={url}]\n- heading "Job" [level=1]\n- heading "Условия" [level=2]\n- heading "Требования" [level=2]\n'+('CAPTCHA' if scenario=='restriction' else '')
                (diag/'problem.snapshot.txt').write_text(snap); (sandbox/'problem.snapshot.txt').write_text(snap+('changed' if scenario=='snapshot' else ''))
                runner.record_failure(diag,'capture','stopped','capture',immediate=True)
                before=runner.load_failures(diag)
                args=argparse.Namespace(run_id='case',pending_id_no_budget='123',detail_hydration=True,unbounded_retries=False,human_instruction='Повторить карточку 123',authorization_id='human-1')
                if scenario!='ok':
                    with self.assertRaises(ValueError): runner.resume_browser(args)
                    self.assertEqual(runner.load_failures(diag)['state'],'NEEDS_HUMAN'); continue
                runner.resume_browser(args)
                auth=runner.load_failures(diag); self.assertEqual(auth['failures'],before['failures']); self.assertEqual(auth['retries_started'],before['retries_started'])
                original=(out/'raw/123.json').read_bytes(); old_snap=(diag/'problem.snapshot.txt').read_bytes()
                with patch.object(runner,'validate_partition'), patch.object(runner,'validate_plan_profession'), patch.object(runner,'queue_from_pages',return_value=queue), patch.object(runner,'unavailable_records',return_value=[]), patch.object(runner,'unresolved_records',return_value=[]):
                    runner.retry_incomplete_observation(argparse.Namespace(run_id='case',id='123',authorization_id='human-1'))
                digest=hashlib.sha256(original).hexdigest()
                self.assertEqual((out/'observations/123'/f'{digest}.json').read_bytes(),original)
                self.assertTrue(any(p.read_bytes()==old_snap for p in (diag/'observations/123').glob('*.problem.snapshot.txt')))
                self.assertFalse((out/'raw/123.json').exists())
                self.assertEqual(runner.load_failures(diag)['bounded_pending']['state'],'reserved')

    def test_user_skip_preserves_evidence_and_accounts_queue(self):
        for scenario in ('ok', 'restriction', 'wrong_id', 'snapshot'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
                root=runner.ROOT; out,service,diag=runner.paths(root,runner.USER_SKIP_RUN); sandbox=root/'sandbox'
                queue=[item(runner.USER_SKIP_ID),item('123')]; url=queue[0]['url']; runner.write(out/'queue.json',queue)
                cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id=runner.USER_SKIP_ID,completed_ids=[])
                runner.write(diag/'checkpoint.json',cp); runner.write(sandbox/'checkpoint.json',cp)
                runner.write(diag/'sandbox.json',{'path':str(sandbox)}); runner.write(sandbox/'browser-tab.json',{'url':url})
                data=dict(state='UNKNOWN_PAGE',source_url=url,heading=None,items=[],unresolved_evidence=dict(complete=False,restriction=scenario=='restriction',known_unavailable=False))
                raw=out/'raw'/f'{runner.USER_SKIP_ID}.json'; runner.write(raw,data); runner.write(sandbox/'raw'/raw.name,data)
                snap=f'- title: "Авито" [url={url}]\n- link "Помощь" [ref=e1]\n'
                (diag/'problem.snapshot.txt').write_text(snap); (sandbox/'problem.snapshot.txt').write_text(snap+('changed' if scenario=='snapshot' else ''))
                runner.record_failure(diag,'capture','stopped','capture',immediate=True); before=runner.load_failures(diag); original=raw.read_bytes()
                args=argparse.Namespace(run_id=runner.USER_SKIP_RUN,id='999' if scenario=='wrong_id' else runner.USER_SKIP_ID,human_instruction='Пропустить 8557245056',authorization_id='skip-1')
                if scenario!='ok':
                    with self.assertRaises(ValueError): runner.mark_user_skipped(args)
                    self.assertEqual(runner.load_failures(diag)['state'],'NEEDS_HUMAN'); continue
                runner.mark_user_skipped(args)
                after=runner.load_failures(diag); self.assertEqual(after['state'],'RECOVERED')
                for field in ('failures','retries_started','retry_limit','budget_start_failures'): self.assertEqual(after[field],before[field])
                self.assertEqual(raw.read_bytes(),original); self.assertEqual((sandbox/'raw'/raw.name).read_bytes(),original)
                checkpoint=runner.read(diag/'checkpoint.json'); self.assertEqual(checkpoint['next_id'],'123'); self.assertEqual(checkpoint['completed_ids'],[])
                records=runner.user_skipped_records(out,queue); self.assertEqual(records[0]['status'],'user_skipped')
                self.assertEqual(records[0]['reason'],'user_directed_incomplete_observation')
                raw.write_text('{}')
                with self.assertRaises(ValueError): runner.user_skipped_records(out,queue)

    def test_user_skip_cli_parses_only_its_own_arguments(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            argv=['run.py','mark-user-skipped','--run-id',runner.USER_SKIP_RUN,'--id',runner.USER_SKIP_ID,
                  '--human-instruction','Пропустить 8557245056','--authorization-id','skip-1']
            with patch('sys.argv',argv), patch.object(runner,'mark_user_skipped') as action:
                runner.main()
            self.assertEqual(action.call_count,1)
            args=action.call_args.args[0]
            self.assertEqual((args.run_id,args.id,args.authorization_id),(runner.USER_SKIP_RUN,runner.USER_SKIP_ID,'skip-1'))
            self.assertFalse(hasattr(args,'file'))

    def test_standing_user_skip_classifies_and_preserves_only_authorized_evidence(self):
        for scenario in ('blocked', 'shell', '404', 'captcha', '429', 'detail', 'unrecognized'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
                root=runner.ROOT; out,service,diag=runner.paths(root,'case'); sandbox=root/'sandbox'; queue=[item('123'),item('456')]; url=queue[0]['url']
                runner.write(out/'queue.json',queue)
                cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='123',completed_ids=[])
                runner.write(diag/'checkpoint.json',cp); runner.write(sandbox/'checkpoint.json',cp); runner.write(diag/'sandbox.json',{'path':str(sandbox)}); runner.write(sandbox/'browser-tab.json',{'url':url})
                ev=dict(complete=scenario=='blocked',restriction=False,known_unavailable=scenario=='blocked')
                data=dict(state='UNKNOWN_PAGE',source_url=url,heading='Объявление не посмотреть' if scenario=='blocked' else None,items=[],unresolved_evidence=ev)
                if scenario=='detail': data.update(state='DETAIL',id='123',fields=[],params_blocks=[])
                runner.write(out/'raw/123.json',data); runner.write(sandbox/'raw/123.json',data)
                title='Ошибка 404. Страница не найдена' if scenario=='404' else 'Авито'
                snap=f'- title: "{title}" [url={url}]\n'
                if scenario=='blocked': snap+='- heading "Объявление не посмотреть" [level=1]\n- text: "Оно заблокировано после проверки."\n'
                if scenario=='captcha': snap+='- heading "CAPTCHA" [level=1]\n'
                if scenario=='429': snap+='- text: "Ошибка 429"\n'
                if scenario=='unrecognized': data['unresolved_evidence']['complete']=True; runner.write(out/'raw/123.json',data); runner.write(sandbox/'raw/123.json',data)
                (diag/'problem.snapshot.txt').write_text(snap); (sandbox/'problem.snapshot.txt').write_text(snap)
                runner.record_failure(diag,'capture','stopped','capture',immediate=True); before=runner.load_failures(diag)
                runner.user_skip_policy(argparse.Namespace(run_id='case',authorization_id='policy-1',human_instruction='Пропускать подтверждённые недоступные и неполные карточки'))
                args=argparse.Namespace(run_id='case',id='123',authorization_id='policy-1')
                if scenario not in ('blocked','shell','404'):
                    with self.assertRaises(ValueError): runner.mark_policy_skipped(args)
                    self.assertEqual(runner.load_failures(diag)['state'],'NEEDS_HUMAN'); continue
                original=(out/'raw/123.json').read_bytes(); runner.mark_policy_skipped(args)
                after=runner.load_failures(diag); self.assertEqual(after['state'],'RECOVERED')
                self.assertEqual((after['failures'],after['retries_started']),(before['failures'],before['retries_started']))
                self.assertEqual((out/'raw/123.json').read_bytes(),original)
                record=runner.user_skipped_records(out,queue)[0]
                self.assertEqual(record['reason'],{'blocked':'unavailable: Оно заблокировано после проверки.','shell':'incomplete_shell','404':'unavailable: exact 404 page title'}[scenario])
                self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'],'456')

    def test_standing_skip_cli_and_capture_continue(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            for command,extra,target in (
                ('user-skip-policy',['--human-instruction','standing instruction','--authorization-id','p1'],'user_skip_policy'),
                ('mark-policy-skipped',['--id','123','--authorization-id','p1'],'mark_policy_skipped')):
                with patch('sys.argv',['run.py',command,'--run-id','case']+extra), patch.object(runner,target) as action:
                    runner.main(); self.assertEqual(action.call_count,1)
            args=argparse.Namespace(run_id='case')
            with patch.object(runner,'_capture_batch',side_effect=['USER_SKIPPED','USER_SKIPPED',None]) as batch:
                runner.capture(args); self.assertEqual(batch.call_count,3)

    def test_detail_skip_requires_wait_matching_identity_and_no_parser_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            root=runner.ROOT;out,_,diag=runner.paths(root,'case');sandbox=root/'sandbox';queue=[item('123'),item('456')];url=queue[0]['url']
            runner.write(out/'queue.json',queue);cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='123',completed_ids=[])
            runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)});runner.write(sandbox/'browser-tab.json',{'url':url})
            data=raw('123');data.update(fields=[],params_blocks=[],parameter_audit=dict(rendered_complete=False,expansion_controls=[]),readiness_wait=dict(attempts=10,interval_ms=500,same_url_and_id=True))
            snap=f'- title: "Работа" [url={url}]\n- heading "Работа" [level=1]\n'
            self.assertEqual(runner.skip_reason(data,snap,'123',url),'incomplete_detail_after_wait')
            for bad,s in ((dict(data,id='999'),snap),(dict(data,readiness_wait=None),snap),(dict(data,fields=[{'name':'x','value':'y'}]),snap),(data,snap+'- heading "Условия" [level=2]\n'),(data,snap+'- heading "CAPTCHA" [level=1]\n'),(data,snap.replace('title: "Работа"','title: "Ошибка 404"'))):
                with self.assertRaises(ValueError):runner.skip_reason(bad,s,'123',url)
            runner.write(out/'raw/123.json',data);runner.write(sandbox/'raw/123.json',data);(diag/'problem.snapshot.txt').write_text(snap);(sandbox/'problem.snapshot.txt').write_text(snap)
            runner.record_failure(diag,'capture','stopped','capture',immediate=True);before=runner.load_failures(diag)
            runner.user_detail_skip_policy(argparse.Namespace(run_id='case',authorization_id='detail-1',human_instruction='Skip individual incomplete cards after wait'))
            runner.mark_policy_skipped(argparse.Namespace(run_id='case',id='123',authorization_id='detail-1'))
            record=runner.user_skipped_records(out,queue)[0];self.assertEqual(record['reason'],'incomplete_detail_after_wait')
            after=runner.load_failures(diag);self.assertEqual(after['failures'],before['failures']);self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'],'456')
            next_data=raw('456');next_data.update(fields=[],params_blocks=[],parameter_audit=dict(rendered_complete=False,expansion_controls=[]),readiness_wait=dict(attempts=10,interval_ms=500,same_url_and_id=True))
            next_snap=f'- title: "Работа" [url={queue[1]["url"]}]\n- heading "Работа" [level=1]\n'
            next_cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id='456',completed_ids=[])
            runner.write(diag/'checkpoint.json',next_cp);runner.write(sandbox/'checkpoint.json',next_cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)});runner.write(sandbox/'browser-tab.json',{'url':queue[1]['url']})
            runner.write(out/'raw/456.json',next_data);runner.write(sandbox/'raw/456.json',next_data);(diag/'problem.snapshot.txt').write_text(next_snap);(sandbox/'problem.snapshot.txt').write_text(next_snap)
            with self.assertRaisesRegex(ValueError,'parser failure'):runner.mark_policy_skipped(argparse.Namespace(run_id='case',id='456',authorization_id='detail-1'))

    def test_user_skip_is_excluded_and_counted_in_final_export(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root=runner.ROOT; out,service,diag=runner.paths(root,runner.USER_SKIP_RUN); sandbox=root/'sandbox'
            runner.write(root/'Data/dataset/sources.json',{'schema_version':'1','exports':[]})
            (root/'Data/README.md').write_text('# Data\n'); (root/'Data/dataset/README.md').write_text('Текущий состав: старый\n')
            skip=item(runner.USER_SKIP_ID); queue=[skip,item()]; url=skip['url']; runner.write(out/'queue.json',queue)
            runner.write(out/'listing.json',{'selected_controls':courier_controls()})
            runner.write(service/'plan.json',{'setup_seconds':1,'profession':'Курьер','segments':'all'})
            cp=dict(state='REVIEW_REQUIRED',stop_reason='identity_or_template_review',next_id=runner.USER_SKIP_ID,completed_ids=[],capture_seconds=0.5)
            runner.write(diag/'checkpoint.json',cp); runner.write(sandbox/'checkpoint.json',cp); runner.write(diag/'sandbox.json',{'path':str(sandbox)}); runner.write(sandbox/'browser-tab.json',{'url':url})
            observation=dict(state='UNKNOWN_PAGE',source_url=url,heading=None,items=[],unresolved_evidence=dict(complete=False,restriction=False,known_unavailable=False))
            runner.write(out/f'raw/{runner.USER_SKIP_ID}.json',observation); runner.write(sandbox/f'raw/{runner.USER_SKIP_ID}.json',observation)
            snap=f'- title: "Авито" [url={url}]\n'; (diag/'problem.snapshot.txt').write_text(snap); (sandbox/'problem.snapshot.txt').write_text(snap)
            runner.record_failure(diag,'capture','stopped','capture',immediate=True)
            runner.mark_user_skipped(argparse.Namespace(run_id=runner.USER_SKIP_RUN,id=runner.USER_SKIP_ID,human_instruction='Пропустить 8557245056',authorization_id='skip-1'))
            runner.write(out/'raw/123.json',raw())
            placements=root/'placements.json'; runner.write(placements,{'123':{'city':'Коммунарка','region':'Москва','evidence':'пос. Коммунарка'}})
            runner.finalize(argparse.Namespace(run_id=runner.USER_SKIP_RUN,placements=placements,review_model='test'))
            manifest=runner.read(service/'manifest.json'); validation=runner.read(out/'exports/validation.json')
            self.assertEqual(manifest['coverage']['user_skipped_count'],1)
            self.assertEqual(manifest['coverage']['user_skipped_observations'][0]['id'],runner.USER_SKIP_ID)
            self.assertEqual(validation['accepted'],1)
            with (out/'exports/vacancies_flat.csv').open(encoding='utf-8-sig',newline='') as stream:
                self.assertEqual([row['id'] for row in csv.DictReader(stream)],['123'])

    def test_deduplicate_queue_empty_mixed_and_tampering(self):
        original_root = runner.ROOT
        for all_duplicate in (True, False):
            with self.subTest(all_duplicate=all_duplicate), tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
                root=runner.ROOT
                for file in runner.AUTOMATION_FILES:
                    dest=root/file; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((original_root/file).read_bytes())
                def freeze(name, identities):
                    out,service,diag=runner.paths(root,name)
                    page=dict(state='LISTING',source_url='https://www.avito.ru/moskva/vakansii',selected_controls=[],items=[item(str(i)) for i in identities])
                    runner.write(out/'listing.json',page);runner.write(out/'listing_pages/0001.json',page)
                    queue=runner.queue_from_pages([page],None);runner.write(out/'queue.json',queue)
                    plan=dict(run_id=name,profession='',profession_scope='all',setup_seconds=1,segments='',selection_mode='first_N_unique_on_frozen_listing',listing_sha256=runner.sha(out/'listing.json'),listing_pages=[dict(file='listing_pages\\0001.json',sha256=runner.sha(out/'listing_pages/0001.json'))])
                    runner.write(service/'plan.json',plan);runner.write(diag/'checkpoint.json',dict(state='READY',completed_ids=[],next_id=queue[0]['id']))
                    runner.save_failures(diag,runner.load_failures(diag))
                    return out,service,diag,queue
                identities=list(range(1,907)); known=identities if all_duplicate else identities[:-2]
                source,source_service,source_diag,queue=freeze('source',identities);known_out,known_service,_,_=freeze('known',known)
                original_ledger=(source_diag/'failures.json').read_bytes(); args=argparse.Namespace(source_run_id='source',run_id='child',against_run_id=['known'])
                with patch.object(runner.subprocess,'check_output',return_value='head'), patch.object(runner,'dirty_automation',return_value=[]): runner.deduplicate_queue(args)
                out,service,diag=runner.paths(root,'child'); child=runner.read(out/'queue.json'); info=runner.read(out/'dedup_selection.json')
                self.assertEqual((info['source_count'],info['duplicate_excluded'],info['novel_count']), (906,906 if all_duplicate else 904,0 if all_duplicate else 2))
                self.assertEqual(child, [] if all_duplicate else queue[-2:]); self.assertEqual((source_diag/'failures.json').read_bytes(),original_ledger)
                runner.validate_partition(out,service)
                (service/'shared-ledger.json').write_bytes((diag/'shared-ledger.json').read_bytes())
                (diag/'shared-ledger.json').unlink()
                runner.validate_partition(out,service)
                with self.assertRaisesRegex(runner.HumanRequired, 'Local shared collection ledger missing'):
                    runner.ensure_allowed(diag, 'capture')
                (diag/'shared-ledger.json').write_bytes((service/'shared-ledger.json').read_bytes())
                with self.assertRaisesRegex(ValueError,'already exists'): runner.deduplicate_queue(args)
                if all_duplicate:
                    self.assertEqual(runner.read(out/'dedup-empty.json')['state'],'DEDUP_EMPTY')
                    for function in (runner.capture,runner.finalize,runner.publish): function(argparse.Namespace(run_id='child'))
                    self.assertEqual((source_diag/'failures.json').read_bytes(),original_ledger)
                else:
                    runner.write(root/'Data/dataset/sources.json',{'schema_version':'1','exports':[]})
                    (root/'Data/README.md').write_text('# Data\n',encoding='utf8');(root/'Data/dataset/README.md').write_text('Текущий состав: old\n',encoding='utf8')
                    def browser(job, output, diagnostics):
                        for candidate in job['queue']: runner.write(output/'raw'/f"{candidate['id']}.json",raw(candidate['id']))
                        return dict(state='CAPTURED',completed_ids=[candidate['id'] for candidate in job['queue']],next_id=None,stop_reason=None), 1
                    with patch.object(runner,'preflight'), patch.object(runner,'browser',side_effect=browser), patch.object(runner,'release_browser'):
                        runner.capture(argparse.Namespace(run_id='child',batch_size=2,retry_reason=None,tab_id=None,config=None))
                    placements=root/'placements.json';runner.write(placements,{i['id']:{'city':'Коммунарка','region':'Москва','evidence':'пос. Коммунарка'} for i in child})
                    runner.finalize(argparse.Namespace(run_id='child',placements=placements,review_model='test'))
                    manifest=runner.read(service/'manifest.json')
                    self.assertEqual(manifest['coverage']['accepted'],2)
                    self.assertEqual(manifest['coverage']['deduplication']['duplicate_excluded'],904)
                known_bytes=(known_out/'queue.json').read_bytes(); (known_out/'queue.json').write_bytes(known_bytes+b' ')
                with self.assertRaisesRegex(ValueError,'known provenance'): runner.validate_partition(out,service)
                (known_out/'queue.json').write_bytes(known_bytes)
                source_bytes=(source/'queue.json').read_bytes(); (source/'queue.json').write_bytes(source_bytes+b' ')
                with self.assertRaisesRegex(ValueError,'source provenance'): runner.validate_partition(out,service)
                (source/'queue.json').write_bytes(source_bytes)

    def test_deduplicate_queue_rejects_source_after_capture(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            out,service,diag=runner.paths(runner.ROOT,'source'); runner.write(out/'raw/123.json',raw()); runner.write(diag/'checkpoint.json',dict(state='READY',completed_ids=['123']))
            args=argparse.Namespace(source_run_id='source',run_id='child',against_run_id=['known'])
            with patch.object(runner,'verified_frozen_queue',return_value=({},[item()])):
                with self.assertRaisesRegex(ValueError,'before any capture'): runner.deduplicate_queue(args)
            self.assertFalse(runner.paths(runner.ROOT,'child')[0].exists())

    def test_branch_only_freeze_caps_at_30_pages(self):
        original_root=runner.ROOT
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            for file in runner.AUTOMATION_FILES:
                dest=runner.ROOT/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((original_root/file).read_bytes())
            counter={'page':0}
            def browser(job,out,diag):
                counter['page']+=1;n=counter['page']
                page=dict(state='LISTING',source_url=f'https://www.avito.ru/moskva/vakansii?p={n}',heading='Работа',
                    observed_at=f'2026-10-30T00:00:{n:02d}Z',selected_controls=[],profession_filter={'observed':False,'selected_count':0},
                    items=[item(str(1000+n))],listing_item_audit={'malformed':0},
                    pagination={'current_pages':[str(n)],'next_url':f'https://www.avito.ru/moskva/vakansii?p={n+1}','end_confirmed':False})
                runner.write(diag/'listing.json',page)
                return {'state':'READY','completed_ids':[]},0.01
            args=argparse.Namespace(run_id='sweep_branch',url='https://www.avito.ru/moskva/vakansii?p=1',limit=None,candidate_limit=None,
                all_pages=True,branch_only=True,max_pages=30,geography='Москва',profession='',profession_scope='all',segments='',
                filter_evidence='base',ui_next=False,tab_id=None,resume=False,accept_saved=False,human_instruction=None,
                agent_model='test',config=None,retry_reason=None)
            with patch.object(runner,'ensure_allowed'),patch.object(runner,'preflight'),patch.object(runner,'browser',side_effect=browser),patch.object(runner,'release_browser'),patch.object(runner,'dirty_automation',return_value=[]),patch.object(runner.subprocess,'check_output',return_value='head'):
                runner.freeze(args)
            out,service,_=runner.paths(runner.ROOT,'sweep_branch');plan=runner.read(service/'plan.json')
            self.assertEqual((len(plan['listing_pages']),plan['listing_end_reason'],len(runner.read(out/'queue.json'))),(30,'page_cap_30',30))

    def test_branch_only_freeze_accepts_verified_empty_result(self):
        original_root=runner.ROOT
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            for file in runner.AUTOMATION_FILES:
                dest=runner.ROOT/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((original_root/file).read_bytes())
            def browser(job,out,diag):
                page=dict(state='EMPTY_RESULT',source_url=job['url'],heading='Ничего не найдено',observed_at='2026-10-30T00:00:00Z',
                    selected_controls=[],profession_filter={'observed':False,'selected_count':0},items=[],empty_evidence='Ничего не найдено')
                runner.write(diag/'listing.json',page)
                return {'state':'READY','completed_ids':[]},0.01
            args=argparse.Namespace(run_id='empty_branch',url='https://www.avito.ru/moskva/vakansii',limit=None,candidate_limit=None,
                all_pages=True,branch_only=True,max_pages=30,geography='Москва',profession='',profession_scope='all',segments='',
                filter_evidence='base',ui_next=False,tab_id=None,resume=False,accept_saved=False,human_instruction=None,
                agent_model='test',config=None,retry_reason=None)
            with patch.object(runner,'ensure_allowed'),patch.object(runner,'preflight'),patch.object(runner,'browser',side_effect=browser),patch.object(runner,'release_browser'),patch.object(runner,'dirty_automation',return_value=[]),patch.object(runner.subprocess,'check_output',return_value='head'):
                runner.freeze(args)
            out,service,diag=runner.paths(runner.ROOT,'empty_branch')
            self.assertEqual(runner.read(out/'queue.json'),[])
            self.assertEqual(runner.read(service/'plan.json')['listing_end_reason'],'empty_result')
            self.assertEqual(runner.read(diag/'checkpoint.json')['state'],'EMPTY_RESULT')
            self.assertTrue((service/'empty-listing.json').exists())

    def test_branch_only_freeze_detects_repeated_page(self):
        original_root=runner.ROOT
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            for file in runner.AUTOMATION_FILES:
                dest=runner.ROOT/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((original_root/file).read_bytes())
            def browser(job,out,diag):
                page=dict(state='LISTING',source_url=job['url'],heading='Работа',observed_at='2026-10-30T00:00:00Z',
                    selected_controls=[],profession_filter={'observed':False,'selected_count':0},items=[item('123')],listing_item_audit={'malformed':0},
                    pagination={'current_pages':['1'],'next_url':job['url'],'end_confirmed':False})
                runner.write(diag/'listing.json',page);return {'state':'READY','completed_ids':[]},0.01
            args=argparse.Namespace(run_id='repeat_branch',url='https://www.avito.ru/moskva/vakansii',limit=None,candidate_limit=None,
                all_pages=True,branch_only=True,max_pages=30,geography='Москва',profession='',profession_scope='all',segments='',
                filter_evidence='base',ui_next=False,tab_id=None,resume=False,accept_saved=False,human_instruction=None,
                agent_model='test',config=None,retry_reason=None)
            with patch.object(runner,'ensure_allowed'),patch.object(runner,'preflight'),patch.object(runner,'browser',side_effect=browser),patch.object(runner,'release_browser'),patch.object(runner,'dirty_automation',return_value=[]),patch.object(runner.subprocess,'check_output',return_value='head'):
                runner.freeze(args)
            _,service,_=runner.paths(runner.ROOT,'repeat_branch')
            self.assertEqual(runner.read(service/'plan.json')['listing_end_reason'],'repeated_next_link')

    def test_partition_url_guard_requires_explicit_audit(self):
        guard = 'Scraper/skills/collect-avito/scripts/browser_batch.js'
        plan = {'partition': {'index': 3}}
        self.assertNotIn(guard, runner.allowed_capture_changes(plan, []))
        self.assertNotIn(guard, runner.allowed_capture_changes(plan, [{'file':guard, 'reason':'other', 'validation':'passed'}]))
        self.assertIn(guard, runner.allowed_capture_changes(plan, [{'file':guard, 'reason':'Audited URL identity guard', 'validation':'URL guard regression passed'}]))
        self.assertNotIn(guard, runner.allowed_capture_changes({}, [{'file':guard, 'reason':'Audited URL identity guard', 'validation':'passed'}]))
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            original = b"      const ready = data.state === 'DETAIL' && data.id === item.id &&\n"
            target = runner.ROOT/guard; target.parent.mkdir(parents=True)
            target.write_bytes(original + b'        safeUrl(p.url()) === safeUrl(item.url) && safeUrl(data.source_url) === safeUrl(item.url) &&\n')
            plan.update(code_git_commit='frozen', automation_sha256={guard:hashlib.sha256(original).hexdigest()})
            with patch.object(runner.subprocess, 'check_output', return_value=original):
                runner.verify_partition_url_guard(plan, {guard})
                target.write_bytes(target.read_bytes() + b'other change')
                with self.assertRaises(ValueError): runner.verify_partition_url_guard(plan, {guard})

    def test_browser_sources_preserve_russian_with_legacy_windows_encoding(self):
        env = dict(os.environ, PYTHONUTF8='0', PYTHONIOENCODING='utf-8')
        code = """import importlib.util,sys
spec=importlib.util.spec_from_file_location('r',sys.argv[1]); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
batch,extractor=r.browser_sources()
assert '\u0421\u043b\u0435\u0434\u0443\u044e\u0449\u0430\u044f \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430' in batch
assert '\u0421\u043b\u0435\u0434\u0443\u044e\u0449\u0430\u044f \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430' in extractor
"""
        subprocess.run([sys.executable, '-c', code, str(SCRIPTS/'run.py')], env=env, check=True)

    def test_ui_next_waits_for_delayed_pagination(self):
        script = (SCRIPTS/'browser_batch.js').read_text()
        job = {'mode':'freeze','url':'https://www.avito.ru/all/vakansii', 'completed_ids':[], 'tab_id':'own', 'ui_next':True}
        body = script.replace('__JOB__', json.dumps(job), 1).replace('__EXTRACTOR__', '"extract"', 1)
        with tempfile.TemporaryDirectory() as directory:
            harness = """const fs=require('fs').promises, pwd=process.cwd(); let calls=0;
const pg={url:()=> 'https://www.avito.ru/all/vakansii',goto:async()=>{},evaluate:async()=>{calls++;return {state:'LISTING',items:[{id:'1'}],listing_item_audit:{malformed:0},pagination:{current_pages:['1'],next_url:calls>=5?'https://www.avito.ru/all/vakansii?p=2':null,end_confirmed:false}}}};
const attachBrowserTab=async()=>pg, snapshot=async()=>({tree:''}), sleep=async()=>{};
(async()=>{BODY; console.log('CALLS='+calls)})().catch(e=>{console.error(e);process.exit(1)});
""".replace('BODY',body)
            result = subprocess.run(['node','-e',harness],cwd=directory,capture_output=True,text=True,check=True)
            self.assertGreaterEqual(int(result.stdout.split('CALLS=')[-1]), 7)
            page = json.loads((Path(directory)/'listing.json').read_text(encoding='utf-8'))
            self.assertTrue(page['pagination']['next_url'])

    def test_direct_freeze_waits_for_complete_controls_and_stops_on_protection(self):
        script = (SCRIPTS/'browser_batch.js').read_text()
        job = {'mode':'freeze','url':'https://www.avito.ru/all/vakansii', 'completed_ids':[], 'tab_id':'own', 'ui_next':False}
        body = script.replace('__JOB__', json.dumps(job), 1).replace('__EXTRACTOR__', '"extract"', 1)
        for scenario in ('delayed', 'short', 'missing', 'protection'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                harness = """const fs=require('fs').promises, pwd=process.cwd(); let calls=0; const scenario=SCENARIO;
const pg={url:()=> 'https://www.avito.ru/all/vakansii',evaluate:async()=>{calls++; if(scenario==='protection' && calls===2) return {state:'NEEDS_HUMAN'}; const ready=scenario==='short' || (scenario==='delayed' && calls>=5); return {state:'LISTING',items:[{id:'1'}],listing_item_audit:{malformed:0},selected_controls:ready?[{label:'Удалённо'}]:[],profession_filter:{observed:ready},pagination:{current_pages:ready&&scenario!=='short'?['1']:[],next_url:ready&&scenario!=='short'?'https://www.avito.ru/all/vakansii?p=2':null,end_confirmed:scenario==='short'}}}};
const attachBrowserTab=async()=>pg, snapshot=async()=>({tree:''}), sleep=async()=>{};
(async()=>{BODY; console.log('CALLS='+calls)})().catch(e=>{console.error(e);process.exit(1)});
""".replace('SCENARIO',json.dumps(scenario)).replace('BODY',body)
                result = subprocess.run(['node','-e',harness],cwd=directory,capture_output=True,text=True,check=True)
                calls = int(result.stdout.split('CALLS=')[-1])
                checkpoint = json.loads((Path(directory)/'checkpoint.json').read_text(encoding='utf-8'))
                if scenario in ('delayed', 'short'):
                    self.assertGreaterEqual(calls, 7 if scenario == 'delayed' else 3)
                    self.assertEqual(checkpoint['state'], 'READY')
                else:
                    self.assertEqual(checkpoint['state'], 'NEEDS_HUMAN' if scenario == 'protection' else 'BROWSER_ERROR')
                    self.assertEqual(calls, 2 if scenario == 'protection' else 30)

    def test_capture_rejects_blank_or_mismatched_actual_url(self):
        script = (SCRIPTS/'browser_batch.js').read_text()
        candidate = item('123'); job = {'mode':'capture','queue':[candidate],'completed_ids':[],'close_tab':False}
        body = script.replace('__JOB__', json.dumps(job), 1).replace('__EXTRACTOR__', '"extract"', 1)
        for actual, source, expected in [('about:blank', candidate['url'], 'REVIEW_REQUIRED'), (candidate['url'], 'about:blank', 'REVIEW_REQUIRED'), (candidate['url'], candidate['url'], 'CAPTURED')]:
            with self.subTest(actual=actual, source=source), tempfile.TemporaryDirectory() as directory:
                harness = """const fs=require('fs').promises, pwd=process.cwd();
const pg={url:()=>ACTUAL,evaluate:async()=>({state:'DETAIL',id:'123',title:'Job',source_url:SOURCE,parameter_audit:{rendered_complete:true},fields:[{name:'x',value:'y'}]})};
const listBrowserTabs=async()=>[], openTab=async()=>pg, snapshot=async()=>({tree:'snapshot'});
(async()=>{BODY; console.log('RESULT='+await fs.readFile('checkpoint.json','utf8'))})().catch(e=>{console.error(e);process.exit(1)});
""".replace('ACTUAL',json.dumps(actual)).replace('SOURCE',json.dumps(source)).replace('BODY',body)
                result = subprocess.run(['node','-e',harness],cwd=directory,capture_output=True,text=True,check=True)
                self.assertEqual(json.loads(result.stdout.split('RESULT=')[-1])['state'],expected)

    def test_partial_suffix_provenance_rejects_queue_tamper_and_wrong_ledger(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'derived'); prov = out/'partial-provenance'
            queue = [item('123'), item('456'), item('789')]
            runner.write(prov/'plan.json', {'partition': {'parent_run_id': 'parent'}})
            runner.write(prov/'queue.json', queue)
            selection = {'accepted_ids':['123'], 'skipped_ids':[], 'unresolved_ids':['456','789'], 'plan_sha256':runner.sha(prov/'plan.json'), 'queue_sha256':runner.sha(prov/'queue.json')}
            runner.write(prov/'selection.json', selection); runner.write(prov/'manifest.json', {'coverage':{'partial':selection}})
            info = {'ledger_parent_run_id':'parent', 'source_hashes':{n:runner.sha(prov/n) for n in ('plan.json','queue.json','selection.json','manifest.json')}}
            runner.write(service/'plan.json', {'partial_suffix':info}); runner.write(diag/'shared-ledger.json', {'parent_run_id':'parent'}); runner.write(out/'queue.json', queue[1:])
            with patch.object(runner, 'validate_plan_profession'):
                runner.validate_partition(out, service)
                runner.write(out/'queue.json', [queue[2]])
                with self.assertRaises(ValueError): runner.validate_partition(out, service)
                runner.write(out/'queue.json', queue[1:]); runner.write(diag/'shared-ledger.json', {'parent_run_id':'other'})
                with self.assertRaises(ValueError): runner.validate_partition(out, service)
                runner.write(diag/'shared-ledger.json', {'parent_run_id':'parent'}); runner.write(prov/'selection.json', dict(selection, unresolved_ids=['789']))
                with self.assertRaises(ValueError): runner.validate_partition(out, service)

    def test_diagnostic_authorization_keeps_budget_and_rejects_replay(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'suffix')
            runner.write(service/'plan.json', {'partial_suffix': {'source_run_id':'partial'}})
            runner.write(out/'queue.json', [item('123')]); runner.write(diag/'checkpoint.json', {'next_id':'123','completed_ids':[]})
            runner.write(diag/'failures.json', {'state':'READY','failures':4,'events':[],'retry_limit':1,'retries_started':0,'budget_start_failures':4})
            args = argparse.Namespace(run_id='suffix', authorization_id='auth', human_instruction='Одна диагностика ID 123')
            with patch.object(runner, 'validate_partition'):
                runner.diagnostic_authorization(args)
                with self.assertRaises(ValueError): runner.diagnostic_authorization(args)
            ledger = runner.read(diag/'failures.json')
            self.assertEqual((ledger['failures'], ledger['retries_started'], ledger['budget_start_failures']), (4,0,4))
            self.assertEqual(ledger['events'][-1]['authorization_id'], 'auth')

    def test_partial_partition_prefix_preserves_queue_and_rejects_gaps_and_bad_raw(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'part')
            queue = [item('123'), item('456'), item('789')]
            runner.write(out/'queue.json', queue); runner.write(service/'plan.json', {'partition': {'index': 1}})
            runner.write(out/'raw/123.json', raw('123')); runner.write(out/'raw/456.json', raw('456'))
            cp = dict(state='REVIEW_REQUIRED', next_id='789', completed_ids=['123', '456'], stop_reason='identity_or_template_review')
            runner.write(diag/'checkpoint.json', cp)
            before = ((service/'plan.json').read_bytes(), (out/'queue.json').read_bytes())
            with patch.object(runner, 'check', return_value={'structural_check_passed': True}):
                selection = runner.partial_prefix_selection(out, service, queue, 'publish prefix')
                self.assertEqual(selection['accepted_ids'], ['123', '456'])
                self.assertEqual(selection['unresolved_ids'], ['789'])
                runner.write(diag/'checkpoint.json', dict(cp, completed_ids=['456']))
                with self.assertRaises(ValueError): runner.partial_prefix_selection(out, service, queue, 'publish prefix')
                runner.write(diag/'checkpoint.json', cp)
                (out/'raw/456.json').unlink()
                with self.assertRaises((ValueError, FileNotFoundError)): runner.partial_prefix_selection(out, service, queue, 'publish prefix')
                runner.write(out/'raw/456.json', dict(raw('456'), parameter_audit={'rendered_complete': False}))
                with self.assertRaises(ValueError): runner.partial_prefix_selection(out, service, queue, 'publish prefix')
            self.assertEqual(before, ((service/'plan.json').read_bytes(), (out/'queue.json').read_bytes()))

    def test_ordinary_partial_prefix_with_verified_skips_and_suffix_guard(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
            out,service,diag=runner.paths(runner.ROOT,'run');queue=[item(str(i)) for i in range(1,6)]
            runner.write(out/'queue.json',queue);runner.write(service/'plan.json',{'selection_mode':'first_N_unique_on_frozen_listing'})
            for i in range(1,5):runner.write(out/f'raw/{i}.json',raw(str(i)))
            cp=dict(state='NEEDS_HUMAN',next_id='5',completed_ids=['1','3'],stop_reason='access_restriction_or_captcha');runner.write(diag/'checkpoint.json',cp)
            with patch.object(runner,'unavailable_records',return_value=[{'id':'2'}]),patch.object(runner,'unresolved_records',return_value=[]),patch.object(runner,'user_skipped_records',return_value=[{'id':'4'}]):
                with self.assertRaisesRegex(ValueError,'instruction'):runner.partial_prefix_selection(out,service,queue,'')
                result=runner.partial_prefix_selection(out,service,queue,'Publish verified prefix')
                self.assertEqual(result['method'],'ordinary_prefix_v1');self.assertEqual(result['accepted_ids'],['1','3']);self.assertEqual(result['skipped_ids'],['2']);self.assertEqual(result['user_skipped_ids'],['4']);self.assertEqual(result['unresolved_ids'],['5']);self.assertEqual(set(result['raw_sha256']),{'1','2','3','4'})
                runner.write(diag/'checkpoint.json',dict(cp,completed_ids=['1']))
                with self.assertRaisesRegex(ValueError,'queue prefix'):runner.partial_prefix_selection(out,service,queue,'Publish verified prefix')
                runner.write(diag/'checkpoint.json',cp);runner.write(out/'raw/5.json',raw('5'))
                with self.assertRaisesRegex(ValueError,'Unresolved raw'):runner.partial_prefix_selection(out,service,queue,'Publish verified prefix')

    def test_archive_partial_pending_keeps_protection_and_evidence(self):
        for scenario in ('ok','no_authorization','bad_snapshot'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
                out,_,diag=runner.paths(runner.ROOT,'run');sandbox=runner.ROOT/'sandbox';queue=[item('123')];url=queue[0]['url'];runner.write(out/'queue.json',queue)
                cp=dict(state='NEEDS_HUMAN',stop_reason='access_restriction_or_captcha',next_id='123',completed_ids=[]);runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp);runner.write(diag/'sandbox.json',{'path':str(sandbox)});runner.write(sandbox/'browser-tab.json',{'url':url})
                data=dict(state='NEEDS_HUMAN',reason='access_restriction_or_captcha',source_url=url,items=[]);runner.write(out/'raw/123.json',data);runner.write(sandbox/'raw/123.json',data)
                snap=f'- title: "Доступ ограничен: проблема с IP" [url={url}]\n';(diag/'problem.snapshot.txt').write_text(snap);(sandbox/'problem.snapshot.txt').write_text(snap+('changed' if scenario=='bad_snapshot' else ''))
                runner.record_failure(diag,'capture','restricted','capture',immediate=True)
                if scenario!='no_authorization':runner.authorize_offline(diag,'Publish verified prefix')
                args=argparse.Namespace(run_id='run',id='123',human_instruction='Publish verified prefix');original=(out/'raw/123.json').read_bytes();before=runner.load_failures(diag)
                if scenario!='ok':
                    with self.assertRaises((ValueError,runner.HumanRequired)):runner.archive_partial_pending(args)
                    self.assertEqual((out/'raw/123.json').read_bytes(),original);continue
                runner.archive_partial_pending(args);digest=hashlib.sha256(original).hexdigest()
                self.assertEqual((out/'observations/123'/f'{digest}.json').read_bytes(),original);self.assertFalse((out/'raw/123.json').exists())
                after=runner.load_failures(diag);self.assertEqual(after,before);self.assertEqual(runner.read(diag/'checkpoint.json'),cp)

    def test_archive_partial_pending_cli_parses_only_its_arguments(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            argv=['run.py','archive-partial-pending','--run-id','run','--id','123','--human-instruction','Publish verified prefix']
            with patch('sys.argv',argv),patch.object(runner,'archive_partial_pending') as action:
                runner.main();self.assertEqual(action.call_args.args[0].id,'123')

    def test_ordinary_partial_finalize_and_validate_excludes_skipped_rows(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            root=runner.ROOT;out,service,diag=runner.paths(root,'run');queue=[item('123'),item('456'),item('789')]
            runner.write(root/'Data/dataset/sources.json',{'schema_version':'1','exports':[]});(root/'Data/README.md').write_text('# Data\n');(root/'Data/dataset/README.md').write_text('Текущий состав: старый\n')
            runner.write(out/'queue.json',queue);runner.write(out/'listing.json',{'selected_controls':courier_controls()});runner.write(out/'raw/123.json',raw('123'));runner.write(out/'raw/456.json',raw('456'))
            runner.write(service/'plan.json',{'setup_seconds':1,'profession':'Курьер','segments':'all','selection_mode':'first_N_unique_on_frozen_listing'})
            runner.write(diag/'checkpoint.json',dict(state='NEEDS_HUMAN',stop_reason='access_restriction_or_captcha',next_id='789',completed_ids=['123'],capture_seconds=1))
            runner.record_failure(diag,'capture','restricted','capture',immediate=True);runner.authorize_offline(diag,'Publish verified prefix')
            placements=root/'placements.json';runner.write(placements,{'123':{'city':'Коммунарка','region':'Москва','evidence':'пос. Коммунарка'}})
            args=argparse.Namespace(run_id='run',placements=placements,review_model='test',partial_prefix=True,partial_instruction='Publish verified prefix')
            with patch.object(runner,'user_skipped_records',return_value=[{'id':'456'}]):
                with self.assertRaises(runner.HumanRequired):runner.finalize(argparse.Namespace(**{**vars(args),'partial_instruction':'Different instruction'}))
                runner.finalize(args);manifest=runner.validate_completed_export(out,service)
                self.assertEqual(manifest['coverage']['partial']['user_skipped_ids'],['456']);self.assertEqual(manifest['coverage']['partial']['unresolved_ids'],['789'])
                with (out/'exports/vacancies_flat.csv').open(encoding='utf-8-sig',newline='') as stream:self.assertEqual([x['id'] for x in csv.DictReader(stream)],['123'])

    def test_instruction_version_tracks_skill_and_rejects_ambiguous_declarations(self):
        with tempfile.TemporaryDirectory() as directory:
            skill = Path(directory)/'SKILL.md'
            skill.write_text('Версия 1.7.29, 29.10.2026.\n', encoding='utf-8')
            self.assertEqual(runner.instruction_version(skill), '1.7.29')
            skill.write_text('Версия 1.7.30, 30.10.2026.\n', encoding='utf-8')
            self.assertEqual(runner.instruction_version(skill), '1.7.30')
            for content in ('No version\n', 'Версия 1.7.29, date\nВерсия 1.7.30, date\n'):
                skill.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    runner.instruction_version(skill)

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
                for call in run.call_args_list:
                    self.assertEqual(call.args[0][1:3], ['-X', 'utf8'])
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
            children = [runner.paths(runner.ROOT, f'child{i}') for i in range(1,8)]
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
                if marker == runner.USER_DELETED_MARKER:
                    self.assertEqual(runner.read(out/'unavailable.json')[0]['status'], 'unavailable_deleted')
                self.assertFalse((diag/'failures.json').exists())
            self.assertFalse(runner.closed_raw_matches({'state':'NEEDS_HUMAN','source_url':item()['url']},'123',item()['url'],runner.CLOSED_MARKERS[0]))
            self.assertFalse(runner.closed_raw_matches({'state':'UNKNOWN_PAGE','heading':'Объявление не посмотреть','source_url':'https://wrong.example'},'123',item()['url'],runner.CLOSED_MARKERS[0]))
            self.assertFalse(runner.closed_raw_matches({'state':'UNKNOWN_PAGE','heading':'Объявление не посмотреть','source_url':item()['url']},'123',item()['url'],'CAPTCHA'))

    def test_rejected_listing_requires_exact_evidence_and_preserves_previous_policy(self):
        for scenario in ('ok','no_flag','wrong_title','restriction','sandbox_tamper','wrong_id'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, patch.object(runner,'ROOT',Path(directory)):
                root=runner.ROOT;out,service,diag=runner.paths(root,'case'); sandbox=root/'sandbox'; candidate=item('123');url=candidate['url']
                runner.write(out/'queue.json',[candidate]);runner.write(service/'plan.json',{})
                cp={'state':'REVIEW_REQUIRED','stop_reason':'identity_or_template_review','next_id':'123','completed_ids':[]};runner.write(diag/'checkpoint.json',cp);runner.write(sandbox/'checkpoint.json',cp)
                data={'state':'UNKNOWN_PAGE','source_url':url,'heading':'Объявление не посмотреть','unresolved_evidence':{'complete':True,'title_body_params_scanned':True,'restriction':scenario=='restriction'}}
                runner.write(out/'raw/123.json',data);runner.write(sandbox/'raw/123.json',data);original=(out/'raw/123.json').read_bytes()
                runner.write(diag/'sandbox.json',{'path':str(sandbox)});runner.write(sandbox/'browser-tab.json',{'url':url})
                title='Другая страница' if scenario=='wrong_title' else 'Ошибка 404. Страница не найдена'
                snap=f'- title: "{title}" [url={url}]\n- text: "{runner.REJECTED_MARKER}"\n';(diag/'problem.snapshot.txt').write_text(snap);(sandbox/'problem.snapshot.txt').write_text(snap+('x' if scenario=='sandbox_tamper' else ''))
                policy={'authorization_id':runner.STANDING_UNAVAILABLE_ID,'human_instruction':runner.STANDING_UNAVAILABLE_INSTRUCTION,'markers':list(runner.PREVIOUS_UNAVAILABLE_MARKERS)};runner.write(out/'standing_unavailable_policy.json',policy)
                ledger=runner.load_failures(diag);ledger['events'].append({'event':'project_unavailable_policy','authorization_id':runner.STANDING_UNAVAILABLE_ID,'instruction':runner.STANDING_UNAVAILABLE_INSTRUCTION});runner.save_failures(diag,ledger)
                args=argparse.Namespace(run_id='case',id='999' if scenario=='wrong_id' else '123',authorization_id=runner.STANDING_UNAVAILABLE_ID,standing_rejected=scenario!='no_flag')
                if scenario=='ok':
                    runner.mark_unavailable(args);record=runner.read(out/'unavailable.json')[0];self.assertEqual(record['status'],'unavailable_rejected');self.assertEqual(record['evidence'],runner.REJECTED_MARKER);self.assertEqual(runner.read(out/'standing_unavailable_policy.json'),policy);self.assertEqual((out/'raw/123.json').read_bytes(),original)
                else:
                    with self.assertRaises((ValueError,runner.HumanRequired)):runner.mark_unavailable(args)
                    self.assertFalse((out/'unavailable.json').exists());self.assertEqual((out/'raw/123.json').read_bytes(),original)

    def test_legacy_standing_policy_is_preserved_and_cannot_authorize_new_marker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'legacy')
            args = argparse.Namespace(run_id='legacy', authorization_id=runner.STANDING_UNAVAILABLE_ID, human_instruction=runner.STANDING_UNAVAILABLE_INSTRUCTION)
            policy = dict(authorization_id=args.authorization_id, human_instruction=args.human_instruction, markers=list(runner.LEGACY_UNAVAILABLE_MARKERS))
            runner.write(out/'standing_unavailable_policy.json', policy)
            runner.standing_unavailable_policy(args)
            self.assertEqual(runner.read(out/'standing_unavailable_policy.json'), policy)
            candidate = item('123'); runner.write(out/'queue.json', [candidate]); runner.write(service/'plan.json', {})
            runner.write(diag/'checkpoint.json', {'state':'REVIEW_REQUIRED','stop_reason':'identity_or_template_review','next_id':'123','completed_ids':[]})
            runner.write(out/'raw/123.json', {'source_url':candidate['url'],'state':'UNKNOWN_PAGE','heading':'Объявление не посмотреть'})
            (diag/'problem.snapshot.txt').write_text(f'# title [url={candidate["url"]}]\n- text: "{runner.USER_DELETED_MARKER}"\n')
            with self.assertRaisesRegex(ValueError, 'outside this authorization'):
                runner.mark_unavailable(argparse.Namespace(run_id='legacy', id='123', authorization_id=args.authorization_id, automatic=True))
            self.assertEqual(runner.read(out/'standing_unavailable_policy.json'), policy)

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
const pg = {url:()=> 'https://www.avito.ru/moskva/vakansii?token=secret#x', goto:async()=>{navigated++}, evaluate:async()=>({state:STATE,selected_controls:[{label:'Любой'}],profession_filter:{observed:true},items:[{id:'1'}],listing_item_audit:{malformed:0},pagination:{current_pages:['1'],end_confirmed:true}})};
let opened=false;
const listBrowserTabs=async()=>opened?[{targetId:'new-id',url:pg.url()}]:[];
const openTab=async()=>{opened=true;return pg};
const attachBrowserTab=async id=>{if(id!=='external-id')throw Error('wrong ID');attached++;return pg};
const snapshot=async()=>({tree:'stopped'});const closeTab=async()=>{closed++}; const sleep=async()=>{};
(async()=>{BODY; const result=JSON.parse(await fs.readFile('browser-tab.json')); console.log('RESULT='+JSON.stringify({closed,attached,navigated,result}));})().catch(e=>{console.error(e);process.exit(1)});
""".replace('STATE', json.dumps(state)).replace('BODY', body)
                result = subprocess.run(['node', '-e', harness], cwd=directory, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout.split('RESULT=')[-1])
                self.assertEqual(data['closed'], int(not external and state == 'LISTING'))
                self.assertEqual(data['attached'], int(external))
                self.assertEqual(data['navigated'], 0)
                self.assertNotIn('secret', data['result']['url'])
                self.assertEqual(data['result']['target_id'], 'external-id' if external else 'new-id')

    @unittest.skipIf(os.name == 'nt', 'POSIX pipe test')
    def test_owned_tab_close_is_acknowledged_by_its_repl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sandbox = root/'sandbox'; sandbox.mkdir()
            runner.write(sandbox/'browser-tab.json', {'ownership':'collector','target_id':'own-id','url':'https://example.test/a'})
            script = """const readline=require('readline');
const p={url:()=> 'https://example.test/a'};globalThis.tlCollectorPage=p;globalThis.tlCollectorExternal=false;
globalThis.listBrowserTabs=async()=>[{targetId:'own-id',url:p.url()}];
globalThis.getTabByTargetId=id=>id==='own-id'?p:null;
globalThis.closeTab=async tab=>{if(tab!==p)throw Error('wrong tab')};
readline.createInterface({input:process.stdin}).on('line',line=>{if(!line.startsWith('await eval('))throw Error('REPL must await');eval(line.replace(/^await /,''))});"""
            proc = subprocess.Popen(['node','-e',script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                runner.retain_repl(proc, root, sandbox, {})
                session = runner.read(root/'browser-session.local.json')
                proc.stdin.close(); proc.stdout.close()
                runner.release_browser(root)
                self.assertTrue(Path(session['release_marker']+'.closed').exists())
                self.assertTrue(Path(session['release_marker']).exists())
                proc.wait(timeout=5)
            finally:
                if proc.poll() is None: proc.kill(); proc.wait()

    @unittest.skipIf(os.name == 'nt', 'POSIX pipe test')
    def test_owned_tab_identity_mismatch_preserves_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); sandbox=root/'sandbox'; sandbox.mkdir()
            runner.write(sandbox/'browser-tab.json', {'ownership':'collector','target_id':'own-id','url':'https://example.test/a'})
            script = """const readline=require('readline');const p={url:()=> 'https://example.test/changed'};
              globalThis.tlCollectorPage=p;globalThis.tlCollectorExternal=false;globalThis.listBrowserTabs=async()=>[{targetId:'own-id',url:p.url()}];
              globalThis.getTabByTargetId=()=>p;globalThis.closeTab=async()=>{throw Error('must not close')};
              readline.createInterface({input:process.stdin}).on('line',line=>{if(!line.startsWith('await eval('))throw Error('REPL must await');eval(line.replace(/^await /,''))});"""
            proc=subprocess.Popen(['node','-e',script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            try:
                runner.retain_repl(proc,root,sandbox,{})
                session=runner.read(root/'browser-session.local.json'); proc.stdin.close(); proc.stdout.close()
                with patch.object(runner,'CLOSE_CONFIRM_TIMEOUT_SECONDS',.3):
                    with self.assertRaisesRegex(RuntimeError,'unconfirmed'): runner.release_browser(root)
                self.assertFalse(Path(session['release_marker']).exists())
                self.assertTrue((root/'browser-session.local.json').exists())
            finally:
                Path(session['release_marker']).touch()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()

    @unittest.skipIf(os.name == 'nt', 'POSIX pipe test')
    def test_owned_tab_without_initial_target_closes_original_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); sandbox=root/'sandbox'; sandbox.mkdir()
            runner.write(sandbox/'browser-tab.json', {'ownership':'collector','target_id':None,'url':'https://example.test/a'})
            script = """const readline=require('readline');const p={url:()=> 'https://example.test/a'};
              globalThis.tlCollectorPage=p;globalThis.tlCollectorExternal=false;globalThis.listBrowserTabs=async()=>[{targetId:'foreign',url:p.url()},{targetId:'own',url:p.url()}];
              globalThis.getTabByTargetId=id=>id==='own'?p:{};globalThis.closeTab=async tab=>{if(tab!==p)throw Error('foreign')};
              readline.createInterface({input:process.stdin}).on('line',line=>{if(!line.startsWith('await eval('))throw Error('REPL must await');eval(line.replace(/^await /,''))});"""
            proc=subprocess.Popen(['node','-e',script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            try:
                runner.retain_repl(proc,root,sandbox,{})
                session=runner.read(root/'browser-session.local.json'); proc.stdin.close(); proc.stdout.close()
                runner.release_browser(root)
                self.assertTrue(Path(session['release_marker']+'.closed').exists())
                proc.wait(timeout=5)
            finally:
                if proc.poll() is None: proc.kill(); proc.wait()

    def test_external_tab_release_never_requests_close(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); marker=root/'release'
            runner.write(root/'browser-session.local.json', {'release_marker':str(marker), 'tab':{'ownership':'external','target_id':'other','url':'https://example.test/a'}})
            runner.release_browser(root)
            self.assertTrue(marker.exists())
            self.assertFalse(Path(str(marker)+'.close').exists())

    def test_unidentified_owned_tab_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); marker=root/'release'
            runner.write(root/'browser-session.local.json', {'release_marker':str(marker), 'tab':{'ownership':'collector','target_id':None,'url':None}})
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                runner.release_browser(root)
            self.assertFalse(marker.exists())
            self.assertTrue((root/'browser-session.local.json').exists())

    def test_confirmed_404_requires_saved_matching_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); out=root/'out'; diag=root/'diag'; sandbox=root/'sandbox'
            (out/'raw').mkdir(parents=True); (sandbox/'raw').mkdir(parents=True); diag.mkdir()
            url='https://example.test/123'; raw=out/'raw'/'123.json'; runner.write(raw,{'source_url':url})
            (sandbox/'raw'/'123.json').write_bytes(raw.read_bytes())
            snap=diag/'problem.snapshot.txt'; snap.write_text(f'- title: "404" [url={url}]')
            original=sandbox/'problem.snapshot.txt'; original.write_bytes(snap.read_bytes())
            tab={'ownership':'collector','target_id':'own','url':url}; runner.write(sandbox/'browser-tab.json',tab)
            runner.write(diag/'browser-session.local.json',{'tab':tab,'sandbox':str(sandbox)})
            record={'id':'123','url':url,'raw_sha256':runner.sha(raw),'snapshot_sha256':runner.sha(snap)}
            with patch.object(runner,'release_browser') as release, patch.object(runner.shutil,'copy2',side_effect=OSError('disk error')):
                with self.assertRaisesRegex(OSError,'disk error'): runner.release_confirmed_404(out,diag,record)
                release.assert_not_called()
            with patch.object(runner,'release_browser') as release:
                self.assertTrue(runner.release_confirmed_404(out,diag,record)); release.assert_called_once_with(diag)
                archive=diag/'observations'/'123'
                self.assertEqual((archive/(runner.sha(raw)+'.raw.json')).read_bytes(),raw.read_bytes())
                self.assertEqual((archive/(runner.sha(snap)+'.problem.snapshot.txt')).read_bytes(),snap.read_bytes())
                self.assertEqual((archive/(runner.sha(sandbox/'browser-tab.json')+'.browser-tab.json')).read_bytes(),(sandbox/'browser-tab.json').read_bytes())
                snap.write_text(f'- title: "404" [url={url}]\n- text: "captcha"'); original.write_bytes(snap.read_bytes())
                record['snapshot_sha256']=runner.sha(snap)
                self.assertFalse(runner.release_confirmed_404(out,diag,record))
                snap.write_text(f'- title: "Other" [url={url}]'); original.write_bytes(snap.read_bytes())
                record['snapshot_sha256']=runner.sha(snap)
                self.assertFalse(runner.release_confirmed_404(out,diag,record))
                snap.write_text(f'- title: "404" [url={url}]'); original.write_bytes(snap.read_bytes())
                record['snapshot_sha256']=runner.sha(snap)
                tab['url']='https://example.test/other'; runner.write(diag/'browser-session.local.json',{'tab':tab,'sandbox':str(sandbox)})
                self.assertFalse(runner.release_confirmed_404(out,diag,record))
                self.assertEqual(release.call_count,1)

    def test_preserve_unreleased_session_before_next_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            diag=Path(directory); source=diag/'browser-session.local.json'
            source.write_bytes(b'{"release_marker":"old"}')
            original=source.read_bytes(); digest=hashlib.sha256(original).hexdigest()
            runner.preserve_browser_session_pointer(diag)
            archived=diag/f'browser-session-retained-{digest}.local.json'
            self.assertEqual(archived.read_bytes(),original)
            self.assertFalse(source.exists())
            source.write_bytes(original); runner.preserve_browser_session_pointer(diag)
            self.assertEqual(archived.read_bytes(),original)

    def test_release_keeps_previous_sessions_and_allows_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            legacy = diag/'browser-session-released.local.json'
            legacy.write_bytes(b'previous session')
            source = diag/'browser-session.local.json'
            session = {'repl_pid': 42, 'release_marker': str(diag/'release-42')}
            runner.write(source, session)
            original = source.read_bytes()
            runner.release_browser(diag)
            archived = diag/f'browser-session-released-{hashlib.sha256(original).hexdigest()}.local.json'
            self.assertEqual(archived.read_bytes(), original)
            self.assertEqual(legacy.read_bytes(), b'previous session')
            self.assertTrue((diag/'release-42').exists())
            source.write_bytes(original)  # retry after destination was already created
            runner.release_browser(diag)
            self.assertFalse(source.exists())
            self.assertEqual(archived.read_bytes(), original)
            runner.release_browser(diag)  # already released

    @unittest.skipIf(os.name == 'nt', 'POSIX terminal transport')
    def test_browser_ignores_prompt_echo_and_imports_complete_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sandbox, diag, out = root/'sandbox', root/'diagnostics', root/'out'
            sandbox.mkdir(); diag.mkdir(); out.mkdir()
            fake = root/'fake_repl.py'
            fake.write_text('''import json, pathlib, sys, time
sandbox = pathlib.Path(sys.argv[1])
line = sys.stdin.readline()
print('repl > ' + line, end='', flush=True)
time.sleep(.1)
print('TL_SESSION=' + str(sandbox), flush=True)
line = sys.stdin.readline()
print('repl > ' + line, end='', flush=True)
(sandbox/'received.txt').write_text(line)
(sandbox/'checkpoint.json').write_text(json.dumps({'state':'READY','completed_ids':[]}))
(sandbox/'listing.json').write_text(json.dumps({'items':list(range(50))}))
print('TL_FINISHED', flush=True)
sys.stdin.read()
''')
            job = {'mode': 'freeze', 'completed_ids': []}
            with patch.object(runner, 'find_aside', return_value='fake'), \
                    patch.object(runner, 'aside_command', return_value=[sys.executable, '-u', str(fake), str(sandbox)]):
                checkpoint, _ = runner.browser(job, out, diag)
            try:
                self.assertEqual(checkpoint['state'], 'READY')
                self.assertEqual(runner.read(diag/'sandbox.json')['path'], str(sandbox))
                self.assertEqual(len(runner.read(diag/'listing.json')['items']), 50)
                received = (sandbox/'received.txt').read_text()
                self.assertTrue(received.startswith('await eval('))
                self.assertTrue(received.endswith(')\n'))
                self.assertGreater(len(received), sum(len(s) for s in runner.browser_sources()))
            finally:
                runner.release_browser(diag)

    @unittest.skipIf(os.name == 'nt', 'POSIX terminal transport')
    def test_interactive_repl_accepts_large_line_before_eof_and_keeper_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            script = ('import sys; assert sys.stdin.isatty(); '
                      'line=sys.stdin.buffer.readline(); '
                      'print("LENGTH="+str(len(line)),flush=True); sys.stdin.buffer.read()')
            proc = runner.launch_repl([sys.executable, '-u', '-c', script], diag, interactive=True)
            try:
                payload = b'x' * (len(runner.browser_sources()[0].encode()) + len(runner.browser_sources()[1].encode())) + b'\n'
                proc.stdin.write(payload); proc.stdin.flush()
                chunks, reader = runner.pipe_chunks(proc.stdout, stoppable=True)
                output = chunks.get(timeout=10)
                self.assertEqual(output.strip(), f'LENGTH={len(payload)}'.encode())
                reader.stop_reading.set(); reader.join(timeout=2)
                self.assertFalse(reader.is_alive())
                runner.retain_repl(proc, diag, None, {})
                info = runner.read(diag/'browser-session.local.json')
                proc.stdin.close(); proc.stdout.close()
                self.assertIsNone(proc.poll())
                Path(info['release_marker']).touch()
                proc.wait(timeout=5)
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait()

    def test_keeper_lifecycle_with_reused_pid_and_old_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            stale = diag/'release-browser-session-12345'
            stale.write_bytes(b'original marker')
            legacy = diag/'browser-session-released.local.json'
            legacy.write_bytes(b'original session')
            markers = []
            for _ in range(2):
                proc = runner.launch_repl([sys.executable, '-u', '-c', 'import sys; print("READY",flush=True); sys.stdin.read()'], diag)
                class ReusedPid:
                    pid = 12345
                    def __getattr__(self, name): return getattr(proc, name)
                try:
                    self.assertEqual(proc.stdout.readline().strip(), b'READY')
                    runner.retain_repl(ReusedPid(), diag, None, {})
                    session = runner.read(diag/'browser-session.local.json')
                    marker = Path(session['release_marker'])
                    self.assertFalse(marker.exists())
                    markers.append(marker)
                    proc.stdin.close(); proc.stdout.close()
                    self.assertIsNone(proc.poll())
                    runner.release_browser(diag)
                    proc.wait(timeout=5)
                    self.assertTrue(marker.exists())
                finally:
                    if proc.poll() is None: proc.kill(); proc.wait()
            self.assertNotEqual(markers[0], markers[1])
            self.assertEqual(stale.read_bytes(), b'original marker')
            self.assertEqual(legacy.read_bytes(), b'original session')
            self.assertEqual(len(list(diag.glob('browser-session-released-*.local.json'))), 2)

    def test_launched_repl_is_detached_and_kept_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            proc = runner.launch_repl([sys.executable, '-u', '-c', 'import sys; print("REVIEW_REQUIRED", flush=True); sys.stdin.read()'], diag)
            try:
                if os.name != 'nt':
                    self.assertEqual(os.getsid(proc.pid), proc.pid)
                self.assertEqual(proc.stdout.readline().strip(), b'REVIEW_REQUIRED')
                runner.retain_repl(proc, diag, None, {})
                info = runner.read(diag/'browser-session.local.json')
                proc.stdin.close(); proc.stdout.close()
                self.assertIsNone(proc.poll())
                Path(info['release_marker']).touch()
                proc.wait(timeout=5)
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait()

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
            if os.name == 'nt':
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
                kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x100000 | 0x1000, False, info['repl_pid'])
                self.assertTrue(handle, ctypes.get_last_error())
                try:
                    exit_code = wintypes.DWORD()
                    self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
                    self.assertEqual(exit_code.value, 259)  # STILL_ACTIVE
                finally:
                    Path(info['release_marker']).touch()
                    self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
                    kernel.CloseHandle(handle)
            else:
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
                log = diag / f"keeper-{info['repl_pid']}.local.log"
                self.assertIn('release_marker_observed', log.read_text())
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
            runner.write(root/'Data/dataset/sources.json', {'exports':[{'run_id':run_id,'path':'Data/transformations/t/vacancies_flat.csv','transformation':{'path':'Data/transformations/t/transformation.json'}}]})
            state = {'commits': 0, 'pushes': 0, 'transform_staged': False}
            def output(command, **kwargs):
                if command[:2] == ['git','branch']:
                    return 'codex/publish_retry'
                if command[:5] == ['git','diff','--cached','--name-only']:
                    return (out/'raw/123.json').relative_to(root).as_posix() if not state['commits'] else ''
                if command[:2] == ['git','add']:
                    state['transform_staged'] = ('Data/transformations/t/vacancies_flat.csv' in command and 'Data/transformations/t/transformation.json' in command)
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
            with patch.object(runner,'ROOT',root), patch.object(runner.subprocess,'check_output',side_effect=output), patch.object(runner.subprocess,'run',side_effect=run), patch.object(runner,'stage_pinned_provenance'), patch.object(runner,'verify_staged_manifest'), patch.object(runner,'verify_staged_run_privacy'):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.publish(args)
                runner.record_failure(diag,'publish','push failed','push')
                args.retry_reason = 'Connection restored'
                runner.publish(args)
            self.assertEqual(state, {'commits':1,'pushes':2,'transform_staged':True})

    def test_publish_stages_pinned_provenance_bytes_and_rejects_changed_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); service = root/'Scraper/runs/run'; service.mkdir(parents=True)
            subprocess.run(['git','init','-q'], cwd=root, check=True)
            subprocess.run(['git','config','core.autocrlf','true'], cwd=root, check=True)
            plan = service/'plan.json'; plan.write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
            manifest_path = service/'manifest.json'; manifest_path.write_bytes(b'{\r\n}\r\n')
            data = root/'Data/item.json'; data.parent.mkdir(parents=True); data.write_bytes(b'{"id":1}\n')
            manifest = {'files': {p.relative_to(root).as_posix(): {'bytes':p.stat().st_size,'sha256':runner.sha(p)} for p in (plan,data)}}
            subprocess.run(['git','add','--','Data/item.json','Scraper/runs/run/plan.json','Scraper/runs/run/manifest.json'],cwd=root,check=True)
            runner.stage_pinned_provenance(root,service,manifest)
            runner.verify_staged_manifest(root,service,manifest)
            runner.check_staged_diff(root)  # CRLF is valid, not trailing whitespace
            bad = root/'notes.txt'; bad.write_bytes(b'bad space \r\n')
            subprocess.run(['git','-c','core.autocrlf=false','add','--','notes.txt'],cwd=root,check=True)
            with self.assertRaisesRegex(ValueError,'trailing whitespace'): runner.check_staged_diff(root)
            subprocess.run(['git','reset','-q','--','notes.txt'],cwd=root,check=True)
            self.assertEqual(subprocess.check_output(['git','show',':Scraper/runs/run/plan.json'],cwd=root),plan.read_bytes())
            original=plan.read_bytes(); plan.write_bytes(b'{\n  \"x\": 2\n}\n')
            subprocess.run(['git','add','--','Scraper/runs/run/plan.json'],cwd=root,check=True)
            plan.write_bytes(original)
            with self.assertRaisesRegex(ValueError,'Staged bytes differ'): runner.verify_staged_manifest(root,service,manifest)

    def test_publish_rejects_unrelated_staged_file_after_normalization(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'publish_guard')
            runner.write(service/'plan.json', {})
            runner.write(service/'manifest.json', dict(status='complete', files={}, collection={}, coverage={'accepted':1}))
            runner.write(runner.ROOT/'Data/dataset/sources.json', {'exports':[{'run_id':'publish_guard'}]})
            def output(command, **kwargs):
                if command[:2] == ['git', 'branch']: return 'codex/publish_guard'
                if command[:4] == ['git', 'diff', '--cached', '--name-only']: return 'unrelated/file.py'
                return ''
            args = argparse.Namespace(run_id='publish_guard', retry_reason=None, summary='Sample', title='Data', prepare_only=False)
            with patch.object(runner.subprocess,'check_output',side_effect=output), patch.object(runner.subprocess,'run'):
                with self.assertRaisesRegex(ValueError, 'unrelated changes'): runner.publish(args)

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

    def test_capture_reloads_checkpoint_after_interrupted_transfer_import(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'resume')
            queue = [item('123'), item('456')]
            runner.write(out / 'queue.json', queue)
            runner.write(service / 'plan.json', {
                'automation_sha256': {},
                'selection_mode': 'first_N_unique_on_frozen_listing',
            })
            runner.write(diag / 'checkpoint.json', {
                'state': 'READY', 'completed_ids': [], 'next_id': '123', 'capture_seconds': 0,
            })
            sandbox = Path(directory) / 'aside-session'
            runner.write(sandbox / 'raw/123.json', raw('123'))
            runner.write(sandbox / 'checkpoint.json', {
                'state': 'CAPTURED', 'completed_ids': ['123'], 'next_id': None,
                'capture_seconds': 1, 'stop_reason': None,
            })
            runner.write(diag / 'sandbox.json', {'path': str(sandbox)})
            seen = []

            def capture_remaining(job, output, diagnostics):
                seen.extend(candidate['id'] for candidate in job['queue'])
                runner.write(output / 'raw/456.json', raw('456'))
                return {
                    'state': 'CAPTURED', 'completed_ids': ['123', '456'],
                    'next_id': None, 'stop_reason': None,
                }, 1

            args = argparse.Namespace(run_id='resume', batch_size=30, config=None,
                                      retry_reason=None, tab_id=None)
            with patch.object(runner, 'validate_partition'), \
                    patch.object(runner, 'validate_plan_profession'), \
                    patch.object(runner, 'preflight'), \
                    patch.object(runner, 'browser', side_effect=capture_remaining), \
                    patch.object(runner, 'release_browser'), \
                    patch.object(runner, 'check', return_value={
                        'structural_check_passed': True,
                    }):
                runner.capture(args)

            self.assertEqual(seen, ['456'])
            self.assertEqual(runner.read(diag / 'checkpoint.json')['completed_ids'], ['123', '456'])

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

    def test_explicit_salary_policy_requires_scope_and_continuation(self):
        instruction = ('Координатор по постоянной инструкции пользователя разрешает проверить сохранённую активную карточку8471771462 с единственной строкой Оклад2 400 - 7 800 ₽ поддержанным review-salary-line, закрепить salary-review-policy для последующих строго аналогичных случаев и продолжить ту же очередь с checkpoint. Raw/историю/вкладку сохранить; недоступной карточку не считать')
        self.assertTrue(runner.salary_policy_direct_authorized(instruction))
        self.assertFalse(runner.salary_policy_direct_authorized(instruction.replace('последующих', 'текущих')))
        self.assertFalse(runner.salary_policy_direct_authorized(instruction.replace('продолжить', 'остановить')))

    def test_offline_salary_review_preserves_raw_and_rejects_other_lines(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'salary')
            data = raw(); data.update(params_blocks=['Условия\nГрафик: Сменный\nОклад105\u00a0000\u00a0₽', 'Требования\nОпыт: Без опыта'], fields=[{'name':'График','value':'Сменный'}, {'name':'Опыт','value':'Без опыта'}], parameter_audit={'rendered_complete':False,'expansion_controls':[]})
            runner.write(out/'raw/123.json', data); before = (out/'raw/123.json').read_bytes()
            runner.write(out/'queue.json', [item('123'), item('456')]); runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[]))
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            runner.authorize_browser(diag, 'Проверить карточку 123, закрепить salary-review-policy для последующих аналогичных случаев и продолжить', 'auth')
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

    def test_standing_technical_salary_policy_preserves_instruction(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, _, diag = runner.paths(runner.ROOT, 'salary-standing')
            data = raw(); data.update(params_blocks=['Условия\nГрафик: Сменный\nОклад37 000 - 88 000 ₽', 'Требования\nОпыт: Без опыта'], fields=[{'name':'График','value':'Сменный'}, {'name':'Опыт','value':'Без опыта'}], parameter_audit={'rendered_complete':False,'expansion_controls':[]})
            runner.write(out/'raw/123.json', data); original = (out/'raw/123.json').read_bytes()
            runner.write(out/'queue.json', [item('123')]); runner.write(diag/'checkpoint.json', dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[]))
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            instruction = 'делай без остановок пока не сделаешь все'
            runner.authorize_browser(diag, instruction, 'auth')
            args = argparse.Namespace(run_id='salary-standing', id='123', authorization_id='auth')
            with self.assertRaises(ValueError): runner.salary_review_policy(args)
            args.standing_technical = True
            runner.salary_review_policy(args); runner.review_salary_line(args)
            self.assertEqual(runner.read(out/'salary_review_policy.json')['human_instruction'], instruction)
            self.assertEqual((out/'raw/123.json').read_bytes(), original)
            self.assertEqual(runner.reviewed_capture(out, item())['fields'][-1]['value'], '37 000 - 88 000 ₽')

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
            archived = diag/'observations'/'123'/(record['snapshot_sha256']+'.problem.snapshot.txt')
            self.assertEqual(archived.read_bytes(), snapshot.read_bytes())
            snapshot.write_text('overwritten by next capture')
            self.assertEqual(runner.sha(archived), record['snapshot_sha256'])
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

    def test_truncated_closed_snapshot_requires_matching_sandbox_evidence(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'truncated')
            url = item()['url']; data = dict(state='UNKNOWN_PAGE', heading='Объявление не посмотреть', source_url=url,
                unresolved_evidence=dict(restriction=False, known_unavailable=True))
            runner.write(out/'raw/123.json', data); runner.write(out/'queue.json', [item('123'), item('456')])
            runner.write(service/'plan.json', dict(selection_mode='saved_incomplete_listing'))
            cp = dict(state='REVIEW_REQUIRED', stop_reason='identity_or_template_review', next_id='123', completed_ids=[])
            runner.write(diag/'checkpoint.json', cp)
            snap = diag/'problem.snapshot.txt'; snap.write_text(f'- title: "404" [url={url[:30]}…]\n- text: "Объявление закрыто."\n', encoding='utf8')
            sandbox = Path(directory)/'sandbox'; runner.write(diag/'sandbox.json', dict(path=str(sandbox)))
            runner.write(sandbox/'checkpoint.json', cp); runner.write(sandbox/'raw/123.json', data)
            (sandbox/'problem.snapshot.txt').write_bytes(snap.read_bytes()); runner.write(sandbox/'browser-tab.json', dict(url=url))
            runner.record_failure(diag, 'capture', 'review', 'capture', immediate=True)
            runner.authorize_browser(diag, 'Продолжить, исключив закрытое объявление 123', 'auth')
            args = argparse.Namespace(run_id='truncated', id='123', authorization_id='auth')
            for altered in (dict(url=item('456')['url']), dict(url='about:blank')):
                runner.write(sandbox/'browser-tab.json', altered)
                with self.assertRaisesRegex(ValueError, 'evidence'): runner.mark_unavailable(args)
            runner.write(sandbox/'browser-tab.json', dict(url=url))
            runner.write(sandbox/'raw/123.json', dict(data, source_url=item('456')['url']))
            with self.assertRaisesRegex(ValueError, 'evidence'): runner.mark_unavailable(args)
            runner.write(sandbox/'raw/123.json', data)
            runner.write(sandbox/'checkpoint.json', dict(cp, next_id='456'))
            with self.assertRaisesRegex(ValueError, 'evidence'): runner.mark_unavailable(args)
            runner.write(sandbox/'checkpoint.json', cp)
            runner.write(out/'raw/123.json', dict(data, source_url=item('456')['url']))
            with self.assertRaisesRegex(ValueError, 'evidence'): runner.mark_unavailable(args)
            runner.write(out/'raw/123.json', data)
            for changed in (dict(data, state='NEEDS_HUMAN'), dict(data, unresolved_evidence=dict(restriction=True, known_unavailable=True))):
                runner.write(out/'raw/123.json', changed); runner.write(sandbox/'raw/123.json', changed)
                with self.assertRaisesRegex(ValueError, 'evidence'): runner.mark_unavailable(args)
            runner.write(out/'raw/123.json', data); runner.write(sandbox/'raw/123.json', data)
            runner.mark_unavailable(args)
            record = runner.unavailable_records(out, runner.read(out/'queue.json'))[0]
            archived = diag/'observations/123'/(record['browser_tab_sha256'] + '.browser-tab.json')
            self.assertEqual(archived.read_bytes(), (sandbox/'browser-tab.json').read_bytes())
            self.assertEqual(runner.sha(archived), record['browser_tab_sha256'])

    def test_preserve_blank_observation_requires_exact_evidence_and_authorization(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'blank')
            sandbox = Path(directory)/'sandbox'
            data = dict(state='UNKNOWN_PAGE', source_url='about:blank', heading=None, items=[], empty_evidence=None, observed_at='now')
            cp = dict(state='REVIEW_REQUIRED', next_id='123', completed_ids=[], stop_reason='identity_or_template_review')
            snapshot = '# note: interactive (clickable / focusable) elements only.\n- title: "" [url=]'
            runner.write(out/'queue.json', [item()]); runner.write(out/'raw/123.json', data)
            runner.write(diag/'checkpoint.json', cp); runner.write(diag/'sandbox.json', {'path': str(sandbox)})
            runner.write(sandbox/'raw/123.json', data); runner.write(sandbox/'checkpoint.json', cp)
            (diag/'problem.snapshot.txt').write_text(snapshot); (sandbox/'problem.snapshot.txt').write_text(snapshot)
            original = (out/'raw/123.json').read_bytes()
            runner.record_failure(diag, 'capture', 'blank', 'capture', immediate=True)
            args = argparse.Namespace(run_id='blank', authorization_id='auth')
            with self.assertRaises(runner.HumanRequired): runner.preserve_stopped_observation(args)
            runner.authorize_browser(diag, 'Сохранить about:blank для ID 123 и продолжить', 'auth')
            (diag/'problem.snapshot.txt').write_text(snapshot + '\nCAPTCHA')
            with self.assertRaises(ValueError): runner.preserve_stopped_observation(args)
            self.assertEqual((out/'raw/123.json').read_bytes(), original)
            (diag/'problem.snapshot.txt').write_text(snapshot)
            runner.preserve_stopped_observation(args)
            self.assertFalse((out/'raw/123.json').exists())
            self.assertEqual((out/'observations/123'/f'{runner.sha(sandbox/"raw/123.json")}.json').read_bytes(), original)
            self.assertEqual(runner.read(diag/'checkpoint.json')['next_id'], '123')
            self.assertEqual((sandbox/'raw/123.json').read_bytes(), original)

    def test_preserve_captcha_observation_requires_authorization_and_retains_raw(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            out, service, diag = runner.paths(runner.ROOT, 'captcha')
            blocked = dict(state='NEEDS_HUMAN', reason='access_restriction_or_captcha', source_url=item()['url'], observed_at='2026-10-28T03:00:00Z')
            runner.write(out/'raw/123.json', blocked); original = (out/'raw/123.json').read_bytes()
            runner.write(out/'queue.json', [item()])
            checkpoint = dict(state='NEEDS_HUMAN', next_id='123', completed_ids=[], stop_reason='access_restriction_or_captcha')
            runner.write(diag/'checkpoint.json', checkpoint)
            sandbox=Path(directory)/'sandbox'; runner.write(diag/'sandbox.json', {'path':str(sandbox)})
            runner.write(sandbox/'checkpoint.json', checkpoint); runner.write(sandbox/'raw/123.json', blocked)
            runner.write(sandbox/'browser-tab.json', {'url':item()['url']})
            (sandbox/'problem.snapshot.txt').write_text('protected evidence', encoding='utf8')
            (diag/'problem.snapshot.txt').write_text('protected evidence', encoding='utf8')
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
            evidence = diag/'observations/123'
            self.assertEqual((evidence/(provenance['snapshot_sha256']+'.problem.snapshot.txt')).read_text(encoding='utf8'), 'protected evidence')
            self.assertEqual(runner.read(evidence/(provenance['browser_tab_sha256']+'.browser-tab.json'))['url'], item()['url'])
            self.assertEqual((sandbox/'raw/123.json').read_bytes(), original)
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

    def test_accept_all_pages_with_pending_browser_error_and_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, service, diag = runner.paths(root, 'saved')
            for file in runner.AUTOMATION_FILES:
                path = root/file; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((runner.ROOT/file).read_bytes())
            url = 'https://www.avito.ru/moskva/vakansii'
            next_url = url + '?p=2'
            page = dict(state='LISTING', source_url=url, selected_controls=[], items=[item('123'), item('456')], pagination={'next_url': next_url})
            runner.write(out/'listing_pages/0001.json', page)
            request = dict(url=url, limit=None, candidate_limit=None, geography='Москва', profession='', segments='', filter_evidence='Москва', all_pages=True, profession_scope='all')
            runner.write(diag/'freeze_request.json', request)
            runner.write(diag/'pending_listing.json', {'url': next_url})
            runner.write(diag/'checkpoint.json', {'state': 'BROWSER_ERROR', 'completed_ids': [], 'next_id': None})
            runner.record_failure(diag, 'freeze', 'browser error', 'freeze')
            args = argparse.Namespace(run_id='saved', resume=True, url=url, limit=None, all_pages=True, geography='Москва', profession='', segments='', filter_evidence='Москва', profession_scope='all', agent_model='gpt-6-luna', retry_reason='Accept incomplete saved pages', accept_saved=True, human_instruction='Собрать только сохранённый префикс')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'preflight') as preflight, patch.object(runner, 'browser') as browser, patch.object(runner, 'dirty_automation', return_value=[]), patch.object(runner.subprocess, 'check_output', return_value='abc'):
                runner.freeze(args)
                preflight.assert_not_called(); browser.assert_not_called()
                plan = runner.read(service/'plan.json')
                self.assertEqual(plan['selection_mode'], 'saved_incomplete_listing')
                self.assertFalse(plan['available_listing_exhausted'])
                self.assertEqual(plan['saved_selection_authorization']['pending']['sha256'], runner.sha(diag/'pending_listing.json'))
                runner.succeeded(diag, 'freeze', 'Accepted incomplete saved pages')
                runner.partition(argparse.Namespace(run_id='saved', batch_size=1, prefix='part'))
                self.assertEqual(runner.read(root/'Data/avito_runs/part_01/queue.json')[0]['id'], '123')
                self.assertEqual(runner.read(root/'Data/avito_runs/part_02/queue.json')[0]['id'], '456')
                self.assertEqual(runner.read(root/'Scraper/runs/part_01/plan.json')['selection_mode'], 'saved_incomplete_listing')

    def test_accept_saved_rejects_unrelated_pending_and_access_stop(self):
        for state, pending_url in [('BROWSER_ERROR', 'https://www.avito.ru/moskva/vakansii?p=9'), ('NEEDS_HUMAN', 'https://www.avito.ru/moskva/vakansii?p=2')]:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); out, service, diag = runner.paths(root, 'saved')
                url = 'https://www.avito.ru/moskva/vakansii'
                runner.write(out/'listing_pages/0001.json', dict(state='LISTING', source_url=url, selected_controls=[], items=[item('123')], pagination={'next_url': url+'?p=2'}))
                runner.write(diag/'freeze_request.json', dict(url=url, limit=None, candidate_limit=None, geography='Москва', profession='', segments='', filter_evidence='Москва', all_pages=True, profession_scope='all'))
                runner.write(diag/'pending_listing.json', {'url': pending_url})
                runner.write(diag/'checkpoint.json', {'state': state, 'completed_ids': [], 'next_id': None})
                args = argparse.Namespace(run_id='saved', resume=True, url=url, limit=None, all_pages=True, geography='Москва', profession='', segments='', filter_evidence='Москва', profession_scope='all', accept_saved=True, human_instruction='Собрать сохранённый префикс')
                with patch.object(runner, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Pending page'):
                    runner.freeze(args)
                self.assertFalse((service/'plan.json').exists())
                self.assertTrue((diag/'pending_listing.json').exists())

    def test_supplement_selects_only_new_ids_and_validates_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, service, diag = runner.paths(root, 'parent')
            for file in runner.AUTOMATION_FILES:
                path = root/file; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((runner.ROOT/file).read_bytes())
            url = 'https://www.avito.ru/moskva/vakansii?p=19'
            first = dict(state='LISTING', source_url=url, heading='Курьер', observed_at='2026-10-29T13:24:00Z', profession_filter={'observed':True, 'selected_count':2}, selected_controls=courier_controls(), items=[item('123')], pagination={'current_pages':['19']})
            runner.write(out/'listing.json', first); runner.write(out/'listing_pages/0001.json', first)
            runner.write(out/'queue.json', runner.queue_from_pages([first], None))
            runner.write(service/'plan.json', dict(run_id='parent', profession='Курьер', profession_scope=None, listing_url=url, listing_sha256=runner.sha(out/'listing.json'), listing_pages=[dict(file='listing_pages/0001.json', sha256=runner.sha(out/'listing_pages/0001.json'), source_url=url)]))
            fresh = dict(first, items=[item('123'), item('456')])
            runner.write(diag/'observation.json', fresh)
            runner.write(diag/'evidence.json', dict(heading='Курьер', observed_at='2026-10-29T13:23:00Z', actual_url=url, source_url=url, current_pages=['19'], listing_state='LISTING', visible_cards=2, protection='not_detected', search={'present':True,'value':''}, profession={'checked':True,'label':'Курьер','marker':'params[144170]/checkbox/3201765'}))
            args=argparse.Namespace(parent_run_id='parent',run_id='supp',observation='observation.json',evidence='evidence.json')
            with patch.object(runner,'ROOT',root), patch.object(runner,'dirty_automation',return_value=[]), patch.object(runner.subprocess,'check_output',return_value='abc'):
                original_evidence = runner.read(diag/'evidence.json')
                runner.write(diag/'evidence.json', dict(original_evidence, search={'present':True,'value':'wrong'}))
                with self.assertRaisesRegex(ValueError, 'parent filters'):
                    runner.supplemental(args)
                self.assertFalse((root/'Data/avito_runs/supp').exists())
                runner.write(diag/'evidence.json', dict(original_evidence, observed_at='2026-10-29T13:00:00Z'))
                with self.assertRaisesRegex(ValueError, 'not contemporaneous'):
                    runner.supplemental(args)
                runner.write(diag/'evidence.json', original_evidence)
                runner.supplemental(args)
                child, child_service, _ = runner.paths(root,'supp')
                self.assertEqual([x['id'] for x in runner.read(child/'queue.json')], ['456'])
                runner.validate_partition(child,child_service)
                runner.write(child/'queue.json', runner.queue_from_pages([fresh], None))
                with self.assertRaisesRegex(ValueError,'new-ID selection'):
                    runner.validate_partition(child,child_service)

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

    def test_http_429_status_does_not_match_vacancy_id(self):
        extractor = (Path(__file__).parent/'extract.js').read_text(encoding='utf8').strip().removesuffix(';')
        harness = r"""const extract=()=>EXTRACTOR;
const body=process.argv[1]; global.location={href:'https://www.avito.ru/moskva/vakansii/job_7875654292',pathname:'/moskva/vakansii/job_7875654292'};
global.document={title:'Авито',readyState:'complete',body:{innerText:body},querySelector:s=>s.includes('item-view/title-info')?{innerText:'Вакансия'}:null,querySelectorAll:()=>[]};
global.getComputedStyle=()=>({visibility:'visible'}); console.log(JSON.stringify(extract()));""".replace('EXTRACTOR',extractor)
        for body,state in [('№ 7875654292 · 5 октября', 'DETAIL'), ('Зарплата 142900 рублей', 'DETAIL'), ('HTTP 429', 'NEEDS_HUMAN'), ('Ошибка 429', 'NEEDS_HUMAN'), ('429 Too Many Requests', 'NEEDS_HUMAN'), ('Доступ ограничен: проблема с IP', 'NEEDS_HUMAN'), ('Подозрительная активность с вашего IP', 'NEEDS_HUMAN')]:
            with self.subTest(body=body):
                result=subprocess.run(['node','-e',harness,body],capture_output=True,text=True,encoding='utf8',check=True)
                self.assertEqual(json.loads(result.stdout)['state'],state)

    def test_windows_checkout_dirty_audit_requires_exact_frozen_bytes(self):
        blob=b'a\nb\n'; file='Scraper/extract.js'
        plan={'code_git_commit':'frozen','uncommitted_automation_files':[file],
              'automation_sha256':{file:hashlib.sha256(blob.replace(b'\n',b'\r\n')).hexdigest()}}
        with patch.object(runner.subprocess,'check_output',return_value=blob):
            self.assertEqual(runner.frozen_dirty_sources(plan),set())
            plan['automation_sha256'][file]='bad'
            self.assertEqual(runner.frozen_dirty_sources(plan),{file})

    def test_unknown_evidence_scans_full_text_and_refuses_incomplete(self):
        extractor = (Path(__file__).parent/'extract.js').read_text(encoding='utf8')
        harness = r"""const extract=()=>EXTRACTOR;
const body=process.argv[1], title=process.argv[2], ready=process.argv[3];
global.location={href:'https://www.avito.ru/moskva/vakansii/example_123'};
global.document={title,readyState:ready,body:{innerText:body},querySelector:()=>null,querySelectorAll:()=>[]};
global.getComputedStyle=()=>({visibility:'visible'});
console.log(JSON.stringify(extract()));""".replace('EXTRACTOR', extractor.rstrip().rstrip(';'))
        for body,title,ready,state,restriction,unavailable in [
            ('ordinary text','Title','complete','UNKNOWN_PAGE',False,False),
            ('x'*6000+' CAPTCHA','Title','complete','NEEDS_HUMAN',None,None),
            ('ordinary text','429 Too Many Requests','complete','NEEDS_HUMAN',None,None),
            ('ordinary text Объявление удалено.','Title','complete','UNKNOWN_PAGE',False,True),
            ('ordinary text','Title','loading','UNKNOWN_PAGE',False,False)]:
            result=subprocess.run(['node','-e',harness,body,title,ready],capture_output=True,text=True,encoding='utf8',check=True)
            data=json.loads(result.stdout); self.assertEqual(data['state'],state)
            if state=='UNKNOWN_PAGE':
                self.assertEqual(data['unresolved_evidence']['restriction'],restriction)
                self.assertEqual(data['unresolved_evidence']['known_unavailable'],unavailable)
                self.assertEqual(data['unresolved_evidence']['complete'],ready=='complete')

    def test_authorized_unresolved_preserve_retry_and_finalize(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = runner.ROOT; run_id = 'remote_1500_2026-10-29_shamil'
            out, service, diag = runner.paths(root, run_id); sandbox = root/'sandbox'
            old_extractor = b'old extractor'; extractor = root/'Scraper/extract.js'
            extractor.parent.mkdir(parents=True); extractor.write_text(runner.HTTP429_OLD_BLOCK+'\n'+runner.HTTP429_OLD_RESTRICTION+'\n', encoding='utf8', newline='')
            queue = [item('123'), item('7995516348')]; queue[1]['listing_rank'] = 2
            runner.write(root/'Data/dataset/sources.json', {'schema_version':'1','exports':[]})
            (root/'Data/README.md').write_text('# Data\n', encoding='utf8'); (root/'Data/dataset/README.md').write_text('Текущий состав: old\n', encoding='utf8')
            runner.write(out/'queue.json', queue); runner.write(out/'listing.json', {'selected_controls':courier_controls()})
            runner.write(out/'raw/123.json', raw())
            runner.write(service/'plan.json', {'setup_seconds':1,'profession':'Курьер','segments':'all','code_git_commit':'old','extractor_sha256':hashlib.sha256(old_extractor).hexdigest()})
            cp = dict(state='REVIEW_REQUIRED', next_id='7995516348', completed_ids=['123'], stop_reason='identity_or_template_review', capture_seconds=1)
            old = dict(state='UNKNOWN_PAGE', source_url=queue[1]['url'], heading=None, items=[], observed_at='old')
            runner.write(out/'raw/7995516348.json', old); runner.write(sandbox/'raw/7995516348.json', old)
            runner.write(diag/'checkpoint.json', cp); runner.write(sandbox/'checkpoint.json', cp); runner.write(diag/'sandbox.json', {'path':str(sandbox)})
            runner.write(sandbox/'browser-tab.json', {'url':queue[1]['url']})
            (diag/'problem.snapshot.txt').write_text('old incomplete snapshot',encoding='utf8'); (sandbox/'problem.snapshot.txt').write_text('old incomplete snapshot',encoding='utf8')
            runner.record_failure(diag, 'capture', 'stopped', 'capture', immediate=True)
            runner.authorize_browser(diag, runner.UNRESOLVED_ANSWER, runner.UNRESOLVED_AUTH)
            runner.note_change(argparse.Namespace(run_id=run_id,file='Scraper/extract.js',reason='Authorized unresolved observation evidence',validation='offline regression'))
            args=argparse.Namespace(run_id=run_id,authorization_id=runner.UNRESOLVED_AUTH,authorization_question=runner.UNRESOLVED_QUESTION)
            with patch.object(runner.subprocess, 'check_output', return_value=old_extractor): runner.unresolved_policy(args)
            runner.preserve_stopped_observation(args)
            self.assertEqual(runner.read(sandbox/'raw/7995516348.json'), old)
            self.assertFalse((out/'raw/7995516348.json').exists())
            new = dict(old, observed_at='new', unresolved_evidence=dict(version=1,complete=True,title_body_params_scanned=True,restriction=False,known_unavailable=False))
            runner.write(out/'raw/7995516348.json', new); runner.write(sandbox/'raw/7995516348.json', new)
            runner.write(diag/'sandbox.json', {'path':str(sandbox)})
            (diag/'problem.snapshot.txt').write_text('new snapshot',encoding='utf8'); (sandbox/'problem.snapshot.txt').write_text('new snapshot',encoding='utf8')
            mark = argparse.Namespace(run_id=run_id,id='7995516348',authorization_id=runner.UNRESOLVED_AUTH)
            for invalid in (dict(restriction=True), dict(known_unavailable=True), dict(complete=False), dict(title_body_params_scanned=False)):
                bad = dict(new, unresolved_evidence=dict(new['unresolved_evidence'], **invalid))
                runner.write(out/'raw/7995516348.json', bad); runner.write(sandbox/'raw/7995516348.json', bad)
                with self.assertRaises(ValueError): runner.mark_unresolved(mark)
                self.assertFalse((out/'unresolved_observations.json').exists())
            runner.write(out/'raw/7995516348.json', new); runner.write(sandbox/'raw/7995516348.json', new)
            runner.write(sandbox/'browser-tab.json', {'url':item('123')['url']})
            with self.assertRaises(ValueError): runner.mark_unresolved(mark)
            runner.write(sandbox/'browser-tab.json', {'url':queue[1]['url']}); runner.mark_unresolved(mark)
            self.assertEqual(runner.read(diag/'checkpoint.json')['completed_ids'], ['123'])
            before_policy = runner.read(out/'unresolved_policy.json'); previous_hash = runner.sha(out/'unresolved_policy.json')
            extractor.write_text(runner.HTTP429_NEW_BLOCK+'\n'+runner.HTTP429_NEW_RESTRICTION+'\n', encoding='utf8', newline='')
            runner.note_change(argparse.Namespace(run_id=run_id,file='Scraper/extract.js',reason=runner.HTTP429_AUDIT_REASON,validation='HTTP status regression'))
            runner.upgrade_unresolved_extractor(argparse.Namespace(run_id=run_id))
            updated = runner.read(out/'unresolved_policy.json')
            self.assertEqual(runner.read(out/'automation'/f'policy-{previous_hash}.json'), before_policy)
            self.assertEqual(updated['extractor_upgrades'][0]['from_sha256'], before_policy['new_extractor_sha256'])
            self.assertEqual(runner.unresolved_records(out, queue)[0]['id'], '7995516348')
            placement = root/'placements.json'; runner.write(placement, {'123':{'city':'Коммунарка','region':'Москва','evidence':'пос. Коммунарка'}})
            runner.finalize(argparse.Namespace(run_id=run_id,placements=placement,review_model='test'))
            manifest = runner.read(service/'manifest.json')
            self.assertEqual(manifest['coverage']['accepted'], 1)
            self.assertEqual(manifest['coverage']['unresolved_observations'][0]['id'], '7995516348')
            with (out/'exports/vacancies_flat.csv').open(encoding='utf-8-sig',newline='') as stream:
                self.assertEqual([r['id'] for r in csv.DictReader(stream)], ['123'])

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
