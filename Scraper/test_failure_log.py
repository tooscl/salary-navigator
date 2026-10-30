"""Retry and human-resume regressions: no browser or network."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

spec = importlib.util.spec_from_file_location('failure_log', Path(__file__).parent/'skills/collect-avito/scripts/failure_log.py')
failure_log = importlib.util.module_from_spec(spec)
spec.loader.exec_module(failure_log)


class FailureBudgetTests(unittest.TestCase):
    def test_exact_demo_duplicate_check_then_success_allows_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            failure_log.record(diag, 'filters', 'CDP Page.enable timeout', 'attach tab')
            failure_log.ensure_allowed(diag, 'filters', 'Open own tab with observed URL')
            failure_log.ensure_allowed(diag, 'filters', 'Open own tab with observed URL')
            failure_log.ensure_allowed(diag, 'filters')
            ledger = failure_log.load(diag)
            self.assertEqual(ledger['state'], 'RETRY_STARTED')
            self.assertEqual(ledger['retries_started'], 1)
            self.assertEqual(sum(e['event'] == 'retry_started' for e in ledger['events']), 1)
            failure_log.succeeded(diag, 'filters', 'Snapshot confirms listing')
            failure_log.ensure_allowed(diag, 'freeze')
            self.assertEqual(failure_log.load(diag)['failures'], 1)

    def test_duplicate_checks_do_not_authorize_duplicate_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            failure_log.record(diag, 'capture', 'Connection lost', 'read')
            failure_log.ensure_allowed(diag, 'capture', 'Restored')
            failure_log.ensure_allowed(diag, 'capture', action=True)
            failure_log.ensure_allowed(diag, 'capture', 'Harmless check')
            with self.assertRaisesRegex(failure_log.HumanRequired, 'action already started'):
                failure_log.ensure_allowed(diag, 'capture', 'Accidental re-execution', action=True)
            with self.assertRaises(ValueError):
                failure_log.record(diag, 'filters', 'Unrelated error', 'read')
            self.assertEqual(failure_log.load(diag)['retries_started'], 1)
            failure_log.record(diag, 'capture', 'Confirmed interrupted process', 'read')
            failure_log.ensure_allowed(diag, 'capture', 'Resume after interruption', action=True)
            self.assertEqual(failure_log.load(diag)['retries_started'], 2)

    def test_concurrent_cli_action_is_blocked_without_spending_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            with failure_log.operation_locked(diag):
                with self.assertRaises(failure_log.HumanRequired):
                    with failure_log.operation_locked(diag):
                        self.fail('Duplicate action acquired the lock')
            with failure_log.operation_locked(diag):
                self.assertEqual(failure_log.load(diag)['retries_started'], 0)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows legacy code page')
    def test_legacy_windows_ledger_reads_and_writes_utf8_without_budget_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            legacy = {'events': [{'event': 'failure', 'message': 'Ошибка'}], 'stages': {},
                      'failures': 2, 'state': 'RETRY_AVAILABLE', 'retry_started': False,
                      'retries_started': 1, 'retry_limit': 5, 'budget_start_failures': 0}
            import json
            (diag/'failures.json').write_bytes(json.dumps(legacy, ensure_ascii=False).encode('cp1251'))
            self.assertEqual(failure_log.load(diag)['events'][0]['message'], 'Ошибка')
            failure_log.save(diag, failure_log.load(diag))
            self.assertEqual(json.loads((diag/'failures.json').read_text(encoding='utf-8')), legacy)

    def test_operation_lock_excludes_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            code = """import importlib.util,pathlib,sys
s=importlib.util.spec_from_file_location('f',sys.argv[1]); f=importlib.util.module_from_spec(s); s.loader.exec_module(f)
with f.operation_locked(pathlib.Path(sys.argv[2])):
 print('locked',flush=True); sys.stdin.readline()
"""
            proc = subprocess.Popen([sys.executable, '-u', '-c', code, str(spec.origin), directory],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertEqual(proc.stdout.readline().strip(), b'locked')
                with self.assertRaises(failure_log.HumanRequired):
                    with failure_log.operation_locked(diag):
                        self.fail('another process holds the lock')
            finally:
                proc.stdin.write(b'\n'); proc.stdin.flush()
                proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 0)
            with failure_log.operation_locked(diag):
                self.assertEqual(failure_log.load(diag)['failures'], 0)

    def test_five_retries_are_shared_across_agents_urls_and_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            event = failure_log.record(diag, 'filters', 'Transient failure', 'read', agent='Luna')
            self.assertEqual(event['retry_budget_remaining'], 5)
            with self.assertRaises(failure_log.HumanRequired):
                failure_log.ensure_allowed(diag, 'filters')
            for attempt in range(1, 6):
                stage = 'filters' if attempt % 2 else 'capture'
                failure_log.ensure_allowed(diag, stage, 'Observed correction')
                failure_log.ensure_allowed(diag, stage, 'Duplicate check')
                event = failure_log.record(diag, stage, 'Transient failure', 'read',
                    'https://www.avito.ru/moskva/vakansii', 'Terra')
                self.assertEqual(event['retry_budget_remaining'], 5-attempt)
                self.assertEqual(event['state'], 'NEEDS_HUMAN' if attempt == 5 else 'RETRY_AVAILABLE')
            self.assertEqual(failure_log.load(diag)['failures'], 6)
            for stage in ['filters', 'freeze', 'capture', 'publish']:
                with self.assertRaises(failure_log.HumanRequired):
                    failure_log.ensure_allowed(diag, stage, 'Must stay stopped')

    def test_success_does_not_reset_global_budget_and_wrong_stage_cannot_finish_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            failure_log.record(diag, 'filters', 'Transient', 'read')
            failure_log.ensure_allowed(diag, 'filters', 'Corrected control')
            with self.assertRaises(failure_log.HumanRequired):
                failure_log.ensure_allowed(diag, 'capture', 'Different stage')
            with self.assertRaises(ValueError):
                failure_log.succeeded(diag, 'capture', 'Wrong stage')
            failure_log.succeeded(diag, 'filters', 'Correct page')
            failure_log.ensure_allowed(diag, 'capture')
            event = failure_log.record(diag, 'capture', 'Another transient', 'read')
            self.assertEqual(event['retry_budget_remaining'], 4)

    def test_access_block_stops_immediately_and_offline_error_cannot_unlock_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            event = failure_log.record(diag, 'capture', 'CAPTCHA', 'open card', immediate=True)
            self.assertEqual(event['retry_budget_remaining'], 0)
            failure_log.succeeded(diag, 'capture', 'Must not clear a hard stop')
            failure_log.authorize_offline(diag, 'Finish saved batch only')
            failure_log.ensure_allowed(diag, 'finalize')
            failure_log.ensure_allowed(diag, 'publish')
            with self.assertRaises(failure_log.HumanRequired):
                failure_log.ensure_allowed(diag, 'capture', 'Not permitted')
            failure_log.record(diag, 'finalize', 'New failure', 'offline')
            for stage in ['publish', 'capture']:
                with self.assertRaises(failure_log.HumanRequired):
                    failure_log.ensure_allowed(diag, stage, 'Must stay stopped')

    def test_explicit_human_browser_resume_keeps_history_and_is_not_replayable_for_more_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            failure_log.record(diag, 'filters', 'Earlier error', 'read', immediate=True)
            original = failure_log.load(diag)['events'][:]
            failure_log.authorize_browser(diag, 'Continue these five vacancies', 'human-turn-1')
            failure_log.ensure_allowed(diag, 'freeze')
            ledger = failure_log.load(diag)
            self.assertEqual(ledger['failures'], 1)
            self.assertEqual(ledger['events'][:len(original)], original)
            self.assertEqual(ledger['retries_started'], 0)
            failure_log.record(diag, 'capture', 'Transient', 'read')
            failure_log.ensure_allowed(diag, 'capture', 'Corrected')
            failure_log.authorize_browser(diag, 'Continue these five vacancies', 'human-turn-1')
            self.assertEqual(failure_log.load(diag)['retries_started'], 1)
            failure_log.record(diag, 'capture', 'CAPTCHA again', 'read', immediate=True)
            with self.assertRaises(failure_log.HumanRequired):
                failure_log.authorize_browser(diag, 'Continue these five vacancies', 'human-turn-1')
            self.assertEqual(failure_log.load(diag)['state'], 'NEEDS_HUMAN')

    def test_legacy_active_retry_requires_recorded_interruption_before_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            failure_log.save(diag, {'events': [dict(event='failure'), dict(event='retry_started', stage='capture')],
                'stages': {}, 'failures': 1, 'state': 'RETRY_STARTED', 'retry_started': True})
            failure_log.ensure_allowed(diag, 'capture', 'Read-only check')
            with self.assertRaises(failure_log.HumanRequired):
                failure_log.ensure_allowed(diag, 'capture', 'Unconfirmed restart', action=True)
            failure_log.record(diag, 'capture', 'Old process confirmed interrupted', 'read')
            failure_log.ensure_allowed(diag, 'capture', 'Resume saved checkpoint', action=True)
            self.assertEqual(failure_log.load(diag)['retries_started'], 2)

    def test_legacy_stopped_ledger_is_not_silently_unblocked(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            failure_log.save(diag, {'events': [dict(event='failure'), dict(event='retry_started', stage='filters')],
                'stages': {'filters': {'failures': 1}}, 'failures': 1,
                'state': 'NEEDS_HUMAN', 'retry_started': True})
            with self.assertRaises(failure_log.HumanRequired):
                failure_log.ensure_allowed(diag, 'freeze', 'New policy is not consent')
            failure_log.authorize_browser(diag, 'Continue', 'human-turn-2')
            failure_log.ensure_allowed(diag, 'freeze')
            self.assertEqual(failure_log.load(diag)['failures'], 1)
            self.assertEqual(len(failure_log.load(diag)['events']), 3)

    def test_resume_requires_stopped_run_and_nonempty_human_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            for instruction, authorization_id in [('', 'turn'), ('Continue', ''), ('Continue', 'turn')]:
                with self.assertRaises(ValueError):
                    failure_log.authorize_browser(diag, instruction, authorization_id)


if __name__ == '__main__':
    unittest.main()
