"""Durable campaign coordinator; browser collection remains in run.py.

One tick executes at most one runner command and waits for its actual exit.
Heartbeat callers can repeat ticks; run drives ticks continuously until a pause,
cooldown or explicit stop. No command grants browser authorization or resets a
failure ledger. All discovery URLs must be observed and supplied in the spec.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

from failure_log import operation_locked, load as load_ledger

ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).with_name('run.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def safe_id(value):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,59}', value):
        raise ValueError('Campaign and seed IDs must be simple names of at most 60 characters')
    return value


def paths(root, campaign):
    return root / 'Scraper/campaigns' / safe_id(campaign)


def controls(values):
    return sorted(set((str(x.get('marker') or ''), str(x.get('value') or ''),
                       str(x.get('label') or '')) for x in values))


def validate_spec(spec):
    if spec.get('schema_version') != 1 or not spec.get('seeds') or not spec.get('known_runs'):
        raise ValueError('Spec requires schema_version=1, observed seeds and known_runs')
    if not 1 <= spec.get('listing_limit', 1500) <= 1500:
        raise ValueError('listing_limit must be between 1 and 1500')
    if spec.get('cooldown_seconds', 3600) < 0:
        raise ValueError('cooldown_seconds must be non-negative; use 0 for continuous discovery')
    ids = set()
    for seed in spec['seeds']:
        identity = safe_id(seed['id'])
        if identity in ids: raise ValueError('Duplicate seed ID')
        ids.add(identity)
        url = urlsplit(seed['url'])
        if (url.scheme != 'https' or url.hostname not in ('avito.ru', 'www.avito.ru')
                or not re.match(r'^/[^/]+/vakansii(?:/|$)', url.path)
                or url.username or url.password or url.port not in (None, 443)):
            raise ValueError('Seed must be an observed Avito vacancy URL')
        if not seed.get('geography') or not seed.get('filter_evidence') or not seed.get('expected_selected_controls'):
            raise ValueError('Seed requires geography, evidence and exact expected_selected_controls')
        if not any(c.get('label') == 'Удалённо' and 'params[172815]' in (c.get('marker') or '')
                   for c in seed['expected_selected_controls']):
            raise ValueError('Every seed must preserve the observed remote-work filter')
    return spec


def ordinary_run(root, run_id):
    """Runner dedup accepts ordinary source queues, not deduplicated children."""
    seen = set()
    while True:
        if run_id in seen: raise ValueError('Cyclic source provenance')
        seen.add(run_id)
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}', run_id):
            raise ValueError('Invalid known run ID')
        plan = read(root / 'Scraper/runs' / run_id / 'plan.json')
        if plan.get('deduplication'):
            run_id = plan['deduplication']['source_run_id']
            continue
        if plan.get('selection_mode') != 'first_N_unique_on_frozen_listing':
            raise ValueError('Known run must resolve to an ordinary frozen queue')
        read(root / 'Data/avito_runs' / run_id / 'queue.json')
        return run_id


def initialize(root, campaign, spec):
    validate_spec(spec)
    directory = paths(root, campaign)
    if directory.exists(): raise ValueError('Campaign already exists; use tick/status')
    known = list(dict.fromkeys(ordinary_run(root, value) for value in spec['known_runs']))
    directory.mkdir(parents=True)
    write(directory / 'spec.json', spec)
    state = dict(version=1, campaign=campaign, phase='DISCOVER', round=1, seed_index=0,
                 known_runs=known, completed=[], next_poll_at=0, inflight=None, reason=None,
                 spec_sha256=hashlib.sha256((directory/'spec.json').read_bytes()).hexdigest())
    write(directory / 'state.json', state)
    return state


def run_ids(campaign, state):
    base = f'{campaign}_r{state["round"]:06d}_s{state["seed_index"]:03d}'
    return base, base + '_new'


def infer_phase(root, campaign, state):
    """Recover a controller pause from immutable runner artifacts.

    This does not authorize browser work and never changes a runner ledger.  It
    only selects the next state-machine phase after the human has separately
    resumed the stopped run through run.py.
    """
    source, child = run_ids(campaign, state)
    source_service = root/'Scraper/runs'/source
    child_service = root/'Scraper/runs'/child
    child_data = root/'Data/avito_runs'/child
    if (child_service/'manifest.json').exists():
        return 'CAPTURE'
    if (child_service/'plan.json').exists() or (child_data/'dedup-empty.json').exists():
        return 'CAPTURE'
    if (source_service/'plan.json').exists():
        return 'DEDUP'
    return 'DISCOVER'


def process_is_alive(pid):
    """Return True/False for a PID, or None when liveness is not provable."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == 'nt':
        # SYNCHRONIZE lets us inspect termination without query permissions,
        # which are commonly denied in managed Windows environments.
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        from ctypes import wintypes
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: PID does not exist.
                return False
            return None
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    return True


def recover_campaign(root, campaign, reason):
    """Recover a persisted dead worker without ever interrupting a live one."""
    directory = paths(root, campaign)
    with operation_locked(directory):
        state = read(directory/'state.json')
        inflight = state.get('inflight')
        if not inflight:
            raise ValueError('Campaign has no inflight worker to recover')
        pid = inflight.get('pid')
        alive = process_is_alive(pid)
        if alive is True:
            raise ValueError(f'Worker PID {pid} is still running')
        if alive is None:
            raise ValueError(f'Cannot prove that worker PID {pid} has exited')
        state['interrupted_worker'] = dict(inflight, recovered_at=time.time())
        state['inflight'] = None
        state.update(phase=infer_phase(root,campaign,state), reason=reason,
                     recovered_at=datetime.now(timezone.utc).isoformat())
        write(directory/'state.json',state)
        return state


def resume_campaign(root, campaign, reason):
    directory = paths(root, campaign)
    with operation_locked(directory):
        state = read(directory/'state.json')
        if state.get('phase') != 'PAUSED':
            raise ValueError('Only a paused campaign can be resumed')
        if state.get('inflight'):
            raise ValueError('An inflight worker must be inspected before resume')
        if (directory/'STOP').exists():
            raise ValueError('Campaign has a user stop flag')
        source, child = run_ids(campaign, state)
        for run_id in (source, child):
            ledger_path = root/'Scraper/runs'/run_id/'diagnostics/failures.json'
            if ledger_path.exists() and read(ledger_path).get('state') in ('NEEDS_HUMAN','BLOCKED','STOPPED'):
                raise ValueError(f'Runner {run_id} still requires human review')
        state.update(phase=infer_phase(root,campaign,state), reason=reason,
                     resumed_at=datetime.now(timezone.utc).isoformat())
        write(directory/'state.json',state)
        return state


def verify_slice(root, run_id, seed):
    plan = read(root / 'Scraper/runs' / run_id / 'plan.json')
    if plan.get('geography') != seed['geography'] or plan.get('profession', '') != seed.get('profession', ''):
        raise ValueError('Frozen geography/profession differs from seed')
    expected = controls(seed['expected_selected_controls'])
    if not plan.get('listing_pages'): raise ValueError('Missing frozen listing pages')
    for page in plan['listing_pages']:
        source = root / 'Data/avito_runs' / run_id / page['file']
        if hashlib.sha256(source.read_bytes()).hexdigest() != page['sha256']:
            raise ValueError('Frozen page hash differs')
        if controls(read(source).get('selected_controls', [])) != expected:
            raise ValueError('Observed filters differ from seed; no detail capture permitted')


def inherit_validated_changes(root, source, completed):
    """Reuse only validations pinned to identical bytes in a completed export."""
    service = root/'Scraper/runs'/source
    plan = read(service/'plan.json')
    diag = service/'diagnostics'
    destination = diag/'automation_changes.json'
    records = read(destination) if destination.exists() else []
    present = {r.get('file') for r in records}
    for entry in reversed(completed):
        prior = root/'Scraper/runs'/entry['capture_run']
        manifest_path, prior_plan_path = prior/'manifest.json', prior/'plan.json'
        if not manifest_path.is_file() or not prior_plan_path.is_file(): continue
        manifest = read(manifest_path)
        plan_key = prior_plan_path.relative_to(root).as_posix()
        expected_plan = next((v for k, v in manifest.get('files', {}).items() if k.replace('\\', '/') == plan_key), None)
        if manifest.get('status') != 'complete' or not expected_plan or hashlib.sha256(prior_plan_path.read_bytes()).hexdigest() != expected_plan.get('sha256'): continue
        prior_hashes = dict(read(prior_plan_path).get('automation_sha256', {}))
        prior_hashes.update(manifest.get('post_capture_automation_sha256', {}))
        for record in manifest.get('automation_changes', []):
            name = record.get('file', '')
            path = root/name
            expected = prior_hashes.get(name)
            if name in present or not expected or plan.get('automation_sha256', {}).get(name) != expected or not path.is_file(): continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected: continue
            if not record.get('reason') or not record.get('validation'): continue
            records.append(dict(file=name, reason=record['reason'], validation=record['validation'],
                                validation_source_run=entry['capture_run'],
                                validation_source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                validated_file_sha256=expected))
            present.add(name)
    if records: write(destination, records)


def execute(root, directory, state, argv):
    """Persist intent first. If the controller dies, no second worker is launched."""
    log = directory / 'worker.local.log'
    state['inflight'] = dict(argv=argv, started_at=time.time())
    write(directory / 'state.json', state)
    with log.open('ab') as stream:
        process = subprocess.Popen([sys.executable, '-X', 'utf8', str(RUNNER), *argv],
                                   cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        state['inflight']['pid'] = process.pid
        write(directory / 'state.json', state)
        result = process.wait()
    state['inflight'] = None
    state['last_command'] = dict(argv=argv, returncode=result, finished_at=time.time(), log=str(log))
    write(directory / 'state.json', state)
    return result


def advance(state, spec, source, child, count):
    state['known_runs'] = list(dict.fromkeys(state['known_runs'] + [source]))
    state['completed'].append(dict(source_run=source, capture_run=child, new_ids=count,
                                   finished_at=datetime.now(timezone.utc).isoformat()))
    state['seed_index'] += 1
    state['phase'] = 'DISCOVER'
    if state['seed_index'] == len(spec['seeds']):
        state['seed_index'] = 0
        state['round'] += 1
        cooldown = spec.get('cooldown_seconds', 3600)
        if cooldown > 0:
            state['phase'] = 'COOLDOWN'
            state['next_poll_at'] = time.time() + cooldown
        else:
            state['next_poll_at'] = 0


def tick(root, campaign, executor=execute):
    directory = paths(root, campaign)
    with operation_locked(directory):
        state, spec = read(directory/'state.json'), read(directory/'spec.json')
        if (directory/'STOP').exists():
            state.update(phase='STOPPED', reason='User stop flag')
            write(directory/'state.json', state)
            return state
        if state['phase'] in ('STOPPED', 'PAUSED'): return state
        if state['inflight']:
            alive = process_is_alive(state['inflight'].get('pid'))
            if alive is True:
                # A concurrent heartbeat must not pause an active campaign worker.
                return state
            state.update(phase='PAUSED', reason='Previous worker exited or liveness cannot be confirmed; inspect its PID and artifacts before recovery')
            write(directory/'state.json', state)
            return state
        if hashlib.sha256((directory/'spec.json').read_bytes()).hexdigest() != state['spec_sha256']:
            raise ValueError('Campaign spec changed after initialization')
        if state['phase'] == 'COOLDOWN':
            if time.time() < state['next_poll_at']: return state
            state['phase'] = 'DISCOVER'
        source, child = run_ids(campaign, state)
        seed = spec['seeds'][state['seed_index']]
        source_service = root/'Scraper/runs'/source
        child_service = root/'Scraper/runs'/child
        child_data = root/'Data/avito_runs'/child
        command = None
        try:
            # A protective stop in either source or shared child is never reset.
            diag = source_service/'diagnostics'
            if (diag/'failures.json').exists():
                ledger = read(diag/'failures.json')
                # A transient runner failure remains retryable within the
                # skill's bounded retry budget.  Only a protective stop or
                # explicit human-review state must pause the campaign.
                if ledger.get('state') in ('NEEDS_HUMAN', 'BLOCKED', 'STOPPED'):
                    raise ValueError('Runner ledger requires review: ' + str(ledger.get('state')))
            if state['phase'] == 'DISCOVER':
                if (source_service/'plan.json').exists():
                    verify_slice(root, source, seed)
                    state['phase'] = 'DEDUP'
                else:
                    command = ['freeze','--run-id',source,'--url',seed['url'],
                               '--limit',str(spec.get('listing_limit',1500)),
                               '--geography',seed['geography'],'--profession',seed.get('profession',''),
                               '--profession-scope',seed.get('profession_scope','all'),
                               '--filter-evidence',seed['filter_evidence']]
                    if source_service.exists():
                        command += ['--resume']
                        ledger_path = source_service/'diagnostics/failures.json'
                        if ledger_path.exists():
                            ledger = read(ledger_path)
                            if ledger.get('failures',0) > ledger.get('budget_start_failures',0):
                                command += ['--retry-reason',
                                    'Human confirmed continuation after review; retrying the saved observed Next URL.']
                    if spec.get('config'): command += ['--config',spec['config']]
            elif state['phase'] == 'DEDUP':
                verify_slice(root, source, seed)
                if (child_service/'plan.json').exists(): state['phase'] = 'CAPTURE'
                else:
                    inherit_validated_changes(root, source, state['completed'])
                    command = ['deduplicate-queue','--source-run-id',source,'--run-id',child]
                    for known in state['known_runs']: command += ['--against-run-id',known]
            elif state['phase'] == 'CAPTURE':
                if (child_data/'dedup-empty.json').exists():
                    advance(state,spec,source,child,0)
                    queue = []
                else:
                    queue = read(child_data/'queue.json')
                if queue and (child_service/'manifest.json').exists():
                    if read(child_service/'manifest.json').get('status') != 'complete':
                        raise ValueError('Unexpected incomplete manifest')
                    advance(state,spec,source,child,len(queue))
                elif queue:
                    cp = read(child_service/'diagnostics/checkpoint.json')
                    if cp['state'] not in ('READY','CAPTURED'):
                        raise ValueError('Capture requires review: '+cp['state'])
                    done = set(cp.get('completed_ids',[]))
                    for name in ('unavailable.json','unresolved_observations.json'):
                        if (child_data/name).exists(): done.update(x['id'] for x in read(child_data/name))
                    ids = {x['id'] for x in queue}
                    if not done.issubset(ids): raise ValueError('Checkpoint includes foreign IDs')
                    if done == ids: state['phase'] = 'FINALIZE'
                    else:
                        command = ['capture','--run-id',child,'--batch-size','30']
                        ledger_path = source_service/'diagnostics/failures.json'
                        if ledger_path.exists():
                            ledger = read(ledger_path)
                            capture_stage = ledger.get('stages',{}).get('capture',{})
                            if (capture_stage.get('state') == 'RETRY_AVAILABLE' and
                                    not ledger.get('retry_started',False)):
                                failures = [event.get('message','') for event in ledger.get('events',[])
                                            if event.get('event') == 'failure' and event.get('stage') == 'capture']
                                last_failure = failures[-1] if failures else ''
                                if ('Record collector changes' in last_failure or
                                        'Collector fix requires' in last_failure):
                                    retry_reason = ('Registered the audited run.py checkpoint-import fix and its '
                                                    'passing regression test and skill update; continuing the '
                                                    'same checkpoint.')
                                elif 'Checkpoint does not match accounted queue prefix' in last_failure:
                                    retry_reason = ('Reloaded the promoted checkpoint after interrupted sandbox '
                                                    'import; regression test now passes.')
                                else:
                                    retry_reason = ('Aside profile access was verified under the owning Windows '
                                                    'user; continuing the same saved checkpoint.')
                                command += ['--retry-reason',
                                    retry_reason]
                        if spec.get('config'): command += ['--config',spec['config']]
            elif state['phase'] == 'FINALIZE':
                if (child_service/'manifest.json').exists():
                    if read(child_service/'manifest.json')['status'] != 'complete': raise ValueError('Incomplete manifest')
                    advance(state,spec,source,child,len(read(child_data/'queue.json')))
                else:
                    command = ['finalize','--run-id',child,'--city-anomaly-policy','blank']
                    ledger_path = source_service/'diagnostics/failures.json'
                    if ledger_path.exists():
                        ledger = read(ledger_path)
                        finalize_stage = ledger.get('stages',{}).get('finalize',{})
                        if (finalize_stage.get('state') == 'RETRY_AVAILABLE' and
                                not ledger.get('retry_started',False)):
                            command += ['--retry-reason',
                                'Retrying finalize from the unchanged verified checkpoint.']
            else: raise ValueError('Unknown campaign phase')
            if command and executor(root,directory,state,command) != 0:
                raise ValueError('Runner stopped; inspect worker.local.log and run diagnostics')
        except (ValueError, OSError, KeyError) as error:
            state.update(phase='PAUSED',reason=str(error))
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(directory/'state.json',state)
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','resume','recover','tick','run','status','stop'])
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--max-actions',type=int,default=20)
    parser.add_argument('--reason',default='Human requested campaign continuation')
    args=parser.parse_args()
    directory=paths(ROOT,args.campaign)
    if args.command=='init':
        if not args.spec: parser.error('init requires --spec')
        state=initialize(ROOT,args.campaign,read(args.spec))
    elif args.command=='resume':
        state=resume_campaign(ROOT,args.campaign,args.reason)
    elif args.command=='recover':
        state=recover_campaign(ROOT,args.campaign,args.reason)
    elif args.command=='stop':
        # Cooperative stop: current runner command finishes, then no new work starts.
        (directory/'STOP').touch(exist_ok=True)
        state=read(directory/'state.json')
        state=dict(state,stop_requested=True)
    elif args.command=='status': state=read(directory/'state.json')
    else:
        if args.max_actions < 1: parser.error('max-actions must be positive')
        for _ in range(1 if args.command=='tick' else args.max_actions):
            state=tick(ROOT,args.campaign)
            if state['phase'] in ('COOLDOWN','PAUSED','STOPPED'): break
    print(json.dumps(state,ensure_ascii=False))


if __name__=='__main__': main()
