"""Persistent per-run failure budget shared by all agents; no automatic retries."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
import time

if os.name == "nt":
    import msvcrt
else:
    import fcntl
import json
import re


class HumanRequired(RuntimeError):
    pass


def ledger_dir(diag):
    marker = diag / 'shared-ledger.json'
    if not marker.exists():
        return diag
    data = json.loads(marker.read_text(encoding='utf-8'))
    parent = data.get('parent_run_id', '')
    if not isinstance(parent, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}', parent):
        raise HumanRequired('Invalid shared ledger pointer')
    target = diag.parent.parent / parent / 'diagnostics'
    if target == diag or not (target / 'failures.json').exists() or (target / 'shared-ledger.json').exists():
        raise HumanRequired('Shared collection ledger missing or invalid; do not reset the budget')
    return target


def _lock(stream, blocking):
    if os.name != 'nt':
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    # Lock the same byte in every process; do not truncate an existing lock file.
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b'\0')
        stream.flush()
    while True:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in (13, 36):
                raise
            if not blocking:
                raise BlockingIOError from exc
            time.sleep(0.05)


def _unlock(stream):
    if os.name == 'nt':
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def locked(diag):
    diag = ledger_dir(diag)
    diag.mkdir(parents=True, exist_ok=True)
    with (diag / 'failure.lock').open('a+b') as stream:
        _lock(stream, True)
        try:
            yield
        finally:
            _unlock(stream)


@contextmanager
def operation_locked(diag):
    """Only one actual CLI action may write/navigate for this run at a time."""
    diag = ledger_dir(diag)
    diag.mkdir(parents=True, exist_ok=True)
    with (diag / 'operation.lock').open('a+b') as stream:
        try:
            _lock(stream, False)
        except BlockingIOError:
            raise HumanRequired('Another collector action is running for this run; do not start a duplicate') from None
        try:
            yield
        finally:
            _unlock(stream)


def _read_ledger(path):
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode('utf-8'))
    except UnicodeDecodeError:
        # Legacy Windows runs used the system code page before UTF-8 was explicit.
        if os.name != 'nt':
            raise
        return json.loads(raw.decode('cp1251'))


def load(diag):
    diag = ledger_dir(diag)
    path = diag / 'failures.json'
    ledger = _read_ledger(path) if path.exists() else {
        'events': [], 'stages': {}, 'failures': 0, 'state': 'READY', 'retry_started': False}
    # Old stopped ledgers stay stopped. The new policy never grants human consent.
    ledger.setdefault('retry_limit', 5)
    if (diag / 'retry-limit.json').exists():
        policy = json.loads((diag / 'retry-limit.json').read_text(encoding='utf-8'))
        if policy != {'retry_limit': 1}:
            raise HumanRequired('Invalid collection retry policy')
        ledger['retry_limit'] = 1
    ledger.setdefault('retries_started', sum(e.get('event') == 'retry_started' for e in ledger['events']))
    ledger.setdefault('budget_start_failures', 0)
    if ledger['state'] == 'RETRY_STARTED':
        ledger.setdefault('retry_action_started', True)  # Legacy format cannot prove the action never began.
    if ledger['state'] == 'RETRY_STARTED' and not ledger.get('active_retry_stage'):
        ledger['active_retry_stage'] = next((e['stage'] for e in reversed(ledger['events'])
            if e.get('event') == 'retry_started'), None)
    return ledger


def save(diag, ledger):
    diag = ledger_dir(diag)
    path = diag / 'failures.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def ensure_allowed(diag, stage, retry_reason=None, *, action=False):
    """Reserve one retry per failed action; duplicate checks reuse that reservation."""
    diag = ledger_dir(diag)
    with locked(diag):
        ledger = load(diag)
        if ledger['state'] == 'NEEDS_HUMAN':
            if stage in ledger.get('offline_authorization', {}).get('stages', []):
                return
            raise HumanRequired('NEEDS_HUMAN: collection stopped; fresh human instruction required')
        if ledger['state'] == 'RETRY_STARTED':
            if ledger.get('active_retry_stage') != stage:
                raise HumanRequired('Finish the active retry with stage-ok or record-error before another stage')
            if action:
                if ledger.get('retry_action_started'):
                    raise HumanRequired('Retry action already started; confirm interruption and record-error before restarting it')
                ledger['retry_action_started'] = True
                save(diag, ledger)
            return  # Duplicate checks are harmless; an action can claim the retry only once.
        if ledger['failures'] > ledger['budget_start_failures'] and ledger['state'] != 'RECOVERED':
            if not retry_reason or not retry_reason.strip():
                raise HumanRequired('Retry requires --retry-reason describing the observed correction')
            if ledger['retries_started'] >= ledger['retry_limit']:
                raise HumanRequired('NEEDS_HUMAN: retry budget exhausted; fresh human instruction required')
            ledger['retries_started'] += 1
            ledger['retry_started'] = True
            ledger['retry_action_started'] = action
            ledger['active_retry_stage'] = stage
            ledger['state'] = 'RETRY_STARTED'
            ledger['events'].append(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage,
                event='retry_started', reason=retry_reason, retry_number=ledger['retries_started']))
            save(diag, ledger)


def succeeded(diag, stage, evidence):
    diag = ledger_dir(diag)
    with locked(diag):
        ledger = load(diag)
        if ledger['state'] == 'RETRY_STARTED':
            if ledger.get('active_retry_stage') != stage:
                raise ValueError('Success must confirm the active retry stage')
            ledger['state'] = 'RECOVERED'
            ledger['retry_started'] = False
            ledger.pop('retry_action_started', None)
            ledger.pop('active_retry_stage', None)
            ledger['events'].append(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage,
                                         event='recovered', evidence=evidence))
            save(diag, ledger)  # Success does not restore the shared retry budget.


def record(diag, stage, message, action, url=None, agent=None, immediate=False):
    diag = ledger_dir(diag)
    with locked(diag):
        ledger = load(diag)
        if ledger['state'] == 'RETRY_STARTED' and ledger.get('active_retry_stage') != stage and not immediate:
            raise ValueError('Record the actual active retry stage; another stage cannot cancel it')
        stopped = ledger['state'] == 'NEEDS_HUMAN'
        ledger.pop('offline_authorization', None)
        ledger['retry_started'] = False
        ledger.pop('retry_action_started', None)
        ledger.pop('active_retry_stage', None)
        current = ledger['stages'].setdefault(stage, {'failures': 0, 'state': 'RETRY_AVAILABLE'})
        current['failures'] += 1
        ledger['failures'] += 1
        ledger['state'] = 'NEEDS_HUMAN' if stopped or immediate or ledger['retries_started'] >= ledger['retry_limit'] or ((diag/'retry-limit.json').exists() and ledger['failures'] - ledger['budget_start_failures'] >= 2) else 'RETRY_AVAILABLE'
        current['state'] = ledger['state']
        event = dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, event='failure',
            attempt=ledger['failures'], state=ledger['state'], message=message,
            action=action, url=url, agent=agent, retry_budget_remaining=0 if ledger['state'] == 'NEEDS_HUMAN' else ledger['retry_limit'] - ledger['retries_started'])
        ledger['events'].append(event)
        save(diag, ledger)
        # Compact report contains no DOM dumps, tokens or local session contents.
        (diag/'failure-report.md').write_text(
            f'{event["state"]}\n\nЭтап: {stage}\nДействие: {action}\n'
            f'Страница: {url or "неизвестна"}\nОшибка: {message}\n'
            f'Неудачных попыток: {ledger["failures"]}; '
            f'осталось повторов: {event["retry_budget_remaining"]}.\n'
            'Смена агента, вкладки или URL не сбрасывает счётчик. '
            'При NEEDS_HUMAN новые переходы запрещены до инструкции человека.\n', encoding='utf-8')
        return event


def authorize_offline(diag, instruction):
    """Caller must have a fresh human instruction; this never permits browser work."""
    diag = ledger_dir(diag)
    if not instruction.strip():
        raise ValueError('Human instruction is required')
    with locked(diag):
        ledger = load(diag)
        authorization = dict(at=datetime.now(timezone.utc).isoformat(), instruction=instruction,
                             stages=['finalize', 'publish'])
        ledger['offline_authorization'] = authorization
        ledger['events'].append(dict(event='human_authorization', **authorization))
        save(diag, ledger)


def authorize_browser(diag, instruction, authorization_id):
    """Explicit human consent starts a bounded new budget, never erases history."""
    diag = ledger_dir(diag)
    if not instruction.strip() or not authorization_id.strip():
        raise ValueError('Fresh human instruction and a stable authorization-id are required')
    with locked(diag):
        ledger = load(diag)
        if any(e.get('event') == 'human_browser_authorization' and e.get('authorization_id') == authorization_id
               for e in ledger['events']):
            if ledger['state'] == 'NEEDS_HUMAN':
                raise HumanRequired('This human authorization was already used; a fresh instruction is required')
            return  # Replayed CLI invocation cannot replenish the budget.
        if ledger['state'] != 'NEEDS_HUMAN':
            raise ValueError('Browser resume requires a stopped run')
        ledger.pop('offline_authorization', None)
        ledger.pop('active_retry_stage', None)
        ledger.pop('retry_action_started', None)
        ledger['budget_start_failures'] = ledger['failures']
        ledger['retries_started'] = 0
        ledger['retry_limit'] = 1 if (diag / 'retry-limit.json').exists() else 5
        ledger['retry_started'] = False
        ledger['state'] = 'READY'
        ledger['events'].append(dict(at=datetime.now(timezone.utc).isoformat(),
            event='human_browser_authorization', instruction=instruction,
            authorization_id=authorization_id, budget_start_failures=ledger['failures'], retry_limit=ledger['retry_limit']))
        save(diag, ledger)
        (diag/'failure-report.md').write_text(
            'RESUMED_BY_HUMAN\n\nИстория ошибок сохранена. Разрешена новая ограниченная попытка '
            f'с {ledger["retry_limit"]} допустимыми повторами; ограничения сайта требуют немедленной остановки.\n', encoding='utf-8')
