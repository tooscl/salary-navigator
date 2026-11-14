"""Reusable Aside collection: freeze, capture/resume, finalize, publish. Run --help."""
import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from queue import Queue, Empty
import subprocess
import sys
import time
import uuid
from threading import Event, Thread
from urllib.parse import urlsplit, parse_qs

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
from utils.build_dataset import build
from utils.title_flags import military_flags
from check_capture import check
from failure_log import ensure_allowed, record as record_failure, succeeded, HumanRequired, authorize_offline, authorize_browser, operation_locked, locked, load as load_failures, save as save_failures
from preflight import find_aside, aside_command
from utils.dataset_pr import validate_automation_changes
from sweep_mode import (init_sweep as sweep_init_impl, register_branch as sweep_register_impl,
    set_branch_status as sweep_status_impl, materialize as sweep_materialize_impl,
    validate_materialized as validate_sweep_materialized, render_report as sweep_report_impl,
    demo_export as sweep_demo_impl)

AUTOMATION_FILES = ['Scraper/skills/collect-avito/references/top-professions.json', 'utils/title_flags.py', 'Scraper/extract.js', 'Scraper/skills/collect-avito/SKILL.md'] + [
    'Scraper/skills/collect-avito/scripts/' + name for name in ('run.py', 'sweep_mode.py', 'browser_batch.js', 'check_capture.py', 'failure_log.py', 'preflight.py')]


class CollectionStopped(RuntimeError, ValueError):
    def __init__(self, message, state):
        super().__init__(message)
        self.immediate = state in ('NEEDS_HUMAN', 'REVIEW_REQUIRED')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_dirty_sources(plan):
    """Exclude only Git blobs proven byte-identical after Windows CRLF checkout."""
    dirty = set(plan.get('uncommitted_automation_files', []))
    for file in list(dirty):
        expected = plan.get('automation_sha256', {}).get(file)
        try:
            blob = subprocess.check_output(['git', 'show', plan['code_git_commit'] + ':' + file], cwd=ROOT, stderr=subprocess.DEVNULL)
        except (KeyError, subprocess.CalledProcessError):
            continue
        if expected in (hashlib.sha256(blob).hexdigest(), hashlib.sha256(blob.replace(b'\n', b'\r\n')).hexdigest()):
            dirty.remove(file)
    return dirty


def instruction_version(path):
    """Read the single declared version from the same skill pinned by the plan."""
    versions = re.findall(r'^Версия\s+(\d+\.\d+\.\d+),', path.read_text(encoding='utf-8'), re.MULTILINE)
    if len(versions) != 1:
        raise ValueError('SKILL.md must declare exactly one instruction version')
    return versions[0]


def dirty_automation(root):
    dirty = []
    for file in AUTOMATION_FILES:
        result = subprocess.run(['git', 'show', 'HEAD:' + file], cwd=root, capture_output=True)
        if result.returncode or result.stdout != (root/file).read_bytes():
            dirty.append(file)
    return dirty


def norm(value):
    return ' '.join(value.split())


def paths(root, run_id):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}', run_id):
        raise ValueError('Use a simple unique run_id, without paths')
    out = root / 'Data/avito_runs' / run_id
    service = root / 'Scraper/runs' / run_id
    return out, service, service / 'diagnostics'


def avito_url(url, identity=None):
    parsed = urlsplit(url)
    host = parsed.hostname or ''
    if (parsed.scheme != 'https' or not (host == 'avito.ru' or host.endswith('.avito.ru'))
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not re.match(r'^/[^/]+/vakansii(?:/|$)', parsed.path)):
        raise ValueError('Expected an observed HTTPS Avito /<region>/vakansii URL; see references/vacancy-entry-routes.md')
    if identity and not parsed.path.endswith('_' + identity):
        raise ValueError('Queue URL and ID differ')
    return url


def same_observed_url(expected, actual):
    """Only permit the observed removal of one literal default localPriority=0."""
    avito_url(expected)
    avito_url(actual)
    if expected == actual:
        return True
    def strip_default(url):
        base, separator, query = url.partition('?')
        if not separator or '#' in query:
            return None
        parts = query.split('&')
        if parts.count('localPriority=0') != 1 or any(p.startswith('localPriority=') and p != 'localPriority=0' for p in parts):
            return None
        rest = [p for p in parts if p != 'localPriority=0']
        return base + ('?' + '&'.join(rest) if rest else '')
    return strip_default(expected) == actual


def selection_profession(listing, scope=None, requested=''):
    if scope == 'top20':
        entries = read(SCRIPTS.parent / 'references/top-professions.json')['entries']
        return '; '.join(sorted((entry['profession'] for entry in entries), key=str.casefold))
    if scope == 'all':
        return ''
    return profession_from_listing(listing, requested)


def profession_from_listing(listing, requested=''):
    """Use checked profession controls, never a title/URL or another filter group."""
    controls = listing.get('selected_controls')
    if not isinstance(controls, list):
        raise ValueError('Listing selected_controls missing; cannot verify profession filter')
    labels = {}
    for control in controls:
        if not isinstance(control, dict):
            raise ValueError('Invalid selected control')
        marker = control.get('marker') or ''
        if not isinstance(marker, str) or not marker.startswith('params[144170]/'):
            continue
        label = norm(control.get('label') or '')
        if control.get('value') == 'all' or marker.endswith('/all'):
            continue
        if not label or label.casefold() in ('все', 'любая', 'любой'):
            raise ValueError('Selected profession has no specific visible label')
        labels.setdefault(label.casefold(), label)
    # Stable order also makes duplicate input/label controls harmless.
    observed = '; '.join(labels[key] for key in sorted(labels))
    audit = listing.get('profession_filter') or {}
    if not observed and (audit.get('observed') is not True or audit.get('selected_count') != 0):
        raise ValueError('Profession filter was not read; an empty selection does not prove an unfiltered listing')
    if requested and norm(requested).casefold() != observed.casefold():
        raise ValueError('Requested profession differs from observed profession filters')
    return observed


def has_checked_profession(listing):
    return any((c.get('marker') or '').startswith('params[144170]/checkbox/')
               for c in listing.get('selected_controls', []) if isinstance(c, dict))


def validate_plan_profession(plan, listing_path):
    if 'profession' not in plan:
        raise ValueError('Plan profession is missing')
    if plan.get('listing_sha256') and sha(listing_path) != plan['listing_sha256']:
        raise ValueError('Frozen listing changed after freeze; preserve the original observation')
    observed = selection_profession(read(listing_path), plan.get('profession_scope'))
    if norm(plan.get('profession') or '') != observed:
        raise ValueError('Plan profession differs from frozen listing filters; do not export inconsistent metadata')
    for index, page in enumerate(plan.get('listing_pages', [])):
        path = listing_path.parent / page['file'].replace('\\', '/')
        if sha(path) != page['sha256']:
            raise ValueError('Frozen listing page changed after freeze')
        if index:
            current = read(path)
            if current.get('ui_transition'):
                validate_ui_transition(read(listing_path), read(listing_path.parent / plan['listing_pages'][index-1]['file'].replace('\\', '/')), current)
            else:
                validate_page_filters(read(listing_path), current)


def observed_profession_correction(out, service, plan, reason):
    """Reconcile a legacy incorrect all-scope from immutable, chained listing evidence."""
    if not reason or not reason.strip() or plan.get('profession_scope') != 'all' or plan.get('profession'):
        raise ValueError('Profession correction requires a reason and an empty legacy all-scope plan')
    info = plan.get('partition')
    if not info:
        raise ValueError('Profession correction requires a partitioned frozen queue')
    parent_path = out/'provenance'/'parent-plan.json'
    parent = read(parent_path)
    if sha(parent_path) != info['parent_plan_sha256'] or parent.get('profession_scope') != 'all' or parent.get('profession'):
        raise ValueError('Parent profession provenance differs')
    expected_child = dict(parent, run_id=plan['run_id'], limit=info['end']-info['start'],
        requested_limit=info['end']-info['start'], available_unique=info['end']-info['start'], partition=info)
    if plan != expected_child:
        raise ValueError('Child plan differs from parent provenance')
    pages = parent.get('listing_pages') or []
    if not pages or plan.get('listing_pages') != pages:
        raise ValueError('Frozen page provenance differs')
    observed = None
    checked = None
    previous = None
    for index, entry in enumerate(pages):
        path = out/entry['file'].replace('\\', '/')
        if sha(path) != entry['sha256']:
            raise ValueError('Frozen page hash differs')
        page = read(path)
        expected = parent['listing_request_url'] if index == 0 else previous
        if not expected or not same_observed_url(expected, page['source_url']) or page['source_url'] != entry['source_url']:
            raise ValueError('Frozen page URL chain differs')
        profession = profession_from_listing(page)
        controls = sorted((c.get('marker'), c.get('value'), c.get('label')) for c in page.get('selected_controls', [])
                          if (c.get('marker') or '').startswith('params[144170]/checkbox/'))
        if not profession or (observed is not None and (profession != observed or controls != checked)):
            raise ValueError('Observed profession controls differ or are empty')
        observed = profession
        checked = controls
        previous = page.get('pagination', {}).get('next_url')
    if sha(out/'listing.json') != pages[0]['sha256']:
        raise ValueError('First listing hash differs')
    return dict(method='observed_checked_controls_v1', reason=reason, original_profession=plan['profession'],
        corrected_profession=observed, original_profession_scope=plan['profession_scope'],
        original_filter_evidence=plan.get('filter_evidence'), observed_source_url=read(out/'listing.json')['source_url'],
        child_plan_sha256=sha(service/'plan.json'), parent_plan_sha256=sha(parent_path),
        parent_queue_sha256=info['parent_queue_sha256'], listing_pages=pages)

def page_number(url):
    values = parse_qs(urlsplit(url).query).get('p', ['1'])
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) < 1:
        raise ValueError('Invalid observed page number')
    return int(values[0])


def validate_ui_transition(first, previous, page):
    validate_page_filters(first, page)
    if str(page_number(page['source_url'])) not in page.get('pagination', {}).get('current_pages', []):
        raise ValueError('Current page marker differs from URL')
    if page_number(page['source_url']) != page_number(previous['source_url']) + 1:
        raise ValueError('Observed page number did not advance exactly once')
    if page.get('heading') != first.get('heading') or selection_profession(page, None) != selection_profession(first, None):
        raise ValueError('Profession controls changed between pages')
    transition = page.get('ui_transition') or {}
    if previous.get('pagination', {}).get('next_button') is True:
        key = lambda c: (c.get('marker') or '', c.get('value') or '', c.get('label') or '')
        if (sorted(map(key, first.get('selected_controls', []))) != sorted(map(key, page.get('selected_controls', [])))
                or not first.get('listing_context', {}).get('geography')
                or first.get('listing_context') != page.get('listing_context')
                or first['listing_context'].get('search') != ''):
            raise ValueError('Button transition changed filters or listing context')
    if (transition.get('method') != 'observed_ui_next' or not isinstance(transition.get('recovered'), bool)
            or (not transition['recovered'] and not transition.get('next_snapshot'))
            or transition.get('from_page') != page_number(previous['source_url'])
            or transition.get('landed_page') != page_number(page['source_url'])
            or transition.get('from_url') != previous['source_url'] or transition.get('next_href') != previous['pagination']['next_url'] or transition.get('landed_url') != page['source_url']
            or (previous['pagination'].get('next_button') is True and (transition.get('next_button') is not True or transition.get('observed_href') is not None))):
        raise ValueError('UI transition evidence differs from previous page')


def validate_page_filters(first, page):
    # The Next link comes from the already filtered listing. Do not reopen UI
    # filters or demand that collapsed controls render all selected professions.
    if urlsplit(first['source_url']).path != urlsplit(page['source_url']).path:
        raise ValueError('Listing region/category changed between pages')


def queue_from_pages(pages, limit, require_full=True):
    if not pages or (limit is not None and limit < 1) or any(p.get('state') != 'LISTING' for p in pages):
        raise ValueError('Expected a listing and positive limit')
    queue, seen = [], set()
    if any((page.get('listing_item_audit') or {}).get('malformed', 0) for page in pages):
        raise CollectionStopped('Listing has a rendered card without ID, URL or title', 'REVIEW_REQUIRED')
    for item in (item for page in pages for item in page.get('items', [])):
        identity = item.get('id', '')
        if not isinstance(identity, str) or not identity.isdigit():
            raise CollectionStopped('Invalid listing ID', 'REVIEW_REQUIRED')
        try:
            avito_url(item['url'], identity)
        except (KeyError, ValueError) as error:
            raise CollectionStopped(f'Invalid listing URL or ID: {error}', 'REVIEW_REQUIRED') from error
        if identity in seen:
            continue
        seen.add(identity)
        queue.append(dict(item, listing_rank=len(queue) + 1))
        if len(queue) == limit:
            return queue
    if require_full and limit is not None:
        raise ValueError(f'Only {len(queue)} unique rendered cards; requested {limit}; do not invent pagination')
    return queue


def queue_from(listing, limit):
    return queue_from_pages([listing], limit)



def verify_comparison(out, plan, queue):
    if plan.get('selection_criterion') != 'hours_11_12':
        return
    path = out/'comparison_ids.json'
    if sha(path) != plan.get('comparison_sha256'):
        raise ValueError('Dataset comparison snapshot changed')
    existing = set(read(path)['ids'])
    if any(item['id'] in existing for item in queue):
        raise ValueError('Candidate already exists in comparison dataset')


def remote_decisions(queue, out, target, criterion="remote"):
    """Validate a contiguous saved prefix; stop exactly at the target."""
    decisions = []
    accepted = 0
    missing = False
    for item in queue:
        path = out / 'raw' / (item['id'] + '.json')
        if not path.exists():
            missing = True
            continue
        if missing or accepted >= target:
            raise ValueError('Saved candidates must be a contiguous prefix ending at the target')
        data = read(path)
        result = check(data, item['id'])
        if not result['structural_check_passed'] or not data.get('parameter_audit', {}).get('rendered_complete'):
            raise ValueError(f'Saved {item["id"]} needs review; never overwrite or silently skip it')
        fields = [(norm(f['name']), norm(f['value'])) for f in data['fields']]
        if any(not name or not value for name, value in fields) or len({name for name, _ in fields}) != len(fields):
            raise ValueError('Empty or duplicate parameter requires review')
        field = 'Количество рабочих часов в день' if criterion == 'hours_11_12' else 'Формат работы'
        values = [value for name, value in fields if name == field]
        if len(values) > 1:
            raise ValueError('Duplicate format parameter requires review')
        match = bool(values and ('11–12' in [norm(v) for v in values[0].split(',')] if criterion == 'hours_11_12' else values == ['Удалённо']))
        decisions.append(dict(id=item['id'], listing_rank=item['listing_rank'],
            accepted=match, parameter=field, observed_value=values[0] if values else None,
            reason='exact_parameter_match' if match else 'parameter_missing_or_different'))
        accepted += int(match)
    return decisions

def pin_aside_target(diag):
    """Keep browser identity stable across preflight, capture and resume locally."""
    target_file = diag / 'aside-target.local.json'
    keys = ('TL_ASIDE_HOST', 'TL_ASIDE_ACCOUNT', 'TL_ASIDE_PROFILE')
    if target_file.exists():
        target = read(target_file)
        if any(os.environ.get(key) and os.environ[key] != target.get(key) for key in keys):
            raise RuntimeError('Aside host/account/profile differs from the pinned run; keep the original target')
        for key in keys:
            if target.get(key):
                os.environ[key] = target[key]
            else:
                os.environ.pop(key, None)
    else:
        executable = find_aside()
        if not executable:
            raise RuntimeError('Aside CLI not found; follow collect-avito/references/FIRST_RUN.md')
        command = aside_command(executable)
        target = dict(TL_ASIDE_HOST=command[command.index('--host')+1],
                      TL_ASIDE_ACCOUNT=command[command.index('--account')+1],
                      TL_ASIDE_PROFILE=command[command.index('--profile')+1])
        write(target_file, target)
        os.environ.update({key: value for key, value in target.items() if value})


def preflight(config_path=None, diag=None):
    if diag is not None:
        pin_aside_target(diag)
    command = [sys.executable, str(SCRIPTS / 'preflight.py'), '--probe-aside']
    if config_path is not None:
        command.extend(['--config', str(config_path)])
    subprocess.run(command, check=True, cwd=ROOT)


def import_sandbox(sandbox, out, diag):
    """Copy even a stopped batch; never replace an earlier raw observation."""
    for name in ['listing.json', 'checkpoint.json', 'timing.json', 'problem.snapshot.txt', 'browser-tab.json', 'next-comparison.json']:
        source = sandbox / name
        if source.exists():
            if name == 'checkpoint.json' and (diag/name).exists():
                previous = read(diag/name)
                write(diag/name, dict(previous, **read(source)))
            else:
                shutil.copy2(source, diag / name)
    raw_dir = sandbox / 'raw'
    if raw_dir.exists():
        (out / 'raw').mkdir(parents=True, exist_ok=True)
        for source in raw_dir.glob('*.json'):
            if not source.stem.isdigit():
                raise ValueError('Unexpected raw filename')
            target = out / 'raw' / source.name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f'Raw observation is immutable: {target.name}; use a new run')
            if not target.exists():
                temporary = target.with_name(target.name + '.tmp')
                shutil.copy2(source, temporary)
                temporary.replace(target)


def pipe_chunks(stream, stoppable=False):
    """Read subprocess stdout without select(), which only accepts sockets on Windows."""
    chunks = Queue()
    stop = Event()

    def read_chunks():
        try:
            while True:
                if stoppable and os.name != 'nt':
                    import select
                    if stop.is_set():
                        break
                    if not select.select([stream], [], [], .1)[0]:
                        continue
                chunk = stream.read1(65536)
                if not chunk:
                    break
                chunks.put(chunk)
        except (OSError, ValueError):
            pass  # EOF/error is reported by the waiting caller as disconnect.
        finally:
            chunks.put(None)

    reader = Thread(target=read_chunks, daemon=True)
    reader.stop_reading = stop
    reader.start()
    return chunks, reader


def preserve_browser_session_pointer(diag):
    """Archive an unreleased session before a later REPL replaces its pointer."""
    path = diag/'browser-session.local.json'
    if not path.exists(): return
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    destination = diag/f'browser-session-retained-{digest}.local.json'
    if destination.exists():
        if destination.read_bytes() != original:
            raise RuntimeError('Retained browser session hash collision; inspect diagnostics')
        path.unlink()
    else:
        path.rename(destination)


def retain_repl(process, diag, sandbox, job):
    """Keep our REPL pipes alive until an explicit local release marker is created."""
    # PIDs may be reused during one run; each keeper needs its own marker.
    release = diag / f'release-browser-session-{process.pid}-{uuid.uuid4().hex}'
    if release.exists():
        raise RuntimeError('Release marker collision; inspect diagnostics')
    # The keeper owns the REPL pipes. A close request is sent only after the
    # caller has validated and persisted its outcome; the acknowledgement must
    # arrive before the ordinary release marker can close those pipes.
    code = """import os,sys,time,threading
if os.name == 'nt':
    import msvcrt
    inp=msvcrt.open_osfhandle(int(sys.argv[1]),os.O_WRONLY)
    out=msvcrt.open_osfhandle(int(sys.argv[2]),os.O_RDONLY)
else:
    inp,out=int(sys.argv[1]),int(sys.argv[2])
release=sys.argv[3]
request=release+'.close'
ack=release+'.closed'
print('keeper_started',flush=True)
observed=bytearray()
def drain():
    while True:
        chunk=os.read(out,65536)
        if not chunk: break
        observed.extend(chunk)
        sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
    print('repl_stdout_eof',flush=True)
if os.name == 'nt':
    t=threading.Thread(target=drain,daemon=True); t.start()
else:
    import select
    t=None
sent=False
while not os.path.exists(release) and (t is None or t.is_alive()):
    if t is None and select.select([out],[],[],.05)[0]:
        chunk=os.read(out,65536)
        if not chunk: break
        observed.extend(chunk)
        sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
    if not sent and os.path.exists(request):
        with open(request,'rb') as f: command=f.read()
        os.write(inp,command+b'\\n'); sent=True
    if sent and not os.path.exists(ack) and b'TL_OWNED_TAB_CLOSED' in observed.replace(b'\\r\\n',b'\\n').splitlines():
        with open(ack,'wb') as f: f.write(b'closed\\n')
    time.sleep(.05)
if os.path.exists(release): print('release_marker_observed',flush=True)
os.close(inp); os.close(out)
"""
    log_path = diag / f'keeper-{process.pid}.local.log'
    with log_path.open('ab', buffering=0) as log:
        log.write(f'handoff_before repl_pid={process.pid} poll={process.poll()}\n'.encode())
        if os.name == 'nt':
            handles = [msvcrt.get_osfhandle(process.stdin.fileno()), msvcrt.get_osfhandle(process.stdout.fileno())]
            startup = subprocess.STARTUPINFO()
            startup.lpAttributeList = {'handle_list': handles}
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            try:
                keeper = subprocess.Popen([sys.executable, '-c', code, *map(str, handles), str(release)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, startupinfo=startup,
                    close_fds=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            finally:
                for handle in handles:
                    os.set_handle_inheritable(handle, False)
        else:
            keeper = subprocess.Popen([sys.executable, '-c', code, str(process.stdin.fileno()),
                str(process.stdout.fileno()), str(release)], pass_fds=(process.stdin.fileno(), process.stdout.fileno()),
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        log.write(f'handoff_after repl_pid={process.pid} poll={process.poll()} keeper_pid={keeper.pid} keeper_poll={keeper.poll()}\n'.encode())
    tab = read(sandbox/'browser-tab.json') if sandbox and (sandbox/'browser-tab.json').exists() else {}
    preserve_browser_session_pointer(diag)
    write(diag/'browser-session.local.json', dict(repl_pid=process.pid, keeper_pid=keeper.pid,
        sandbox=str(sandbox) if sandbox else None, tab=tab, release_marker=str(release)))


CLOSE_CONFIRM_TIMEOUT_SECONDS = 15


def release_browser(diag, close_owned=True):
    path = diag/'browser-session.local.json'
    if path.exists():
        session = read(path)
        tab = session.get('tab') or {}
        if close_owned and tab.get('ownership') == 'collector':
            target, url = tab.get('target_id'), tab.get('url')
            if (target is not None and (not isinstance(target, str) or not target)) or not isinstance(url, str) or not url:
                raise RuntimeError('Cannot prove owned browser tab identity; session preserved')
            marker = Path(session['release_marker'])
            request, ack = Path(str(marker)+'.close'), Path(str(marker)+'.closed')
            # The original REPL holds the Page returned by openTab and records
            # whether it was supplied externally. Check that ownership and URL
            # there; never attach or close an arbitrary target.
            expression = "(async()=>{const p=globalThis.tlCollectorPage;const id=" + json.dumps(target) + ";const url=" + json.dumps(url) + ";const clean=v=>{try{let u=new URL(v);return u.origin+u.pathname}catch(_){return null}};if(!p||globalThis.tlCollectorExternal!==false||clean(p.url())!==url)throw Error('Owned tab identity changed');if(id!==null){const found=(await listBrowserTabs()).filter(t=>t.targetId===id);if(found.length!==1||clean(found[0].url)!==url)throw Error('Owned tab target changed')}await closeTab(p);globalThis.tlCollectorPage=null;console.log('TL_OWNED_TAB_CLOSED')})().catch(e=>console.log('TL_CLOSE_ERROR='+String(e)))"
            expression = 'await eval(' + json.dumps(expression) + ')'
            temporary = request.with_name(request.name+'.tmp')
            temporary.write_text(expression, encoding='utf8')
            temporary.replace(request)
            deadline = time.monotonic()+CLOSE_CONFIRM_TIMEOUT_SECONDS
            while not ack.exists() and time.monotonic()<deadline:
                time.sleep(.05)
            if not ack.exists():
                raise RuntimeError('Owned tab close unconfirmed; session preserved')
        Path(session['release_marker']).touch()
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        destination = diag/f'browser-session-released-{digest}.local.json'
        if destination.exists():
            if destination.read_bytes() != original:
                raise RuntimeError('Released browser session hash collision; inspect diagnostics')
            path.unlink()
        else:
            path.rename(destination)


def launch_repl(command, cwd, interactive=False):
    if interactive and os.name != 'nt':
        import pty
        import tty
        master, slave = pty.openpty()
        try:
            # Raw mode disables echo and the canonical line-length limit: the
            # injected extractor is much larger than a terminal input line.
            tty.setraw(slave)
            process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                                       cwd=cwd, start_new_session=True)
            process.stdin = os.fdopen(os.dup(master), 'wb')
            process.stdout = os.fdopen(os.dup(master), 'rb')
            return process
        finally:
            os.close(master)
            os.close(slave)
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=cwd, start_new_session=True)

def browser_sources():
    return ((SCRIPTS / 'browser_batch.js').read_text(encoding='utf-8'),
            (ROOT / 'Scraper/extract.js').read_text(encoding='utf-8'))


def browser(job, out, diag):
    # One-shot Aside buffers stdout until completion. An interactive handshake
    # records its sandbox BEFORE collection, so interruption cannot orphan raw.
    executable = find_aside()
    if not executable:
        raise RuntimeError('Aside CLI not found; follow collect-avito/references/FIRST_RUN.md')
    process = launch_repl(aside_command(executable) + ['repl'], ROOT, interactive=True)
    incoming, reader = pipe_chunks(process.stdout, stoppable=os.name != 'nt')
    chunks = bytearray()
    def until(pattern, timeout, offset=0):
        deadline = time.monotonic() + timeout
        while True:
            output = chunks[offset:].decode('utf-8', errors='replace')
            found = re.search(pattern, output)
            if found:
                return found
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Aside timed out; sandbox pointer and partial raw preserved')
            try:
                chunk = incoming.get(timeout=remaining)
            except Empty:
                raise RuntimeError('Aside timed out; sandbox pointer and partial raw preserved') from None
            if chunk is None:
                raise RuntimeError('Aside disconnected; partial raw preserved')
            chunks.extend(chunk)
    sandbox = None
    preserve = True  # Any failure after launching REPL keeps its own session alive.
    stamp = str(time.time_ns())
    started = time.perf_counter()
    try:
        process.stdin.write(b'console.log("TL_SESSION="+pwd)\n'); process.stdin.flush()
        # Aside's interactive readline may echo our input even with terminal
        # echo disabled. Only a complete output line is a protocol marker.
        sandbox = Path(until(r'(?m)^TL_SESSION=([^\r\n]+)\r?\n', 20)[1].strip())
        if not sandbox.is_absolute() or not sandbox.is_dir():
            raise RuntimeError('Aside returned an invalid sandbox path')
        write(diag / 'sandbox.json', {'path': str(sandbox)})
        # One REPL RPC per navigation avoids Aside's per-evaluation 120s budget
        # becoming a timeout on the entire queue. No model calls between cards.
        jobs = [dict(job, close_tab=False)] if job['mode'] == 'freeze' else [
            dict(job, queue=[item], close_tab=False)
            for index,item in enumerate(job['queue'])]
        done = list(job['completed_ids'])
        for part in jobs:
            part['completed_ids'] = done
            script, extractor = browser_sources()
            script = script.replace('__JOB__', json.dumps(part, ensure_ascii=False), 1)
            script = script.replace('__EXTRACTOR__', json.dumps(extractor), 1)
            expression = 'await eval(' + json.dumps('(async()=>{' + script + '})()') + ')\n'
            offset = len(chunks)
            process.stdin.write(expression.encode()); process.stdin.flush()
            until(r'(?m)^TL_FINISHED\r?\n', 140, offset)
            import_sandbox(sandbox, out, diag)
            cp = read(diag/'checkpoint.json')
            cp['capture_seconds'] = job.get('previous_capture_seconds', 0) + time.perf_counter()-started
            write(diag/'checkpoint.json', cp)
            done = cp['completed_ids']
            if cp['state'] not in ('READY', 'CAPTURED'):
                break
            if job.get('selection_queue') is not None:
                decisions = remote_decisions(job['selection_queue'], out, job['selection_target'], job.get('selection_criterion', 'remote'))
                if sum(d['accepted'] for d in decisions) >= job['selection_target']:
                    break
    finally:
        if os.name != 'nt':
            reader.stop_reading.set()
            reader.join(timeout=2)
        if preserve and process.poll() is None and process.stdin and process.stdout:
            retain_repl(process, diag, sandbox, job)
        if process.stdin:
            process.stdin.close()
        if preserve and os.name != 'nt' and process.stdout:
            process.stdout.close()  # keeper owns its duplicate after handoff
        if not preserve:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate(); process.wait(timeout=3)
        if not preserve:
            process.stdout.close()
            reader.join(timeout=1)
        (diag / f'aside-{stamp}.local.log').write_text(chunks.decode('utf-8', errors='replace'), encoding='utf-8')
        if sandbox:
            import_sandbox(sandbox, out, diag)
    return read(diag / 'checkpoint.json'), time.perf_counter()-started


def archive_partial_listing(args):
    """Preserve a rejected pre-plan listing before one explicitly justified correction."""
    out, service, diag = paths(ROOT, args.run_id)
    if not args.reason.strip() or (service/'plan.json').exists() or (out/'queue.json').exists() or any((out/'raw').glob('*')):
        raise ValueError('Only an unplanned, uncaptured rejected listing can be archived')
    if (out/'listing_pages').exists() and any((out/'listing_pages').iterdir()):
        raise ValueError('Saved listing pages exist; do not discard partial freeze')
    if not (diag/'pending_listing.json').exists() or not (diag/'listing.json').exists():
        raise ValueError('No rejected pending listing to preserve')
    ledger = load_failures(diag)
    if ledger['state'] != 'RETRY_AVAILABLE' or not ledger['events'] or ledger['events'][-1].get('stage') != 'freeze':
        raise ValueError('Archive requires the recorded freeze failure and available correction')
    archive = service/'rejected-partial-listing'
    if archive.exists():
        raise ValueError('Rejected listing already archived')
    archive.mkdir(parents=True)
    names = ('freeze_request.json', 'pending_listing.json', 'listing.json', 'checkpoint.json', 'sandbox.json', 'timing.json', 'browser-tab.json')
    hashes = {}
    for name in names:
        source = diag/name
        if source.exists():
            target = archive/name
            shutil.copy2(source,target)
            hashes[name] = sha(target)
    write(archive/'provenance.json', dict(reason=args.reason, files=hashes, at=datetime.now(timezone.utc).isoformat()))
    for name in hashes: (diag/name).unlink()
    if (out/'listing_pages').exists(): (out/'listing_pages').rmdir()
    print('Rejected partial listing archived; failure budget preserved')


def archive_uncertain_freeze(args):
    out, service, diag = paths(ROOT, args.run_id)
    if not args.reason.strip() or (service/'plan.json').exists() or (out/'queue.json').exists() or any((out/'raw').glob('*')):
        raise ValueError('Only an unplanned, uncaptured freeze can be archived')
    ledger = load_failures(diag)
    if ledger.get('stages', {}).get('freeze', {}).get('state') != 'RETRY_AVAILABLE':
        raise ValueError('Archive requires a recorded freeze failure and available correction')
    archive = service/'uncertain-freeze'
    if archive.exists(): raise ValueError('Uncertain freeze already archived')
    archive.mkdir(parents=True)
    files = {}
    for root, names in ((out, ('listing_pages',)), (diag, ('freeze_request.json','pending_listing.json','listing.json','checkpoint.json','sandbox.json','timing.json','browser-tab.json','failures.json'))):
        for name in names:
            source = root/name
            if not source.exists(): continue
            target = archive/('data' if root == out else 'diagnostics')/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir(): shutil.copytree(source, target)
            else: shutil.copy2(source, target)
            for item in ([target] if target.is_file() else sorted(target.rglob('*'))):
                if item.is_file(): files[item.relative_to(archive).as_posix()] = sha(item)
    write(archive/'provenance.json', dict(reason=args.reason, files=files, at=datetime.now(timezone.utc).isoformat(),
        original_code_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        original_automation_sha256={file: hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + file], cwd=ROOT)).hexdigest() for file in AUTOMATION_FILES},
        corrected_automation_sha256={file: sha(ROOT/file) for file in AUTOMATION_FILES}))
    for name in ('freeze_request.json','pending_listing.json','listing.json','checkpoint.json','sandbox.json','timing.json','browser-tab.json'):
        (diag/name).unlink(missing_ok=True)
    if (out/'listing_pages').exists(): shutil.rmtree(out/'listing_pages')
    print('Uncertain freeze archived with hashes; failure history preserved')


def reselect(args):
    """Preserve a mistaken freeze before any card capture, keeping its failure ledger."""
    out, service, diag = paths(ROOT, args.run_id)
    ensure_allowed(diag, 'reselect')
    if not args.reason.strip():
        raise ValueError('A reason is required')
    if (service/'manifest.json').exists() or any((out/'raw').glob('*')):
        raise ValueError('Reselection is only allowed before card capture')
    archive = service/'superseded-freeze'
    if archive.exists():
        index = 2
        while (service/f'superseded-freeze-{index:04d}').exists():
            index += 1
        archive = service/f'superseded-freeze-{index:04d}'
    if not (service/'plan.json').is_file() or not (out/'queue.json').is_file():
        raise ValueError('A complete frozen plan and queue are required')
    archive.mkdir(parents=True)
    shutil.copytree(out, archive/'data')
    shutil.copy2(service/'plan.json', archive/'plan.json')
    write(archive/'reason.json', {'reason': args.reason, 'at': datetime.now(timezone.utc).isoformat()})
    write(archive/'hashes.json', {p.relative_to(archive).as_posix(): sha(p) for p in sorted(archive.rglob('*')) if p.is_file()})
    shutil.rmtree(out)
    (service/'plan.json').unlink()
    for name in ('freeze_request.json', 'pending_listing.json', 'listing.json', 'checkpoint.json', 'sandbox.json', 'timing.json'):
        (diag/name).unlink(missing_ok=True)
    print('Original freeze archived; failure history and retry budget preserved. Freeze the corrected observed listing.')


def continuation(args):
    """Seed a new freeze from immutable saved pages and the parent's observed Next."""
    parent_out, parent_service, parent_diag = paths(ROOT, args.parent_run_id)
    out, service, diag = paths(ROOT, args.run_id)
    ensure_allowed(parent_diag, 'continuation')
    if out.exists() or service.exists() or args.run_id == args.parent_run_id:
        raise ValueError('Continuation requires a new run ID')
    plan = read(parent_service/'plan.json')
    validate_plan_profession(plan, parent_out/'listing.json')
    queue = read(parent_out/'queue.json')
    pages = [read(parent_out/p['file'].replace('\\', '/')) for p in plan['listing_pages']]
    if not plan.get('requested_all_pages') or not pages or queue_from_pages(pages, None) != queue:
        raise ValueError('Continuation requires a complete saved all-pages parent queue')
    next_url = pages[-1].get('pagination', {}).get('next_url')
    if not next_url or any(same_observed_url(next_url, page['source_url']) for page in pages):
        raise ValueError('Parent has no unvisited observed Next')
    avito_url(next_url)
    request = read(parent_diag/'freeze_request.json')
    with locked(parent_diag):
        save_failures(parent_diag, load_failures(parent_diag))
    out.mkdir(parents=True); diag.mkdir(parents=True)
    for page in plan['listing_pages']:
        target = out/page['file'].replace('\\', '/'); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_out/page['file'].replace('\\', '/'), target)
    provenance = out/'continuation-provenance'; provenance.mkdir()
    shutil.copy2(parent_service/'plan.json', provenance/'parent-plan.json')
    shutil.copy2(parent_out/'queue.json', provenance/'parent-queue.json')
    write(diag/'freeze_request.json', request)
    write(diag/'shared-ledger.json', {'parent_run_id': args.parent_run_id})
    for name in ('automation_changes.json', 'aside-target.local.json'):
        if (parent_diag/name).exists():
            shutil.copy2(parent_diag/name, diag/name)
    info = dict(parent_run_id=args.parent_run_id, parent_plan_sha256=sha(provenance/'parent-plan.json'),
                parent_queue_sha256=sha(provenance/'parent-queue.json'), parent_page_count=len(pages), next_url=next_url)
    write(diag/'continuation.json', info)
    print(json.dumps(dict(run_id=args.run_id, continuation=info, request=request), ensure_ascii=False))


def freeze(args):
    out, service, diag = paths(ROOT, args.run_id)
    existing = out.exists() or (service/'plan.json').exists() or (diag/'freeze_request.json').exists()
    ensure_allowed(diag, 'freeze', getattr(args, 'retry_reason', None), action=True)
    if existing and not args.resume:
        raise ValueError('Run already exists; use freeze --resume if no queue exists, otherwise capture')
    if (service / 'manifest.json').exists() or (service / 'plan.json').exists():
        raise ValueError('Queue already frozen; use capture, not freeze')
    avito_url(args.url)
    candidate_limit = getattr(args, 'candidate_limit', None)
    all_pages = getattr(args, 'all_pages', False)
    branch_only = getattr(args, 'branch_only', False)
    max_pages = getattr(args, 'max_pages', None)
    if (all_pages and (args.limit is not None or candidate_limit is not None)) or (not all_pages and (args.limit is None or args.limit < 1)) or (candidate_limit is not None and candidate_limit < args.limit):
        raise ValueError('Use a positive --limit or --all-pages; candidate limit requires a numeric target')
    if branch_only and not all_pages:
        raise ValueError('Sweep branch freeze uses --branch-only with --all-pages')
    if max_pages is not None and (not branch_only or not 1 <= max_pages <= 30):
        raise ValueError('--max-pages is only for branch-only freeze and must be 1..30')
    accept_saved = getattr(args, 'accept_saved', False)
    instruction = getattr(args, 'human_instruction', None)
    if accept_saved and (not args.resume or not instruction or not instruction.strip()):
        raise ValueError('Accepting saved pages requires --resume and a human instruction')
    if not accept_saved:
        preflight(getattr(args, 'config', None), diag)
    started = time.perf_counter()
    diag.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    request = dict(url=args.url, limit=args.limit, candidate_limit=getattr(args, 'candidate_limit', None), geography=args.geography,
                   profession=args.profession, segments=args.segments, filter_evidence=args.filter_evidence)
    request.update(all_pages=all_pages, profession_scope=getattr(args, 'profession_scope', None))
    if branch_only:
        request.update(branch_only=True, max_pages=max_pages)
    ui_next = getattr(args, 'ui_next', False)
    if ui_next:
        if not getattr(args, 'tab_id', None): raise ValueError('UI Next requires an explicit --tab-id')
        request['ui_next'] = True
    if (out/'listing_pages').exists() and not (diag/'freeze_request.json').exists():
        raise ValueError('Frozen pages lack their original request; cannot safely resume')
    route_override = None
    if getattr(args, 'selection_criterion', 'remote') != 'remote':
        request['selection_criterion'] = args.selection_criterion
    if existing and (diag/'freeze_request.json').exists() and read(diag / 'freeze_request.json') != request:
        original = read(diag/'freeze_request.json')
        authorization_id = getattr(args, 'resume_route_authorization_id', None)
        first_path = out/'listing_pages/0001.json'
        if not (args.resume and ui_next and authorization_id and not original.get('ui_next')
                and dict(request, **{'ui_next': True}) == dict(original, **{'ui_next': True})
                and first_path.exists()):
            raise ValueError('Resume must preserve the original filter request')
        with locked(diag):
            ledger = load_failures(diag)
            auth_index = next((i for i in range(len(ledger['events']) - 1, -1, -1) if ledger['events'][i].get('event') == 'human_browser_authorization' and ledger['events'][i].get('authorization_id') == authorization_id), -1)
            auth = ledger['events'][auth_index] if auth_index >= 0 else None
            prior = [event for event in ledger['events'] if event.get('event') == 'freeze_route_override']
            last_failure = max((i for i, event in enumerate(ledger['events']) if event.get('event') == 'failure'), default=-1)
            standing = bool(auth and auth.get('unbounded_retries') and ledger.get('unbounded_retries'))
            if not auth or not auth.get('instruction') or ledger['state'] == 'NEEDS_HUMAN' or (not prior and not standing and auth_index <= last_failure):
                raise HumanRequired('UI Next resume requires a fresh recorded browser authorization')
            route_override = dict(authorization_id=authorization_id, first_page_sha256=sha(first_path), mode='ui_next')
            if prior and any(event.get('authorization_id') != authorization_id or event.get('first_page_sha256') != route_override['first_page_sha256'] for event in prior):
                raise HumanRequired('Existing freeze route override differs')
            if not prior:
                preserved = {}
                archive = diag/'pre-ui-next'
                archive.mkdir(exist_ok=True)
                for name in ('listing.json', 'pending_listing.json', 'checkpoint.json', 'sandbox.json', 'browser-tab.json'):
                    source, target = diag/name, archive/name
                    if source.exists():
                        if target.exists() and sha(source) != sha(target):
                            raise HumanRequired('Existing pre-UI Next evidence differs')
                        if not target.exists():
                            shutil.copy2(source, target)
                        preserved[name] = sha(target)
                ledger['events'].append(dict(event='freeze_route_override', at=datetime.now(timezone.utc).isoformat(), preserved=preserved, **route_override))
                save_failures(diag, ledger)
    if not (diag / 'freeze_request.json').exists():
        write(diag / 'freeze_request.json', request)
    page_dir = out / 'listing_pages'
    page_paths = sorted(page_dir.glob('*.json'), key=lambda p: int(p.stem)) if page_dir.exists() else []
    if [p.name for p in page_paths] != [f'{i:04d}.json' for i in range(1, len(page_paths)+1)]:
        raise ValueError('Listing page sequence is incomplete')
    pages = [read(path) for path in page_paths]
    for index, page in enumerate(pages):
        expected_url = args.url if index == 0 else pages[index-1].get('pagination', {}).get('next_url')
        if index and ui_next:
            validate_ui_transition(pages[0], pages[index-1], page)
        elif not expected_url or not same_observed_url(expected_url, page.get('source_url', '')):
            raise ValueError('Frozen listing page does not match the observed Next link')
        if index:
            validate_page_filters(pages[0], page)
    pending = diag / 'pending_listing.json'
    if pages and pending.exists() and read(pending) == {'url': pages[-1]['source_url']}:
        pending.unlink()
    cp = read(diag/'checkpoint.json') if (diag/'checkpoint.json').exists() else dict(state='READY', completed_ids=[])
    target = candidate_limit or args.limit
    comparison = None
    if getattr(args, 'selection_criterion', 'remote') == 'hours_11_12':
        if candidate_limit is None:
            raise ValueError('Hours selection requires --candidate-limit')
        comparison_path = out/'comparison_ids.json'
        if comparison_path.exists():
            comparison = read(comparison_path)
        else:
            dataset = ROOT/'Data/dataset/vacancies.csv'
            with dataset.open(encoding='utf-8-sig', newline='') as stream:
                ids = sorted({row['id'] for row in csv.DictReader(stream)})
            comparison = dict(dataset_sha256=sha(dataset), ids=ids)
            write(comparison_path, comparison)
    existing_ids = set(comparison['ids']) if comparison else set()
    listing_end_reason = None
    saved_pending = None
    if accept_saved:
        if candidate_limit is not None or not pages or any((out/'raw').glob('*')):
            raise ValueError('Saved-page acceptance requires complete saved pages and an ordinary request')
        if pending.exists():
            pending_data = read(pending)
            if pending_data != {'url': pages[-1].get('pagination', {}).get('next_url')} or cp.get('state') != 'BROWSER_ERROR':
                raise ValueError('Pending page must match the observed Next and a stopped browser error')
            saved_pending = dict(data=pending_data, sha256=sha(pending), checkpoint=cp)
        target = len(queue_from_pages(pages, None))
        if not target or (not all_pages and target >= args.limit):
            raise ValueError('Saved pages must contain a nonempty incomplete selection')
    empty_saved = bool(pages and pages[0].get('state') == 'EMPTY_RESULT')
    while not empty_saved and (not pages or target is None or len([x for x in queue_from_pages(pages, None, False) if x['id'] not in existing_ids]) < target):
        if pages and not ui_next and pages[-1].get('pagination', {}).get('next_button') is True:
            raise ValueError('Observed Next button requires UI Next mode')
        button_next = bool(pages and ui_next and pages[-1].get('pagination', {}).get('next_button') is True)
        url = args.url if not pages else (pages[-1]['source_url'] if button_next else pages[-1].get('pagination', {}).get('next_url'))
        if not url:
            listing_end_reason = 'confirmed_end' if pages[-1].get('pagination', {}).get('end_confirmed') else 'no_next_link'
            break
        if max_pages is not None and len(pages) >= max_pages:
            listing_end_reason = f'page_cap_{max_pages}'
            break
        avito_url(url)
        if pages and not ui_next:
            validate_page_filters(pages[0], {'source_url': url})
        if not button_next and any(same_observed_url(url, p['source_url']) for p in pages):
            listing_end_reason = 'repeated_next_link'
            break
        pending_value = {'ui_next_from': url} if button_next else {'url': url}
        recovering = pending.exists() and read(pending) == pending_value
        if pending.exists() and not recovering:
            raise ValueError('Pending listing does not match the observed Next link')
        if recovering and (diag / 'sandbox.json').exists():
            import_sandbox(Path(read(diag / 'sandbox.json')['path']), out, diag)
        if recovering and not ui_next and (diag/'checkpoint.json').exists() and read(diag/'checkpoint.json')['state'] == 'READY' and (diag/'listing.json').exists():
            cp = read(diag/'checkpoint.json')
        else:
            for name in ('sandbox.json', 'listing.json', 'checkpoint.json'):
                (diag/name).unlink(missing_ok=True)
            write(pending, pending_value)
            cp, _ = browser(dict(mode='freeze', url=url, completed_ids=[], tab_id=getattr(args, 'tab_id', None), allow_navigation=bool(pages), ui_next=ui_next, allow_empty=branch_only, previous=pages[-1] if pages and ui_next else None, first=pages[0] if pages and ui_next else None, geography=args.geography), out, diag)
        if cp['state'] != 'READY':
            raise CollectionStopped(f'Stopped: {cp["state"]}; inspect local diagnostics', cp['state'])
        page = read(diag / 'listing.json')
        if page.get('state') == 'EMPTY_RESULT':
            if not branch_only or pages or page.get('items') or not page.get('empty_evidence'):
                raise ValueError('Unexpected or unverified empty listing')
            path = page_dir / '0001.json'
            write(path, page)
            pages.append(page); page_paths.append(path)
            pending.unlink(missing_ok=True)
            listing_end_reason = 'empty_result'
            release_browser(diag)
            break
        if pages and ui_next:
            validate_ui_transition(pages[0], pages[-1], page)
        elif not same_observed_url(url, page.get('source_url', '')):
            raise ValueError('Loaded listing URL differs from observed link')
        if pages:
            validate_page_filters(pages[0], page)
        else:
            if ui_next and (page_number(page['source_url']) != 1 or '1' not in page.get('pagination', {}).get('current_pages', [])):
                raise ValueError('First page marker differs from URL')
            if getattr(args, 'profession_scope', None) == 'all' and has_checked_profession(page):
                raise ValueError('All-scope conflicts with observed checked profession controls')
            selection_profession(page, getattr(args, 'profession_scope', None), args.profession)
            if candidate_limit is not None and getattr(args, 'selection_criterion', 'remote') == 'remote' and not any(c.get('label') == 'Удалённо' and 'params[172815]' in (c.get('marker') or '') for c in page.get('selected_controls', [])):
                raise ValueError('Remote selection requires the observed Удалённо filter')
        queue_from_pages([page], target, False)  # validate every observed card before persisting
        path = page_dir / f'{len(pages)+1:04d}.json'
        write(path, page)
        pages.append(page); page_paths.append(path)
        pending.unlink(missing_ok=True)
        release_browser(diag)
    listing = pages[0]
    profession = selection_profession(listing, getattr(args, 'profession_scope', None), args.profession)
    if candidate_limit is not None and getattr(args, 'selection_criterion', 'remote') == 'remote' and not any(c.get('label') == 'Удалённо' and 'params[172815]' in (c.get('marker') or '') for c in listing.get('selected_controls', [])):
        raise ValueError('Remote selection requires the observed Удалённо filter')
    if candidate_limit is not None and candidate_limit < args.limit:
        raise ValueError('Candidate limit must be at least the target limit')
    queue = [] if listing.get('state') == 'EMPTY_RESULT' else [x for x in queue_from_pages(pages, None, require_full=False) if x['id'] not in existing_ids][:target]
    continuation_info = read(diag/'continuation.json') if (diag/'continuation.json').exists() else None
    if continuation_info:
        provenance = out/'continuation-provenance'
        if sha(provenance/'parent-plan.json') != continuation_info['parent_plan_sha256'] or sha(provenance/'parent-queue.json') != continuation_info['parent_queue_sha256']:
            raise ValueError('Continuation parent provenance changed')
        original = read(provenance/'parent-queue.json')
        if queue[:len(original)] != original:
            raise ValueError('Continuation parent queue differs from saved pages')
        queue = queue[len(original):]
    if not queue:
        if continuation_info and listing_end_reason:
            write(service/'continuation-empty.json', dict(continuation=continuation_info, listing_end_reason=listing_end_reason,
                available_listing_exhausted=listing_end_reason == 'confirmed_end', listing_pages=[dict(file=p.relative_to(out).as_posix(), sha256=sha(p)) for p in page_paths]))
            print(json.dumps({'state': 'EMPTY', 'listing_end_reason': listing_end_reason}))
            return
        if not (branch_only and listing_end_reason == 'empty_result'):
            raise ValueError('No vacancies were observed; cannot create an empty capture queue')
    write(out / 'listing.json', listing)
    write(out / 'queue.json', queue)
    write(service / 'plan.json', dict(run_id=args.run_id, geography=args.geography,
        profession=profession, segments=args.segments, filter_evidence=args.filter_evidence,
        listing_url=listing['source_url'], listing_request_url=args.url, selected_controls=listing.get('selected_controls', []),
        listing_sha256=sha(out / 'listing.json'),
        **({'resume_route_override': route_override} if route_override else {}),
        profession_scope=getattr(args, 'profession_scope', None),
        profession_source='selected_ui_context' if getattr(args, 'profession_scope', None) else 'observed_controls',
        listing_pages=[dict(file=path.relative_to(out).as_posix(), sha256=sha(path), source_url=page['source_url'])
                       for path, page in zip(page_paths, pages)],
        selection_mode='saved_incomplete_listing' if accept_saved else ('first_N_remote_parameter_matches' if candidate_limit is not None else 'first_N_unique_on_frozen_listing'), limit=args.limit if candidate_limit is not None else len(queue), requested_all_pages=all_pages,
        saved_selection_authorization={'instruction': instruction, 'at': datetime.now(timezone.utc).isoformat(), 'original_request': request, 'pending': saved_pending} if accept_saved else None,
        requested_limit=args.limit, available_unique=len(queue),
        listing_end_reason=listing_end_reason, available_listing_exhausted=listing_end_reason == 'confirmed_end', continuation=continuation_info,
        branch_only=branch_only, max_pages=max_pages,
        candidate_limit=candidate_limit, selection_criterion=getattr(args, 'selection_criterion', 'remote'),
        comparison_sha256=sha(out/'comparison_ids.json') if comparison else None,
        created_at=datetime.now(timezone.utc).isoformat(), instruction_version=instruction_version(ROOT/'Scraper/skills/collect-avito/SKILL.md'),
        extractor_sha256=sha(ROOT / 'Scraper/extract.js'), instruction_sha256=sha(SCRIPTS.parent / 'SKILL.md'),
        code_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        setup_seconds=time.perf_counter() - started, live_route_verified=False,
        execution_model=args.agent_model,
        automation_sha256={file: sha(ROOT/file) for file in AUTOMATION_FILES},
        uncommitted_automation_files=dirty_automation(ROOT)))
    cp.update(state='READY' if queue else 'EMPTY_RESULT', next_id=queue[0]['id'] if queue else None, capture_seconds=0)
    write(diag / 'checkpoint.json', cp)
    if not queue:
        write(service/'empty-listing.json', {'state':'EMPTY_RESULT','evidence':listing.get('empty_evidence'),
            'listing_end_reason':listing_end_reason,'listing_sha256':sha(out/'listing.json')})
    print(json.dumps({'state': 'READY', 'queued': len(queue), 'heading': listing.get('heading'),
                      'selected_controls': listing.get('selected_controls'),
                      'requested_limit': args.limit, 'listing_end_reason': listing_end_reason,
                      'message': 'Конец выдачи подтверждён' if listing_end_reason == 'confirmed_end' else 'Обход остановлен с ограничением' if listing_end_reason else 'Запрошенная очередь готова'}, ensure_ascii=False))


CLOSED_MARKERS = ('Объявление закрыто.', 'Объявление снято с публикации.')
REVIEW_MARKER = 'Оно ещё на проверке.'
BLOCKED_LISTING_MARKER = 'Оно заблокировано после проверки.'
DELETED_MARKER = 'Объявление удалено.'
USER_DELETED_MARKER = 'Пользователь его удалил.'
NO_LONGER_AVAILABLE_MARKER = 'Объявление больше недоступно.'
UNAVAILABLE_TO_VIEW_MARKER = 'Объявление недоступно для просмотра.'
REJECTED_MARKER = 'Оно отклонено, но снова появится в поиске, если его исправят.'
OTHER_UNAVAILABLE_MARKERS = (DELETED_MARKER, USER_DELETED_MARKER, NO_LONGER_AVAILABLE_MARKER, UNAVAILABLE_TO_VIEW_MARKER, REJECTED_MARKER)
ALL_UNAVAILABLE_MARKERS = (*CLOSED_MARKERS, REVIEW_MARKER, BLOCKED_LISTING_MARKER, *OTHER_UNAVAILABLE_MARKERS)
PREVIOUS_UNAVAILABLE_MARKERS = tuple(marker for marker in ALL_UNAVAILABLE_MARKERS if marker != REJECTED_MARKER)
LEGACY_UNAVAILABLE_MARKERS = tuple(marker for marker in PREVIOUS_UNAVAILABLE_MARKERS if marker != USER_DELETED_MARKER)
STANDING_UNAVAILABLE_ID = 'verified-unavailable-v1'
STANDING_UNAVAILABLE_SOURCE = 'project_skill'
STANDING_UNAVAILABLE_INSTRUCTION = ('Исключать отдельные карточки только по подтверждённому статусу недоступности, '
    'сохраняя raw, локальное доказательство, ID и причину; CAPTCHA, 429 и общие ограничения сайта останавливают сбор.')


def closed_raw_matches(data, identity, url, marker):
    if marker not in ALL_UNAVAILABLE_MARKERS or data.get('source_url') != url:
        return False
    if data.get('state') == 'UNKNOWN_PAGE':
        return marker in (CLOSED_MARKERS[0], REVIEW_MARKER, BLOCKED_LISTING_MARKER, *OTHER_UNAVAILABLE_MARKERS) and norm(data.get('heading') or '') == 'Объявление не посмотреть'
    return data.get('state') == 'DETAIL' and data.get('id') == identity and bool(data.get('title')) and marker == CLOSED_MARKERS[1]


def standing_unavailable_policy(args):
    """Register the human instruction once in the shared ledger and bind this child."""
    out, service, diag = paths(ROOT, args.run_id)
    if args.authorization_id != STANDING_UNAVAILABLE_ID or args.human_instruction != STANDING_UNAVAILABLE_INSTRUCTION:
        raise ValueError('Standing authorization must match the project policy')
    with locked(diag):
        ledger = load_failures(diag)
        if ledger['state'] == 'NEEDS_HUMAN':
            raise HumanRequired('Resolve the current stop before enabling this policy')
        existing = [event for event in ledger['events'] if event.get('event') == 'project_unavailable_policy' and event.get('authorization_id') == args.authorization_id]
        if existing and existing[-1]['instruction'] != args.human_instruction:
            raise ValueError('Authorization ID already identifies a different instruction')
        if not existing:
            ledger['events'].append(dict(event='project_unavailable_policy', at=datetime.now(timezone.utc).isoformat(),
                authorization_id=args.authorization_id, instruction=args.human_instruction, source=STANDING_UNAVAILABLE_SOURCE))
            save_failures(diag, ledger)
    policy = dict(authorization_id=args.authorization_id, human_instruction=args.human_instruction,
                  markers=list(ALL_UNAVAILABLE_MARKERS))
    path = out/'standing_unavailable_policy.json'
    if path.exists() and read(path) != policy:
        if read(path) not in (dict(policy, markers=list(LEGACY_UNAVAILABLE_MARKERS)), dict(policy, markers=list(PREVIOUS_UNAVAILABLE_MARKERS))):
            raise ValueError('Existing standing policy differs; preserve history')
    if not path.exists(): write(path, policy)
    print('Standing unavailable policy recorded for this run')


def closed_policy(args):
    out, service, diag = paths(ROOT, args.run_id)
    ledger = load_failures(diag)
    if ledger['state'] in ('NEEDS_HUMAN', 'RETRY_STARTED'):
        raise HumanRequired('Current browser authorization required')
    events = ledger['events']
    indexes = [i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization' and event.get('authorization_id') == args.authorization_id]
    if not indexes or any(event.get('state') == 'NEEDS_HUMAN' for event in events[indexes[-1]+1:]):
        raise HumanRequired('Current browser authorization required')
    instruction = events[indexes[-1]]['instruction']
    question = getattr(args, 'authorization_question', None)
    direct = ('закрыт' in instruction.lower() and 'последующ' in instruction.lower()
              and not re.search(r'\b(?:нет|не|кроме|только)\b', instruction.lower()))
    contextual = (question and re.match(r'^\s*да(?:\s|[,!.]|$)', instruction.lower())
                  and not re.search(r'\b(?:нет|не|кроме|только)\b', instruction.lower())
                  and 'закрыт' in question.lower() and 'последующ' in question.lower()
                  and 'все' in question.lower() and 'разреш' in question.lower()
                  and 'пропуск' in question.lower() and '?' in question)
    if not direct and not contextual:
        raise ValueError('Authorization must explicitly cover subsequent closed vacancies')
    path = out/'closed_policy.json'
    policy = dict(authorization_id=args.authorization_id, human_instruction=instruction, markers=list(CLOSED_MARKERS))
    if question:
        policy['authorization_question'] = question
    if path.exists() and read(path) != policy:
        raise ValueError('Existing closed policy differs; preserve history')
    if not path.exists(): write(path, policy)
    print('Closed-listing policy recorded for this run')


def review_policy(args):
    out, service, diag = paths(ROOT, args.run_id)
    ledger = load_failures(diag)
    if ledger['state'] in ('NEEDS_HUMAN', 'RETRY_STARTED'):
        raise HumanRequired('Current browser authorization required')
    events = ledger['events']
    indexes = [i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization' and event.get('authorization_id') == args.authorization_id]
    if not indexes or any(event.get('state') == 'NEEDS_HUMAN' for event in events[indexes[-1]+1:]):
        raise HumanRequired('Current browser authorization required')
    instruction = events[indexes[-1]]['instruction']
    if 'проверке' not in instruction.lower() or 'последующ' not in instruction.lower():
        raise ValueError('Authorization must explicitly cover subsequent listings under review')
    path = out/'review_policy.json'
    policy = dict(authorization_id=args.authorization_id, human_instruction=instruction, markers=[REVIEW_MARKER])
    if path.exists() and read(path) != policy:
        raise ValueError('Existing review policy differs; preserve history')
    if not path.exists(): write(path, policy)
    print('Under-review policy recorded for this run')


def blocked_listing_policy(args):
    out, service, diag = paths(ROOT, args.run_id)
    ledger = load_failures(diag)
    if ledger['state'] in ('NEEDS_HUMAN', 'RETRY_STARTED'):
        raise HumanRequired('Current browser authorization required')
    events = ledger['events']
    indexes = [i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization' and event.get('authorization_id') == args.authorization_id]
    if not indexes or any(event.get('state') == 'NEEDS_HUMAN' for event in events[indexes[-1]+1:]):
        raise HumanRequired('Current browser authorization required')
    instruction = events[indexes[-1]]['instruction']
    if 'заблокировано после проверки' not in instruction.lower() or 'последующ' not in instruction.lower():
        raise ValueError('Authorization must explicitly cover subsequent listings blocked after review')
    path = out/'blocked_listing_policy.json'
    policy = dict(authorization_id=args.authorization_id, human_instruction=instruction, markers=[BLOCKED_LISTING_MARKER])
    if path.exists() and read(path) != policy:
        raise ValueError('Existing blocked-listing policy differs; preserve history')
    if not path.exists(): write(path, policy)
    print('Blocked-listing policy recorded for this run')



UNRESOLVED_QUESTION = 'Разрешаете сохранить наблюдение по ID `7995516348` как неразрешённое, добавить проверяемую процедуру с тестом и продолжить оставшуюся очередь с этой карточки?'
UNRESOLVED_ANSWER = 'Да, разрешаю. Подобные вакансии можешь тоже сохарянть'
UNRESOLVED_AUTH = 'user-2026-10-29-unknown-page-policy-7995516348'


def unresolved_policy(args):
    out, service, diag = paths(ROOT, args.run_id)
    if args.authorization_id != UNRESOLVED_AUTH or args.authorization_question != UNRESOLVED_QUESTION:
        raise ValueError('Exact authorization question and ID required')
    ledger = load_failures(diag)
    events = ledger['events']
    auth = [i for i, e in enumerate(events) if e.get('event') == 'human_browser_authorization' and e.get('authorization_id') == args.authorization_id and e.get('instruction') == UNRESOLVED_ANSWER]
    if not auth or ledger['state'] in ('NEEDS_HUMAN', 'RETRY_STARTED') or any(e.get('state') == 'NEEDS_HUMAN' for e in events[auth[-1]+1:]):
        raise HumanRequired('Current exact browser authorization required')
    plan = read(service/'plan.json')
    if args.run_id != 'remote_1500_2026-10-29_shamil':
        raise ValueError('This authorization is limited to the named run')
    old = plan['extractor_sha256']; new = sha(ROOT/'Scraper/extract.js')
    original = subprocess.check_output(['git', 'show', plan['code_git_commit'] + ':Scraper/extract.js'], cwd=ROOT)
    if hashlib.sha256(original).hexdigest() != old:
        crlf = original.replace(b'\n', b'\r\n')
        if hashlib.sha256(crlf).hexdigest() != old: raise ValueError('Frozen extractor bytes cannot be reconstructed exactly')
        original = crlf
    if new == old: raise ValueError('Extractor upgrade must change the frozen extractor')
    changes = read(diag/'automation_changes.json') if (diag/'automation_changes.json').exists() else []
    changes = [dict(c, file=c.get('file', '').replace('\\', '/')) for c in changes]
    if not any(c.get('file', '').replace('\\', '/') == 'Scraper/extract.js' and c.get('reason') == 'Authorized unresolved observation evidence' and c.get('validation', '').strip() for c in changes):
        raise ValueError('Register the extractor change and test via note-change first')
    for digest, content in ((old, original), (new, (ROOT/'Scraper/extract.js').read_bytes())):
        dest = out/'automation'/f'extractor-{digest}.js'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != content: raise ValueError('Extractor archive collision')
        if not dest.exists(): dest.write_bytes(content)
    policy = dict(version=1, run_id=args.run_id, authorization_id=args.authorization_id, question=args.authorization_question,
        answer=UNRESOLVED_ANSWER, old_extractor_sha256=old, new_extractor_sha256=new)
    path = out/'unresolved_policy.json'
    if path.exists() and read(path) != policy: raise ValueError('Existing policy differs')
    if not path.exists(): write(path, policy)


HTTP429_OLD_BLOCK = '  const blockingText = /Доступ ограничен|проблема с IP|подтвердите.{0,30}(человек|робот)|проверка безопасности|Слишком много запросов|429 Too Many Requests/i;'
HTTP429_NEW_BLOCK = '  const blockingText = /Доступ ограничен|проблема с IP|подтвердите.{0,30}(человек|робот)|проверка безопасности|Слишком много запросов/i;'
HTTP429_OLD_RESTRICTION = '  const restrictionEvidence = /captcha|капч|429|too many requests|доступ ограничен|ограничени[ея] доступа|проблем[аы] с ip|ip.?адрес|подтвердите.{0,30}(человек|робот)|проверка безопасности|слишком много запросов|доступ заблокирован/i.test(normalizedEvidence);'
HTTP429_NEW_RESTRICTION = '  const http429 = /(?:\\bHTTP\\s*|ошибка\\s*|код\\s*)(?<!\\d)429(?!\\d)|(?<!\\d)429(?!\\d)\\s*(?:Too Many Requests|слишком много запросов)/i.test(normalizedEvidence) ||\n    /^\\s*429\\s*$/.test(normalizedEvidence) || headings.some(s => /^\\s*429\\s*$/.test(s));\n  const restrictionEvidence = http429 || /captcha|капч|too many requests|доступ ограничен|ограничени[ея] доступа|проблем[аы] с ip|подозрительная активность с вашего ip|ip.?адрес|подтвердите.{0,30}(человек|робот)|проверка безопасности|слишком много запросов|доступ заблокирован/i.test(normalizedEvidence);'
HTTP429_AUDIT_REASON = 'Audited HTTP 429 status boundary'


def policy_extractor_hashes(out, policy):
    hashes = {policy['old_extractor_sha256'], policy['new_extractor_sha256']}
    for index, change in enumerate(policy.get('extractor_upgrades', [])):
        previous = out/'automation'/('policy-' + change['previous_policy_sha256'] + '.json')
        if not previous.is_file() or sha(previous) != change['previous_policy_sha256']:
            raise ValueError('Prior policy archive differs')
        old_policy = read(previous)
        if (old_policy.get('new_extractor_sha256') != change['from_sha256'] or old_policy.get('extractor_upgrades', []) != policy['extractor_upgrades'][:index]
                or any(old_policy.get(k) != policy.get(k) for k in ('version', 'run_id', 'authorization_id', 'question', 'answer', 'old_extractor_sha256'))
                or change.get('reason') != HTTP429_AUDIT_REASON): raise ValueError('Extractor upgrade history differs')
        expected_to = policy['extractor_upgrades'][index+1]['from_sha256'] if index+1 < len(policy['extractor_upgrades']) else policy['new_extractor_sha256']
        if change['to_sha256'] != expected_to: raise ValueError('Extractor upgrade chain differs')
        hashes.update((change['from_sha256'], change['to_sha256']))
    for digest in hashes:
        archived = out/'automation'/f'extractor-{digest}.js'
        if not archived.is_file() or sha(archived) != digest: raise ValueError('Extractor archive differs')
    return hashes - {policy['old_extractor_sha256']}


def upgrade_unresolved_extractor(args):
    out, service, diag = paths(ROOT, args.run_id)
    if args.run_id != 'remote_1500_2026-10-29_shamil' or (service/'manifest.json').exists(): raise ValueError('Wrong or completed run')
    path = out/'unresolved_policy.json'; policy = read(path)
    if (policy.get('authorization_id') != UNRESOLVED_AUTH or policy.get('question') != UNRESOLVED_QUESTION or policy.get('answer') != UNRESOLVED_ANSWER): raise ValueError('Original authorization differs')
    policy_extractor_hashes(out, policy)
    before = policy['new_extractor_sha256']; after = sha(ROOT/'Scraper/extract.js')
    original = (out/'automation'/f'extractor-{before}.js').read_bytes()
    old = original.decode('utf8'); new = (ROOT/'Scraper/extract.js').read_text(encoding='utf8')
    if old.count(HTTP429_OLD_BLOCK) != 1 or old.count(HTTP429_OLD_RESTRICTION) != 1 or old.replace(HTTP429_OLD_BLOCK, HTTP429_NEW_BLOCK).replace(HTTP429_OLD_RESTRICTION, HTTP429_NEW_RESTRICTION) != new:
        raise ValueError('Upgrade must be exactly the audited HTTP 429 classifier fix')
    changes = read(diag/'automation_changes.json') if (diag/'automation_changes.json').exists() else []
    if not any(c.get('file', '').replace('\\', '/') == 'Scraper/extract.js' and c.get('reason') == HTTP429_AUDIT_REASON and c.get('validation', '').strip() for c in changes): raise ValueError('Record the extractor regression via note-change')
    previous_hash = sha(path); snapshot = out/'automation'/f'policy-{previous_hash}.json'; target = out/'automation'/f'extractor-{after}.js'
    for dest, content in ((snapshot, path.read_bytes()), (target, (ROOT/'Scraper/extract.js').read_bytes())):
        if dest.exists() and dest.read_bytes() != content: raise ValueError('Upgrade archive collision')
        if not dest.exists(): dest.write_bytes(content)
    policy['extractor_upgrades'] = policy.get('extractor_upgrades', []) + [dict(from_sha256=before,to_sha256=after,previous_policy_sha256=previous_hash,reason=HTTP429_AUDIT_REASON)]
    policy['new_extractor_sha256'] = after; write(path, policy)
    policy_extractor_hashes(out, policy)
    print(json.dumps(dict(previous_extractor_sha256=before,current_extractor_sha256=after,previous_policy_sha256=previous_hash)))


USER_SKIP_RUN = 'avito-yakutsk-20261106'
USER_SKIP_ID = '8557245056'


def user_skipped_records(out, queue):
    path = out/'user_skipped_observations.json'
    records = read(path) if path.exists() else []
    if not records: return []
    diag = paths(ROOT, out.name)[2]; seen = set()
    for record in records:
        identity = record['id']; matches = [item for item in queue if item['id'] == identity]
        if identity in seen: raise ValueError('Duplicate user skip ID')
        seen.add(identity)
        raw = out/'raw'/f'{identity}.json'
        if len(matches) != 1 or record.get('url') != matches[0]['url'] or sha(raw) != record.get('raw_sha256'):
            raise ValueError('User skip queue or raw differs')
        archive = diag/'observations'/identity
        for key, suffix in (('snapshot_sha256', 'problem.snapshot.txt'), ('browser_tab_sha256', 'browser-tab.json')):
            digest = record.get(key, ''); saved = archive/(digest+'.'+suffix)
            if not saved.is_file() or sha(saved) != digest: raise ValueError('User skip evidence archive differs')
        snap = (archive/(record['snapshot_sha256']+'.problem.snapshot.txt')).read_text(encoding='utf8')
        if read(archive/(record['browser_tab_sha256']+'.browser-tab.json')).get('url') != record['url']:
            raise ValueError('User skip browser URL differs')
        data = read(raw)
        if record.get('status') != 'user_skipped' or not record.get('instruction') or not record.get('authorization_id'):
            raise ValueError('User skip evidence or instruction differs')
        if record.get('reason') == 'user_directed_incomplete_observation':
            if out.name != USER_SKIP_RUN or identity != USER_SKIP_ID or USER_SKIP_ID not in record['instruction'] or skip_reason(data, snap, identity, record['url']) != 'incomplete_shell':
                raise ValueError('Original user skip evidence differs')
        else:
            policy_file = 'user_detail_skip_policy.json' if record.get('reason') == 'incomplete_detail_after_wait' else 'user_skip_policy.json'
            policy = read(out/policy_file)
            expected_scope = ['unrestricted_incomplete_detail_after_wait'] if policy_file == 'user_detail_skip_policy.json' else ['exact_unavailable_or_404', 'unrestricted_incomplete_shell']
            if (policy.get('run_id') != out.name or policy.get('authorization_id') != record['authorization_id']
                    or policy.get('instruction') != record['instruction'] or policy.get('scope') != expected_scope
                    or not any(e.get('event') == 'user_skip_policy' and all(e.get(k) == policy.get(k) for k in policy) for e in load_failures(diag)['events'])
                    or skip_reason(data, snap, identity, record['url']) != record.get('reason')):
                raise ValueError('Standing user skip evidence differs')
        events = load_failures(diag)['events']
        if not any(e.get('event') == 'user_skipped_observation' and all(e.get(key) == record.get(key) for key in record) for e in events):
            raise ValueError('User skip authorization event differs')
    return records


def skip_reason(data, snapshot, identity, url):
    if data.get('source_url') != url or f'[url={url}]' not in snapshot:
        raise ValueError('Skip URL differs from frozen queue')
    if re.search(r'captcha|капч|доступ ограничен|ограничени[ея] доступа|проблема с ip|подозрительн|too many requests|http\s*429|(?:ошибка|код)\s*429|проверка безопасности|^- heading "(?:Войдите|Авторизация|Требуется вход)', snapshot, re.I | re.M):
        raise ValueError('Access or security restriction is never skippable')
    ev = data.get('unresolved_evidence', {})
    if data.get('state') == 'UNKNOWN_PAGE' and ev.get('restriction') is False:
        markers = [m for m in ALL_UNAVAILABLE_MARKERS if any(norm(line) == f'- text: "{m}"' for line in snapshot.splitlines())]
        if len(markers) == 1 and ev.get('known_unavailable') is True and closed_raw_matches(data, identity, url, markers[0]):
            return 'unavailable: ' + markers[0]
        if (ev.get('known_unavailable') is False and any(line == f'- title: "{title}" [url={url}]' for line in snapshot.splitlines() for title in ('Ошибка 404. Страница не найдена', 'Ошибка 404', '404'))):
            return 'unavailable: exact 404 page title'
        if (ev.get('complete') is False and ev.get('known_unavailable') is False and data.get('heading') is None
                and not data.get('title') and not data.get('fields') and data.get('items') == [] and '\n- heading ' not in snapshot):
            return 'incomplete_shell'
    if data.get('state') == 'DETAIL':
        markers = [m for m in ALL_UNAVAILABLE_MARKERS if any(norm(line) == f'- text: "{m}"' for line in snapshot.splitlines())]
        if len(markers) == 1 and closed_raw_matches(data, identity, url, markers[0]): return 'unavailable: ' + markers[0]
        if (not markers and data.get('id') == identity and check(data, identity)['structural_check_passed']
                and f'- heading "{data["title"]}" [level=1]' in snapshot and data.get('parameter_audit', {}).get('rendered_complete') is False
                and data.get('parameter_audit', {}).get('expansion_controls') == [] and data.get('fields') == [] and data.get('params_blocks') == []
                and data.get('readiness_wait') == dict(attempts=10, interval_ms=500, same_url_and_id=True)
                and not re.search(r'^- title: "(?:Ошибка 404(?:\. Страница не найдена)?|404)"', snapshot, re.M)
                and not re.search(r'^\s*- heading "(?:Условия|Требования)"', snapshot, re.M)):
            return 'incomplete_detail_after_wait'
    raise ValueError('Observation is outside the standing skip policy')


def mark_user_skipped(args):
    if args.run_id != USER_SKIP_RUN or args.id != USER_SKIP_ID or args.id not in args.human_instruction or not args.authorization_id.strip():
        raise ValueError('Explicit instruction and authorization are limited to the specified run and ID')
    out, service, diag = paths(ROOT, args.run_id)
    if (service/'manifest.json').exists() or (out/'user_skipped_observations.json').exists():
        raise ValueError('Completed run or existing user skip cannot be replaced')
    cp = read(diag/'checkpoint.json'); queue = read(out/'queue.json')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected exactly one pending ID')
    item = matches[0]; index = queue.index(item)
    prior = {r['id'] for r in unavailable_records(out, queue)} | {r['id'] for r in unresolved_records(out, queue)} | {r['id'] for r in user_skipped_records(out, queue)}
    if cp.get('completed_ids') != [x['id'] for x in queue[:index] if x['id'] not in prior]: raise ValueError('Checkpoint prefix differs')
    incomplete_shell_evidence(out, diag, cp, item)
    ledger = load_failures(diag)
    failures = [e for e in ledger['events'] if e.get('event') == 'failure']
    if (ledger['state'] != 'NEEDS_HUMAN' or not failures or failures[-1].get('stage') != 'capture'
            or any(e.get('authorization_id') == args.authorization_id for e in ledger['events'])):
        raise HumanRequired('Stopped ledger and new authorization ID required')
    raw = out/'raw'/f'{args.id}.json'; snapshot = diag/'problem.snapshot.txt'
    sandbox = Path(read(diag/'sandbox.json')['path']); tab = sandbox/'browser-tab.json'
    archive = diag/'observations'/args.id; archive.mkdir(parents=True, exist_ok=True)
    for source in (snapshot, tab):
        destination = archive/(sha(source)+'.'+source.name)
        if destination.exists() and destination.read_bytes() != source.read_bytes(): raise ValueError('Evidence archive collision')
        if not destination.exists(): shutil.copy2(source, destination)
    record = dict(id=args.id, url=item['url'], status='user_skipped', reason='user_directed_incomplete_observation',
        raw_sha256=sha(raw), snapshot_sha256=sha(snapshot), browser_tab_sha256=sha(tab),
        authorization_id=args.authorization_id, instruction=args.human_instruction)
    ledger['events'].append(dict(event='user_skipped_observation', at=datetime.now(timezone.utc).isoformat(), **record))
    ledger['state'] = 'RECOVERED'; ledger['retry_started'] = False
    ledger.pop('active_retry_stage', None); ledger.pop('retry_action_started', None)
    save_failures(diag, ledger)
    write(out/'user_skipped_observations.json', [record])
    cp.update(state='READY', next_id=queue[index+1]['id'] if index+1 < len(queue) else None, stop_reason=None)
    write(diag/'checkpoint.json', cp); (diag/'sandbox.json').unlink(missing_ok=True)
    print(json.dumps({'user_skipped': args.id, 'next_id': cp['next_id']}))


def user_skip_policy(args):
    out, service, diag = paths(ROOT, args.run_id)
    if not args.authorization_id.strip() or not args.human_instruction.strip(): raise ValueError('Explicit standing user instruction and ID required')
    if (service/'manifest.json').exists(): raise ValueError('Completed run is immutable')
    policy = dict(version=1, run_id=args.run_id, authorization_id=args.authorization_id, instruction=args.human_instruction,
        scope=['exact_unavailable_or_404', 'unrestricted_incomplete_shell'])
    path = out/'user_skip_policy.json'
    with locked(diag):
        ledger = load_failures(diag)
        if path.exists():
            if read(path) != policy: raise ValueError('Existing standing policy differs')
            return
        if any(e.get('authorization_id') == args.authorization_id for e in ledger['events']): raise ValueError('Authorization ID already used')
        ledger['events'].append(dict(event='user_skip_policy', at=datetime.now(timezone.utc).isoformat(), **policy))
        save_failures(diag, ledger); write(path, policy)
    print('Standing user skip policy recorded for this run')


def user_detail_skip_policy(args):
    out, service, diag = paths(ROOT, args.run_id)
    if not args.authorization_id.strip() or not args.human_instruction.strip(): raise ValueError('Explicit standing user instruction and ID required')
    if (service/'manifest.json').exists(): raise ValueError('Completed run is immutable')
    policy = dict(version=1, run_id=args.run_id, authorization_id=args.authorization_id, instruction=args.human_instruction,
                  scope=['unrestricted_incomplete_detail_after_wait'])
    path = out/'user_detail_skip_policy.json'
    with locked(diag):
        ledger = load_failures(diag)
        if path.exists():
            if read(path) != policy: raise ValueError('Existing detail skip policy differs')
            return
        if any(e.get('authorization_id') == args.authorization_id for e in ledger['events']): raise ValueError('Authorization ID already used')
        ledger['events'].append(dict(event='user_skip_policy', at=datetime.now(timezone.utc).isoformat(), **policy))
        save_failures(diag, ledger); write(path, policy)
    print('Standing incomplete DETAIL policy recorded for this run')


def mark_policy_skipped(args):
    out, service, diag = paths(ROOT, args.run_id)
    if (service/'manifest.json').exists(): raise ValueError('Completed run is immutable')
    policies = [read(out/name) for name in ('user_skip_policy.json','user_detail_skip_policy.json') if (out/name).exists()]
    matches_policy = [p for p in policies if p.get('run_id') == args.run_id and p.get('authorization_id') == args.authorization_id]
    if len(matches_policy) != 1:
        raise ValueError('Standing user skip policy differs')
    policy = matches_policy[0]
    cp = read(diag/'checkpoint.json'); queue = read(out/'queue.json')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1 or cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != args.id:
        raise ValueError('Expected exact pending review ID')
    item = matches[0]; index = queue.index(item)
    prior = {r['id'] for r in unavailable_records(out, queue)} | {r['id'] for r in unresolved_records(out, queue)} | {r['id'] for r in user_skipped_records(out, queue)}
    if args.id in prior or cp.get('completed_ids') != [x['id'] for x in queue[:index] if x['id'] not in prior]: raise ValueError('Checkpoint prefix differs')
    raw = out/'raw'/f'{args.id}.json'; snapshot = diag/'problem.snapshot.txt'
    sandbox = Path(read(diag/'sandbox.json')['path']); tab = sandbox/'browser-tab.json'
    if ((sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or (sandbox/'problem.snapshot.txt').read_bytes() != snapshot.read_bytes()
            or any(read(sandbox/'checkpoint.json').get(k) != cp.get(k) for k in ('state','next_id','stop_reason','completed_ids'))
            or read(tab).get('url') != item['url']): raise ValueError('Stopped sandbox differs')
    reason = skip_reason(read(raw), snapshot.read_text(encoding='utf8'), args.id, item['url'])
    if (reason == 'incomplete_detail_after_wait') != (policy.get('scope') == ['unrestricted_incomplete_detail_after_wait']):
        raise ValueError('Observation is outside this policy scope')
    if reason == 'incomplete_detail_after_wait':
        previous = [r for r in user_skipped_records(out, queue) if r.get('reason') == reason]
        if previous:
            previous_index = next(i for i, candidate in enumerate(queue) if candidate['id'] == previous[-1]['id'])
            if not any(candidate['id'] in cp['completed_ids'] for candidate in queue[previous_index + 1:index]):
                raise ValueError('Consecutive incomplete DETAIL observations may indicate a parser failure')
    ledger = load_failures(diag)
    if not any(e.get('event') == 'user_skip_policy' and e.get('run_id') == args.run_id and e.get('authorization_id') == args.authorization_id and e.get('instruction') == policy['instruction'] for e in ledger['events']):
        raise HumanRequired('Standing policy authorization event missing')
    if ledger['state'] == 'NEEDS_HUMAN':
        failures = [e for e in ledger['events'] if e.get('event') == 'failure']
        if not failures or failures[-1].get('stage') != 'capture': raise HumanRequired('Unrelated human stop cannot be cleared')
    archive = diag/'observations'/args.id; archive.mkdir(parents=True, exist_ok=True)
    for source in (snapshot, tab):
        target = archive/(sha(source)+'.'+source.name)
        if target.exists() and target.read_bytes() != source.read_bytes(): raise ValueError('Evidence archive collision')
        if not target.exists(): shutil.copy2(source, target)
    record = dict(id=args.id, url=item['url'], status='user_skipped', reason=reason, raw_sha256=sha(raw),
        snapshot_sha256=sha(snapshot), browser_tab_sha256=sha(tab), authorization_id=args.authorization_id, instruction=policy['instruction'])
    records = read(out/'user_skipped_observations.json') if (out/'user_skipped_observations.json').exists() else []
    ledger['events'].append(dict(event='user_skipped_observation', at=datetime.now(timezone.utc).isoformat(), **record))
    if ledger['state'] == 'NEEDS_HUMAN': ledger['state'] = 'RECOVERED'
    save_failures(diag, ledger); records.append(record); write(out/'user_skipped_observations.json', records)
    cp.update(state='READY', next_id=queue[index+1]['id'] if index+1 < len(queue) else None, stop_reason=None)
    write(diag/'checkpoint.json', cp); (diag/'sandbox.json').unlink(missing_ok=True)
    print(json.dumps({'user_skipped': args.id, 'reason': reason, 'next_id': cp['next_id']}))


def unresolved_records(out, queue):
    path = out/'unresolved_observations.json'
    records = read(path) if path.exists() else []
    policy_path = out/'unresolved_policy.json'
    if records and not policy_path.exists(): raise ValueError('Missing unresolved policy')
    policy = read(policy_path) if records else None
    if records:
        if (policy.get('version') != 1 or policy.get('run_id') != out.name or policy.get('run_id') != 'remote_1500_2026-10-29_shamil' or policy.get('authorization_id') != UNRESOLVED_AUTH or policy.get('question') != UNRESOLVED_QUESTION or policy.get('answer') != UNRESOLVED_ANSWER): raise ValueError('Unresolved policy authorization differs')
        allowed_extractors = policy_extractor_hashes(out, policy)
    ids = set()
    for record in records:
        identity = record['id']; matches = [item for item in queue if item['id'] == identity]
        raw = out/'raw'/f'{identity}.json'
        if identity in ids or len(matches) != 1 or record.get('url') != matches[0]['url'] or not raw.is_file() or sha(raw) != record.get('raw_sha256'):
            raise ValueError('Unresolved observation does not match queue or raw')
        archive = paths(ROOT, out.name)[2]/'observations'/identity
        for key, name in (('snapshot_sha256', 'problem.snapshot.txt'), ('browser_tab_sha256', 'browser-tab.json')):
            saved = archive/(record.get(key, '') + '.' + name)
            if not saved.is_file() or sha(saved) != record.get(key): raise ValueError('Unresolved evidence archive differs')
        if read(archive/(record['browser_tab_sha256'] + '.browser-tab.json')).get('url') != record['url']: raise ValueError('Unresolved tab URL differs')
        avito_url(record['url'], identity)
        data = read(raw); evidence = data.get('unresolved_evidence', {})
        if (data.get('state') != 'UNKNOWN_PAGE' or data.get('source_url') != record['url'] or evidence != dict(version=1, complete=True, title_body_params_scanned=True, restriction=False, known_unavailable=False)
                or record.get('authorization_id') != policy['authorization_id'] or record.get('extractor_sha256') not in allowed_extractors):
            raise ValueError('Unresolved observation lacks complete new evidence')
        ids.add(identity)
    return records


def mark_unresolved(args):
    out, service, diag = paths(ROOT, args.run_id)
    if (service/'manifest.json').exists(): raise ValueError('Completed run is immutable')
    policy = read(out/'unresolved_policy.json')
    if policy['run_id'] != args.run_id or policy['authorization_id'] != args.authorization_id or sha(ROOT/'Scraper/extract.js') != policy['new_extractor_sha256']:
        raise ValueError('Policy or upgraded extractor differs')
    ledger = load_failures(diag)
    if ledger['state'] == 'NEEDS_HUMAN': raise HumanRequired('Current browser authorization required')
    cp = read(diag/'checkpoint.json'); queue = read(out/'queue.json')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != args.id:
        raise ValueError('Expected pending review checkpoint')
    excluded = {r['id'] for r in unavailable_records(out, queue)} | {r['id'] for r in unresolved_records(out, queue)} | {r['id'] for r in user_skipped_records(out, queue)}
    matches = [x for x in queue if x['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected one queue ID')
    index = queue.index(matches[0])
    if cp.get('completed_ids') != [x['id'] for x in queue[:index] if x['id'] not in excluded]: raise ValueError('Checkpoint prefix mismatch')
    raw = out/'raw'/f'{args.id}.json'; snapshot = diag/'problem.snapshot.txt'; pointer = diag/'sandbox.json'
    sandbox = Path(read(pointer)['path']); tab = read(sandbox/'browser-tab.json')
    if (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or (sandbox/'problem.snapshot.txt').read_bytes() != snapshot.read_bytes() or any(read(sandbox/'checkpoint.json').get(k) != cp.get(k) for k in ('next_id', 'state', 'stop_reason', 'completed_ids')) or tab.get('url') != matches[0]['url']:
        raise ValueError('New sandbox evidence or browser URL mismatch')
    records = unresolved_records(out, queue)
    record = dict(id=args.id, url=matches[0]['url'], status='unresolved_observation', raw_sha256=sha(raw), snapshot_sha256=sha(snapshot), browser_tab_sha256=sha(sandbox/'browser-tab.json'), extractor_sha256=policy['new_extractor_sha256'], authorization_id=args.authorization_id)
    # Validate before mutating the registry or checkpoint.
    data = read(raw); ev = data.get('unresolved_evidence', {})
    avito_url(matches[0]['url'], args.id)
    if data.get('source_url') != matches[0]['url'] or data.get('state') != 'UNKNOWN_PAGE' or ev != dict(version=1, complete=True, title_body_params_scanned=True, restriction=False, known_unavailable=False):
        raise ValueError('New observation lacks complete unrestricted evidence')
    archive = diag/'observations'/args.id
    archive.mkdir(parents=True, exist_ok=True)
    for source in (snapshot, sandbox/'browser-tab.json'):
        dest = archive/(sha(source) + '.' + source.name)
        if dest.exists() and dest.read_bytes() != source.read_bytes(): raise ValueError('Evidence archive collision')
        if not dest.exists(): shutil.copy2(source, dest)
    records.append(record); write(out/'unresolved_observations.json', records)
    cp.update(state='READY', next_id=queue[index+1]['id'] if index+1 < len(queue) else None, stop_reason=None)
    write(diag/'checkpoint.json', cp); pointer.unlink(missing_ok=True)
    print(json.dumps({'unresolved': args.id, 'next_id': cp['next_id']}))


def unavailable_records(out, queue):
    path = out/'unavailable.json'
    records = read(path) if path.exists() else []
    ids = set()
    for record in records:
        identity = record['id']
        matches = [item for item in queue if item['id'] == identity]
        raw = out/'raw'/f'{identity}.json'
        if identity in ids or len(matches) != 1 or record.get('url') != matches[0]['url'] or not raw.is_file() or sha(raw) != record.get('raw_sha256'):
            raise ValueError('Unavailable record does not match frozen queue or raw')
        data = read(raw)
        redirect_valid = (record.get('status') == 'unavailable_redirect' and data.get('state') == 'LISTING'
            and data.get('source_url') == record.get('observed_url')
            and record.get('evidence') == 'detail_redirected_to_category_listing_twice'
            and len(record.get('observation_sha256', [])) == 2)
        if not (redirect_valid or closed_raw_matches(data, identity, record['url'], record.get('evidence'))) or not record.get('authorization_id'):
            raise ValueError('Invalid unavailable observation')
        ids.add(identity)
    return records


def mark_unavailable(args):
    out, service, diag = paths(ROOT, args.run_id)
    if (service/'manifest.json').exists():
        raise ValueError('Completed run is immutable')
    ledger = load_failures(diag)
    offline_standing = getattr(args, 'offline_standing', False)
    if offline_standing and args.authorization_id != STANDING_UNAVAILABLE_ID: raise HumanRequired('Offline handling requires the standing project policy')
    if not offline_standing and (ledger['state'] == 'NEEDS_HUMAN' or (ledger['state'] == 'RETRY_STARTED' and not getattr(args, 'automatic', False))):
        raise HumanRequired('Resume browser authorization is required after the stop')
    events = ledger['events']
    auth = [i for i, event in enumerate(events) if event.get('event') in ('human_browser_authorization', 'project_unavailable_policy') and event.get('authorization_id') == args.authorization_id]
    latest_auth = max((i for i, event in enumerate(events) if event.get('event') in ('human_browser_authorization', 'project_unavailable_policy', 'recurring_ip_authorization', 'recurring_incomplete_authorization', 'recurring_502_authorization', 'recurring_unavailable_authorization', 'recurring_reviewed_authorization')), default=-1)
    if not auth or latest_auth < 0 or (not offline_standing and any(event.get('state') == 'NEEDS_HUMAN' for event in events[latest_auth+1:])):
        raise HumanRequired('Current human authorization required')
    instruction = events[auth[-1]]['instruction']
    policies = [read(path) for path in (out/'closed_policy.json', out/'review_policy.json', out/'blocked_listing_policy.json', out/'standing_unavailable_policy.json') if path.exists()]
    policy = next((p for p in policies if p.get('authorization_id') == args.authorization_id and p.get('human_instruction') == instruction and p.get('markers') in (list(CLOSED_MARKERS), [REVIEW_MARKER], [BLOCKED_LISTING_MARKER], list(ALL_UNAVAILABLE_MARKERS), list(PREVIOUS_UNAVAILABLE_MARKERS), list(LEGACY_UNAVAILABLE_MARKERS))), None)
    if offline_standing and (ledger['state'] != 'NEEDS_HUMAN' or not policy or events[auth[-1]].get('event') != 'project_unavailable_policy'):
        raise HumanRequired('Stopped run and recorded standing policy are required')
    if args.id not in instruction and not policy:
        raise ValueError('Human authorization must identify the unavailable ID or authorize its policy')
    cp = read(diag/'checkpoint.json')
    queue = read(out/'queue.json')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != args.id or args.id in cp.get('completed_ids', []):
        raise ValueError('Expected the reviewed pending checkpoint ID')
    if read(service/'plan.json').get('selection_mode') == 'first_N_remote_parameter_matches':
        raise ValueError('Unavailable handling is not supported for remote candidate selection')
    matches = [item for item in queue if item['id'] == args.id]
    prior_excluded = {record['id'] for record in unavailable_records(out, queue)} | {record['id'] for record in unresolved_records(out, queue)} | {record['id'] for record in user_skipped_records(out, queue)}
    pending_index = next((i for i, item in enumerate(queue) if item['id'] == args.id), None)
    if pending_index is None or cp.get('completed_ids') != [item['id'] for item in queue[:pending_index] if item['id'] not in prior_excluded]:
        raise ValueError('Checkpoint completed IDs must match the reviewed queue prefix')
    if len(matches) != 1:
        raise ValueError('Expected one frozen queue item')
    raw = out/'raw'/f'{args.id}.json'; data = read(raw)
    snap = diag/'problem.snapshot.txt'; snapshot = snap.read_text(encoding='utf-8')
    url = matches[0]['url']
    markers = [marker for marker in ALL_UNAVAILABLE_MARKERS if any(norm(line) == f'- text: "{marker}"' for line in snapshot.splitlines())]
    if len(markers) != 1 or not closed_raw_matches(data, args.id, url, markers[0]):
        raise ValueError('Missing exact closed-page evidence for the pending URL')
    tab_evidence = None
    if f'[url={url}]' not in snapshot:
        # A truncated snapshot URL is not identity evidence. Bind the original
        # browser tab observation to the same stopped sandbox and full queue URL.
        titles = re.findall(r'^- title: .*?\[url=([^\]]+)\]', snapshot, re.MULTILINE)
        evidence = data.get('unresolved_evidence', {})
        if (len(titles) != 1 or not titles[0].endswith('…')
                or not url.startswith(titles[0][:-1])
                or evidence.get('restriction') is not False or evidence.get('known_unavailable') is not True):
            raise ValueError('Missing truncated closed-page URL evidence')
        pointer = diag/'sandbox.json'
        if not pointer.exists(): raise ValueError('Missing exact closed-page URL evidence')
        sandbox = Path(read(pointer)['path']); tab = sandbox/'browser-tab.json'
        sandbox_cp = read(sandbox/'checkpoint.json')
        if (not tab.is_file() or read(tab).get('url') != url
                or (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes()
                or (sandbox/'problem.snapshot.txt').read_bytes() != snap.read_bytes()
                or any(sandbox_cp.get(k) != cp.get(k) for k in ('state', 'stop_reason', 'next_id', 'completed_ids'))):
            raise ValueError('Stopped sandbox URL evidence differs')
        avito_url(url, args.id)
        tab_evidence = tab
    standing_rejected = (markers[0] == REJECTED_MARKER and getattr(args, 'standing_rejected', False)
        and args.authorization_id == STANDING_UNAVAILABLE_ID and policy
        and policy['markers'] == list(PREVIOUS_UNAVAILABLE_MARKERS))
    if markers[0] == REJECTED_MARKER:
        ev = data.get('unresolved_evidence', {})
        pointer = diag/'sandbox.json'
        if (data.get('state') != 'UNKNOWN_PAGE' or ev.get('complete') is not True or ev.get('title_body_params_scanned') is not True or ev.get('restriction') is not False
                or f'- title: "Ошибка 404. Страница не найдена" [url={url}]' not in snapshot
                or re.search(r'captcha|капч|доступ ограничен|проблема с ip|too many requests|http\s*429', snapshot, re.I)
                or not pointer.exists()):
            raise ValueError('Rejected listing requires exact unrestricted 404 and original sandbox')
        sandbox = Path(read(pointer)['path']); sandbox_cp = read(sandbox/'checkpoint.json')
        if ((sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or (sandbox/'problem.snapshot.txt').read_bytes() != snap.read_bytes()
                or read(sandbox/'browser-tab.json').get('url') != url
                or any(sandbox_cp.get(k) != cp.get(k) for k in ('state','next_id','stop_reason','completed_ids'))):
            raise ValueError('Rejected listing sandbox evidence differs')
        evidence_archive = diag/'observations'/args.id; evidence_archive.mkdir(parents=True, exist_ok=True)
        for source in (snap, sandbox/'browser-tab.json'):
            dest = evidence_archive/(sha(source)+'.'+source.name)
            if dest.exists() and dest.read_bytes() != source.read_bytes(): raise ValueError('Evidence archive collision')
            if not dest.exists(): shutil.copy2(source,dest)
        tab_evidence = sandbox/'browser-tab.json'
    if policy and markers[0] not in policy['markers'] and not standing_rejected:
        raise ValueError('Observed marker is outside this authorization')
    records = unavailable_records(out, queue)
    if any(record['id'] == args.id for record in records):
        raise ValueError('Unavailable ID already recorded')
    pointer = diag/'sandbox.json'
    if pointer.exists():
        sandbox = Path(read(pointer)['path'])
        if (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or read(sandbox/'checkpoint.json').get('next_id') != args.id:
            raise ValueError('Sandbox differs from stopped observation')
    if offline_standing:
        avito_url(url, args.id)
        plan = read(service/'plan.json'); validate_partition(out, service); validate_plan_profession(plan, out/'listing.json')
        if not plan.get('listing_pages') or queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']], len(queue)) != queue: raise ValueError('Frozen queue differs')
        if (not pointer.exists() or (sandbox/'problem.snapshot.txt').read_bytes() != snap.read_bytes() or read(sandbox/'browser-tab.json').get('url') != url
                or any(read(sandbox/'checkpoint.json').get(k) != cp.get(k) for k in ('state','next_id','stop_reason','completed_ids'))
                or re.search(r'captcha|капч|доступ ограничен|проблема с ip|too many requests|http\s*429', snapshot, re.I)
                or data.get('state') != 'DETAIL' or f'- heading "{data["title"]}" [level=1]' not in snapshot):
            raise ValueError('Stopped evidence, heading, identity or sandbox differs')
        evidence_archive = diag/'observations'/args.id; evidence_archive.mkdir(parents=True, exist_ok=True)
        for source in (snap, sandbox/'browser-tab.json'):
            dest = evidence_archive/(sha(source)+'.'+source.name)
            if dest.exists() and dest.read_bytes() != source.read_bytes(): raise ValueError('Evidence archive collision')
            if not dest.exists(): shutil.copy2(source,dest)
    tab_digest = sha(tab_evidence) if tab_evidence else None
    if tab_evidence:
        archive = diag/'observations'/args.id; archive.mkdir(parents=True, exist_ok=True)
        destination = archive/(tab_digest + '.browser-tab.json')
        if destination.exists() and destination.read_bytes() != tab_evidence.read_bytes():
            raise ValueError('Browser tab archive collision')
        if not destination.exists(): shutil.copy2(tab_evidence, destination)
    # Preserve every verified snapshot before the next capture replaces it.
    archive = diag/'observations'/args.id; archive.mkdir(parents=True, exist_ok=True)
    archived_snapshot = archive/(sha(snap) + '.problem.snapshot.txt')
    if archived_snapshot.exists() and archived_snapshot.read_bytes() != snap.read_bytes():
        raise ValueError('Snapshot archive collision')
    if not archived_snapshot.exists(): shutil.copy2(snap, archived_snapshot)
    records.append(dict(browser_tab_sha256=tab_digest, id=args.id, url=url, status=('unavailable_under_review' if markers[0] == REVIEW_MARKER else 'unavailable_blocked_after_review' if markers[0] == BLOCKED_LISTING_MARKER else 'unavailable_rejected' if markers[0] == REJECTED_MARKER else 'unavailable_deleted' if markers[0] in (DELETED_MARKER, USER_DELETED_MARKER) else 'unavailable_other' if markers[0] in OTHER_UNAVAILABLE_MARKERS else 'unavailable_closed'), evidence=markers[0],
        snapshot_sha256=sha(snap), raw_sha256=sha(raw), observed_at=data.get('observed_at'),
        authorization_id=args.authorization_id, human_instruction=instruction,
        recorded_at=datetime.now(timezone.utc).isoformat()))
    write(out/'unavailable.json', records)
    index = next(i for i, item in enumerate(queue) if item['id'] == args.id)
    cp.update(state='READY', next_id=queue[index+1]['id'] if index+1 < len(queue) else None, stop_reason=None)
    write(diag/'checkpoint.json', cp)
    pointer.unlink(missing_ok=True)
    print(json.dumps({'unavailable': args.id, 'next_id': cp['next_id']}))


def salary_line_review(data, identity, url):
    """Validate a saved complete parameter block with a joined salary label/value."""
    if not check(data, identity)['structural_check_passed'] or data.get('source_url') != url:
        raise ValueError('Saved identity or URL mismatch')
    audit = data.get('parameter_audit', {})
    if audit.get('rendered_complete') is not False or audit.get('expansion_controls') != []:
        raise ValueError('Saved parameter audit is not the supported case')
    blocks = data.get('params_blocks')
    if not isinstance(blocks, list) or len(blocks) != 2 or not blocks[0].startswith('Условия\n') or not blocks[1].startswith('Требования\n'):
        raise ValueError('Expected complete conditions and requirements blocks')
    lines = [line.strip() for block in blocks for line in block.splitlines() if line.strip()]
    salary_lines = [line for line in lines if re.fullmatch(r'Оклад\s*[0-9][0-9\s]*(?:\s*[-–—]\s*[0-9][0-9\s]*)?\s*₽', line)]
    if len(salary_lines) != 1:
        raise ValueError('Expected exactly one joined salary line')
    salary_line = salary_lines[0]
    parsed = []
    for line in lines:
        if line in ('Условия', 'Требования') or line == salary_line: continue
        match = re.fullmatch(r'([^:]+):\s+(.+)', line)
        if not match: raise ValueError('Unrecognised parameter line')
        parsed.append((norm(match.group(1)), norm(match.group(2))))
    fields = data.get('fields') or []
    saved = [(norm(f['name']), norm(f['value'])) for f in fields]
    if parsed != saved or len({name for name, value in saved}) != len(saved) or any(name == 'Оклад' for name, value in saved):
        raise ValueError('Saved parameters do not match all visible lines')
    value = salary_line[len('Оклад'):].strip()
    result = dict(data)
    result['fields'] = fields + [dict(name='Оклад', value=value, values=[value], values_parse='one_visible_value', evidence_raw=salary_line)]
    result['parameter_audit'] = dict(audit, rendered_complete=True, offline_review='joined_salary_line_v1')
    return result


def reviewed_capture(out, item):
    path = out/'raw'/f"{item['id']}.json"
    data = read(path)
    review_path = out/'offline_reviews'/f"{item['id']}.json"
    if review_path.exists():
        review = read(review_path)
        if review.get('raw_sha256') != sha(path) or review.get('method') not in ('joined_salary_line_v1', 'address_button_v1') or not review.get('authorization_id'):
            raise ValueError('Offline review provenance mismatch')
        data = salary_line_review(data, item['id'], item['url']) if review['method'] == 'joined_salary_line_v1' else address_button_review(data, item['id'], item['url'])
    return data


def address_button_review(data, identity, url):
    if not check(data, identity)['structural_check_passed'] or data.get('source_url') != url: raise ValueError('Saved identity or URL mismatch')
    audit=data.get('parameter_audit', {}); blocks=data.get('params_blocks'); fields=data.get('fields') or []
    if audit.get('rendered_complete') is not False or not isinstance(blocks,list) or len(blocks)!=2 or not blocks[0].startswith('Условия\n') or not blocks[1].startswith('Требования\n'): raise ValueError('Expected both complete parameter blocks')
    lines=[line.strip() for block in blocks for line in block.split('\n') if line.strip()]; parsed=[]
    for line in lines:
        if line in ('Условия','Требования'): continue
        match=re.fullmatch(r'([^:]+):\s+(.+)',line)
        if not match: raise ValueError('Unrecognised parameter line')
        parsed.append((norm(match.group(1)),norm(match.group(2))))
    saved=[(norm(f['name']),norm(f['value'])) for f in fields]
    if parsed!=saved or len({name for name,value in saved})!=len(saved): raise ValueError('Fields do not match all parameter lines')
    addresses=[value for name,value in saved if name=='Где предстоит работать']
    controls=[norm(x) for x in audit.get('expansion_controls',[])]
    if len(addresses)!=1 or controls!=addresses or not re.search(r'ещ[её]|показать|раскрыть',addresses[0],re.I): raise ValueError('Not an address-button false positive')
    sections=[x.get('text') for x in data.get('source_sections',[]) if x.get('marker')=='item-view/item-params']
    if sections != blocks: raise ValueError('Parameter source sections differ')
    result=dict(data);result['parameter_audit']=dict(audit,rendered_complete=True,offline_review='address_button_v1')
    return result


def review_address_button(args):
    out,service,diag=paths(ROOT,args.run_id);cp=read(diag/'checkpoint.json');queue=read(out/'queue.json');ledger=load_failures(diag)
    if cp.get('state')!='REVIEW_REQUIRED' or cp.get('stop_reason')!='identity_or_template_review' or cp.get('next_id')!=args.id: raise ValueError('Expected pending review')
    policies=[e for e in ledger['events'] if (e.get('event')=='recurring_incomplete_policy' and e.get('source')==args.source) or (e.get('event')=='human_browser_authorization' and e.get('authorization_id')==args.source and e.get('unbounded_retries') is True and ledger.get('unbounded_retries') is True)]
    if len(policies)!=1 or (ledger['state']=='NEEDS_HUMAN' and getattr(args,'automatic',False)): raise HumanRequired('Recorded request policy required')
    direct = policies[0].get('event') == 'human_browser_authorization'
    if direct and ledger['state']=='NEEDS_HUMAN':
        failures=[e for e in ledger['events'] if e.get('event')=='failure']
        if not failures or failures[-1].get('stage')!='capture': raise HumanRequired('Unrelated stop cannot be cleared')
    matches=[x for x in queue if x['id']==args.id]
    if len(matches)!=1: raise ValueError('Expected one queue ID')
    index=queue.index(matches[0]);excluded={r['id'] for r in unavailable_records(out,queue)}|{r['id'] for r in unresolved_records(out,queue)}|{r['id'] for r in user_skipped_records(out,queue)}
    if cp.get('completed_ids') != [x['id'] for x in queue[:index] if x['id'] not in excluded]: raise ValueError('Checkpoint prefix mismatch')
    plan=read(service/'plan.json');validate_partition(out,service);validate_plan_profession(plan,out/'listing.json')
    if plan.get('run_id')!=args.run_id or queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']],len(queue))!=queue: raise ValueError('Frozen queue differs')
    url=matches[0]['url'];avito_url(url,args.id);raw=out/'raw'/f'{args.id}.json';data=read(raw);address_button_review(data,args.id,url)
    snap=diag/'problem.snapshot.txt';snapshot=snap.read_text(encoding='utf8');sandbox=Path(read(diag/'sandbox.json')['path']);tab=sandbox/'browser-tab.json';address=data['parameter_audit']['expansion_controls'][0]
    if (f'[url={url}]' not in snapshot or f'- heading "{data["title"]}" [level=1]' not in snapshot or f'- text: "Где предстоит работать:"\n- button "{address}"' not in snapshot
        or re.search(r'captcha|капч|доступ ограничен|проблема с ip|too many requests|(?<!\d)429(?!\d)|проверка безопасности|^- heading "(?:Войдите|Авторизация|Требуется вход)',snapshot,re.I|re.M)
        or (sandbox/'raw'/raw.name).read_bytes()!=raw.read_bytes() or (sandbox/'problem.snapshot.txt').read_bytes()!=snap.read_bytes() or read(tab).get('url')!=url
        or any(read(sandbox/'checkpoint.json').get(k)!=cp.get(k) for k in ('state','next_id','stop_reason','completed_ids'))): raise ValueError('Address-button snapshot or sandbox differs')
    evidence=diag/'observations'/args.id;evidence.mkdir(parents=True,exist_ok=True)
    for src in (snap,tab):
        dst=evidence/(sha(src)+'.'+src.name)
        if dst.exists() and dst.read_bytes()!=src.read_bytes(): raise ValueError('Evidence archive collision')
        if not dst.exists(): shutil.copy2(src,dst)
    dest=out/'offline_reviews'/f'{args.id}.json'
    if dest.exists(): raise ValueError('Offline review already exists')
    write(dest,dict(id=args.id,raw_sha256=sha(raw),method='address_button_v1',authorization_id=args.source,human_instruction=policies[0]['instruction'],snapshot_sha256=sha(snap),browser_tab_sha256=sha(tab),reviewed_at=datetime.now(timezone.utc).isoformat()))
    cp['completed_ids'].append(args.id);cp.update(state='READY',next_id=queue[index+1]['id'] if index+1<len(queue) else None,stop_reason=None);write(diag/'checkpoint.json',cp);(diag/'sandbox.json').unlink()
    if direct and ledger['state']=='NEEDS_HUMAN':
        ledger['events'].append(dict(event='address_button_review_recovered',at=datetime.now(timezone.utc).isoformat(),id=args.id,authorization_id=args.source,raw_sha256=sha(raw)))
        ledger['state']='RECOVERED'; ledger['retry_started']=False; save_failures(diag,ledger)
    print(json.dumps({'reviewed':args.id,'next_id':cp['next_id']}))


def salary_policy_direct_authorized(instruction):
    lower = instruction.lower()
    return 'продолж' in lower and ('исправ' in lower or all(term in lower for term in ('проверить', 'salary-review-policy', 'последующ', 'аналогич')))


def salary_review_policy(args):
    out, service, diag = paths(ROOT, args.run_id)
    ledger = load_failures(diag); events = ledger['events']
    if ledger['state'] in ('NEEDS_HUMAN', 'RETRY_STARTED'):
        raise HumanRequired('Current browser authorization required')
    auth = [i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization' and event.get('authorization_id') == args.authorization_id]
    if not auth or any(event.get('state') == 'NEEDS_HUMAN' for event in events[auth[-1]+1:]):
        raise HumanRequired('Current browser authorization required')
    instruction = events[auth[-1]]['instruction']
    lower = instruction.lower()
    direct = salary_policy_direct_authorized(instruction)
    dialogue = re.search(r'вопрос пользователю:\s*«([^»]+)»\s*ответ пользователя:\s*«([^»]+)»', lower)
    contextual = False
    if dialogue:
        question, answer = dialogue.groups()
        contextual = (all(term in question for term in ('разреш', 'обработ', 'карточк', 'проверк', 'последующ', 'аналогич', 'продолж'))
                      and re.match(r'^\s*да(?:\s|[,!.]|$)', answer)
                      and 'всех таких случа' in answer
                      and not re.search(r'\b(?:нет|не|кроме|только)\b', answer))
    standing = getattr(args, 'standing_technical', False)
    if not standing and (args.id not in instruction or not (direct or contextual)):
        raise ValueError('Authorization must request fixing this case and continuing')
    cp = read(diag/'checkpoint.json'); queue = read(out/'queue.json')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected one queue ID')
    if cp.get('next_id') != args.id:
        review_path = out/'offline_reviews'/f'{args.id}.json'
        if args.id not in cp.get('completed_ids', []) or not review_path.exists():
            raise ValueError('Expected pending or already reviewed completed ID')
        review = read(review_path)
        if review.get('authorization_id') != args.authorization_id or review.get('human_instruction') != instruction:
            raise ValueError('Existing review authorization differs')
        reviewed_capture(out, matches[0])
    else:
        salary_line_review(read(out/'raw'/f'{args.id}.json'), args.id, matches[0]['url'])
    path = out/'salary_review_policy.json'
    policy = dict(authorization_id=args.authorization_id, human_instruction=instruction, method='joined_salary_line_v1', initial_id=args.id)
    if path.exists() and read(path) != policy: raise ValueError('Existing salary review policy differs')
    if not path.exists(): write(path, policy)
    print('Salary-line review policy recorded for this run')


def review_salary_line(args):
    out, service, diag = paths(ROOT, args.run_id)
    if (service/'manifest.json').exists(): raise ValueError('Completed run is immutable')
    ledger = load_failures(diag); events = ledger['events']
    source=getattr(args,'source',None)
    if source:
        auth=[i for i,e in enumerate(events) if e.get('event')=='recurring_incomplete_policy' and e.get('source')==source]
        if len(auth)!=1 or args.authorization_id!=source: raise HumanRequired('Recorded request policy required')
        if ledger['state']=='NEEDS_HUMAN' and getattr(args,'automatic',False): raise HumanRequired('Automatic review requires active capture')
    else:
        if ledger['state'] == 'NEEDS_HUMAN' or (ledger['state'] == 'RETRY_STARTED' and not getattr(args, 'automatic', False)): raise HumanRequired('Current human authorization required')
        auth = [i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization' and event.get('authorization_id') == args.authorization_id]
        latest = max((i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization'), default=-1)
        if not auth or latest < 0 or any(event.get('state') == 'NEEDS_HUMAN' for event in events[latest+1:]): raise HumanRequired('Current browser authorization required')
        policy_path = out/'salary_review_policy.json'; policy = read(policy_path) if policy_path.exists() else {}
        policy_valid = policy.get('authorization_id') == args.authorization_id and policy.get('human_instruction') == events[auth[-1]]['instruction'] and policy.get('method') == 'joined_salary_line_v1'
        if args.id not in events[auth[-1]]['instruction'] and not policy_valid: raise HumanRequired('Authorization must identify this ID or cover this salary review')
    cp = read(diag/'checkpoint.json'); queue = read(out/'queue.json')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != args.id:
        raise ValueError('Expected the pending review checkpoint')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected one queue item')
    excluded = {record['id'] for record in unavailable_records(out, queue)} | {record['id'] for record in unresolved_records(out, queue)} | {record['id'] for record in user_skipped_records(out, queue)}
    index = next(i for i, item in enumerate(queue) if item['id'] == args.id)
    if cp.get('completed_ids') != [item['id'] for item in queue[:index] if item['id'] not in excluded]:
        raise ValueError('Checkpoint prefix mismatch')
    raw = out/'raw'/f'{args.id}.json'; salary_line_review(read(raw), args.id, matches[0]['url'])
    pointer = diag/'sandbox.json'
    if pointer.exists():
        sandbox = Path(read(pointer)['path'])
        if (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or read(sandbox/'checkpoint.json').get('next_id') != args.id:
            raise ValueError('Sandbox differs from stopped observation')
    if source:
        plan=read(service/'plan.json');validate_partition(out,service);validate_plan_profession(plan,out/'listing.json')
        if plan.get('run_id')!=args.run_id or queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']],len(queue))!=queue: raise ValueError('Frozen queue differs')
        data=read(raw);snap=diag/'problem.snapshot.txt';snapshot=snap.read_text(encoding='utf8');url=matches[0]['url'];avito_url(url,args.id)
        lines=[line.strip() for block in data['params_blocks'] for line in block.split('\n') if line.strip()];salary=next(line for line in lines if re.fullmatch(r'Оклад\s*[0-9][0-9\s]*(?:\s*[-–—]\s*[0-9][0-9\s]*)?\s*₽',line));amount=norm(salary[len('Оклад'):])
        sections=[x.get('text') for x in data.get('source_sections',[]) if x.get('marker')=='item-view/item-params']
        if (not pointer.exists() or sections!=data['params_blocks'] or f'[url={url}]' not in snapshot or f'- heading "{data["title"]}" [level=1]' not in snapshot
            or 'Оклад' not in snapshot or amount not in norm(snapshot) or re.search(r'captcha|капч|доступ ограничен|проблема с ip|too many requests|http\s*429',snapshot,re.I)
            or (sandbox/'problem.snapshot.txt').read_bytes()!=snap.read_bytes() or read(sandbox/'browser-tab.json').get('url')!=url
            or any(read(sandbox/'checkpoint.json').get(k)!=cp.get(k) for k in ('state','next_id','stop_reason','completed_ids'))): raise ValueError('Salary snapshot or sandbox differs')
        evidence=diag/'observations'/args.id;evidence.mkdir(parents=True,exist_ok=True)
        for src in (snap,sandbox/'browser-tab.json'):
            dst=evidence/(sha(src)+'.'+src.name)
            if dst.exists() and dst.read_bytes()!=src.read_bytes(): raise ValueError('Evidence archive collision')
            if not dst.exists(): shutil.copy2(src,dst)
    dest = out/'offline_reviews'/f'{args.id}.json'
    if dest.exists(): raise ValueError('Offline review already exists')
    write(dest, dict(id=args.id, raw_sha256=sha(raw), method='joined_salary_line_v1', authorization_id=args.authorization_id,
        human_instruction=events[auth[-1]]['instruction'], reviewed_at=datetime.now(timezone.utc).isoformat(),
        **(dict(snapshot_sha256=sha(snap),browser_tab_sha256=sha(sandbox/'browser-tab.json')) if source else {})))
    cp['completed_ids'].append(args.id)
    cp.update(state='READY', next_id=queue[index+1]['id'] if index+1 < len(queue) else None, stop_reason=None)
    write(diag/'checkpoint.json', cp); pointer.unlink(missing_ok=True)
    print(json.dumps({'reviewed': args.id, 'next_id': cp['next_id']}))


def validate_supplement_observation(parent, page, evidence):
    if (page.get('state') != 'LISTING' or page.get('heading') != evidence.get('heading')
            or page.get('source_url') != evidence.get('actual_url')
            or page.get('pagination', {}).get('current_pages') != evidence.get('current_pages')
            or evidence.get('listing_state') != 'LISTING' or evidence.get('protection') != 'not_detected'
            or evidence.get('visible_cards') != len(page.get('items', []))
            or evidence.get('search') != {'present': True, 'value': ''}
            or not evidence.get('profession', {}).get('checked')
            or evidence.get('profession', {}).get('label') != parent['profession']
            or not any(c.get('marker') == evidence.get('profession', {}).get('marker') and c.get('label') == parent['profession'] for c in page.get('selected_controls', []))
            or selection_profession(page) != parent['profession']
            or page.get('profession_filter', {}).get('observed') is not True
            or page.get('profession_filter', {}).get('selected_count') != sum(1 for c in page.get('selected_controls', []) if (c.get('marker') or '').startswith('params[144170]/'))
            or urlsplit(page['source_url']).path != urlsplit(parent['listing_url']).path):
        raise ValueError('Supplement listing evidence differs from parent filters')
    try:
        observed_at = datetime.fromisoformat(page['observed_at'].replace('Z', '+00:00'))
        evidence_at = datetime.fromisoformat(evidence['observed_at'].replace('Z', '+00:00'))
        if observed_at.tzinfo is None or evidence_at.tzinfo is None or abs((observed_at - evidence_at).total_seconds()) > 120:
            raise ValueError('Supplement evidence is not contemporaneous')
    except (KeyError, TypeError) as error:
        raise ValueError('Supplement observation times are required') from error
    if (not same_observed_url(evidence['source_url'], page['source_url'])
            or page.get('pagination', {}).get('current_pages') != [str(page_number(page['source_url']))]):
        raise ValueError('Supplement page marker differs from source')


def validate_supplement(out, service):
    plan = read(service/'plan.json'); info = plan['supplement']
    prov = out/'supplement-provenance'
    if set(info.get('source_hashes', {})) != {'parent-plan.json', 'parent-queue.json', 'observation.json', 'evidence.json'}:
        raise ValueError('Supplement provenance set differs')
    for name, digest in info['source_hashes'].items():
        if sha(prov/name) != digest:
            raise ValueError('Supplement provenance changed')
    parent = read(prov/'parent-plan.json'); original = read(prov/'parent-queue.json')
    page = read(out/'listing.json'); evidence = read(prov/'evidence.json')
    validate_supplement_observation(parent, page, evidence)
    if sha(out/'listing.json') != info['source_hashes']['observation.json']:
        raise ValueError('Supplement listing differs from observation')
    expected = [x for x in queue_from_pages([page], None) if x['id'] not in {i['id'] for i in original}]
    if read(out/'queue.json') != expected or not expected:
        raise ValueError('Supplement queue is not the new-ID selection')
    if read(service/'diagnostics/shared-ledger.json') != {'parent_run_id': info['parent_run_id']}:
        raise ValueError('Supplement shared ledger differs')
    validate_plan_profession(plan, out/'listing.json')


def supplemental(args):
    parent_out, parent_service, parent_diag = paths(ROOT, args.parent_run_id)
    out, service, diag = paths(ROOT, args.run_id)
    ensure_allowed(parent_diag, 'supplemental')
    if out.exists() or service.exists() or args.run_id == args.parent_run_id:
        raise ValueError('Supplement requires a new run ID')
    parent = read(parent_service/'plan.json'); validate_plan_profession(parent, parent_out/'listing.json')
    original = read(parent_out/'queue.json'); pages = [read(parent_out/x['file'].replace('\\', '/')) for x in parent['listing_pages']]
    if queue_from_pages(pages, None) != original:
        raise ValueError('Parent queue differs from frozen pages')
    if Path(args.observation).name != args.observation or Path(args.evidence).name != args.evidence:
        raise ValueError('Observation and evidence must be local diagnostic filenames')
    observation = parent_diag/args.observation; evidence = parent_diag/args.evidence
    page = read(observation); validate_supplement_observation(parent, page, read(evidence))
    known = {x['id'] for x in original}
    queue = [x for x in queue_from_pages([page], None) if x['id'] not in known]
    if not queue: raise ValueError('No new IDs in observed listing')
    out.mkdir(parents=True); diag.mkdir(parents=True); prov = out/'supplement-provenance'; prov.mkdir()
    files = {'parent-plan.json': parent_service/'plan.json', 'parent-queue.json': parent_out/'queue.json', 'observation.json': observation, 'evidence.json': evidence}
    for name, path in files.items(): shutil.copy2(path, prov/name)
    shutil.copy2(observation, out/'listing.json'); write(out/'queue.json', queue)
    info = dict(parent_run_id=args.parent_run_id, source_hashes={name: sha(path) for name, path in files.items()})
    plan = dict(parent, run_id=args.run_id, listing_url=page['source_url'], listing_sha256=sha(out/'listing.json'), listing_pages=[],
        selected_controls=page['selected_controls'], selection_mode='observed_supplement_new_ids', limit=len(queue), requested_limit=len(queue),
        available_unique=len(queue), requested_all_pages=False, available_listing_exhausted=False, listing_end_reason=None,
        partition=None, continuation=None, saved_selection_authorization=None, supplement=info, instruction_version=instruction_version(SCRIPTS.parent/'SKILL.md'),
        instruction_sha256=sha(SCRIPTS.parent/'SKILL.md'), code_git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        automation_sha256={file: sha(ROOT/file) for file in AUTOMATION_FILES}, uncommitted_automation_files=dirty_automation(ROOT))
    write(service/'plan.json', plan); write(diag/'shared-ledger.json', {'parent_run_id': args.parent_run_id})
    write(diag/'checkpoint.json', dict(state='READY', completed_ids=[], next_id=queue[0]['id'], capture_seconds=0))
    validate_supplement(out, service)
    print(json.dumps({'run_id':args.run_id, 'queued':len(queue)}, ensure_ascii=False))


def verified_frozen_queue(out, service):
    plan = read(service/'plan.json'); queue = read(out/'queue.json')
    if plan.get('selection_mode') != 'first_N_unique_on_frozen_listing' or any(plan.get(k) for k in ('partition','continuation','partial_suffix','supplement','deduplication')):
        raise ValueError('Deduplication requires an ordinary frozen queue')
    validate_plan_profession(plan, out/'listing.json')
    if not queue or not plan.get('listing_pages') or queue_from_pages([read(out/p['file'].replace('\\', '/')) for p in plan['listing_pages']], len(queue)) != queue:
        raise ValueError('Frozen queue differs from listing pages')
    if len({x['id'] for x in queue}) != len(queue): raise ValueError('Frozen queue IDs are not unique')
    manifest_path = service/'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path); files = {k.replace('\\', '/'): v for k,v in manifest['files'].items()}
        for f in (service/'plan.json', out/'queue.json'):
            key = f.relative_to(ROOT).as_posix(); record = files.get(key)
            if not record or record['sha256'] != sha(f) or record['bytes'] != f.stat().st_size:
                raise ValueError('Frozen artifact does not match its completed manifest')
    return plan, queue


def dataset_keys(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'source', 'id'}.issubset(reader.fieldnames or []):
            raise ValueError('Dataset lacks source and id columns')
        keys = [(row['source'], row['id']) for row in reader]
    if any(not source or not identity for source, identity in keys) or len(keys) != len(set(keys)):
        raise ValueError('Dataset has empty or duplicate keys')
    return sorted(keys)


def validate_deduplication(out, service):
    plan = read(service/'plan.json'); info = plan['deduplication']; prov = out/'dedup-provenance'
    source_out, source_service, _ = paths(ROOT, info['source_run_id'])
    ledger_pointers = [p for p in (service/'diagnostics/shared-ledger.json', service/'shared-ledger.json') if p.exists()]
    if not ledger_pointers or any(read(p) != {'parent_run_id': info['source_run_id']} for p in ledger_pointers):
        raise ValueError('Deduplicated child must use source ledger')
    source_plan_path = prov/'source-plan.json'; source_queue_path = prov/'source-queue.json'
    originals = (source_service/'plan.json', source_out/'queue.json', source_service/'dedup-child.json')
    present = [p.exists() for p in originals]
    if any(present) and not all(present): raise ValueError('Incomplete dedup source provenance')
    if all(present) and read(originals[2]) != {'child_run_id': plan['run_id']}:
        raise ValueError('Source child assignment differs')
    for local, original, expected in ((source_plan_path, originals[0], info['source_plan_sha256']), (source_queue_path, originals[1], info['source_queue_sha256'])):
        if sha(local) != expected or (all(present) and sha(original) != expected): raise ValueError('Dedup source provenance changed')
    parent = read(source_plan_path); source_queue = read(source_queue_path)
    validate_plan_profession(parent, out/'listing.json')
    if queue_from_pages([read(out/x['file'].replace('\\', '/')) for x in parent['listing_pages']], len(source_queue)) != source_queue:
        raise ValueError('Dedup source queue differs from frozen listing')
    known = set()
    for entry in info['against']:
        known_out, known_service, _ = paths(ROOT, entry['run_id'])
        for name, original, digest in ((entry['plan_file'],known_service/'plan.json',entry['plan_sha256']), (entry['queue_file'],known_out/'queue.json',entry['queue_sha256'])):
            if sha(prov/name) != digest or sha(original) != digest: raise ValueError('Dedup known provenance changed')
        known_plan, known_queue = verified_frozen_queue(known_out, known_service)
        known.update(x['id'] for x in known_queue)
    dataset = info.get('dataset')
    if dataset:
        manifest_path = prov/'dataset-manifest.json'; keys_path = prov/'dataset-keys.json'
        if sha(manifest_path) != dataset['manifest_sha256'] or sha(keys_path) != dataset['keys_sha256']:
            raise ValueError('Dataset dedup provenance changed')
        manifest = read(manifest_path); keys = read(keys_path)
        if manifest['sha256'] != dataset['csv_sha256'] or manifest['rows'] != len(keys) or keys != sorted(keys) or len(keys) != len({tuple(k) for k in keys}):
            raise ValueError('Dataset dedup basis differs')
        known.update(identity for source, identity in keys if source == 'avito')
    eligible = [item for item in source_queue if item['id'] not in known]
    novel = eligible[:info['selection_limit']] if info.get('selection_limit') is not None else eligible
    excluded = [item['id'] for item in source_queue if item['id'] in known]
    if 'eligible_count' in info and (info['eligible_count'] != len(eligible) or info['deferred_ids'] != [x['id'] for x in eligible[len(novel):]]):
        raise ValueError('Dedup deferred selection differs')
    if plan.get('run_id') != service.name or plan.get('limit') != len(novel) or len({e['run_id'] for e in info['against']}) != len(info['against']): raise ValueError('Deduplicated child metadata differs')
    if (len(source_queue) != info['source_count'] or len(novel) != info['novel_count'] or excluded != info['duplicate_ids'] or len(excluded) != info['duplicate_excluded'] or read(out/'queue.json') != novel):
        raise ValueError('Deduplicated queue or counts differ')
    for field in ('geography','profession','profession_scope','segments','filter_evidence','listing_url','listing_request_url','listing_sha256','listing_pages','selected_controls'):
        if plan.get(field) != parent.get(field): raise ValueError('Deduplicated filters differ from source')
    if read(out/'dedup_selection.json') != info: raise ValueError('Dedup selection differs from plan')
    if not novel:
        empty = read(out/'dedup-empty.json')
        if empty != {'state':'DEDUP_EMPTY','source_count':len(source_queue),'duplicate_excluded':len(excluded),'novel_count':0}: raise ValueError('Empty dedup result differs')
    elif (out/'dedup-empty.json').exists(): raise ValueError('Nonempty dedup queue cannot have empty marker')
    return info


def deduplicate_queue(args):
    source_out, source_service, source_diag = paths(ROOT, args.source_run_id)
    out, service, diag = paths(ROOT, args.run_id)
    ensure_allowed(source_diag, 'deduplicate-queue')
    if out.exists() or service.exists() or (source_service/'dedup-child.json').exists(): raise ValueError('Derivative or source assignment already exists')
    against = args.against_run_id or []
    use_dataset = getattr(args, 'against_dataset', False)
    limit = getattr(args, 'limit', None)
    if limit is not None and limit <= 0: raise ValueError('Selection limit must be positive')
    if (not against and not use_dataset) or len(set(against)) != len(against) or args.source_run_id in against or args.run_id in against or args.run_id == args.source_run_id:
        raise ValueError('Distinct source, child and known run IDs required')
    parent, source_queue = verified_frozen_queue(source_out, source_service)
    with locked(source_diag):
        if not (source_diag/'failures.json').exists(): save_failures(source_diag, load_failures(source_diag))
    cp = read(source_diag/'checkpoint.json')
    if (source_service/'manifest.json').exists() or (source_service/'partition.json').exists() or any((source_out/'raw').glob('*.json')) or cp.get('completed_ids') or cp.get('state') != 'READY':
        raise ValueError('Source must be frozen before any capture')
    known_runs = []; known = set()
    for index, run_id in enumerate(against):
        known_out, known_service, _ = paths(ROOT, run_id)
        _, queue = verified_frozen_queue(known_out, known_service)
        known.update(item['id'] for item in queue)
        known_runs.append(dict(run_id=run_id,plan_file=f'known-{index:03d}-plan.json',queue_file=f'known-{index:03d}-queue.json',plan_sha256=sha(known_service/'plan.json'),queue_sha256=sha(known_out/'queue.json')))
    dataset_manifest = ROOT/'Data/dataset/manifest.json'; dataset_csv = ROOT/'Data/dataset/vacancies.csv'
    dataset_info = None; keys = None
    if use_dataset:
        manifest = read(dataset_manifest)
        if sha(dataset_csv) != manifest['sha256']: raise ValueError('Materialized dataset does not match manifest')
        keys = dataset_keys(dataset_csv)
        if len(keys) != manifest['rows']: raise ValueError('Dataset row count differs from manifest')
        known.update(identity for source, identity in keys if source == 'avito')
    eligible = [item for item in source_queue if item['id'] not in known]
    novel = eligible[:limit] if limit is not None else eligible
    excluded = [item['id'] for item in source_queue if item['id'] in known]
    info = dict(source_run_id=args.source_run_id,source_plan_sha256=sha(source_service/'plan.json'),source_queue_sha256=sha(source_out/'queue.json'),against=known_runs,source_count=len(source_queue),duplicate_excluded=len(excluded),novel_count=len(novel),duplicate_ids=excluded)
    out.mkdir(parents=True); diag.mkdir(parents=True); prov=out/'dedup-provenance'; prov.mkdir()
    if use_dataset:
        shutil.copy2(dataset_manifest, prov/'dataset-manifest.json'); write(prov/'dataset-keys.json', keys)
        info['dataset'] = dict(csv_sha256=manifest['sha256'], manifest_sha256=sha(prov/'dataset-manifest.json'), keys_sha256=sha(prov/'dataset-keys.json'), rows=len(keys))
    if limit is not None:
        info.update(selection_limit=limit, eligible_count=len(eligible), deferred_ids=[x['id'] for x in eligible[len(novel):]])
    shutil.copy2(source_service/'plan.json',prov/'source-plan.json'); shutil.copy2(source_out/'queue.json',prov/'source-queue.json')
    for archive in sorted(source_service.glob('superseded-freeze*')):
        if not archive.is_dir(): continue
        target = prov/'rejected-freezes'/archive.name
        shutil.copytree(archive, target)
        actual = {p.relative_to(target).as_posix(): sha(p) for p in sorted(target.rglob('*')) if p.is_file() and p != target/'hashes.json'}
        if (target/'hashes.json').exists() and read(target/'hashes.json') != actual:
            raise ValueError('Superseded freeze archive hash differs')
        if not (target/'hashes.json').exists(): write(target/'hashes.json', actual)
    for entry in known_runs:
        known_out, known_service, _ = paths(ROOT, entry['run_id'])
        shutil.copy2(known_service/'plan.json',prov/entry['plan_file']); shutil.copy2(known_out/'queue.json',prov/entry['queue_file'])
    shutil.copy2(source_out/'listing.json',out/'listing.json')
    for page in parent['listing_pages']:
        target = out/page['file'].replace('\\', '/'); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source_out/page['file'].replace('\\', '/'),target)
    write(out/'queue.json',novel); write(out/'dedup_selection.json',info)
    if not novel: write(out/'dedup-empty.json',dict(state='DEDUP_EMPTY',source_count=len(source_queue),duplicate_excluded=len(excluded),novel_count=0))
    plan = dict(parent,run_id=args.run_id,limit=len(novel),available_unique=len(novel),deduplication=info,
        instruction_version=instruction_version(SCRIPTS.parent/'SKILL.md'),instruction_sha256=sha(SCRIPTS.parent/'SKILL.md'),
        code_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        automation_sha256={file:sha(ROOT/file) for file in AUTOMATION_FILES},uncommitted_automation_files=dirty_automation(ROOT))
    write(service/'plan.json',plan); write(diag/'shared-ledger.json',{'parent_run_id':args.source_run_id})
    write(diag/'checkpoint.json',dict(state='READY' if novel else 'DEDUP_EMPTY',completed_ids=[],next_id=novel[0]['id'] if novel else None,capture_seconds=0))
    for name in ('automation_changes.json','aside-target.local.json'):
        if (source_diag/name).exists(): shutil.copy2(source_diag/name,diag/name)
    write(source_service/'dedup-child.json',{'child_run_id':args.run_id})
    validate_deduplication(out,service)
    print(json.dumps({'state':'READY' if novel else 'DEDUP_EMPTY','run_id':args.run_id,'source_count':len(source_queue),'duplicate_excluded':len(excluded),'novel_count':len(novel)},ensure_ascii=False))


def validate_partition(out, service):
    plan = read(service/'plan.json')
    if plan.get('sweep'):
        validate_sweep_materialized(ROOT, out, service)
        return
    if plan.get('deduplication'):
        validate_deduplication(out, service)
        return
    if (service/'dedup-child.json').exists(): raise ValueError('Source queue assigned to deduplicated child')
    if plan.get('supplement'):
        validate_supplement(out, service)
        return
    suffix = plan.get('partial_suffix')
    if suffix:
        provenance = out/'partial-provenance'
        for name, digest in suffix['source_hashes'].items():
            if sha(provenance/name) != digest: raise ValueError('Partial source provenance changed')
        source_plan = read(provenance/'plan.json'); source_queue = read(provenance/'queue.json')
        selection = read(provenance/'selection.json'); manifest = read(provenance/'manifest.json')
        ids = [x['id'] for x in source_queue]; unresolved = selection['unresolved_ids']; cut = len(ids)-len(unresolved)
        if (not source_plan.get('partition') or not 0 < cut < len(ids) or ids[cut:] != unresolved
                or selection['plan_sha256'] != sha(provenance/'plan.json') or selection['queue_sha256'] != sha(provenance/'queue.json')
                or manifest.get('coverage', {}).get('partial') != selection or read(out/'queue.json') != source_queue[cut:]
                or set(selection['accepted_ids']) | set(selection['skipped_ids']) | set(selection.get('unresolved_observation_ids', [])) != set(ids[:cut])):
            raise ValueError('Derivative queue differs from partial source suffix')
        if read(service/'diagnostics/shared-ledger.json') != {'parent_run_id': suffix['ledger_parent_run_id']} or suffix['ledger_parent_run_id'] != source_plan['partition']['parent_run_id']:
            raise ValueError('Derivative shared ledger differs from frozen parent')
        validate_plan_profession(source_plan, out/'listing.json')
        return
    info = plan.get('partition')
    if not info:
        continuation_info = plan.get('continuation')
        if continuation_info:
            provenance = out/'continuation-provenance'
            if sha(provenance/'parent-plan.json') != continuation_info['parent_plan_sha256'] or sha(provenance/'parent-queue.json') != continuation_info['parent_queue_sha256']:
                raise ValueError('Continuation parent provenance changed')
            parent = read(provenance/'parent-plan.json')
            original = read(provenance/'parent-queue.json')
            pages = [read(out/p['file'].replace('\\', '/')) for p in plan['listing_pages']]
            count = continuation_info['parent_page_count']
            if count != len(parent['listing_pages']) or plan['listing_pages'][:count] != parent['listing_pages']:
                raise ValueError('Continuation pages differ from parent provenance')
            if not pages or pages[count-1].get('pagination', {}).get('next_url') != continuation_info['next_url']:
                raise ValueError('Continuation Next differs from saved parent')
            if count >= len(pages) or not same_observed_url(continuation_info['next_url'], pages[count]['source_url']):
                raise ValueError('Continuation did not follow the saved Next')
            combined = queue_from_pages(pages, None)
            if combined[:len(original)] != original or combined[len(original):] != read(out/'queue.json'):
                raise ValueError('Continuation queue differs from listing or parent')
            if read(service/'diagnostics'/'shared-ledger.json') != {'parent_run_id': continuation_info['parent_run_id']}:
                raise ValueError('Continuation shared ledger pointer changed')
        elif (service/'diagnostics'/'shared-ledger.json').exists():
            raise ValueError('Shared ledger child is missing partition metadata')
        if (service/'partition.json').exists():
            raise ValueError('Parent queue is partitioned; use its child runs')
        return
    provenance = out/'provenance'
    if sha(provenance/'parent-plan.json') != info['parent_plan_sha256'] or sha(provenance/'parent-queue.json') != info['parent_queue_sha256']:
        raise ValueError('Parent freeze provenance changed')
    parent = read(provenance/'parent-plan.json')
    queue = read(provenance/'parent-queue.json')
    validate_plan_profession(parent, out/'listing.json')
    pages = [read(out/item['file'].replace('\\', '/')) for item in parent['listing_pages']]
    if queue_from_pages(pages, len(queue)) != queue:
        raise ValueError('Parent queue does not match the frozen listing')
    start, end, size, index = info['start'], info['end'], info['batch_size'], info['index']
    if size < 1 or index < 1 or start != (index-1)*size or end != min(start+size, len(queue)) or start >= end:
        raise ValueError('Invalid partition range')
    if info['total'] != len(queue) or read(out/'queue.json') != queue[start:end]:
        raise ValueError('Child queue is not the specified contiguous parent range')
    siblings = info.get('siblings', [])
    if len(siblings) != (len(queue)+size-1)//size or len(set(siblings)) != len(siblings) or siblings[index-1] != plan.get('run_id'):
        raise ValueError('Invalid sibling assignments')
    parent_service = service.parent / info['parent_run_id']
    parent_partition = read(parent_service/'partition.json')
    if parent_partition != dict(children=siblings, batch_size=size, total=len(queue), parent_plan_sha256=info['parent_plan_sha256'], parent_queue_sha256=info['parent_queue_sha256']):
        raise ValueError('Partition group differs from parent record')
    marker = read(service/'diagnostics'/'shared-ledger.json')
    if marker != {'parent_run_id': info['parent_run_id']}:
        raise ValueError('Shared failure ledger pointer changed')


def partial_suffix(args):
    source_out, source_service, _ = paths(ROOT, args.source_run_id)
    out, service, diag = paths(ROOT, args.run_id)
    if out.exists() or service.exists(): raise ValueError('Derivative run already exists')
    manifest = validate_completed_export(source_out, source_service)
    selection = manifest['coverage'].get('partial')
    if not selection or not selection.get('unresolved_ids'): raise ValueError('Source must be a finalized partial partition')
    source_plan = read(source_service/'plan.json'); queue = read(source_out/'queue.json')
    count = len(selection['unresolved_ids']); remaining = queue[-count:]
    if [item['id'] for item in remaining] != selection['unresolved_ids']: raise ValueError('Source unresolved IDs are not the queue suffix')
    out.mkdir(parents=True); diag.mkdir(parents=True); provenance = out/'partial-provenance'; provenance.mkdir()
    source_files = {'plan.json': source_service/'plan.json', 'queue.json': source_out/'queue.json', 'selection.json': source_out/'exports/partial_selection.json', 'manifest.json': source_service/'manifest.json'}
    for name, path in source_files.items(): shutil.copy2(path, provenance/name)
    shutil.copy2(source_out/'listing.json', out/'listing.json')
    for page in source_plan['listing_pages']:
        target = out/page['file'].replace('\\', '/'); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source_out/page['file'].replace('\\', '/'), target)
    write(out/'queue.json', remaining)
    info = dict(source_run_id=args.source_run_id, ledger_parent_run_id=source_plan['partition']['parent_run_id'], source_hashes={name: sha(path) for name,path in source_files.items()})
    plan = dict(source_plan, run_id=args.run_id, partition=None, partial_suffix=info, limit=count, requested_limit=count, available_unique=count,
                instruction_version=instruction_version(SCRIPTS.parent/'SKILL.md'), instruction_sha256=sha(SCRIPTS.parent/'SKILL.md'),
                code_git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                automation_sha256={file: sha(ROOT/file) for file in AUTOMATION_FILES}, uncommitted_automation_files=dirty_automation(ROOT))
    write(service/'plan.json', plan); write(diag/'shared-ledger.json', {'parent_run_id': info['ledger_parent_run_id']})
    write(diag/'checkpoint.json', dict(state='READY', completed_ids=[], next_id=remaining[0]['id'], capture_seconds=0))
    validate_partition(out, service)
    print(json.dumps({'run_id':args.run_id,'queued':count,'next_id':remaining[0]['id']}))


def partition(args):
    parent_out, parent_service, parent_diag = paths(ROOT, args.run_id)
    ensure_allowed(parent_diag, 'partition')
    if args.batch_size < 1:
        raise ValueError('Partition size must be positive')
    plan = read(parent_service/'plan.json')
    if plan.get('selection_mode') not in ('first_N_unique_on_frozen_listing', 'saved_incomplete_listing') or plan.get('partition') or plan.get('continuation'):
        raise ValueError('Partition supports an ordinary frozen parent queue only')
    if (parent_service/'manifest.json').exists() or (parent_service/'partition.json').exists() or any((parent_out/'raw').glob('*.json')):
        raise ValueError('Partition requires an unpartitioned, uncaptured parent')
    validate_plan_profession(plan, parent_out/'listing.json')
    queue = read(parent_out/'queue.json')
    if not queue or not plan.get('listing_pages') or queue_from_pages([read(parent_out/p['file'].replace('\\', '/')) for p in plan['listing_pages']], len(queue)) != queue:
        raise ValueError('Parent queue differs from frozen listing')
    ids = [f'{args.prefix}_{i+1:02d}' for i in range((len(queue)+args.batch_size-1)//args.batch_size)]
    for child in ids:
        out, service, _ = paths(ROOT, child)
        if child == args.run_id or out.exists() or service.exists():
            raise ValueError(f'Child run already exists: {child}')
    # This collection's explicit policy allows one correction after a failure.
    write(parent_diag/'retry-limit.json', {'retry_limit': 1})
    # Ensure there is always a durable single ledger before pointers are written.
    with locked(parent_diag):
        save_failures(parent_diag, load_failures(parent_diag))
    parent_plan_hash, parent_queue_hash = sha(parent_service/'plan.json'), sha(parent_out/'queue.json')
    for index, child in enumerate(ids, 1):
        out, service, diag = paths(ROOT, child)
        out.mkdir(parents=True); diag.mkdir(parents=True)
        shutil.copy2(parent_out/'listing.json', out/'listing.json')
        for page in plan['listing_pages']:
            target = out/page['file'].replace('\\', '/'); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent_out/page['file'].replace('\\', '/'), target)
        provenance = out/'provenance'; provenance.mkdir()
        shutil.copy2(parent_service/'plan.json', provenance/'parent-plan.json')
        shutil.copy2(parent_out/'queue.json', provenance/'parent-queue.json')
        start = (index-1)*args.batch_size; end = min(index*args.batch_size, len(queue))
        write(out/'queue.json', queue[start:end])
        info = dict(parent_run_id=args.run_id, parent_plan_sha256=parent_plan_hash,
            parent_queue_sha256=parent_queue_hash, index=index, batch_size=args.batch_size,
            start=start, end=end, total=len(queue), siblings=ids)
        write(service/'plan.json', dict(plan, run_id=child, limit=end-start,
            requested_limit=end-start, available_unique=end-start, partition=info))
        write(diag/'checkpoint.json', dict(state='READY', completed_ids=[], next_id=queue[start]['id'], capture_seconds=0))
        write(diag/'shared-ledger.json', {'parent_run_id': args.run_id})
        for name in ('automation_changes.json', 'aside-target.local.json'):
            if (parent_diag/name).exists(): shutil.copy2(parent_diag/name, diag/name)
    write(parent_service/'partition.json', dict(children=ids, batch_size=args.batch_size, total=len(queue),
        parent_plan_sha256=parent_plan_hash, parent_queue_sha256=parent_queue_hash))
    for child in ids:
        child_out, child_service, _ = paths(ROOT, child)
        validate_partition(child_out, child_service)
    print(json.dumps(dict(children=ids, total=len(queue), batch_size=args.batch_size)))


def allowed_capture_changes(plan, changes):
    permitted = {'Scraper/skills/collect-avito/scripts/run.py', 'Scraper/skills/collect-avito/SKILL.md'}
    if any(c.get('file', '').replace('\\', '/') == 'Scraper/extract.js' and c.get('reason') == 'Authorized unresolved observation evidence' and c.get('validation', '').strip() for c in changes): permitted.add('Scraper/extract.js')
    if any(c.get('file', '').replace('\\', '/') == 'Scraper/extract.js' and c.get('reason') == 'Bound expansion labels to avoid address false positives' and c.get('validation', '').strip() for c in changes): permitted.add('Scraper/extract.js')
    guard = 'Scraper/skills/collect-avito/scripts/browser_batch.js'
    for c in changes:
        file = c.get('file', '').replace('\\', '/')
        if file == 'Scraper/skills/collect-avito/scripts/preflight.py' and c.get('reason') == 'Explicit pinned Aside target flags' and c.get('validation', '').strip(): permitted.add(file)
        if c.get('validation', '').strip() and ((file == guard and c.get('reason') in ('Wait for pagination evidence in UI Next mode', 'Bounded readiness for incomplete detail shell', 'Bounded hydration wait for matching detail', 'Record bounded wait for incomplete detail')) or (file == 'Scraper/skills/collect-avito/scripts/failure_log.py' and c.get('reason') in ('Windows cross-platform shared file locking', 'Explicit UTF-8 browser scripts and shared ledger on Windows', 'Guard incomplete shell preservation and authorize one pending capture without new budget'))): permitted.add(file)
    if plan.get('partial_suffix') or (plan.get('partition') and any(c.get('file') == guard and c.get('reason') == 'Audited URL identity guard' and c.get('validation', '').strip() for c in changes)):
        permitted.add(guard)
    return permitted


def audited_address_label_fix(plan):
    before = '/ещ[её]|показать|раскрыть/i.test(s)'.encode('utf8')
    after = '/^(?:ещ[её]|показать|раскрыть)(?:\\s|$)/iu.test(s)'.encode('utf8')
    content = (ROOT/'Scraper/extract.js').read_bytes()
    return content.count(after) == 1 and hashlib.sha256(content.replace(after, before, 1)).hexdigest() == plan.get('extractor_sha256')


def validate_extractor_upgrade(out, plan, changed):
    if 'Scraper/extract.js' not in changed or audited_address_label_fix(plan): return
    policy = read(out/'unresolved_policy.json')
    if policy.get('old_extractor_sha256') != plan['extractor_sha256'] or policy.get('new_extractor_sha256') != sha(ROOT/'Scraper/extract.js'):
        raise ValueError('Extractor upgrade policy differs')


def verify_partition_url_guard(plan, changed):
    guard = 'Scraper/skills/collect-avito/scripts/browser_batch.js'
    if not plan.get('partition') or guard not in changed: return
    original = subprocess.check_output(['git', 'show', plan['code_git_commit'] + ':' + guard], cwd=ROOT)
    if hashlib.sha256(original).hexdigest() != plan['automation_sha256'][guard]:
        raise ValueError('Frozen browser source does not match recorded hash')
    before = b"      const ready = data.state === 'DETAIL' && data.id === item.id &&\n"
    after = before + b'        safeUrl(p.url()) === safeUrl(item.url) && safeUrl(data.source_url) === safeUrl(item.url) &&\n'
    if original.count(before) != 1 or (ROOT/guard).read_bytes() != original.replace(before, after, 1):
        raise ValueError('Partition browser change is not the audited URL guard')


def release_confirmed_404(out, diag, record):
    """Release only a persisted, unrestricted exact 404 observation."""
    raw = out/'raw'/f"{record['id']}.json"
    snap = diag/'problem.snapshot.txt'
    if not raw.is_file() or not snap.is_file() or sha(raw) != record.get('raw_sha256') or sha(snap) != record.get('snapshot_sha256'):
        return False
    session_path = diag/'browser-session.local.json'
    if not session_path.is_file(): return False
    session = read(session_path)
    tab = session.get('tab') or {}
    sandbox_path = session.get('sandbox')
    if tab.get('ownership') != 'collector' or tab.get('url') != record.get('url') or not sandbox_path:
        return False
    sandbox = Path(sandbox_path)
    original_raw = sandbox/'raw'/f"{record['id']}.json"
    original_snap, original_tab = sandbox/'problem.snapshot.txt', sandbox/'browser-tab.json'
    if (not original_raw.is_file() or not original_snap.is_file() or not original_tab.is_file()
            or sha(original_raw) != record['raw_sha256'] or sha(original_snap) != record['snapshot_sha256']
            or read(original_tab) != tab or read(raw).get('source_url') != record['url']):
        return False
    text = snap.read_text(encoding='utf8')
    titles = ('Ошибка 404. Страница не найдена', 'Ошибка 404', '404')
    if not any(f'- title: "{title}" [url={record["url"]}]' in text.splitlines() for title in titles):
        return False
    if re.search(r'captcha|капч|доступ ограничен|проблема с ip|too many requests|http\s*429', text, re.I):
        return False
    # A closed REPL may be garbage-collected by Aside. Preserve exact local
    # evidence before closing, including the original tab identity.
    archive = diag/'observations'/record['id']
    archive.mkdir(parents=True, exist_ok=True)
    for source, suffix in ((original_raw, 'raw.json'), (original_snap, 'problem.snapshot.txt'), (original_tab, 'browser-tab.json')):
        digest = sha(source)
        destination = archive/(digest+'.'+suffix)
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise RuntimeError('404 evidence archive collision; session preserved')
        else:
            shutil.copy2(source, destination)
            if sha(destination) != digest:
                raise RuntimeError('404 evidence archive verification failed; session preserved')
    release_browser(diag)
    return True


def capture(args):
    while _capture_batch(args) == 'USER_SKIPPED':
        pass


def _capture_batch(args):
    out, service, diag = paths(ROOT, args.run_id)
    validate_partition(out, service)
    if (out/'dedup-empty.json').exists():
        print(json.dumps({'state':'DEDUP_EMPTY','run_id':args.run_id})); return
    bounded = load_failures(diag).get('bounded_pending', {})
    if bounded.get('state') == 'reserved':
        cp = read(diag/'checkpoint.json')
        if cp.get('state') != 'READY' or cp.get('next_id') != bounded['id'] or (out/'raw'/f"{bounded['id']}.json").exists():
            raise HumanRequired('Preserve the authorized pending observation before capture')
        args.batch_size = 1
    ensure_allowed(diag, 'capture', getattr(args, 'retry_reason', None), action=True)
    if (service / 'manifest.json').exists():
        raise ValueError('Completed run is immutable')
    plan = read(service / 'plan.json')
    if plan.get('branch_only'):
        raise ValueError('Sweep branch runs contain listing evidence only; materialize the sweep capture queue')
    validate_plan_profession(plan, out / 'listing.json')
    preflight(getattr(args, 'config', None), diag)
    changed = {file for file, value in plan['automation_sha256'].items() if sha(ROOT/file) != value}
    changes = read(diag/'automation_changes.json') if (diag/'automation_changes.json').exists() else []
    changes = [dict(c, file=c.get('file', '').replace('\\', '/')) for c in changes]
    permitted = allowed_capture_changes(plan, changes)
    if changed - permitted:
        raise ValueError('Unapproved collector source changed; preserve old run')
    validate_extractor_upgrade(out, plan, changed)
    verify_partition_url_guard(plan, changed)
    validate_automation_changes(changes, changed)
    standing_unavailable_policy(argparse.Namespace(run_id=args.run_id, authorization_id=STANDING_UNAVAILABLE_ID,
        human_instruction=STANDING_UNAVAILABLE_INSTRUCTION))
    queue = read(out / 'queue.json')
    verify_comparison(out, plan, queue)
    # Recover an interrupted transfer before considering another navigation.
    if (diag / 'sandbox.json').exists():
        import_sandbox(Path(read(diag / 'sandbox.json')['path']), out, diag)
    # import_sandbox may promote a newer checkpoint alongside recovered raw.
    # Read it only after the transfer so the prefix check uses one snapshot.
    cp = read(diag / 'checkpoint.json')
    if plan.get('selection_mode') == 'first_N_remote_parameter_matches':
        decisions = remote_decisions(queue, out, plan['limit'], plan.get('selection_criterion', 'remote'))
        accepted = sum(d['accepted'] for d in decisions)
        if accepted == plan['limit']:
            print(json.dumps(dict(state='CAPTURED', captured=len(decisions), accepted=accepted, remaining=0)))
            return
        if len(decisions) == len(queue):
            raise CollectionStopped('Candidate limit exhausted before target; preserve run without export', 'REVIEW_REQUIRED')
        pending = queue[len(decisions):len(decisions)+args.batch_size]
        cp2, elapsed = browser(dict(mode='capture', queue=pending, completed_ids=[d['id'] for d in decisions],
            selection_queue=queue, selection_target=plan['limit'], selection_criterion=plan.get('selection_criterion', 'remote'), tab_id=getattr(args, 'tab_id', None),
            previous_capture_seconds=cp.get('capture_seconds', 0)), out, diag)
        cp2['capture_seconds'] = cp.get('capture_seconds', 0) + elapsed
        write(diag / 'checkpoint.json', cp2)
        if cp2['state'] != 'CAPTURED':
            raise CollectionStopped('Stopped; inspect diagnostics', cp2['state'])
        decisions = remote_decisions(queue, out, plan['limit'], plan.get('selection_criterion', 'remote'))
        if cp2['completed_ids'] != [d['id'] for d in decisions]:
            raise ValueError('Checkpoint does not match the saved candidate prefix')
        accepted = sum(d['accepted'] for d in decisions)
        if accepted < plan['limit'] and len(decisions) == len(queue):
            raise CollectionStopped('Candidate limit exhausted before target; preserve run without export', 'REVIEW_REQUIRED')
        release_browser(diag)
        print(json.dumps(dict(state='CAPTURED' if accepted == plan['limit'] else 'IN_PROGRESS',
            captured=len(decisions), accepted=accepted, remaining=max(0, plan['limit']-accepted))))
        return
    unavailable = {record['id'] for record in unavailable_records(out, queue)} | {record['id'] for record in unresolved_records(out, queue)} | {record['id'] for record in user_skipped_records(out, queue)}
    pending = []
    done = []
    for item in queue:
        if item['id'] in unavailable:
            if pending: raise ValueError('Accounted observation follows a queue gap')
            continue
        path = out / 'raw' / (item['id'] + '.json')
        if path.exists():
            if pending: raise ValueError('Saved observation follows a queue gap')
            data = reviewed_capture(out, item)
            if not check(data, item['id'])['structural_check_passed'] or not data.get('parameter_audit', {}).get('rendered_complete'):
                raise ValueError(f'Saved {item["id"]} needs review; never overwrite or silently skip it')
            done.append(item['id'])
        else:
            pending.append(item)
    if cp.get('completed_ids') != done:
        raise ValueError('Checkpoint does not match accounted queue prefix')
    if not pending:
        print(json.dumps({'state': 'CAPTURED', 'captured': len(done), 'remaining': 0}))
        return
    if bounded.get('state') == 'reserved' and pending[0]['id'] != bounded['id']:
        raise ValueError('First pending queue ID differs from bounded authorization')
    cp2, elapsed = browser(dict(mode='capture', queue=pending[:args.batch_size], completed_ids=done, tab_id=getattr(args, 'tab_id', None),
                               previous_capture_seconds=cp.get('capture_seconds', 0)), out, diag)
    if bounded.get('state') == 'reserved' and cp2.get('state') == 'CAPTURED':
        if not pending or pending[0]['id'] != bounded['id'] or cp2.get('completed_ids') != done + [bounded['id']]:
            raise ValueError('Bounded capture did not complete exactly the authorized ID')
        captured = reviewed_capture(out, pending[0])
        if not check(captured, bounded['id'])['structural_check_passed'] or not captured.get('parameter_audit', {}).get('rendered_complete'):
            raise ValueError('Bounded pending capture is not structurally complete')
    cp2['capture_seconds'] = cp.get('capture_seconds', 0) + elapsed
    write(diag / 'checkpoint.json', cp2)
    print(json.dumps(dict(cp2, captured=len(cp2['completed_ids']), remaining=len(queue)-len(unavailable)-len(cp2['completed_ids']))))
    if cp2['state'] != 'CAPTURED':
        if cp2['state'] == 'REVIEW_REQUIRED' and bounded.get('state') != 'reserved':
            for policy_name in ('user_skip_policy.json','user_detail_skip_policy.json'):
                if not (out/policy_name).exists(): continue
                policy = read(out/policy_name)
                try:
                    mark_policy_skipped(argparse.Namespace(run_id=args.run_id, id=cp2['next_id'], authorization_id=policy['authorization_id']))
                except (ValueError, HumanRequired, OSError, KeyError):
                    pass
                else:
                    succeeded(diag, 'capture', 'Standing policy accounted for the stopped observation')
                    records = user_skipped_records(out, queue)
                    if records and records[-1].get('id') == cp2['next_id']:
                        release_confirmed_404(out, diag, records[-1])
                    print(json.dumps({'state':'IN_PROGRESS', 'user_skipped':cp2['next_id']})); return 'USER_SKIPPED'
            for policy_path in (out/'closed_policy.json', out/'review_policy.json', out/'blocked_listing_policy.json', out/'standing_unavailable_policy.json'):
                if not policy_path.exists(): continue
                policy = read(policy_path)
                try:
                    mark_unavailable(argparse.Namespace(run_id=args.run_id, id=cp2['next_id'], authorization_id=policy['authorization_id'], automatic=True))
                except (ValueError, HumanRequired, OSError, KeyError):
                    continue
                records = unavailable_records(out, queue)
                if records and records[-1].get('id') == cp2['next_id']:
                    release_confirmed_404(out, diag, records[-1])
                print(json.dumps({'state': 'IN_PROGRESS', 'unavailable': cp2['next_id']}))
                return
            unresolved_policy_path = out/'unresolved_policy.json'
            if unresolved_policy_path.exists():
                policy = read(unresolved_policy_path)
                try:
                    mark_unresolved(argparse.Namespace(run_id=args.run_id, id=cp2['next_id'], authorization_id=policy['authorization_id']))
                except (ValueError, HumanRequired, OSError, KeyError):
                    pass
                else:
                    print(json.dumps({'state': 'IN_PROGRESS', 'unresolved': cp2['next_id']}))
                    return
            policies=[e for e in load_failures(diag)['events'] if e.get('event')=='recurring_incomplete_policy']
            if len(policies)==1:
                try:
                    review_address_button(argparse.Namespace(run_id=args.run_id,id=cp2['next_id'],source=policies[0]['source'],automatic=True))
                except (ValueError, HumanRequired, OSError, KeyError):
                    pass
                else:
                    print(json.dumps({'state':'IN_PROGRESS','reviewed':cp2['next_id']})); return
                try:
                    review_salary_line(argparse.Namespace(run_id=args.run_id,id=cp2['next_id'],source=policies[0]['source'],authorization_id=policies[0]['source'],automatic=True))
                except (ValueError, HumanRequired, OSError, KeyError, StopIteration):
                    pass
                else:
                    print(json.dumps({'state':'IN_PROGRESS','reviewed':cp2['next_id']})); return
            salary_policy = out/'salary_review_policy.json'
            if salary_policy.exists():
                policy = read(salary_policy)
                try:
                    review_salary_line(argparse.Namespace(run_id=args.run_id, id=cp2['next_id'], authorization_id=policy['authorization_id'], automatic=True))
                except (ValueError, HumanRequired, OSError, KeyError):
                    pass
                else:
                    print(json.dumps({'state': 'IN_PROGRESS', 'reviewed': cp2['next_id']}))
                    return
        raise CollectionStopped('Stopped; inspect diagnostics and repair the shared collector before resuming', cp2['state'])
    expected = done + [item['id'] for item in pending[:args.batch_size]]
    if cp2.get('completed_ids') != expected:
        raise ValueError('Checkpoint does not match completed batch')
    for item in pending[:args.batch_size]:
        data = reviewed_capture(out, item)
        if not check(data, item['id'])['structural_check_passed'] or not data.get('parameter_audit', {}).get('rendered_complete'):
            raise ValueError('Completed batch failed Python validation; session preserved')
    release_browser(diag)


def placement(data, item, regions, override=None):
    location = norm('\n'.join(data.get('location_raw', [])))
    if not location:
        raise ValueError(f'Missing visible location: {item["id"]}')
    if override:
        city, region = override['city'], override['region']
        if norm(override.get('evidence', '')) not in location or not override.get('evidence'):
            raise ValueError('Placement override requires an exact quote from visible location')
        if not region:
            raise ValueError('Placement region is required')
        if not city and override.get('city_unknown_confirmed') is not True:
            raise ValueError('Empty city requires explicit confirmation')
        if city and not re.search(r'(?<!\w)' + re.escape(city) + r'(?!\w)', location):
            raise ValueError('Placement city must occur in visible location')
        return city, region, location
    # The first line is the address; later lines are nearby metro/district UI.
    # Match whole locality components only, stopping before street/object names.
    matches = []
    address = '\n'.join(data.get('location_raw', [])).split('\n', 1)[0]
    for component in re.split(r'[,;]', address):
        component = norm(component)
        if re.search(r'(?:^|\s)(?:ул\.|улица|проспект|пр-т|шоссе|пер\.|переулок|пл\.|площадь|тракт|линия|дом)(?:\s|$)|^д\.\s*\d', component, re.I):
            break
        locality = re.sub(r'^(?:г\.|город |пос\.|посёлок |поселок |д\.|деревня |с\.|село |пгт\.?\s)\s*', '', component)
        if locality in regions:
            matches.append((locality, regions[locality]))
        elif locality != component:
            raise ValueError(f'Unknown locality {locality}; require a reviewed placement entry')
        elif (component and component not in regions.values() and
              not re.search(r'обл\.|область|край|республика|округ|район|р-н|мкр|микрорайон|муниципальн|городск|^м\.', component, re.I)):
            raise ValueError(f'Unknown address component {component}; require a reviewed placement entry')
    if not matches:
        raise ValueError(f'Unknown placement for {item["id"]}: add a reviewed --placements entry')
    city, region = matches[-1]
    if len({value for _, value in matches}) > 1:
        raise ValueError('Ambiguous placement regions; require a reviewed placement entry')
    return city, region, location


def reviewed_row(data, item, regions, plan, override=None):
    result = check(data, item['id'])
    if not result['structural_check_passed']:
        raise ValueError(result['errors'])
    if not data.get('parameter_audit', {}).get('rendered_complete'):
        raise ValueError('Template completeness not verified; patch collector, do not auto-approve')
    fields = {}
    for field in data['fields']:
        name, value = norm(field['name']), norm(field['value'])
        if not name or not value or name in fields:
            raise ValueError('Empty or duplicate parameter')
        fields[name] = value
    anomaly = ''
    city_source = 'card'
    try:
        city, region, location = placement(data, item, regions, override)
        if not city:
            city_source = 'missing'
            anomaly = 'city_not_visible_in_card'
    except ValueError as error:
        policy = plan.get('city_anomaly_policy')
        if override is not None or policy not in ('blank', 'search_filter'):
            raise
        anomaly = str(error)
        location = norm('\n'.join(data.get('location_raw', [])))
        city, region, city_source = '', '', 'missing'
        # Region can still be explicit even when the locality is not.
        region_matches = {value for value in regions.values()
            if value in location or value.replace('область','обл.') in location}
        if len(region_matches) == 1:
            region = region_matches.pop()
        if policy == 'search_filter':
            search_city = plan.get('geography')
            if search_city not in regions or not plan.get('filter_evidence'):
                raise ValueError('Search fallback requires a confirmed single city and filter evidence')
            avito_url(plan['listing_url'])
            city, region, city_source = search_city, regions[search_city], 'search_filter'
    row = dict(id=data['id'], city=city, profession=plan.get('profession') or '',
        segments=plan.get('segments') or '', title=data['title'], salary_raw=data.get('salary_raw') or '',
        source_url=data['source_url'], observed_at=data['observed_at'], description=data.get('description') or '',
        location_observed=location, employer_name_observed='; '.join(dict.fromkeys(data.get('employer_raw', []))),
        date_raw=data.get('date_raw') or '', views_total=data.get('views_raw') or '', views_today=data.get('today_views_raw') or '',
        listing_rank=item['listing_rank'], listing_promoted=str(item['promoted']) if item.get('promoted') is not None else '',
        full_parameters_read='True', analysis_status='validated', export_eligibility='included',
        read_status_parameters='observed',
        read_status_description='observed' if data.get('description', '') and data['description'].strip() else 'missing',
        read_status_location='observed' if location else 'missing',
        read_limitations='rendered_DOM_only; contact_controls_not_opened; live_route_unverified',
        enrichment_status='not_run', enriched_fields='', enrichment_version='')
    if not row['description']:
        row['read_limitations'] += '; missing_description'
    if not location:
        row['read_limitations'] += '; missing_location'
    row.update(military_flags(data['title']))
    row.update(city__source=city_source,
               city__evidence=plan['listing_url'] if city_source == 'search_filter' else location,
               city_anomaly=anomaly)
    memberships = item.get('sweep_memberships') or []
    if memberships:
        first = memberships[0]
        row.update(discovery_slices='; '.join(record['branch_id'] for record in memberships),
            discovery_branch_count=str(len(memberships)),
            discovery_memberships_json=json.dumps(memberships, ensure_ascii=False, separators=(',', ':')),
            first_discovery_filter=first.get('filter') or '',
            first_discovery_value=first.get('value') or '',
            first_discovery_page=first.get('page') or '',
            first_discovery_position=first.get('position') or '')
    # Parameter labels may be new, but cannot replace identity or provenance.
    if set(fields) & set(row):
        raise ValueError('Parameter collides with metadata')
    row.update(fields)
    review = dict(id=data['id'], structural_check=result, parameter_audit=data['parameter_audit'],
        review_method='deterministic_rendered_template_v1', semantic_review_performed=False,
        city=city, region=region, location_evidence=location, city_source=city_source,
        city_anomaly=anomaly, analysis_status='validated')
    return row, review, region


def partial_prefix_selection(out, service, queue, instruction):
    if not instruction or not instruction.strip():
        raise ValueError('Explicit partial publication instruction required')
    plan = read(service/'plan.json')
    if not plan.get('partition') and plan.get('selection_mode') != 'first_N_unique_on_frozen_listing':
        raise ValueError('Partial prefix requires a frozen partition or ordinary queue')
    cp = read(service/'diagnostics/checkpoint.json')
    unavailable = unavailable_records(out, queue)
    unresolved = unresolved_records(out, queue)
    user_skipped = user_skipped_records(out, queue)
    unavailable_ids = {r['id'] for r in unavailable}
    unresolved_ids = {r['id'] for r in unresolved}
    user_skipped_ids = {r['id'] for r in user_skipped}
    if (unavailable_ids & unresolved_ids) or (user_skipped_ids & (unavailable_ids | unresolved_ids)): raise ValueError('Conflicting prefix exclusions')
    excluded = unavailable_ids | unresolved_ids | user_skipped_ids
    ids = [x['id'] for x in queue]
    next_id = cp.get('next_id')
    if next_id not in ids or cp.get('state') not in ('REVIEW_REQUIRED', 'BROWSER_ERROR', 'NEEDS_HUMAN'):
        raise ValueError('Expected a stopped pending partition ID')
    cut = ids.index(next_id)
    if cut == 0 or cp.get('completed_ids') != [i for i in ids[:cut] if i not in excluded] or excluded - set(ids[:cut]):
        raise ValueError('Completed IDs must be the accounted queue prefix')
    if any((out/'raw'/f'{i}.json').exists() for i in ids[cut:]):
        raise ValueError('Unresolved raw must be preserved or reviewed before partial export')
    accepted = []
    for item in queue[:cut]:
        if item['id'] in excluded: continue
        data = reviewed_capture(out, item)
        if not check(data, item['id'])['structural_check_passed'] or data.get('source_url') != item['url'] or not data.get('parameter_audit', {}).get('rendered_complete'):
            raise ValueError('Prefix contains missing or unreviewed raw')
        accepted.append(item['id'])
    return dict(method='partition_prefix_v1' if plan.get('partition') else 'ordinary_prefix_v1', plan_sha256=sha(service/'plan.json'), queue_sha256=sha(out/'queue.json'),
        accepted_ids=accepted, skipped_ids=[i for i in ids[:cut] if i in unavailable_ids], unresolved_observation_ids=[i for i in ids[:cut] if i in unresolved_ids], user_skipped_ids=[i for i in ids[:cut] if i in user_skipped_ids], unresolved_ids=ids[cut:],
        stop_state=cp['state'], stop_reason=cp.get('stop_reason'), next_id=next_id,
        raw_sha256={i: sha(out/'raw'/f'{i}.json') for i in ids[:cut]},
        authorization=instruction)


def public_provenance(args):
    """Keep private provenance bytes locally and create hash-linked public copies."""
    out,service,diag=paths(ROOT,args.run_id)
    if (service/'manifest.json').exists(): raise ValueError('Finalized run is immutable')
    cp=read(diag/'checkpoint.json')
    if cp.get('state')!='CAPTURED' or cp.get('next_id') is not None: raise ValueError('Completed capture required')
    pending=[]
    for path in sorted((out/'observations').glob('*/*.provenance.json')):
        data=read(path)
        if 'sandbox_path' not in data: continue
        digest=path.name.removesuffix('.provenance.json');raw=path.parent/(digest+'.json')
        if not re.fullmatch(r'[0-9a-f]{64}',digest) or not raw.exists() or sha(raw)!=digest or data.get('raw_sha256',data.get('sha256'))!=digest: raise ValueError('Archived raw provenance differs')
        private=diag/'private-provenance'/path.parent.name/path.name
        if private.exists() and private.read_bytes()!=path.read_bytes(): raise ValueError('Private provenance archive collision')
        public={k:v for k,v in data.items() if k!='sandbox_path'}
        public.update(public_schema='redacted_provenance_v1',private_provenance_sha256=sha(path))
        if re.search(r'/Users/|/home/|\.aside[/\\]|TL_ASIDE_|profileId|target_id',json.dumps(public,ensure_ascii=False),re.I): raise ValueError('Unexpected private locator in public provenance')
        pending.append((path,private,public))
    for path,private,public in pending:
        private.parent.mkdir(parents=True,exist_ok=True)
        if not private.exists(): shutil.copy2(path,private)
        write(path,public)
    print(json.dumps({'public_provenance':len(pending)}))


def validate_provenance_redaction(out, service, manifest):
    info=manifest.get('provenance_redaction')
    if not info: return
    diag=service/'diagnostics'; audit_name=(out/'exports'/'provenance_redaction_audit.json').relative_to(ROOT).as_posix()
    if info.get('audit_file')!=audit_name: raise ValueError('Invalid provenance redaction audit path')
    audit_path=ROOT/audit_name; original=diag/'private-provenance'/'original-manifest.json'
    if not original.is_file() or sha(original)!=info.get('original_manifest_sha256') or not audit_path.is_file() or sha(audit_path)!=info.get('audit_sha256'):
        raise ValueError('Private manifest or provenance redaction audit differs')
    audit=read(audit_path); previous=read(original)
    if audit.get('version')!=1 or audit.get('run_id')!=out.name or audit.get('original_manifest_sha256')!=sha(original): raise ValueError('Provenance redaction audit differs')
    expected=json.loads(json.dumps(previous,ensure_ascii=False)); seen=set()
    for record in audit.get('files',[]):
        name=record.get('file',''); relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or not name.startswith(out.relative_to(ROOT).as_posix()+'/observations/') or not name.endswith('.provenance.json') or name in seen:
            raise ValueError('Invalid redaction audit path')
        seen.add(name); public_path=ROOT/relative; private=diag/'private-provenance'/public_path.parent.name/public_path.name
        if not private.is_file() or sha(private)!=record.get('private_sha256') or not public_path.is_file() or sha(public_path)!=record.get('public_sha256'):
            raise ValueError('Private or public provenance bytes differ')
        old=read(private); public={k:v for k,v in old.items() if k!='sandbox_path'};public.update(public_schema='redacted_provenance_v1',private_provenance_sha256=sha(private))
        original_entry=previous.get('files',{}).get(name,{})
        if ('sandbox_path' not in old or read(public_path)!=public or original_entry.get('sha256')!=sha(private)
                or original_entry.get('bytes')!=private.stat().st_size): raise ValueError('Provenance redaction is not an exact projection')
        expected['files'][name]=dict(bytes=public_path.stat().st_size,sha256=sha(public_path))
    if not seen: raise ValueError('Empty provenance redaction audit')
    expected['files'][info['audit_file']]=dict(bytes=audit_path.stat().st_size,sha256=sha(audit_path));expected['provenance_redaction']=info
    if expected!=manifest: raise ValueError('Redacted manifest differs outside the audited projection')


def redact_finalized_provenance(args):
    """Create an audited public provenance revision without changing finalized raw or exports."""
    out,service,diag=paths(ROOT,args.run_id); manifest=validate_completed_export(out,service)
    if manifest.get('provenance_redaction'):
        validate_provenance_redaction(out,service,manifest); print('Finalized provenance already redacted'); return
    manifest_path=service/'manifest.json'; original_bytes=manifest_path.read_bytes(); original_digest=sha(manifest_path)
    manifest=json.loads(json.dumps(manifest,ensure_ascii=False))
    private_root=diag/'private-provenance'; original=private_root/'original-manifest.json'; audit_path=out/'exports'/'provenance_redaction_audit.json'
    if audit_path.exists() or (original.exists() and original.read_bytes()!=original_bytes): raise ValueError('Existing redaction evidence differs')
    marker=re.compile(r'/Users/|/home/|\.aside[/\\]|TL_ASIDE_|profileId|target_id',re.I); pending=[]
    for path in sorted((out/'observations').glob('*/*.provenance.json')):
        data=read(path)
        if 'sandbox_path' not in data: continue
        name=path.relative_to(ROOT).as_posix();digest=path.name.removesuffix('.provenance.json');raw=path.parent/(digest+'.json');previous=manifest['files'].get(name,{})
        if not re.fullmatch(r'[0-9a-f]{64}',digest) or not raw.is_file() or sha(raw)!=digest or data.get('raw_sha256',data.get('sha256'))!=digest or previous.get('sha256')!=sha(path) or previous.get('bytes')!=path.stat().st_size:
            raise ValueError('Original provenance or archived raw differs from completed manifest')
        private=private_root/path.parent.name/path.name
        if private.exists() and private.read_bytes()!=path.read_bytes(): raise ValueError('Private provenance archive collision')
        public={k:v for k,v in data.items() if k!='sandbox_path'};public.update(public_schema='redacted_provenance_v1',private_provenance_sha256=sha(path))
        encoded=(json.dumps(public,ensure_ascii=False,indent=2)+'\n').encode('utf8')
        if marker.search(encoded.decode('utf8')): raise ValueError('Redacted provenance contains private locator')
        pending.append((path,private,public,dict(file=name,private_sha256=sha(path),public_sha256=hashlib.sha256(encoded).hexdigest())))
    if not pending: raise ValueError('No finalized provenance requires redaction')
    original.parent.mkdir(parents=True,exist_ok=True)
    if not original.exists(): original.write_bytes(original_bytes)
    for path,private,_,_ in pending:
        private.parent.mkdir(parents=True,exist_ok=True)
        if not private.exists(): shutil.copy2(path,private)
    audit=dict(version=1,run_id=args.run_id,original_manifest_sha256=original_digest,files=[record for _,_,_,record in pending])
    try:
        for path,_,public,record in pending:
            write(path,public)
            if sha(path)!=record['public_sha256']: raise ValueError('Redacted provenance hash differs')
            manifest['files'][record['file']]=dict(bytes=path.stat().st_size,sha256=sha(path))
        write(audit_path,audit);audit_name=audit_path.relative_to(ROOT).as_posix()
        manifest['files'][audit_name]=dict(bytes=audit_path.stat().st_size,sha256=sha(audit_path))
        manifest['provenance_redaction']=dict(audit_file=audit_name,audit_sha256=sha(audit_path),original_manifest_sha256=original_digest)
        write(manifest_path,manifest);validate_provenance_redaction(out,service,manifest)
    except (OSError, ValueError, KeyError):
        for path,private,_,_ in pending: shutil.copy2(private,path)
        manifest_path.write_bytes(original_bytes);audit_path.unlink(missing_ok=True)
        raise
    print(json.dumps({'redacted_provenance':len(pending),'original_manifest_sha256':original_digest}))


def finalize(args):
    out, service, diag = paths(ROOT, args.run_id)
    validate_partition(out, service)
    if (out/'dedup-empty.json').exists():
        print(json.dumps({'state':'DEDUP_EMPTY','run_id':args.run_id})); return
    ensure_allowed(diag, 'finalize', getattr(args, 'retry_reason', None), action=True)
    if (service / 'manifest.json').exists():
        raise ValueError('Finalized run exists; do not overwrite')
    if getattr(args, 'partial_prefix', False) and load_failures(diag)['state'] == 'NEEDS_HUMAN':
        if load_failures(diag).get('offline_authorization', {}).get('instruction') != getattr(args, 'partial_instruction', None):
            raise HumanRequired('Partial instruction must match the recorded offline authorization')
    started = time.perf_counter()
    queue, plan = read(out / 'queue.json'), read(service / 'plan.json')
    if plan.get('branch_only'):
        raise ValueError('Sweep branch runs are not exported; materialize the unique sweep queue')
    validate_plan_profession(plan, out / 'listing.json')
    correction = None
    if getattr(args, 'profession_from_observed_controls', False):
        correction = observed_profession_correction(out, service, plan, getattr(args, 'profession_correction_reason', None))
        plan = dict(plan, profession=correction['corrected_profession'])
    elif getattr(args, 'profession_correction_reason', None):
        raise ValueError('Correction reason requires --profession-from-observed-controls')
    elif plan.get('profession_scope') == 'all' and has_checked_profession(read(out/'listing.json')):
        raise ValueError('All-scope conflicts with observed profession; use audited correction')
    plan = dict(plan, city_anomaly_policy=getattr(args,'city_anomaly_policy','blank'))
    changes = read(diag / 'automation_changes.json') if (diag / 'automation_changes.json').exists() else []
    changes = [dict(c, file=c.get('file', '').replace('\\', '/')) for c in changes]
    changed = {file for file,value in plan.get('automation_sha256', {}).items() if sha(ROOT/file) != value}
    reviewed_changes = {change['file'] for change in changes}
    allowed_changes = {'Scraper/skills/collect-avito/scripts/run.py', 'Scraper/skills/collect-avito/scripts/failure_log.py', 'Scraper/skills/collect-avito/SKILL.md'}
    allowed_changes |= allowed_capture_changes(plan, changes)
    if changed - (reviewed_changes & allowed_changes):
        raise ValueError('Unregistered collector source changed after capture; use a new run')
    validate_extractor_upgrade(out, plan, changed)
    verify_partition_url_guard(plan, changed)
    registry_path = ROOT / 'Data/dataset/sources.json'
    registry = read(registry_path)
    registered = [e for e in registry['exports'] if e['run_id'] == args.run_id]
    validate_automation_changes(changes, frozen_dirty_sources(plan) | changed)
    regions = {}
    for entry in registry['exports']:
        for city, region in entry['placement_regions'].items():
            if city in regions and regions[city] != region:
                raise ValueError(f'Ambiguous city {city}; require disambiguated placement mapping')
            regions[city] = region
    overrides = read(args.placements) if args.placements else {}
    decisions = None
    selected_queue = queue
    verify_comparison(out, plan, queue)
    if plan.get('selection_mode') == 'first_N_remote_parameter_matches':
        decisions = remote_decisions(queue, out, plan['limit'], plan.get('selection_criterion', 'remote'))
        if sum(d['accepted'] for d in decisions) != plan['limit']:
            raise ValueError('Target not reached; incomplete candidate selection cannot be exported')
        selected_ids = {d['id'] for d in decisions if d['accepted']}
        selected_queue = [item for item in queue if item['id'] in selected_ids]
    unavailable = unavailable_records(out, queue)
    if unavailable and decisions is not None:
        raise ValueError('Unavailable handling is not supported for remote candidate selection')
    unresolved = unresolved_records(out, queue)
    user_skipped = user_skipped_records(out, queue)
    if {r['id'] for r in unavailable} & {r['id'] for r in unresolved}: raise ValueError('Observation cannot be both unavailable and unresolved')
    if {r['id'] for r in user_skipped} & ({r['id'] for r in unavailable} | {r['id'] for r in unresolved}): raise ValueError('User skip conflicts with another outcome')
    if user_skipped and decisions is not None: raise ValueError('User skip is not supported for remote candidate selection')
    excluded = {record['id'] for record in unavailable} | {record['id'] for record in unresolved} | {record['id'] for record in user_skipped}
    partial = partial_prefix_selection(out, service, queue, getattr(args, 'partial_instruction', None)) if getattr(args, 'partial_prefix', False) else None
    if partial:
        selected_queue = queue[:len(partial['accepted_ids']) + len(partial['skipped_ids']) + len(partial.get('unresolved_observation_ids', [])) + len(partial.get('user_skipped_ids', []))]
    rows, reviews, mapping, unknown_cities = [], [], {}, {}
    for item in selected_queue:
        if item['id'] in excluded:
            continue
        data = reviewed_capture(out, item)
        row, review, region = reviewed_row(data, item, regions, plan, overrides.get(item['id']))
        offline_path = out/'offline_reviews'/f"{item['id']}.json"
        if offline_path.exists(): review['offline_review'] = read(offline_path)
        rows.append(row); reviews.append(review)
        if row['city']:
            mapping[row['city']] = region
        else:
            unknown_cities[item['id']] = {'region': region, 'evidence': row['location_observed'],
                                        'region_unknown': not bool(region),
                                        **({'location_missing': True} if not row['location_observed'] else {})}
    columns = list(dict.fromkeys(k for row in rows for k in row))
    exports = out / 'exports'
    exports.mkdir(exist_ok=True)  # unfinished drafts may be retried; completed runs cannot
    if partial:
        write(exports/'partial_selection.json', partial)
    if correction:
        write(exports/'profession_correction.json', correction)
    with (exports / 'vacancies_flat.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    write(exports / 'vacancies.json', rows)
    if decisions is not None:
        write(exports / 'selection.json', dict(criterion=({'field': 'Количество рабочих часов в день', 'value': '11–12'} if plan.get('selection_criterion') == 'hours_11_12' else {'field': 'Формат работы', 'value': 'Удалённо'}),
            candidate_limit=len(queue), target=plan['limit'], decisions=decisions))
    for review in reviews:
        write(out / 'reviewed' / (review['id'] + '.json'), review)
    write(exports / 'validation.json', dict(planned=len(queue), accepted=len(rows),
        unique_ids=len({r['id'] for r in rows}), unavailable=unavailable, unresolved_observations=unresolved, user_skipped_observations=user_skipped, errors=[], review_method='deterministic_rendered_template_v1'))
    entry = dict(run_id=args.run_id, source='avito',
        path=str((exports / 'vacancies_flat.csv').relative_to(ROOT)), sha256=sha(exports / 'vacancies_flat.csv'),
        placement_regions=mapping, enrichment_status='not_run')
    if correction:
        entry['profession_correction'] = dict(path=str((exports/'profession_correction.json').relative_to(ROOT)),
            sha256=sha(exports/'profession_correction.json'))
    if unknown_cities:
        entry['unknown_city_placements'] = unknown_cities
    if any(row.get('city__source') == 'search_filter' for row in rows):
        entry['search_city'] = plan['geography']
        entry['search_city_evidence'] = plan['listing_url']
    if registered and registered != [entry]:
        raise ValueError('Registered draft differs; preserve it and resolve collision')
    registry['exports'] = [e for e in registry['exports'] if e['run_id'] != args.run_id] + [entry]
    csv_bytes, manifest_bytes = build(ROOT, registry)  # validates before changing shared registry
    write(registry_path, registry)
    from utils.build_dataset import write_outputs
    write_outputs(ROOT / 'Data/dataset', csv_bytes, manifest_bytes)
    manifest = dict(run_id=args.run_id, dataset_id=args.run_id, status='complete',
        schema=dict(columns=columns, unknown_policy='empty_not_zero'), collection=plan,
        coverage=dict(planned=len(queue), accepted=len(rows), unavailable=unavailable, unresolved_observations=unresolved, user_skipped_observations=user_skipped, user_skipped_count=len(user_skipped), user_skip_policy=read(out/'user_skip_policy.json') if (out/'user_skip_policy.json').exists() else None, user_detail_skip_policy=read(out/'user_detail_skip_policy.json') if (out/'user_detail_skip_policy.json').exists() else None, deduplication=plan.get('deduplication'), sweep=plan.get('sweep'), unique_ids=[r['id'] for r in rows], errors=[], partial=partial,
            limitations=['Visible rendered parameters only; enrichment not run; placement on site; live route unverified']),
        timing=dict(setup_seconds=plan['setup_seconds'], capture_seconds=read(diag/'checkpoint.json').get('capture_seconds'),
            review_export_seconds=time.perf_counter()-started, publication_seconds=None),
        review_model=args.review_model, automation_changes=changes,
        profession_correction=correction,
        post_capture_automation_sha256={file: sha(ROOT/file) for file in sorted(changed)}, files={})
    service_files = [service / 'plan.json']
    if plan.get('deduplication'):
        # Publish only the ledger pointer; the shared error ledger remains local.
        pointer = read(diag/'shared-ledger.json')
        if pointer != {'parent_run_id': plan['deduplication']['source_run_id']}:
            raise ValueError('Deduplicated child must use source ledger')
        write(service/'shared-ledger.json', pointer)
        service_files.append(service/'shared-ledger.json')
    for path in sorted(list(out.rglob('*')) + service_files):
        if path.is_file():
            manifest['files'][str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    catalog = ROOT / 'Data/README.md'
    link = f'\n- [{args.run_id}](avito_runs/{args.run_id}/exports/vacancies_flat.csv): {len(rows)} вакансий.\n'
    if link not in catalog.read_text():
        catalog.write_text(catalog.read_text() + link)
    dataset_doc = ROOT / 'Data/dataset/README.md'
    meta = json.loads(manifest_bytes)
    dataset_doc.write_text(re.sub(r'Текущий состав:.*?(?=\n)',
        f'Текущий состав: **{meta["rows"]} вакансий, {len(meta["columns"])} колонок из {len(registry["exports"])} выгрузок**.',
        dataset_doc.read_text(), count=1))
    write(service / 'manifest.json', manifest)  # completion marker is the last write
    print(json.dumps({'accepted': len(rows), 'batch_columns': len(columns),
                      'dataset_rows': meta['rows'], 'dataset_columns': len(meta['columns'])}))


def note_change(args):
    _, service, diag = paths(ROOT, args.run_id)
    relative = Path(args.file)
    if relative.is_absolute() or '..' in relative.parts or not (ROOT / relative).is_file():
        raise ValueError('Change must reference a reusable file inside the repository')
    path = diag / 'automation_changes.json'
    changes = read(path) if path.exists() else []
    changes.append(dict(file=relative.as_posix(), reason=args.reason, validation=args.validation))
    write(path, changes)


def record_error(args):
    _, _, diag = paths(ROOT, args.run_id)
    print(json.dumps(record_failure(diag, args.stage, args.message, args.action,
        args.url, args.agent, args.immediate), ensure_ascii=False))


def check_stage(args):
    _, _, diag = paths(ROOT, args.run_id)
    ensure_allowed(diag, args.stage, args.retry_reason)


def stage_ok(args):
    _, _, diag = paths(ROOT, args.run_id)
    succeeded(diag, args.stage, args.evidence)


def resume_offline(args):
    _, _, diag = paths(ROOT, args.run_id)
    authorize_offline(diag, args.human_instruction)
    print('Human instruction recorded: finalize/publish allowed; browser remains stopped')


def archive_partial_pending(args):
    """Archive a protected pending observation for authorized offline partial export."""
    out, service, diag = paths(ROOT, args.run_id)
    ledger=load_failures(diag); auth=ledger.get('offline_authorization', {})
    if ledger['state'] != 'NEEDS_HUMAN' or not args.human_instruction.strip() or auth.get('instruction') != args.human_instruction:
        raise HumanRequired('Matching current offline publication authorization required')
    events=ledger['events']; authorizations=[i for i,e in enumerate(events) if e.get('event')=='human_authorization' and e.get('instruction')==args.human_instruction]
    if not authorizations or any(e.get('event')=='failure' for e in events[authorizations[-1]+1:]): raise HumanRequired('Offline authorization must follow the last failure')
    cp=read(diag/'checkpoint.json'); queue=read(out/'queue.json'); matches=[item for item in queue if item['id']==args.id]
    if len(matches)!=1 or cp.get('state')!='NEEDS_HUMAN' or cp.get('stop_reason')!='access_restriction_or_captcha' or cp.get('next_id')!=args.id:
        raise ValueError('Expected exact protected pending checkpoint')
    item=matches[0]; index=queue.index(item); url=item['url']; avito_url(url,args.id)
    excluded={r['id'] for r in unavailable_records(out,queue)}|{r['id'] for r in unresolved_records(out,queue)}|{r['id'] for r in user_skipped_records(out,queue)}
    if cp.get('completed_ids') != [x['id'] for x in queue[:index] if x['id'] not in excluded] or excluded-{x['id'] for x in queue[:index]}:
        raise ValueError('Checkpoint prefix differs')
    raw=out/'raw'/f'{args.id}.json'; data=read(raw); snapshot=diag/'problem.snapshot.txt'; snap=snapshot.read_text(encoding='utf8')
    sandbox=Path(read(diag/'sandbox.json')['path']); tab=sandbox/'browser-tab.json'; sandbox_cp=read(sandbox/'checkpoint.json')
    if (data.get('state')!='NEEDS_HUMAN' or data.get('reason')!='access_restriction_or_captcha' or data.get('source_url')!=url
            or f'[url={url}]' not in snap or not re.search(r'captcha|капч|доступ ограничен|проблема с ip',snap,re.I)
            or (sandbox/'raw'/raw.name).read_bytes()!=raw.read_bytes() or (sandbox/'problem.snapshot.txt').read_bytes()!=snapshot.read_bytes()
            or any(sandbox_cp.get(k)!=cp.get(k) for k in ('state','stop_reason','next_id','completed_ids')) or read(tab).get('url')!=url):
        raise ValueError('Protected raw, snapshot, sandbox or URL differs')
    if any((out/'raw'/f'{x["id"]}.json').exists() for x in queue[index+1:]): raise ValueError('Additional suffix raw exists')
    archive=out/'observations'/args.id; archive.mkdir(parents=True,exist_ok=True);digest=sha(raw); target=archive/(digest+'.json')
    if target.exists() and target.read_bytes()!=raw.read_bytes(): raise ValueError('Raw archive collision')
    if not target.exists(): shutil.copy2(raw,target)
    evidence=diag/'observations'/args.id; evidence.mkdir(parents=True,exist_ok=True)
    for source in (snapshot,tab):
        dest=evidence/(sha(source)+'.'+source.name)
        if dest.exists() and dest.read_bytes()!=source.read_bytes(): raise ValueError('Evidence archive collision')
        if not dest.exists(): shutil.copy2(source,dest)
    write(archive/(digest+'.provenance.json'),dict(reason='protected_pending_for_offline_partial',raw_sha256=digest,snapshot_sha256=sha(snapshot),browser_tab_sha256=sha(tab),url=url,authorization=args.human_instruction))
    raw.unlink()
    print(json.dumps({'archived_pending':args.id,'raw_sha256':digest,'browser_state':'NEEDS_HUMAN'}))


def preserve_stopped_observation(args):
    """Preserve a blocked observation before retrying exactly the checkpoint ID."""
    out, service, diag = paths(ROOT, args.run_id)
    ledger = load_failures(diag)
    if ledger['state'] in ('NEEDS_HUMAN', 'RETRY_STARTED'):
        raise HumanRequired('Resolve the current stop or active retry before preserving an observation')
    events = ledger['events']
    authorizations = [i for i, event in enumerate(events) if event.get('event') in ('human_browser_authorization', 'recurring_ip_authorization') and event.get('authorization_id') == args.authorization_id]
    if not authorizations or any(event.get('state') == 'NEEDS_HUMAN' for event in events[authorizations[-1]+1:]):
        raise HumanRequired('Current browser authorization required after the stop')
    cp = read(diag/'checkpoint.json')
    identity = cp.get('next_id')
    captcha = cp.get('state') == 'NEEDS_HUMAN' and cp.get('stop_reason') == 'access_restriction_or_captcha'
    blank = cp.get('state') == 'REVIEW_REQUIRED' and cp.get('stop_reason') == 'identity_or_template_review'
    unresolved_retry = blank and (out/'unresolved_policy.json').exists() and read(out/'unresolved_policy.json').get('authorization_id') == args.authorization_id
    if not (captcha or blank) or not identity or identity in cp.get('completed_ids', []):
        raise ValueError('Expected a supported stopped observation at the pending checkpoint ID')
    queue = read(out/'queue.json')
    if len([item for item in queue if item['id'] == identity]) != 1:
        raise ValueError('Checkpoint ID must occur once in the frozen queue')
    recurring = events[authorizations[-1]].get('event') == 'recurring_ip_authorization'
    if recurring:
        if not captcha or events[authorizations[-1]].get('pending_id') != identity:
            raise HumanRequired('Recurring authorization is limited to the stopped IP observation')
        excluded = {r['id'] for r in unavailable_records(out, queue)} | {r['id'] for r in unresolved_records(out, queue)} | {r['id'] for r in user_skipped_records(out, queue)}
        index = next(i for i, item in enumerate(queue) if item['id'] == identity)
        if cp.get('completed_ids') != [item['id'] for item in queue[:index] if item['id'] not in excluded]: raise ValueError('Checkpoint prefix mismatch')
    if blank:
        instruction = events[authorizations[-1]].get('instruction', '')
        if not unresolved_retry and (identity not in instruction or 'about:blank' not in instruction):
            raise HumanRequired('Authorization must identify the blank-tab observation and pending ID')
        excluded = {record['id'] for record in unavailable_records(out, queue)} | {record['id'] for record in unresolved_records(out, queue)} | {record['id'] for record in user_skipped_records(out, queue)}
        index = next(i for i, item in enumerate(queue) if item['id'] == identity)
        if cp.get('completed_ids') != [item['id'] for item in queue[:index] if item['id'] not in excluded]:
            raise ValueError('Checkpoint prefix mismatch')
    source = out/'raw'/f'{identity}.json'
    data = read(source)
    if captcha:
        if data.get('state') != 'NEEDS_HUMAN' or data.get('reason') != 'access_restriction_or_captcha' or data.get('source_url') != next(item['url'] for item in queue if item['id'] == identity):
            raise ValueError('Raw observation does not match the stopped queue item')
        reason = data['reason']
        snapshot = diag/'problem.snapshot.txt'
        if recurring and (sha(source) != events[authorizations[-1]].get('raw_sha256') or sha(snapshot) != events[authorizations[-1]].get('snapshot_sha256')):
            raise ValueError('IP evidence changed after recurring claim')
    elif unresolved_retry:
        snapshot = diag/'problem.snapshot.txt'
        url = next(item['url'] for item in queue if item['id'] == identity)
        avito_url(url, identity)
        if data.get('state') != 'UNKNOWN_PAGE' or data.get('source_url') != url or not snapshot.exists():
            raise ValueError('Expected the original unknown observation at the queue URL')
        reason = 'unknown_page_before_authorized_retry'
    else:
        snapshot = diag/'problem.snapshot.txt'
        if (data.get('state') != 'UNKNOWN_PAGE' or data.get('source_url') != 'about:blank'
                or data.get('heading') is not None or data.get('items') != [] or data.get('empty_evidence') is not None
                or not snapshot.exists() or snapshot.read_text(encoding='utf-8').strip() != '# note: interactive (clickable / focusable) elements only.\n- title: "" [url=]'):
            raise ValueError('Not an evidenced blank-tab observation')
        reason = 'blank_tab_before_navigation'
    sandbox_pointer = diag/'sandbox.json'
    missing_claim = {}
    if sandbox_pointer.exists():
        sandbox = Path(read(sandbox_pointer)['path'])
        missing_claim = events[authorizations[-1]] if recurring else {}
        if recurring and missing_claim.get('missing_sandbox'):
            log = diag/missing_claim['log_file']
            if (sandbox.exists() or sha(sandbox_pointer) != missing_claim['sandbox_pointer_sha256']
                    or sha(diag/'checkpoint.json') != missing_claim['checkpoint_sha256']
                    or sha(diag/'browser-tab.json') != missing_claim['browser_tab_sha256']
                    or sha(diag/'browser-session.local.json') != missing_claim['session_sha256']
                    or sha(log) != missing_claim['log_sha256']):
                raise ValueError('Missing-sandbox evidence changed after recurring claim')
        else:
            missing_claim = {}
        other = sandbox/'raw'/source.name
        if not missing_claim and (not other.exists() or other.read_bytes() != source.read_bytes()):
            raise ValueError('Sandbox raw differs; inspect interrupted transfer')
        if not missing_claim:
            sandbox_cp = read(sandbox/'checkpoint.json')
            if sandbox_cp.get('next_id') != identity or sandbox_cp.get('state') != cp.get('state') or sandbox_cp.get('stop_reason') != cp.get('stop_reason') or (recurring and sandbox_cp.get('completed_ids') != cp.get('completed_ids')):
                raise ValueError('Sandbox checkpoint differs; inspect interrupted transfer')
            if (blank or (captcha and snapshot.exists())) and (not (sandbox/'problem.snapshot.txt').exists() or (sandbox/'problem.snapshot.txt').read_bytes() != snapshot.read_bytes()):
                raise ValueError('Sandbox snapshot differs; inspect interrupted transfer')
            if captcha and (sandbox/'browser-tab.json').exists() and read(sandbox/'browser-tab.json').get('url') != data.get('source_url'):
                raise ValueError('Original browser URL differs from queue')
    elif blank or recurring:
        raise ValueError('Blank-tab preservation requires the original sandbox')
    archive = out/'observations'/identity
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive/(sha(source) + '.json')
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError('Observation archive collision')
    if not destination.exists():
        shutil.copy2(source, destination)
    snapshot_saved = blank or (captcha and snapshot.exists())
    tab_source = diag/'browser-tab.json' if recurring and missing_claim.get('missing_sandbox') else sandbox/'browser-tab.json' if sandbox_pointer.exists() else None
    tab_saved = bool(tab_source and tab_source.exists() and (unresolved_retry or captcha))
    if snapshot_saved or tab_saved:
        evidence_archive = diag/'observations'/identity
        evidence_archive.mkdir(parents=True, exist_ok=True)
        if snapshot_saved:
            snapshot_dest = evidence_archive/(sha(snapshot) + '.problem.snapshot.txt')
            if snapshot_dest.exists() and snapshot_dest.read_bytes() != snapshot.read_bytes(): raise ValueError('Snapshot archive collision')
            if not snapshot_dest.exists(): shutil.copy2(snapshot, snapshot_dest)
        if tab_saved:
            tab = tab_source
            if unresolved_retry and read(tab).get('url') != url: raise ValueError('Original browser URL differs from queue')
            tab_dest = evidence_archive/(sha(tab) + '.browser-tab.json')
            if tab_dest.exists() and tab_dest.read_bytes() != tab.read_bytes(): raise ValueError('Browser tab archive collision')
            if not tab_dest.exists(): shutil.copy2(tab, tab_dest)
    write(archive/(sha(source) + '.provenance.json'), dict(authorization_id=args.authorization_id,
        original_file=str(source.relative_to(out)), sha256=sha(source), observed_at=data.get('observed_at'),
        preserved_at=datetime.now(timezone.utc).isoformat(), reason=reason,
        snapshot_sha256=sha(snapshot) if snapshot_saved else None, sandbox_path=str(sandbox) if sandbox_pointer.exists() else None, browser_tab_sha256=sha(tab) if tab_saved else None,
        **(dict(sandbox_available=False, local_log_sha256=missing_claim['log_sha256']) if missing_claim else {})))
    # The old sandbox remains on disk; removing only its pointer prevents reimport of
    # the already archived stopped observation into the pending raw slot.
    sandbox_pointer.unlink(missing_ok=True)
    source.unlink()
    if recurring:
        cp.update(state='READY', stop_reason=None); write(diag/'checkpoint.json', cp)
    print(json.dumps({'pending_id': identity, 'preserved': str(destination.relative_to(out))}))


def recapture_cleared_restriction(args):
    """Archive one old restriction and replace its raw only after a verified fresh capture."""
    out, service, diag = paths(ROOT, args.run_id)
    if not args.tab_id or not args.authorization_id or not args.retry_reason: raise ValueError('Explicit owned tab, authorization ID and retry reason required')
    cp, queue = read(diag/'checkpoint.json'), read(out/'queue.json')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1 or cp.get('state') != 'NEEDS_HUMAN' or cp.get('stop_reason') != 'access_restriction_or_captcha' or cp.get('next_id') != args.id:
        raise ValueError('Expected exact pending access-restriction checkpoint')
    item=matches[0]; index=queue.index(item); url=item['url']; avito_url(url,args.id)
    excluded={r['id'] for r in unavailable_records(out,queue)}|{r['id'] for r in unresolved_records(out,queue)}|{r['id'] for r in user_skipped_records(out,queue)}
    if cp.get('completed_ids') != [x['id'] for x in queue[:index] if x['id'] not in excluded]: raise ValueError('Checkpoint prefix differs')
    plan=read(service/'plan.json');validate_partition(out,service);validate_plan_profession(plan,out/'listing.json')
    if queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']],len(queue)) != queue: raise ValueError('Frozen queue differs')
    ledger=load_failures(diag); events=ledger['events']
    auth=[i for i,e in enumerate(events) if e.get('event')=='human_browser_authorization' and e.get('authorization_id')==args.authorization_id]
    if (len(auth)!=1 or ledger['state'] not in ('READY','RECOVERED','RETRY_AVAILABLE') or args.id not in events[auth[0]].get('instruction','')
            or 'ip' not in events[auth[0]].get('instruction','').lower() or 'снят' not in events[auth[0]].get('instruction','').lower()
            or any(e.get('state')=='NEEDS_HUMAN' for e in events[auth[0]+1:])):
        raise HumanRequired('Current explicit authorization confirming cleared IP restriction for this ID required')
    if any(e.get('event')=='cleared_restriction_recapture_started' and (e.get('id')==args.id or e.get('authorization_id')==args.authorization_id) for e in events):
        raise HumanRequired('This restriction recapture was already attempted')
    raw=out/'raw'/f'{args.id}.json'; original=read(raw); snap=diag/'problem.snapshot.txt'; snapshot=snap.read_text(encoding='utf8')
    tab=diag/'browser-tab.json'; old_tab=read(tab); pointer=diag/'sandbox.json'; old_sandbox=Path(read(pointer)['path'])
    if (original.get('state')!='NEEDS_HUMAN' or original.get('reason')!='access_restriction_or_captcha' or original.get('source_url')!=url
            or f'[url={url}]' not in snapshot or not re.search(r'доступ ограничен|проблема с ip|captcha|капч',snapshot,re.I)
            or old_tab.get('url')!=url or not old_tab.get('target_id') or args.tab_id==old_tab['target_id']):
        raise ValueError('Saved restriction evidence, queue URL or replacement tab differs')
    if (old_sandbox/'raw'/raw.name).exists() and (old_sandbox/'raw'/raw.name).read_bytes()!=raw.read_bytes(): raise ValueError('Original sandbox raw differs')
    if (old_sandbox/'problem.snapshot.txt').exists() and (old_sandbox/'problem.snapshot.txt').read_bytes()!=snap.read_bytes(): raise ValueError('Original sandbox snapshot differs')
    stage=diag/'cleared-recapture'/args.id
    if stage.exists(): raise ValueError('Recapture staging already exists; inspect before retry')
    archive=out/'observations'/args.id; archive.mkdir(parents=True,exist_ok=True); digest=sha(raw); saved=archive/(digest+'.json')
    if saved.exists() and saved.read_bytes()!=raw.read_bytes(): raise ValueError('Observation archive collision')
    if not saved.exists(): shutil.copy2(raw,saved)
    evidence=diag/'observations'/args.id; evidence.mkdir(parents=True,exist_ok=True)
    for source in (snap,tab):
        target=evidence/(sha(source)+'.'+source.name)
        if target.exists() and target.read_bytes()!=source.read_bytes(): raise ValueError('Evidence archive collision')
        if not target.exists(): shutil.copy2(source,target)
    write(archive/(digest+'.provenance.json'),dict(reason=original['reason'],raw_sha256=digest,snapshot_sha256=sha(snap),browser_tab_sha256=sha(tab),
        authorization_id=args.authorization_id,original_url=url,sandbox_path=str(old_sandbox),sandbox_available=old_sandbox.exists()))
    changed={f for f,v in plan['automation_sha256'].items() if sha(ROOT/f)!=v}; changes=read(diag/'automation_changes.json') if (diag/'automation_changes.json').exists() else []
    if changed-allowed_capture_changes(plan,changes): raise ValueError('Unapproved collector source changed')
    validate_extractor_upgrade(out,plan,changed);verify_partition_url_guard(plan,changed);validate_automation_changes(changes,changed)
    preflight(getattr(args,'config',None),diag)
    ensure_allowed(diag,'capture',args.retry_reason,action=True)
    ledger=load_failures(diag);ledger['events'].append(dict(event='cleared_restriction_recapture_started',at=datetime.now(timezone.utc).isoformat(),id=args.id,authorization_id=args.authorization_id,raw_sha256=digest,snapshot_sha256=sha(snap)));save_failures(diag,ledger)
    try:
        cp2,elapsed=browser(dict(mode='capture',queue=[item],completed_ids=[],tab_id=args.tab_id,previous_capture_seconds=0),stage/'data',stage/'diagnostics')
        staged=stage/'data'/'raw'/raw.name
        if cp2.get('state')!='CAPTURED' or cp2.get('completed_ids')!=[args.id] or not staged.exists(): raise ValueError('Fresh observation did not capture the pending ID')
        fresh=read(staged)
        if not check(fresh,args.id)['structural_check_passed'] or fresh.get('source_url')!=url or not fresh.get('parameter_audit',{}).get('rendered_complete'):
            raise ValueError('Fresh observation failed identity or complete-template verification')
        if sha(raw)!=digest: raise ValueError('Original raw changed during recapture')
        replacement=raw.with_suffix('.json.replacement');shutil.copy2(staged,replacement);replacement.replace(raw)
        cp['completed_ids'].append(args.id);cp.update(state='READY',stop_reason=None,next_id=queue[index+1]['id'] if index+1<len(queue) else None,capture_seconds=cp.get('capture_seconds',0)+elapsed)
        write(diag/'checkpoint.json',cp);pointer.unlink(missing_ok=True)
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as error:
        record_failure(diag,'capture',str(error),'recapture-cleared-restriction',immediate=True)
        raise
    ledger=load_failures(diag);ledger['events'].append(dict(event='cleared_restriction_recapture_completed',at=datetime.now(timezone.utc).isoformat(),id=args.id,authorization_id=args.authorization_id,old_raw_sha256=digest,new_raw_sha256=sha(raw)));save_failures(diag,ledger)
    succeeded(diag,'capture','Cleared restriction recaptured with verified identity and complete parameters')
    print(json.dumps({'recaptured':args.id,'next_id':cp['next_id']}))


def retry_redirect_observation(args):
    """Archive a detail-to-listing redirect and retry the same frozen queue ID."""
    out, service, diag = paths(ROOT, args.run_id)
    cp, queue = read(diag/'checkpoint.json'), read(out/'queue.json')
    identity = cp.get('next_id')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or identity != args.id:
        raise ValueError('Expected the pending identity review checkpoint')
    matches = [item for item in queue if item['id'] == identity]
    if len(matches) != 1:
        raise ValueError('Expected one frozen queue item')
    raw = out/'raw'/f'{identity}.json'; data = read(raw)
    observed = args.observed_url.rstrip('/')
    parsed = urlsplit(observed)
    if (data.get('state') != 'LISTING' or data.get('source_url', '').rstrip('/') != observed
            or parsed.scheme != 'https' or parsed.hostname not in ('avito.ru', 'www.avito.ru')
            or not re.fullmatch(r'/[^/]+/vakansii', parsed.path) or parsed.query
            or any(item.get('id') == identity for item in data.get('items', []))):
        raise ValueError('Stopped raw is not the confirmed detail-to-listing redirect')
    archive = out/'observations'/identity; archive.mkdir(parents=True, exist_ok=True)
    destination = archive/(sha(raw) + '.json')
    if destination.exists() and destination.read_bytes() != raw.read_bytes():
        raise ValueError('Observation archive collision')
    if not destination.exists(): shutil.copy2(raw, destination)
    pointer = diag/'sandbox.json'
    if not pointer.exists(): raise ValueError('Redirect retry requires the original sandbox')
    sandbox = Path(read(pointer)['path'])
    if (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes():
        raise ValueError('Sandbox raw differs from the stopped redirect')
    write(archive/(sha(raw) + '.provenance.json'), dict(authorization_id=args.authorization_id,
        original_url=matches[0]['url'], observed_url=observed, raw_sha256=sha(raw),
        preserved_at=datetime.now(timezone.utc).isoformat(), reason='detail_redirected_to_category_listing'))
    raw.unlink(); pointer.unlink(); cp.update(state='READY', stop_reason=None)
    write(diag/'checkpoint.json', cp)
    print(json.dumps({'pending_id': identity, 'redirect_preserved': str(destination.relative_to(out))}))


def mark_redirect_unavailable(args):
    """Exclude one queue ID only after two preserved, matching category redirects."""
    out, service, diag = paths(ROOT, args.run_id)
    cp, queue = read(diag/'checkpoint.json'), read(out/'queue.json')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('next_id') != args.id:
        raise ValueError('Expected the pending redirect review')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected one frozen queue item')
    raw = out/'raw'/f'{args.id}.json'; data = read(raw)
    observed = args.observed_url.rstrip('/')
    if data.get('state') != 'LISTING' or data.get('source_url', '').rstrip('/') != observed:
        raise ValueError('Current observation is not the confirmed category redirect')
    previous = sorted((out/'observations'/args.id).glob('*.json'))
    previous = [path for path in previous if not path.name.endswith('.provenance.json') and read(path).get('state') == 'LISTING' and read(path).get('source_url', '').rstrip('/') == observed]
    if not previous: raise ValueError('A matching prior redirect observation is required')
    records = unavailable_records(out, queue)
    records.append(dict(id=args.id, url=matches[0]['url'], status='unavailable_redirect',
        evidence='detail_redirected_to_category_listing_twice', observed_url=observed,
        observation_sha256=[sha(previous[-1]), sha(raw)], raw_sha256=sha(raw),
        authorization_id=args.authorization_id, recorded_at=datetime.now(timezone.utc).isoformat()))
    write(out/'unavailable.json', records)
    index = next(i for i,item in enumerate(queue) if item['id'] == args.id)
    cp.update(state='READY', next_id=queue[index+1]['id'] if index+1 < len(queue) else None, stop_reason=None)
    write(diag/'checkpoint.json', cp); (diag/'sandbox.json').unlink(missing_ok=True)
    print(json.dumps({'unavailable_redirect': args.id, 'next_id': cp['next_id']}))


def visible_http_502(data, snapshot, url):
    evidence=data.get('unresolved_evidence', {})
    return (data.get('state')=='UNKNOWN_PAGE' and data.get('source_url')==url and data.get('heading') is None
        and not data.get('title') and not data.get('fields') and data.get('items')==[]
        and evidence.get('complete') is True and evidence.get('restriction') is False and evidence.get('known_unavailable') is False
        and f'- title: "HTTP 502" [url={url}]' in snapshot and '- text: "502 Error\\n' in snapshot
        and 'Bad gateway' in snapshot and 'Сайт www.avito.ru не отвечает.' in snapshot
        and not re.search(r'captcha|капч|доступ ограничен|проблема с ip|too many requests|(?<!\d)429(?!\d)|http\s*(?!502\b)\d{3}\b|объявление (?:закрыто|снято|удалено|недоступно)', snapshot, re.I))


def incomplete_shell_evidence(out, diag, cp, item):
    identity, url = item['id'], item['url']
    avito_url(url, identity)
    raw = out/'raw'/f'{identity}.json'; data = read(raw); evidence = data.get('unresolved_evidence', {})
    snapshot = diag/'problem.snapshot.txt'; snap = snapshot.read_text(encoding='utf8')
    sandbox = Path(read(diag/'sandbox.json')['path']); tab = sandbox/'browser-tab.json'
    if (cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != identity
            or data.get('state') != 'UNKNOWN_PAGE' or data.get('source_url') != url
            or evidence.get('complete') is not False or evidence.get('restriction') is not False or evidence.get('known_unavailable') is not False
            or data.get('heading') is not None or data.get('title') or data.get('fields') or data.get('items') != []
            or f'[url={url}]' not in snap or '\n- heading ' in snap
            or re.search(r'captcha|капч|(?<!\d)429(?!\d)|доступ ограничен|подозрительн|объявление (?:закрыто|снято|удалено|недоступно)', snap, re.I)
            or (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes()
            or (sandbox/'problem.snapshot.txt').read_bytes() != snapshot.read_bytes()
            or any(read(sandbox/'checkpoint.json').get(k) != cp.get(k) for k in ('state', 'next_id', 'stop_reason', 'completed_ids'))
            or read(tab).get('url') != url):
        raise ValueError('Incomplete shell evidence or sandbox differs')
    return data


def detail_hydration_evidence(out, diag, cp, item):
    identity, url = item['id'], item['url']
    avito_url(url, identity)
    raw = out/'raw'/f'{identity}.json'; data = read(raw)
    snapshot = diag/'problem.snapshot.txt'; snap = snapshot.read_text(encoding='utf8')
    sandbox = Path(read(diag/'sandbox.json')['path']); tab = sandbox/'browser-tab.json'
    if (cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != identity
            or data.get('state') != 'DETAIL' or data.get('id') != identity or data.get('source_url') != url
            or not check(data, identity)['structural_check_passed'] or data.get('parameter_audit', {}).get('rendered_complete') is not False
            or data.get('fields') != [] or data.get('params_blocks') != []
            or f'[url={url}]' not in snap or f'- heading "{data["title"]}" [level=1]' not in snap
            or '- heading "Условия"' not in snap or '- heading "Требования"' not in snap
            or re.search(r'captcha|капч|(?<!\d)429(?!\d)|доступ ограничен|подозрительн|объявление (?:закрыто|снято|удалено|недоступно)', snap, re.I)
            or (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes()
            or (sandbox/'problem.snapshot.txt').read_bytes() != snapshot.read_bytes()
            or any(read(sandbox/'checkpoint.json').get(k) != cp.get(k) for k in ('state', 'next_id', 'stop_reason', 'completed_ids'))
            or read(tab).get('url') != url):
        raise ValueError('Incomplete DETAIL evidence or sandbox differs')
    return data


def retry_incomplete_observation(args):
    """Preserve an authorized, evidenced incomplete shell before retrying its ID."""
    out, service, diag = paths(ROOT, args.run_id)
    ledger = load_failures(diag)
    events = ledger['events']
    auth = [i for i, e in enumerate(events) if e.get('event') in ('human_browser_authorization', 'recurring_incomplete_authorization', 'recurring_502_authorization') and e.get('authorization_id') == args.authorization_id]
    if ledger['state'] in ('NEEDS_HUMAN', 'RETRY_STARTED') or not auth or any(e.get('event') == 'failure' for e in events[auth[-1]+1:]):
        raise HumanRequired('Fresh browser authorization after the stop is required')
    if (events[auth[-1]].get('event') == 'human_browser_authorization'
            and args.id not in events[auth[-1]].get('instruction', '')
            and not getattr(args, 'standing_incomplete', False)):
        raise HumanRequired('Authorization must identify the pending ID')
    if events[auth[-1]].get('event') in ('recurring_incomplete_authorization', 'recurring_502_authorization') and events[auth[-1]].get('pending_id') != args.id:
        raise HumanRequired('Recurring authorization must match the pending ID')
    if events[auth[-1]].get('no_new_budget') and ledger.get('bounded_pending') != dict(id=args.id, state='reserved', authorization_id=args.authorization_id):
        raise HumanRequired('Bounded authorization does not match this pending observation')
    cp, queue = read(diag/'checkpoint.json'), read(out/'queue.json')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != args.id:
        raise ValueError('Expected the pending incomplete observation')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected one frozen queue item')
    plan = read(service/'plan.json')
    if plan.get('run_id') != args.run_id or not plan.get('listing_pages') or not plan.get('listing_sha256'):
        raise ValueError('Missing frozen plan provenance')
    validate_partition(out, service)
    validate_plan_profession(plan, out/'listing.json')
    pages = [read(out/p['file'].replace('\\', '/')) for p in plan['listing_pages']]
    if queue_from_pages(pages, len(queue)) != queue:
        raise ValueError('Frozen queue differs from listing pages')
    excluded = {r['id'] for r in unavailable_records(out, queue)} | {r['id'] for r in unresolved_records(out, queue)} | {r['id'] for r in user_skipped_records(out, queue)}
    index = queue.index(matches[0])
    if cp.get('completed_ids') != [item['id'] for item in queue[:index] if item['id'] not in excluded]:
        raise ValueError('Checkpoint prefix mismatch')
    url = matches[0]['url']; avito_url(url, args.id)
    raw = out/'raw'/f'{args.id}.json'; data = read(raw); evidence = data.get('unresolved_evidence', {})
    http502 = events[auth[-1]].get('event') == 'recurring_502_authorization'
    detail = events[auth[-1]].get('detail_hydration') is True
    if detail:
        detail_hydration_evidence(out, diag, cp, matches[0])
    if not http502 and not detail and (data.get('state') != 'UNKNOWN_PAGE' or data.get('source_url') != url
            or evidence.get('complete') is not False or evidence.get('restriction') is not False
            or evidence.get('known_unavailable') is not False or data.get('heading') is not None
            or data.get('title') or data.get('fields') or data.get('items') != []):
        raise ValueError('Observation is not an unrestricted incomplete page shell')
    snapshot = diag/'problem.snapshot.txt'; pointer = diag/'sandbox.json'
    sandbox = Path(read(pointer)['path']); tab = sandbox/'browser-tab.json'
    if events[auth[-1]].get('event') in ('recurring_incomplete_authorization', 'recurring_502_authorization') and (sha(raw) != events[auth[-1]].get('raw_sha256') or sha(snapshot) != events[auth[-1]].get('snapshot_sha256')):
        raise ValueError('Incomplete evidence changed after recurring claim')
    snap = snapshot.read_text(encoding='utf8')
    if http502 and not visible_http_502(data,snap,url): raise ValueError('Not the evidenced visible HTTP 502 page')
    if (f'[url={url}]' not in snap or re.search(r'captcha|429|доступ ограничен|подозрительн|объявление (?:закрыто|удалено|недоступно)', snap, re.I)
            or (not detail and '\n- heading ' in snap) or (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes()
            or (sandbox/'problem.snapshot.txt').read_bytes() != snapshot.read_bytes()
            or any(read(sandbox/'checkpoint.json').get(k) != cp.get(k) for k in ('state', 'next_id', 'stop_reason', 'completed_ids'))
            or read(tab).get('url') != url):
        raise ValueError('Incomplete shell evidence or sandbox differs')
    archive = out/'observations'/args.id; archive.mkdir(parents=True, exist_ok=True)
    destination = archive/(sha(raw)+'.json')
    if destination.exists() and destination.read_bytes() != raw.read_bytes(): raise ValueError('Observation archive collision')
    if not destination.exists(): shutil.copy2(raw, destination)
    evidence_archive = diag/'observations'/args.id; evidence_archive.mkdir(parents=True, exist_ok=True)
    for source in (snapshot, tab):
        dest = evidence_archive/(sha(source)+'.'+source.name)
        if dest.exists() and dest.read_bytes() != source.read_bytes(): raise ValueError('Evidence archive collision')
        if not dest.exists(): shutil.copy2(source, dest)
    write(archive/(sha(raw)+'.provenance.json'), dict(authorization_id=args.authorization_id,
        original_url=url, raw_sha256=sha(raw), snapshot_sha256=sha(snapshot), browser_tab_sha256=sha(tab),
        plan_sha256=sha(service/'plan.json'), queue_sha256=sha(out/'queue.json'), checkpoint_sha256=sha(diag/'checkpoint.json'),
        sandbox_path=str(sandbox), preserved_at=datetime.now(timezone.utc).isoformat(),
        reason='detail_hydration' if detail else 'visible_http_502' if http502 else 'unrestricted_incomplete_page_shell'))
    raw.unlink(); (diag/'sandbox.json').unlink(missing_ok=True)
    cp.update(state='READY', stop_reason=None); write(diag/'checkpoint.json', cp)
    print(json.dumps({'pending_id': args.id, 'incomplete_preserved': str(destination.relative_to(out))}))


def resume_browser(args):
    out, service, diag = paths(ROOT, args.run_id)
    correction = getattr(args, 'correct_record_error', False)
    if correction and ((service/'plan.json').exists() or (out/'queue.json').exists() or any((out/'raw').glob('*')) or (diag/'checkpoint.json').exists()):
        raise HumanRequired('Administrative correction is limited to a pre-freeze run without checkpoint or raw')
    pending_id = getattr(args, 'pending_id_no_budget', None)
    detail_hydration = getattr(args, 'detail_hydration', False)
    if correction and pending_id:
        raise ValueError('Administrative correction cannot authorize a pending capture')
    if pending_id and getattr(args, 'retain_unbounded', False):
        raise ValueError('Pending-only and retained unbounded authorization are mutually exclusive')
    if detail_hydration and not pending_id:
        raise ValueError('DETAIL hydration authorization requires a pending ID')
    if pending_id and getattr(args, 'unbounded_retries', False):
        raise ValueError('Pending-only and unbounded authorization are mutually exclusive')
    if pending_id:
        if pending_id not in args.human_instruction or not args.authorization_id.strip():
            raise ValueError('Explicit instruction must identify the pending ID')
        cp = read(diag/'checkpoint.json')
        if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('next_id') != pending_id:
            raise ValueError('Authorization must match the pending review checkpoint')
        if detail_hydration:
            queue = read(out/'queue.json')
            matching = [item for item in queue if item['id'] == pending_id]
            if len(matching) != 1: raise ValueError('Expected one pending queue ID')
            detail_hydration_evidence(out, diag, cp, matching[0])
        with locked(diag):
            ledger = load_failures(diag)
            if ledger['state'] != 'NEEDS_HUMAN' or (not detail_hydration and ledger['retries_started'] < ledger['retry_limit']):
                raise HumanRequired('Requires an exhausted stopped budget')
            if any(e.get('authorization_id') == args.authorization_id for e in ledger['events']):
                raise HumanRequired('Authorization ID already used')
            ledger['state'] = 'READY'
            ledger['bounded_pending'] = dict(id=pending_id, state='reserved', authorization_id=args.authorization_id)
            ledger['events'].append(dict(event='human_browser_authorization', at=datetime.now(timezone.utc).isoformat(), instruction=args.human_instruction,
                authorization_id=args.authorization_id, pending_id=pending_id, no_new_budget=True, detail_hydration=detail_hydration))
            save_failures(diag, ledger)
        print('One pending capture authorized without replenishing the retry budget')
        return
    authorize_browser(diag, args.human_instruction, args.authorization_id, unbounded_retries=getattr(args, 'unbounded_retries', False), correct_record_error=correction, retain_unbounded=getattr(args, 'retain_unbounded', False))
    print('Human browser authorization recorded; history retained')


def register_standing_run(args):
    """Record an explicit goal-scoped standing instruction before a new run starts."""
    out, service, diag = paths(ROOT, args.run_id)
    fields = (args.goal_source, args.authorization_id, args.technical_instruction, args.ip_instruction)
    if any(not value.strip() for value in fields):
        raise ValueError('Goal source, authorization ID and exact instructions are required')
    if (service/'plan.json').exists() or (out/'queue.json').exists() or any((out/'raw').glob('*')):
        raise HumanRequired('Standing registration is only permitted before freeze')
    with locked(diag):
        ledger = load_failures(diag)
        event = dict(event='standing_run_authorization', goal_source=args.goal_source,
                     authorization_id=args.authorization_id, technical_instruction=args.technical_instruction,
                     ip_instruction=args.ip_instruction, interval_seconds=1800)
        prior = [e for e in ledger['events'] if e.get('event') == 'standing_run_authorization']
        if prior:
            if len(prior) != 1 or any(prior[0].get(k) != v for k, v in event.items()):
                raise HumanRequired('Standing authorization already differs')
            return
        if ledger['state'] != 'READY' or ledger['failures'] or ledger['events']:
            raise HumanRequired('Standing registration requires a fresh run ledger')
        now = datetime.now(timezone.utc).isoformat()
        ledger['unbounded_retries'] = True
        ledger['events'].append(dict(event, at=now))
        ledger['events'].append(dict(event='recurring_ip_policy', at=now, source=args.authorization_id,
            instruction=args.ip_instruction, additional_instruction=args.technical_instruction,
            goal_source=args.goal_source, interval_seconds=1800))
        save_failures(diag, ledger)
    print('Standing goal instructions recorded without authorizing a stopped run')


def recurring_ip_policy(args):
    """Record the user's recurring instruction once, without authorizing a read."""
    _, _, diag = paths(ROOT, args.run_id)
    if not args.source.strip() or not args.human_instruction.strip() or not args.additional_instruction.strip():
        raise ValueError('Exact user instructions and stable source are required')
    with locked(diag):
        ledger = load_failures(diag)
        if ledger['state'] != 'NEEDS_HUMAN': raise HumanRequired('Policy registration requires a stopped run')
        policy = dict(event='recurring_ip_policy', source=args.source, instruction=args.human_instruction,
                      additional_instruction=args.additional_instruction, interval_seconds=1800)
        existing = [e for e in ledger['events'] if e.get('event') == 'recurring_ip_policy' and e.get('source') == args.source]
        if existing:
            if any(any(e.get(k) != v for k,v in policy.items()) for e in existing): raise ValueError('Policy source already has different terms')
            return
        ledger['events'].append(dict(policy, at=datetime.now(timezone.utc).isoformat()))
        save_failures(diag, ledger)
    print('Recurring IP policy recorded without authorizing a browser action')


def recurring_ip_attempt(args):
    """Claim one IP-limited pending read under a previously recorded policy."""
    out, service, diag = paths(ROOT, args.run_id)
    cp = read(diag/'checkpoint.json'); queue = read(out/'queue.json')
    if cp.get('state') != 'NEEDS_HUMAN' or cp.get('stop_reason') != 'access_restriction_or_captcha' or cp.get('next_id') != args.id:
        raise ValueError('Expected the stopped pending IP observation')
    matches = [x for x in queue if x['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected one pending queue ID')
    url = matches[0]['url']; avito_url(url, args.id)
    plan = read(service/'plan.json')
    if plan.get('run_id') != args.run_id or not plan.get('listing_pages') or not plan.get('listing_sha256'): raise ValueError('Missing frozen plan')
    validate_partition(out, service); validate_plan_profession(plan, out/'listing.json')
    if queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']], len(queue)) != queue: raise ValueError('Frozen queue differs')
    excluded = {r['id'] for r in unavailable_records(out, queue)} | {r['id'] for r in unresolved_records(out, queue)} | {r['id'] for r in user_skipped_records(out, queue)}
    if cp.get('completed_ids') != [x['id'] for x in queue[:queue.index(matches[0])] if x['id'] not in excluded]: raise ValueError('Checkpoint prefix mismatch')
    raw = out/'raw'/f'{args.id}.json'; data = read(raw); snap = diag/'problem.snapshot.txt'
    sandbox = Path(read(diag/'sandbox.json')['path']); tab = sandbox/'browser-tab.json'; text = snap.read_text(encoding='utf8')
    missing_log = getattr(args, 'missing_sandbox_log', None)
    missing = not sandbox.exists()
    if missing:
        if not missing_log: raise ValueError('Original sandbox missing; explicit local log evidence required')
        log = Path(missing_log).resolve()
        if log.parent != diag.resolve() or not re.fullmatch(r'aside-\d+\.local\.log', log.name):
            raise ValueError('Expected an original local Aside log in this run diagnostics')
        log_text = log.read_text(encoding='utf8'); session = read(diag/'browser-session.local.json')
        saved_tab = read(diag/'browser-tab.json')
        marker = f'TL_SANDBOX={sandbox}\n'
        if (session.get('sandbox') != str(sandbox) or saved_tab.get('url') != url
                or marker not in log_text or text not in log_text
                or log_text.find(marker) > log_text.find(text)
                or not re.search(r'(?m)^TL_FINISHED\r?\n\[ok\s*\|', log_text[log_text.find(text)+len(text):])):
            raise ValueError('Missing-sandbox local session or completed RPC evidence differs')
    elif missing_log:
        raise ValueError('Missing-sandbox recovery is not applicable while sandbox exists')
    if (data.get('state') != 'NEEDS_HUMAN' or data.get('reason') != 'access_restriction_or_captcha' or data.get('source_url') != url
            or f'- title: "Доступ ограничен: проблема с IP" [url={url}]' not in text
            or 'Страница не загрузилась из-за проблем с IP' not in text or re.search(r'\b429\b', text)
            or (not missing and ((sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or (sandbox/'problem.snapshot.txt').read_bytes() != snap.read_bytes()
            or any(read(sandbox/'checkpoint.json').get(k) != cp.get(k) for k in ('state','next_id','stop_reason','completed_ids')) or read(tab).get('url') != url))):
        raise ValueError('IP evidence, identity or original sandbox differs')
    with locked(diag):
        ledger = load_failures(diag); events = ledger['events']
        policies = [e for e in events if e.get('event') == 'recurring_ip_policy' and e.get('source') == args.source]
        failures = [e for e in events if e.get('event') == 'failure']
        if ledger['state'] != 'NEEDS_HUMAN' or len(policies) != 1 or not failures:
            raise HumanRequired('Requires a stopped run and registered recurring policy')
        if any(e.get('authorization_id') == args.authorization_id for e in events) or not args.authorization_id.strip():
            raise HumanRequired('Attempt ID must be new and nonempty')
        last = datetime.fromisoformat(failures[-1]['at'])
        previous = [datetime.fromisoformat(e['at']) for e in events if e.get('event') in ('recurring_ip_authorization', 'recurring_incomplete_authorization', 'recurring_502_authorization', 'recurring_unavailable_authorization', 'recurring_reviewed_authorization') and e.get('source') == args.source]
        now = datetime.now(timezone.utc)
        if (now - max([last] + previous)).total_seconds() < policies[0]['interval_seconds']:
            raise HumanRequired('Thirty-minute interval has not elapsed')
        ledger['state'] = 'READY'
        ledger['bounded_pending'] = dict(id=args.id, state='reserved', authorization_id=args.authorization_id)
        events.append(dict(event='recurring_ip_authorization', at=now.isoformat(), source=args.source,
            authorization_id=args.authorization_id, pending_id=args.id, no_new_budget=True,
            failure_at=failures[-1]['at'], raw_sha256=sha(raw), snapshot_sha256=sha(snap),
            **(dict(missing_sandbox=True, checkpoint_sha256=sha(diag/'checkpoint.json'),
                    browser_tab_sha256=sha(diag/'browser-tab.json'), session_sha256=sha(diag/'browser-session.local.json'),
                    log_file=log.name, log_sha256=sha(log), sandbox_pointer_sha256=sha(diag/'sandbox.json')) if missing else {})))
        save_failures(diag, ledger)
    print('One recurring IP attempt claimed without new retry budget')


def recurring_reviewed_continuation(args):
    out,service,diag=paths(ROOT,args.run_id);cp=read(diag/'checkpoint.json');queue=read(out/'queue.json')
    if cp.get('state')!='READY' or cp.get('next_id')!=args.id or (out/'raw'/f'{args.id}.json').exists() or (diag/'sandbox.json').exists(): raise ValueError('Expected next unread pending ID')
    matches=[i for i,x in enumerate(queue) if x['id']==args.id]
    if len(matches)!=1 or matches[0]==0: raise ValueError('Invalid next ID')
    index=matches[0]-1; previous=queue[index];review=read(out/'offline_reviews'/f"{previous['id']}.json")
    if review.get('method') not in ('address_button_v1','joined_salary_line_v1') or review.get('authorization_id')!=args.source: raise ValueError('Expected reviewed observation from this policy')
    reviewed_capture(out,previous); evidence=diag/'observations'/previous['id']/(review['snapshot_sha256']+'.problem.snapshot.txt')
    if sha(evidence)!=review['snapshot_sha256'] or f'[url={previous["url"]}]' not in evidence.read_text(encoding='utf8'): raise ValueError('Review evidence differs')
    plan=read(service/'plan.json');validate_partition(out,service);validate_plan_profession(plan,out/'listing.json')
    if plan.get('run_id')!=args.run_id or queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']],len(queue))!=queue: raise ValueError('Frozen queue differs')
    excluded={r['id'] for r in unavailable_records(out,queue)}|{r['id'] for r in unresolved_records(out,queue)}|{r['id'] for r in user_skipped_records(out,queue)}
    if cp.get('completed_ids')!=[x['id'] for x in queue[:index+1] if x['id'] not in excluded]: raise ValueError('Checkpoint prefix mismatch')
    with locked(diag):
        ledger=load_failures(diag);events=ledger['events'];policies=[e for e in events if e.get('event')=='recurring_ip_policy' and e.get('source')==args.source];failures=[e for e in events if e.get('event')=='failure']
        if ledger['state']!='NEEDS_HUMAN' or ledger['retries_started']<ledger['retry_limit'] or len(policies)!=1 or not failures or not any(e.get('event')=='recurring_incomplete_policy' and e.get('source')==args.source for e in events): raise HumanRequired('Stopped exhausted budget and policies required')
        if datetime.fromisoformat(review['reviewed_at'])<datetime.fromisoformat(failures[-1]['at']): raise ValueError('Review predates last failure')
        if not args.authorization_id.strip() or any(e.get('authorization_id')==args.authorization_id for e in events): raise HumanRequired('Attempt ID must be new')
        times=[datetime.fromisoformat(failures[-1]['at'])]+[datetime.fromisoformat(e['at']) for e in events if e.get('event') in ('recurring_ip_authorization','recurring_incomplete_authorization','recurring_502_authorization','recurring_unavailable_authorization','recurring_reviewed_authorization') and e.get('source')==args.source];now=datetime.now(timezone.utc)
        if (now-max(times)).total_seconds()<policies[0]['interval_seconds']: raise HumanRequired('Thirty-minute interval has not elapsed')
        ledger['state']='READY';ledger['bounded_pending']=dict(id=args.id,state='reserved',authorization_id=args.authorization_id)
        events.append(dict(event='recurring_reviewed_authorization',at=now.isoformat(),source=args.source,authorization_id=args.authorization_id,pending_id=args.id,no_new_budget=True,reviewed_id=previous['id'],snapshot_sha256=review['snapshot_sha256']));save_failures(diag,ledger)
    print('One post-review pending read claimed without new budget')


def recurring_unavailable_continuation(args):
    out, service, diag = paths(ROOT, args.run_id); cp=read(diag/'checkpoint.json'); queue=read(out/'queue.json')
    if cp.get('state') != 'READY' or cp.get('next_id') != args.id or (out/'raw'/f'{args.id}.json').exists() or (diag/'sandbox.json').exists(): raise ValueError('Expected next unread pending ID')
    plan=read(service/'plan.json'); validate_partition(out,service); validate_plan_profession(plan,out/'listing.json')
    if plan.get('run_id') != args.run_id or not plan.get('listing_pages') or queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']],len(queue)) != queue: raise ValueError('Frozen queue differs')
    records=unavailable_records(out,queue)
    if not records: raise ValueError('No standing unavailable observation')
    record=records[-1]; index=next((i for i,x in enumerate(queue) if x['id']==record['id']),None)
    if index is None or index+1>=len(queue) or queue[index+1]['id']!=args.id or record.get('authorization_id')!=STANDING_UNAVAILABLE_ID: raise ValueError('Unavailable record does not precede pending ID')
    excluded={r['id'] for r in records}|{r['id'] for r in unresolved_records(out,queue)}|{r['id'] for r in user_skipped_records(out,queue)}
    if cp.get('completed_ids') != [x['id'] for x in queue[:index+1] if x['id'] not in excluded]: raise ValueError('Checkpoint prefix mismatch')
    evidence=diag/'observations'/record['id']/(record['snapshot_sha256']+'.problem.snapshot.txt')
    if sha(evidence)!=record['snapshot_sha256'] or f'[url={record["url"]}]' not in evidence.read_text(encoding='utf8'): raise ValueError('Unavailable evidence differs')
    with locked(diag):
        ledger=load_failures(diag); events=ledger['events']; policies=[e for e in events if e.get('event')=='recurring_ip_policy' and e.get('source')==args.source]; failures=[e for e in events if e.get('event')=='failure']
        if ledger['state']!='NEEDS_HUMAN' or ledger['retries_started']<ledger['retry_limit'] or len(policies)!=1 or not failures or not any(e.get('event')=='project_unavailable_policy' and e.get('authorization_id')==STANDING_UNAVAILABLE_ID for e in events): raise HumanRequired('Stopped exhausted budget and standing policies required')
        if datetime.fromisoformat(record['recorded_at']) < datetime.fromisoformat(failures[-1]['at']): raise ValueError('Unavailable decision predates last failure')
        if not args.authorization_id.strip() or any(e.get('authorization_id')==args.authorization_id for e in events): raise HumanRequired('Attempt ID must be new')
        times=[datetime.fromisoformat(failures[-1]['at'])]+[datetime.fromisoformat(e['at']) for e in events if e.get('event') in ('recurring_ip_authorization','recurring_incomplete_authorization','recurring_502_authorization','recurring_unavailable_authorization', 'recurring_reviewed_authorization') and e.get('source')==args.source]
        now=datetime.now(timezone.utc)
        if (now-max(times)).total_seconds()<policies[0]['interval_seconds']: raise HumanRequired('Thirty-minute interval has not elapsed')
        ledger['state']='READY';ledger['bounded_pending']=dict(id=args.id,state='reserved',authorization_id=args.authorization_id)
        events.append(dict(event='recurring_unavailable_authorization',at=now.isoformat(),source=args.source,authorization_id=args.authorization_id,pending_id=args.id,no_new_budget=True,unavailable_id=record['id'],snapshot_sha256=record['snapshot_sha256']))
        save_failures(diag,ledger)
    print('One post-unavailable pending read claimed without new budget')


def recurring_incomplete_policy(args):
    _, _, diag = paths(ROOT, args.run_id)
    if not args.source.strip() or not args.human_instruction.strip(): raise ValueError('Exact instruction and source required')
    with locked(diag):
        ledger = load_failures(diag); events = ledger['events']
        if not any(e.get('event') == 'recurring_ip_policy' and e.get('source') == args.source for e in events): raise HumanRequired('Recurring interval policy missing')
        previous = [e for e in events if e.get('event') == 'recurring_incomplete_policy' and e.get('source') == args.source]
        if previous:
            if len(previous) != 1 or previous[0].get('instruction') != args.human_instruction: raise ValueError('Existing policy differs')
            return
        events.append(dict(event='recurring_incomplete_policy', source=args.source, instruction=args.human_instruction, at=datetime.now(timezone.utc).isoformat()))
        save_failures(diag, ledger)
    print('Recurring incomplete-shell instruction recorded')


def recurring_incomplete_attempt(args):
    out, service, diag = paths(ROOT, args.run_id)
    cp, queue = read(diag/'checkpoint.json'), read(out/'queue.json')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != args.id: raise ValueError('Expected pending incomplete observation')
    matches = [x for x in queue if x['id'] == args.id]
    if len(matches) != 1: raise ValueError('Pending ID differs from queue')
    url = matches[0]['url']; avito_url(url, args.id)
    plan = read(service/'plan.json')
    if plan.get('run_id') != args.run_id or not plan.get('listing_pages') or not plan.get('listing_sha256'): raise ValueError('Missing frozen plan')
    validate_partition(out, service); validate_plan_profession(plan, out/'listing.json')
    if queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']], len(queue)) != queue: raise ValueError('Frozen queue differs')
    excluded = {r['id'] for r in unavailable_records(out, queue)} | {r['id'] for r in unresolved_records(out, queue)} | {r['id'] for r in user_skipped_records(out, queue)}
    if cp.get('completed_ids') != [x['id'] for x in queue[:queue.index(matches[0])] if x['id'] not in excluded]: raise ValueError('Checkpoint prefix mismatch')
    raw = out/'raw'/f'{args.id}.json'; data = read(raw); ev = data.get('unresolved_evidence', {}); snap = diag/'problem.snapshot.txt'
    sandbox = Path(read(diag/'sandbox.json')['path']); tab = sandbox/'browser-tab.json'; text = snap.read_text(encoding='utf8')
    is502=getattr(args,'visible_http_502',False)
    if is502 and not visible_http_502(data,text,url): raise ValueError('Not the evidenced visible HTTP 502 page')
    if ((not is502 and (data.get('state') != 'UNKNOWN_PAGE' or data.get('source_url') != url or ev.get('complete') is not False or ev.get('restriction') is not False
            or ev.get('known_unavailable') is not False or data.get('heading') is not None or data.get('title') or data.get('fields') or data.get('items') != []
            or f'[url={url}]' not in text or re.search(r'captcha|429|доступ ограничен|подозрительн|объявление (?:закрыто|удалено|недоступно)', text, re.I) or '\n- heading ' in text))
            or (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or (sandbox/'problem.snapshot.txt').read_bytes() != snap.read_bytes()
            or any(read(sandbox/'checkpoint.json').get(k) != cp.get(k) for k in ('state','next_id','stop_reason','completed_ids')) or read(tab).get('url') != url):
        raise ValueError('Incomplete shell evidence or sandbox differs')
    with locked(diag):
        ledger=load_failures(diag); events=ledger['events']
        interval=[e for e in events if e.get('event')=='recurring_ip_policy' and e.get('source')==args.source]
        policies=[e for e in events if e.get('event')=='recurring_incomplete_policy' and e.get('source')==args.source]
        failures=[e for e in events if e.get('event')=='failure']
        if ledger['state']!='NEEDS_HUMAN' or ledger['retries_started']<ledger['retry_limit'] or len(interval)!=1 or len(policies)!=1 or not failures: raise HumanRequired('Stopped exhausted budget and policies required')
        if not args.authorization_id.strip() or any(e.get('authorization_id')==args.authorization_id for e in events): raise HumanRequired('Attempt ID must be new')
        times=[datetime.fromisoformat(failures[-1]['at'])]+[datetime.fromisoformat(e['at']) for e in events if e.get('event') in ('recurring_ip_authorization','recurring_incomplete_authorization','recurring_502_authorization','recurring_unavailable_authorization', 'recurring_reviewed_authorization') and e.get('source')==args.source]
        now=datetime.now(timezone.utc)
        if (now-max(times)).total_seconds()<interval[0]['interval_seconds']: raise HumanRequired('Thirty-minute interval has not elapsed')
        ledger['state']='READY'; ledger['bounded_pending']=dict(id=args.id,state='reserved',authorization_id=args.authorization_id)
        events.append(dict(event='recurring_502_authorization' if is502 else 'recurring_incomplete_authorization',at=now.isoformat(),source=args.source,authorization_id=args.authorization_id,pending_id=args.id,no_new_budget=True,raw_sha256=sha(raw),snapshot_sha256=sha(snap)))
        save_failures(diag,ledger)
    print('One recurring '+('visible HTTP 502' if is502 else 'incomplete-shell')+' attempt claimed without new budget')


def recurring_ip_after_preflight(args):
    """Reclaim an already preserved IP observation after the evidenced preflight failure."""
    out, service, diag = paths(ROOT, args.run_id)
    cp, queue = read(diag/'checkpoint.json'), read(out/'queue.json')
    if cp.get('state') != 'READY' or cp.get('stop_reason') is not None or cp.get('next_id') != args.id or (out/'raw'/f'{args.id}.json').exists() or (diag/'sandbox.json').exists():
        raise ValueError('Expected preserved pending ID without a new raw or sandbox')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected one pending queue ID')
    url = matches[0]['url']; avito_url(url, args.id)
    plan = read(service/'plan.json')
    if plan.get('run_id') != args.run_id or not plan.get('listing_sha256') or not plan.get('listing_pages'): raise ValueError('Missing frozen plan')
    validate_partition(out, service); validate_plan_profession(plan, out/'listing.json')
    if queue_from_pages([read(out/p['file'].replace('\\','/')) for p in plan['listing_pages']], len(queue)) != queue: raise ValueError('Frozen queue differs')
    excluded = {r['id'] for r in unavailable_records(out, queue)} | {r['id'] for r in unresolved_records(out, queue)} | {r['id'] for r in user_skipped_records(out, queue)}
    if cp.get('completed_ids') != [x['id'] for x in queue[:queue.index(matches[0])] if x['id'] not in excluded]: raise ValueError('Checkpoint prefix mismatch')
    expected_raw = set(cp['completed_ids']) | excluded
    if {p.stem for p in (out/'raw').glob('*.json')} != expected_raw: raise ValueError('Raw observations do not match the completed prefix')
    with locked(diag):
        ledger = load_failures(diag); events = ledger['events']; bounded = ledger.get('bounded_pending', {})
        if ledger['state'] != 'NEEDS_HUMAN' or ledger['retries_started'] < ledger['retry_limit'] or bounded.get('state') != 'failed' or bounded.get('id') != args.id:
            raise HumanRequired('Expected a failed exhausted bounded attempt')
        policies = [e for e in events if e.get('event') == 'recurring_ip_policy' and e.get('source') == args.source]
        claims = [e for e in events if e.get('event') == 'recurring_ip_authorization' and e.get('authorization_id') == bounded.get('authorization_id') and e.get('source') == args.source]
        if len(policies) != 1 or len(claims) != 1 or len(events) < 2 or events[-2].get('event') != 'bounded_pending_started' or events[-2].get('id') != args.id:
            raise ValueError('Missing matching prior policy, claim or started action')
        failure = events[-1]
        command = ['/Library/Developer/CommandLineTools/usr/bin/python3', str(SCRIPTS/'preflight.py'), '--probe-aside', '--config', str(ROOT/'Scraper/collect-avito.local.json')]
        expected_message = str(subprocess.CalledProcessError(2, command))
        if failure.get('event') != 'failure' or failure.get('stage') != 'capture' or failure.get('state') != 'NEEDS_HUMAN' or failure.get('message') != expected_message:
            raise ValueError('Last failure is not the evidenced preflight command failure')
        claim = claims[0]; archive = out/'observations'/args.id; digest = claim.get('raw_sha256', '')
        source = archive/(digest+'.json'); provenance = read(archive/(digest+'.provenance.json'))
        snapshot = diag/'observations'/args.id/(claim.get('snapshot_sha256','')+'.problem.snapshot.txt')
        tab = diag/'observations'/args.id/(provenance.get('browser_tab_sha256','')+'.browser-tab.json')
        original = read(source); sandbox = Path(provenance.get('sandbox_path',''))
        if (sha(source) != digest or provenance.get('sha256') != digest or provenance.get('authorization_id') != bounded['authorization_id']
                or provenance.get('snapshot_sha256') != claim.get('snapshot_sha256') or sha(snapshot) != claim.get('snapshot_sha256')
                or sha(tab) != provenance.get('browser_tab_sha256') or read(tab).get('url') != url
                or original.get('source_url') != url or original.get('state') != 'NEEDS_HUMAN' or original.get('reason') != 'access_restriction_or_captcha'
                or f'- title: "Доступ ограничен: проблема с IP" [url={url}]' not in snapshot.read_text(encoding='utf8')
                or (sandbox/'raw'/f'{args.id}.json').read_bytes() != source.read_bytes()
                or (sandbox/'problem.snapshot.txt').read_bytes() != snapshot.read_bytes()
                or read(sandbox/'browser-tab.json').get('url') != url
                or read(sandbox/'checkpoint.json').get('completed_ids') != cp.get('completed_ids')
                or read(sandbox/'checkpoint.json').get('next_id') != args.id):
            raise ValueError('Original IP provenance or sandbox differs')
        if not args.authorization_id.strip() or any(e.get('authorization_id') == args.authorization_id for e in events): raise HumanRequired('Attempt ID must be new and nonempty')
        now = datetime.now(timezone.utc); last = datetime.fromisoformat(failure['at'])
        attempts = [datetime.fromisoformat(e['at']) for e in events if e.get('event') in ('recurring_ip_authorization', 'recurring_incomplete_authorization', 'recurring_502_authorization', 'recurring_unavailable_authorization', 'recurring_reviewed_authorization') and e.get('source') == args.source]
        if (now-max([last]+attempts)).total_seconds() < policies[0]['interval_seconds']: raise HumanRequired('Thirty-minute interval has not elapsed')
        ledger['state'] = 'READY'; ledger['bounded_pending'] = dict(id=args.id, state='reserved', authorization_id=args.authorization_id)
        events.append(dict(event='recurring_ip_authorization', at=now.isoformat(), source=args.source, authorization_id=args.authorization_id,
            pending_id=args.id, no_new_budget=True, failure_at=failure['at'], raw_sha256=digest,
            snapshot_sha256=claim['snapshot_sha256'], preserved_before_preflight=True, previous_authorization_id=bounded['authorization_id']))
        save_failures(diag, ledger)
    print('One preserved IP observation reclaimed after preflight failure; no new budget')


def diagnostic_pending_capture(args):
    """One authorized pending-card observation, with no budget reset."""
    out, service, diag = paths(ROOT, args.run_id)
    validate_partition(out, service)
    cp, queue = read(diag/'checkpoint.json'), read(out/'queue.json')
    identity = cp.get('next_id')
    missing_tab = re.fullmatch(r'Error: No open browser tab found for targetId ([A-Za-z0-9_-]+)', cp.get('stop_reason') or '')
    if (cp.get('state') != 'BROWSER_ERROR' or (cp.get('stop_reason') not in ('Error: CDP command timeout: Page.navigate', 'Error: Session with given id not found.') and not missing_tab)
            or not identity or identity != args.id or not args.human_instruction.strip() or not args.authorization_id.strip()
            or not args.connection_evidence.strip() or not (args.tab_id or '').strip()):
        raise ValueError('Requires supported connection-error checkpoint, pending ID, explicit instruction, restored connection evidence and own tab')
    pending = [item for item in queue if item['id'] == identity]
    if len(pending) != 1 or (out/'raw'/f'{identity}.json').exists():
        raise ValueError('Pending queue ID must be unique and have no saved raw')
    if missing_tab:
        old_id = missing_tab.group(1)
        pointer = read(diag/'sandbox.json'); sandbox = Path(pointer['path'])
        saved_tab = read(diag/'browser-tab.json'); sandbox_tab = read(sandbox/'browser-tab.json'); saved_cp = read(sandbox/'checkpoint.json')
        if (saved_tab != sandbox_tab or saved_tab.get('target_id') != old_id or saved_tab.get('ownership') != 'external'
                or saved_tab.get('url') != pending[0]['url'] or args.tab_id == old_id
                or any(saved_cp.get(k) != cp.get(k) for k in ('state','stop_reason','next_id','completed_ids'))):
            raise ValueError('Missing tab evidence, pending URL or replacement target differs')
    index = queue.index(pending[0])
    excluded = {record['id'] for record in unavailable_records(out, queue)} | {record['id'] for record in unresolved_records(out, queue)} | {record['id'] for record in user_skipped_records(out, queue)}
    if cp['completed_ids'] != [item['id'] for item in queue[:index] if item['id'] not in excluded] or excluded - {item['id'] for item in queue[:index]}:
        raise ValueError('Checkpoint must match the saved queue prefix')
    for item in queue[:index]:
        if item['id'] in excluded: continue
        data = reviewed_capture(out, item)
        if not check(data, item['id'])['structural_check_passed'] or not data.get('parameter_audit', {}).get('rendered_complete'):
            raise ValueError('Completed prefix is not structurally verified')
    plan = read(service/'plan.json')
    validate_plan_profession(plan, out/'listing.json')
    changed = {file for file, value in plan['automation_sha256'].items() if sha(ROOT/file) != value}
    changes = read(diag/'automation_changes.json') if (diag/'automation_changes.json').exists() else []
    changes = [dict(c, file=c.get('file', '').replace('\\', '/')) for c in changes]
    if changed - allowed_capture_changes(plan, changes):
        raise ValueError('Unapproved collector source changed')
    validate_extractor_upgrade(out, plan, changed)
    verify_partition_url_guard(plan, changed)
    validate_automation_changes(changes, changed)
    preflight(getattr(args, 'config', None), diag)
    archive = diag/(f'diagnostic-original-{identity}' if missing_tab else 'diagnostic-original')
    with locked(diag):
        ledger = load_failures(diag)
        if ledger['state'] not in ('NEEDS_HUMAN', 'RETRY_AVAILABLE'):
            raise HumanRequired('Diagnostic requires a stopped or retry-available ledger')
        if ledger['state'] == 'RETRY_AVAILABLE' and not ledger.get('unbounded_retries') and ledger['retries_started'] >= ledger['retry_limit']:
            raise HumanRequired('Diagnostic retry budget exhausted')
        if any(e.get('event') == 'pending_diagnostic_started' and (e.get('authorization_id') == args.authorization_id or (e.get('run_id') == args.run_id and (not missing_tab or e.get('pending_id') == identity))) for e in ledger['events']):
            raise HumanRequired('Pending diagnostic or authorization already consumed; no second action')
        if archive.exists():
            raise ValueError('Diagnostic evidence archive already exists')
        archive.mkdir()
        hashes = {}
        for name in ('checkpoint.json', 'sandbox.json', 'browser-tab.json', 'browser-session.local.json', 'timing.json', 'problem.snapshot.txt'):
            source = diag/name
            if source.exists():
                shutil.copy2(source, archive/name)
                hashes[name] = sha(archive/name)
        write(archive/'hashes.json', hashes)
        bounded_retry = ledger['state'] == 'RETRY_AVAILABLE'
        if bounded_retry:
            ledger['retries_started'] += 1
            ledger['retry_started'] = True
            ledger['retry_action_started'] = True
            ledger['active_retry_stage'] = 'capture'
            ledger['state'] = 'RETRY_STARTED'
            ledger['events'].append(dict(at=datetime.now(timezone.utc).isoformat(), stage='capture', event='retry_started',
                reason='Authorized one-card diagnostic after restored connection: ' + args.connection_evidence, retry_number=ledger['retries_started']))
        ledger['events'].append(dict(event='pending_diagnostic_started', at=datetime.now(timezone.utc).isoformat(),
            run_id=args.run_id, pending_id=identity, authorization_id=args.authorization_id,
            instruction=args.human_instruction, connection_evidence=args.connection_evidence, original_hashes=hashes))
        save_failures(diag, ledger)
    # The existing error REPL stays alive; a spent retry is never replenished.
    try:
        cp2, elapsed = browser(dict(mode='capture', queue=pending, completed_ids=cp['completed_ids'], tab_id=args.tab_id,
            previous_capture_seconds=cp.get('capture_seconds', 0)), out, diag)
        if cp2.get('state') != 'CAPTURED' or cp2.get('completed_ids') != cp['completed_ids'] + [identity]:
            raise RuntimeError('Diagnostic did not complete pending ID: ' + str(cp2.get('state')))
        data = reviewed_capture(out, pending[0])
        if not check(data, identity)['structural_check_passed'] or not data.get('parameter_audit', {}).get('rendered_complete'):
            raise RuntimeError('Diagnostic raw failed structural verification')
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        record_failure(diag, 'capture', str(error), 'diagnostic-pending-capture', immediate=True)
        raise
    with locked(diag):
        ledger = load_failures(diag)
        ledger['events'].append(dict(event='pending_diagnostic_result', at=datetime.now(timezone.utc).isoformat(),
            authorization_id=args.authorization_id, pending_id=identity, state=cp2.get('state'),
            completed_ids=cp2.get('completed_ids'), checkpoint_sha256=sha(diag/'checkpoint.json')))
        save_failures(diag, ledger)
    if bounded_retry:
        succeeded(diag, 'capture', 'Authorized pending-ID diagnostic completed with verified raw')
    print(json.dumps({'diagnostic_state': cp2.get('state'), 'pending_id': identity, 'ledger_state': load_failures(diag)['state']}, ensure_ascii=False))


def external_threshold_authorization(args):
    """Record human authorization at a stricter external stop without resetting budget."""
    _, _, diag = paths(ROOT, args.run_id)
    if not args.human_instruction.strip() or not args.authorization_id.strip():
        raise ValueError('Human instruction and stable authorization ID required')
    with locked(diag):
        ledger = load_failures(diag)
        prior = [e for e in ledger['events'] if e.get('authorization_id') == args.authorization_id]
        if prior:
            if len(prior) == 1 and prior[0].get('event') == 'external_threshold_authorization' and prior[0].get('instruction') == args.human_instruction:
                return
            raise ValueError('Authorization ID already used for a different instruction')
        if ledger['state'] != 'RETRY_AVAILABLE' or ledger['failures'] < 2 or ledger['retries_started'] >= ledger['retry_limit']:
            raise HumanRequired('Requires externally stopped technical run with remaining retry budget')
        ledger['events'].append(dict(event='external_threshold_authorization', authorization_id=args.authorization_id,
            instruction=args.human_instruction, failures=ledger['failures'], retries_started=ledger['retries_started'],
            at=datetime.now(timezone.utc).isoformat()))
        save_failures(diag, ledger)
    print('External threshold authorization recorded; history and retry budget unchanged')


def diagnostic_authorization(args):
    out, service, diag = paths(ROOT, args.run_id)
    validate_partition(out, service)
    plan = read(service/'plan.json'); cp = read(diag/'checkpoint.json'); queue = read(out/'queue.json')
    if not plan.get('partial_suffix') or not queue or cp.get('completed_ids') or cp.get('next_id') != queue[0]['id']:
        raise ValueError('Diagnostic authorization requires untouched derivative suffix')
    if queue[0]['id'] not in args.human_instruction or not args.authorization_id.strip() or not args.human_instruction.strip():
        raise ValueError('Diagnostic instruction must identify the pending ID')
    with locked(diag):
        ledger = load_failures(diag)
        if ledger['state'] != 'READY': raise HumanRequired('Diagnostic authorization requires READY ledger')
        if any(e.get('authorization_id') == args.authorization_id for e in ledger['events']):
            raise ValueError('Authorization ID already recorded')
        ledger['events'].append(dict(event='diagnostic_authorization', authorization_id=args.authorization_id,
            instruction=args.human_instruction, run_id=args.run_id, next_id=queue[0]['id'], at=datetime.now(timezone.utc).isoformat()))
        save_failures(diag, ledger)
    print('Diagnostic instruction recorded; failure budget unchanged')


def validate_completed_export(out, service):
    manifest = read(service/'manifest.json')
    if manifest.get('status') != 'complete':
        raise ValueError('Post-finalize review requires a complete export')
    validate_partition(out, service)
    plan = read(service/'plan.json')
    validate_plan_profession(plan, out/'listing.json')
    for path, expected in manifest['files'].items():
        if sha(ROOT/path.replace('\\', '/')) != expected['sha256']:
            raise ValueError(f'Published artifact changed: {path}')
    validate_provenance_redaction(out,service,manifest)
    correction_path = out/'exports/profession_correction.json'
    correction = read(correction_path) if correction_path.exists() else None
    if correction and observed_profession_correction(out, service, plan, correction['reason']) != correction:
        raise ValueError('Profession correction evidence differs')
    queue = read(out/'queue.json')
    verified_unavailable = unavailable_records(out, queue); verified_unresolved = unresolved_records(out, queue); verified_user_skipped = user_skipped_records(out, queue)
    if manifest['coverage'].get('unavailable', []) != verified_unavailable or manifest['coverage'].get('unresolved_observations', []) != verified_unresolved: raise ValueError('Manifest observation coverage differs')
    if manifest['coverage'].get('user_skipped_observations', []) != verified_user_skipped or manifest['coverage'].get('user_skipped_count', 0) != len(verified_user_skipped): raise ValueError('Manifest user-skip coverage differs')
    if manifest['coverage'].get('user_skip_policy') != (read(out/'user_skip_policy.json') if (out/'user_skip_policy.json').exists() else None): raise ValueError('Manifest user-skip policy differs')
    if manifest['coverage'].get('user_detail_skip_policy') != (read(out/'user_detail_skip_policy.json') if (out/'user_detail_skip_policy.json').exists() else None): raise ValueError('Manifest incomplete-detail policy differs')
    excluded = {r['id'] for r in verified_unavailable} | {r['id'] for r in verified_unresolved} | {r['id'] for r in verified_user_skipped}
    partial = manifest['coverage'].get('partial')
    if partial:
        if read(out/'exports/partial_selection.json') != partial or partial['plan_sha256'] != sha(service/'plan.json') or partial['queue_sha256'] != sha(out/'queue.json'):
            raise ValueError('Partial selection provenance differs')
        ids = [x['id'] for x in queue]
        cut = len(partial['accepted_ids']) + len(partial['skipped_ids']) + len(partial.get('unresolved_observation_ids', [])) + len(partial.get('user_skipped_ids', []))
        valid_mode = (plan.get('partition') and partial.get('method') == 'partition_prefix_v1') or (not plan.get('partition') and plan.get('selection_mode') == 'first_N_unique_on_frozen_listing' and partial.get('method') == 'ordinary_prefix_v1')
        if not valid_mode or not partial.get('authorization', '').strip() or not 0 < cut < len(ids) or partial['unresolved_ids'] != ids[cut:] or partial['next_id'] != ids[cut] or partial['skipped_ids'] != [i for i in ids[:cut] if i in {r['id'] for r in unavailable_records(out, queue)}] or partial.get('unresolved_observation_ids', []) != [i for i in ids[:cut] if i in {r['id'] for r in unresolved_records(out, queue)}] or partial.get('user_skipped_ids', []) != [i for i in ids[:cut] if i in {r['id'] for r in verified_user_skipped}] or partial['accepted_ids'] != [i for i in ids[:cut] if i not in excluded] or excluded - set(ids[:cut]):
            raise ValueError('Partial selection is not the accounted prefix')
        if partial['raw_sha256'] != {i: sha(out/'raw'/f'{i}.json') for i in ids[:cut]}:
            raise ValueError('Partial raw hashes differ')
        if any((out/'raw'/f'{i}.json').exists() for i in ids[cut:]): raise ValueError('Unresolved suffix raw exists')
        queue = queue[:cut]
    expected = [item for item in queue if item['id'] not in excluded]
    for item in expected:
        data = reviewed_capture(out, item)
        if (not check(data, item['id'])['structural_check_passed'] or data.get('source_url') != item['url']
                or not data.get('parameter_audit', {}).get('rendered_complete')):
            raise ValueError('Raw capture fails structural or identity validation')
    with (out/'exports/vacancies_flat.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [r['id'] for r in rows] != [item['id'] for item in expected] or any(r['profession'] != (correction['corrected_profession'] if correction else plan['profession']) for r in rows):
        raise ValueError('Export IDs or profession differ from evidence')
    if manifest.get('profession_correction') != correction or manifest['coverage']['accepted'] != len(rows):
        raise ValueError('Manifest correction or accepted count differs')
    return manifest


POST_FINALIZE_TESTED = ['Scraper/skills/collect-avito/scripts/run.py', 'Scraper/skills/collect-avito/SKILL.md',
                        'Scraper/test_aside_runner.py', 'utils/build_dataset.py', 'utils/test_build_dataset.py']


def post_finalize_review(args):
    out, service, diag = paths(ROOT, args.run_id)
    reviews = sorted((out/'exports').glob('post_finalize_review*.json'))
    dest = out/'exports'/('post_finalize_review.json' if not reviews else f'post_finalize_review_{len(reviews)+1:03}.json')
    if dest.exists() or not args.reason.strip():
        raise ValueError('A new review and a reason are required')
    manifest = validate_completed_export(out, service)
    baseline = dict(manifest['collection'].get('automation_sha256', {}))
    baseline.update(manifest.get('post_capture_automation_sha256', {}))
    changes = manifest.get('automation_changes', []) + (read(diag/'automation_changes.json') if (diag/'automation_changes.json').exists() else [])
    noted = {c['file'] for c in changes}
    changed = {f for f, value in baseline.items() if sha(ROOT/f) != value}
    if changed - noted or not changed:
        raise ValueError('Post-finalize source changes must be recorded and nonempty')
    validate_automation_changes(changes, changed)
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest', 'Scraper.test_aside_runner', 'utils.test_build_dataset'], check=True, cwd=ROOT)
    subprocess.run([sys.executable, '-X', 'utf8', 'utils/build_dataset.py', '--check'], check=True, cwd=ROOT)
    write(dest, dict(method='post_finalize_validation_v1', reason=args.reason,
        original_manifest_sha256=sha(service/'manifest.json'), previous_automation_sha256=baseline, automation_changes=changes,
        previous_review_sha256=sha(reviews[-1]) if reviews else None,
        current_automation_sha256={f: sha(ROOT/f) for f in sorted(set(baseline) | set(POST_FINALIZE_TESTED) | noted)},
        validation='Partition, plan, page hashes, URL chain, raw, correction, CSV IDs and profession, dataset check and offline regression suites passed'))


def check_staged_diff(root):
    """Keep Git checks strict except non-marker text in staged data separator lines."""
    if subprocess.check_output(['git', 'ls-files', '-u', '-z'], cwd=root):
        raise ValueError('Unmerged index entries')
    result = subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=root, text=True, capture_output=True)
    if result.returncode == 0:
        return
    if result.returncode != 2 or result.stderr:
        raise ValueError('Git diff check failed: ' + result.stderr + result.stdout)
    for diagnostic in result.stdout.splitlines():
        match = re.fullmatch(r'(.+):(\d+): leftover conflict marker', diagnostic)
        if not match:
            raise ValueError('Git diff check failed: ' + diagnostic)
        path, number = match.group(1), int(match.group(2))
        if not (path.startswith('Data/') and Path(path).suffix.lower() in {'.csv', '.json'}):
            raise ValueError('Git diff check failed: ' + diagnostic)
        data = subprocess.check_output(['git', 'show', ':' + path], cwd=root).decode('utf-8-sig').split('\n')
        line = data[number-1].removesuffix('\r') if 0 < number <= len(data) else ''
        if not re.fullmatch(r'={7}\s*[^=\s].*', line):
            raise ValueError('Git diff check failed: ' + diagnostic)


def stage_pinned_provenance(root, service, manifest):
    """Keep the exact bytes of this run's manifest and pinned service artifacts."""
    prefix = service.relative_to(root).as_posix() + '/'
    pinned = [name.replace('\\', '/') for name in manifest['files'] if name.replace('\\', '/').startswith(prefix)]
    pinned.append((service/'manifest.json').relative_to(root).as_posix())
    subprocess.run(['git', '-c', 'core.autocrlf=false', 'add', '--renormalize', '--', *pinned], cwd=root, check=True)


def verify_staged_run_privacy(root, out, service):
    """Reject private locators in staged run data and service metadata."""
    paths = subprocess.check_output(['git','ls-files','--cached','-z','--',str(out.relative_to(root)),str((service/'plan.json').relative_to(root)),str((service/'manifest.json').relative_to(root))],cwd=root).decode('utf8').split('\0')
    paths = [name for name in paths if name and Path(name).suffix.lower() in ('.json','.csv','.md','.txt')]
    if any('\n' in name or '\r' in name for name in paths): raise ValueError('Invalid staged run artifact name')
    if not paths: return
    query=b''.join((':'+name+'\n').encode('utf8') for name in paths)
    from io import BytesIO
    stream=BytesIO(subprocess.run(['git','cat-file','--batch'],cwd=root,input=query,stdout=subprocess.PIPE,check=True).stdout)
    marker=re.compile(rb'/Users/|/home/|\.aside[/\\]|TL_ASIDE_|profileId|target_id',re.I)
    for _ in paths:
        header=stream.readline().rstrip(b'\n').split()
        if len(header)!=3 or header[1]!=b'blob' or not header[2].isdigit(): raise ValueError('Cannot inspect staged run artifact')
        content=stream.read(int(header[2]))
        if stream.read(1)!=b'\n' or marker.search(content): raise ValueError('Staged run artifact contains private locator')
    if stream.read(1): raise ValueError('Unexpected staged run artifact output')


def verify_staged_manifest(root, service, manifest):
    """Check every manifest file against the Git index, including exact byte count."""
    entries = [(name.replace('\\', '/'), value['bytes'], value['sha256']) for name, value in manifest['files'].items()]
    manifest_path = service/'manifest.json'
    entries.append((manifest_path.relative_to(root).as_posix(), manifest_path.stat().st_size, sha(manifest_path)))
    for name, _, _ in entries:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\n' in name or '\r' in name:
            raise ValueError('Invalid manifest path for staged byte verification')
    query = b''.join((':' + name + '\n').encode('utf8') for name, _, _ in entries)
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=root, input=query, stdout=subprocess.PIPE, check=True).stdout
    from io import BytesIO
    stream = BytesIO(result)
    for name, size, digest in entries:
        header = stream.readline().rstrip(b'\n').split()
        if len(header) != 3 or header[1] != b'blob' or not header[2].isdigit():
            raise ValueError(f'Missing staged manifest artifact: {name}')
        actual_size = int(header[2]); content = stream.read(actual_size); delimiter = stream.read(1)
        if delimiter != b'\n' or actual_size != size or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'Staged bytes differ from manifest: {name}')
    if stream.read(1): raise ValueError('Unexpected staged artifact output')


def publish(args):
    out, service, diag = paths(ROOT, args.run_id)
    validate_partition(out, service)
    if (out/'dedup-empty.json').exists():
        print(json.dumps({'state':'DEDUP_EMPTY','run_id':args.run_id})); return
    ensure_allowed(diag, 'publish', getattr(args, 'retry_reason', None), action=True)
    manifest = read(service / 'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Only completed exports can be published')
    validate_provenance_redaction(out,service,manifest)
    if manifest.get('coverage', {}).get('partial'):
        validate_completed_export(out, service)
    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=ROOT, text=True).strip()
    git('fetch', 'origin', 'main')
    git('merge-base', '--is-ancestor', 'origin/main', 'HEAD')
    if git('branch', '--show-current') == 'main':
        raise ValueError('Publish from a codex/<run_id> branch')
    for path, expected in manifest['files'].items():
        if sha(ROOT / path.replace('\\', '/')) != expected['sha256']:
            raise ValueError(f'Published artifact changed: {path}')
    expected_automation = dict(manifest['collection'].get('automation_sha256', {}))
    expected_automation.update(manifest.get('post_capture_automation_sha256', {}))
    if any(sha(ROOT/file) != value for file,value in expected_automation.items()):
        reviews = sorted((out/'exports').glob('post_finalize_review*.json'))
        if not reviews:
            raise ValueError('Collector source changed after collection; document and resolve before publication')
        previous = None
        for index, path in enumerate(reviews):
            expected_name = 'post_finalize_review.json' if index == 0 else f'post_finalize_review_{index+1:03}.json'
            if path.name != expected_name or read(path).get('previous_review_sha256') != previous:
                raise ValueError('Post-finalize review chain differs')
            previous = sha(path)
        review = read(reviews[-1])
        validate_completed_export(out, service)
        if (review.get('method') != 'post_finalize_validation_v1' or review.get('original_manifest_sha256') != sha(service/'manifest.json')
                or review.get('previous_automation_sha256') != expected_automation
                or any(sha(ROOT/f) != h for f, h in review.get('current_automation_sha256', {}).items())
                or not (set(expected_automation) | set(POST_FINALIZE_TESTED)).issubset(review.get('current_automation_sha256', {}))
                or not review.get('reason', '').strip()):
            raise ValueError('Post-finalize review no longer matches source or manifest')
    # Existing fast batch validations suffice for data-only runs; changed code needs its own tests.
    subprocess.run([sys.executable, '-X', 'utf8', 'utils/build_dataset.py', '--check'], check=True, cwd=ROOT)
    files = [str(out.relative_to(ROOT)), str((service/'plan.json').relative_to(ROOT)),
        str((service/'manifest.json').relative_to(ROOT)), 'Data/dataset/sources.json',
        'Data/dataset/parts', 'Data/dataset/manifest.json', 'Data/README.md', 'Data/dataset/README.md']
    if manifest['collection'].get('deduplication'):
        pointer = service/'shared-ledger.json'
        if str(pointer.relative_to(ROOT)) not in manifest['files']:
            raise ValueError('Dedup public ledger pointer missing from manifest')
        files.append(str(pointer.relative_to(ROOT)))
    source_entries = [e for e in read(ROOT/'Data/dataset/sources.json')['exports'] if e['run_id'] == args.run_id]
    if len(source_entries) != 1:
        raise ValueError('Expected exactly one registered source for this run')
    transformation = source_entries[0].get('transformation')
    if transformation:
        # build_dataset --check above verified lineage and confined these paths.
        files += [source_entries[0]['path'], transformation['path']]
    for archive in sorted(service.glob('superseded-freeze*')):
        if archive.is_dir(): files.append(str(archive.relative_to(ROOT)))
    files += [change['file'] for change in manifest.get('automation_changes', [])]
    if any(sha(ROOT/file) != value for file,value in expected_automation.items()):
        files += [change['file'] for change in review.get('automation_changes', [])]
    staged = [path.replace('\\', '/') for path in git('diff', '--cached', '--name-only').splitlines()]
    allowed_paths = [path.replace('\\', '/') for path in files]
    if any(not any(path == own or path.startswith(own + '/') for own in allowed_paths) for path in staged):
        raise ValueError('Stage contains unrelated changes; inspect before publication')
    git('add', '--', *files)
    stage_pinned_provenance(ROOT, service, manifest)
    body = diag / 'pr.md'
    info = manifest.get('collection', {}).get('deduplication')
    summary = args.summary
    if info:
        summary += f" Дедупликация: исходная очередь {info['source_count']}, повторов исключено {info['duplicate_excluded']}, выбрано новых ID {info['novel_count']}; основа: {', '.join(x['run_id'] for x in info['against'])}{' общий датасет' if info.get('dataset') else ''}."
    subprocess.run([sys.executable, 'utils/dataset_pr.py', '--run-id', args.run_id,
        '--summary', summary, '--output', str(body)], check=True, cwd=ROOT)
    verify_staged_manifest(ROOT, service, manifest)
    verify_staged_run_privacy(ROOT, out, service)
    check_staged_diff(ROOT)
    if args.prepare_only:
        print(f'Staged explicit run files; review body at {body}')
        return
    # A retry after add/commit/push must resume, not fail because our own stage is present.
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', f'data: {args.run_id} ({manifest["coverage"]["accepted"]} вакансий)')
    git('push', '-u', 'origin', 'HEAD')
    existing = subprocess.run(['gh', 'pr', 'view', '--json', 'url,state'], cwd=ROOT, text=True, capture_output=True)
    if existing.returncode == 0 and json.loads(existing.stdout).get('state') == 'OPEN':
        result = json.loads(existing.stdout)['url']
    else:
        result = subprocess.check_output(['gh', 'pr', 'create', '--base', 'main', '--title', args.title,
                                          '--body-file', str(body)], cwd=ROOT, text=True)
    print(result.strip())


def sweep_metadata():
    return dict(instruction_version=instruction_version(SCRIPTS.parent/'SKILL.md'),
        instruction_sha256=sha(SCRIPTS.parent/'SKILL.md'),
        code_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        automation_sha256={file:sha(ROOT/file) for file in AUTOMATION_FILES},
        uncommitted_automation_files=dirty_automation(ROOT))


def sweep_init(args):
    print(json.dumps(sweep_init_impl(ROOT, args.run_id, args.spec, sweep_metadata()), ensure_ascii=False))


def sweep_snapshot(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.name):
        raise ValueError('Use a path-safe snapshot name')
    avito_url(args.url)
    sweep_data = ROOT/'Data/avito_sweeps'/args.run_id/'observations'/args.name
    sweep_service = ROOT/'Scraper/sweeps'/args.run_id
    if not (sweep_service/'plan.json').exists():
        raise ValueError('Initialize the sweep before observations')
    target = sweep_service/'observations'/f'{args.name}.json'
    if target.exists() or sweep_data.exists():
        raise ValueError('Sweep snapshot name already exists')
    diag = sweep_service/'observations'/f'{args.name}.diagnostics'
    preflight(args.config, diag)
    checkpoint, _ = browser(dict(mode='freeze', url=args.url, completed_ids=[], tab_id=args.tab_id,
        allow_navigation=False, ui_next=False), sweep_data, diag)
    if checkpoint.get('state') != 'READY':
        raise CollectionStopped('Sweep snapshot requires a stable listing or verified empty result', checkpoint.get('state'))
    listing = read(diag/'listing.json')
    if listing.get('state') not in ('LISTING','EMPTY_RESULT'):
        raise ValueError('Sweep snapshot is not a listing')
    write(target, listing)
    release_browser(diag)
    print(json.dumps({'state':'READY','snapshot':str(target.relative_to(ROOT)),
        'selected_controls':listing.get('selected_controls',[]),
        'available_filter_controls':listing.get('available_filter_controls',[])},ensure_ascii=False))


def sweep_register_branch(args):
    result = sweep_register_impl(ROOT, args.run_id, args.branch_id, args.branch_run_id, args.reset_evidence)
    print(json.dumps(result, ensure_ascii=False))


def sweep_set_branch_status(args):
    print(json.dumps(sweep_status_impl(ROOT, args.run_id, args.branch_id, args.status, args.reason), ensure_ascii=False))


def sweep_materialize(args):
    print(json.dumps(sweep_materialize_impl(ROOT, args.run_id, args.capture_run_id,
        sweep_metadata(), args.allow_partial), ensure_ascii=False))


def sweep_report(args):
    print(json.dumps(sweep_report_impl(ROOT, args.run_id), ensure_ascii=False))


def sweep_demo_export(args):
    print(json.dumps(sweep_demo_impl(ROOT, args.run_id, args.source_run_id, args.limit), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['register-standing-run', 'redact-finalized-provenance', 'archive-partial-pending', 'recapture-cleared-restriction', 'user-detail-skip-policy', 'user-skip-policy', 'mark-policy-skipped', 'mark-user-skipped', 'public-provenance', 'review-address-button', 'recurring-reviewed-continuation', 'recurring-unavailable-continuation', 'recurring-incomplete-policy', 'recurring-incomplete-attempt', 'recurring-ip-after-preflight', 'recurring-ip-policy', 'recurring-ip-attempt', 'sweep-init', 'sweep-snapshot', 'sweep-register-branch', 'sweep-set-branch-status', 'sweep-materialize', 'sweep-report', 'sweep-demo-export', 'archive-uncertain-freeze', 'archive-partial-listing', 'partition', 'partial-suffix', 'deduplicate-queue', 'continuation', 'supplemental', 'freeze', 'capture', 'finalize', 'post-finalize-review', 'publish', 'reselect', 'standing-unavailable-policy', 'closed-policy', 'review-policy', 'blocked-listing-policy', 'salary-review-policy', 'review-salary-line', 'mark-unavailable', 'mark-redirect-unavailable', 'unresolved-policy', 'upgrade-unresolved-extractor', 'mark-unresolved', 'preserve-stopped-observation', 'retry-redirect-observation', 'retry-incomplete-observation', 'note-change', 'record-error', 'check-stage', 'stage-ok', 'resume-offline', 'resume-browser', 'diagnostic-authorization', 'external-threshold-authorization', 'diagnostic-pending-capture']:
        command = commands.add_parser(name)
        command.add_argument('--run-id', required=True)
        if name == 'register-standing-run':
            for flag in ('goal-source', 'authorization-id', 'technical-instruction', 'ip-instruction'):
                command.add_argument('--' + flag, required=True)
        elif name == 'archive-partial-pending':
            command.add_argument('--id', required=True); command.add_argument('--human-instruction', required=True)
        elif name == 'recapture-cleared-restriction':
            command.add_argument('--id', required=True); command.add_argument('--authorization-id', required=True); command.add_argument('--tab-id', required=True); command.add_argument('--retry-reason', required=True); command.add_argument('--config', type=Path)
        elif name in ('user-skip-policy', 'user-detail-skip-policy'):
            command.add_argument('--authorization-id', required=True); command.add_argument('--human-instruction', required=True)
        elif name == 'mark-policy-skipped':
            command.add_argument('--id', required=True); command.add_argument('--authorization-id', required=True)
        elif name == 'mark-user-skipped':
            command.add_argument('--id', required=True); command.add_argument('--authorization-id', required=True); command.add_argument('--human-instruction', required=True)
        elif name == 'review-address-button':
            command.add_argument('--source', required=True); command.add_argument('--id', required=True)
        elif name == 'recurring-incomplete-policy':
            command.add_argument('--source', required=True); command.add_argument('--human-instruction', required=True)
        elif name == 'recurring-ip-policy':
            command.add_argument('--source', required=True); command.add_argument('--human-instruction', required=True); command.add_argument('--additional-instruction', required=True)
        elif name in ('recurring-reviewed-continuation', 'recurring-unavailable-continuation', 'recurring-incomplete-attempt', 'recurring-ip-attempt', 'recurring-ip-after-preflight'):
            command.add_argument('--source', required=True); command.add_argument('--id', required=True); command.add_argument('--authorization-id', required=True)
            if name == 'recurring-incomplete-attempt': command.add_argument('--visible-http-502', action='store_true', help='Require an exact saved visible HTTP 502 page')
            if name == 'recurring-ip-attempt': command.add_argument('--missing-sandbox-log', type=Path, help='Original completed Aside log when the stopped sandbox is missing')
        elif name in ('freeze', 'capture', 'diagnostic-pending-capture'):
            command.add_argument('--tab-id', help='Explicitly authorized Aside browser targetId')
            command.add_argument('--config', type=Path, help='Local config also used by setup_workspace/preflight')
            for target in ('host', 'account', 'profile'): command.add_argument('--'+target, help='Explicit Aside target; must match the pinned run')
        if name in ('register-standing-run', 'redact-finalized-provenance', 'archive-partial-pending', 'recapture-cleared-restriction', 'user-detail-skip-policy', 'user-skip-policy', 'mark-policy-skipped', 'mark-user-skipped', 'public-provenance', 'review-address-button', 'recurring-reviewed-continuation', 'recurring-unavailable-continuation', 'recurring-incomplete-policy', 'recurring-incomplete-attempt', 'recurring-ip-policy', 'recurring-ip-attempt', 'recurring-ip-after-preflight'):
            pass
        elif name == 'sweep-init':
            command.add_argument('--spec', type=Path, required=True)
        elif name == 'sweep-snapshot':
            command.add_argument('--name', required=True)
            command.add_argument('--url', required=True)
            command.add_argument('--tab-id', required=True)
            command.add_argument('--config', type=Path)
        elif name == 'sweep-register-branch':
            command.add_argument('--branch-id', required=True)
            command.add_argument('--branch-run-id', required=True)
            command.add_argument('--reset-evidence', type=Path)
        elif name == 'sweep-set-branch-status':
            command.add_argument('--branch-id', required=True)
            command.add_argument('--status', choices=['unavailable','skipped','error'], required=True)
            command.add_argument('--reason', required=True)
        elif name == 'sweep-materialize':
            command.add_argument('--capture-run-id', required=True)
            command.add_argument('--allow-partial', action='store_true')
        elif name == 'sweep-demo-export':
            command.add_argument('--source-run-id', required=True)
            command.add_argument('--limit', type=int,
                help='Optional deterministic row limit for a review-sized demonstration CSV')
        elif name == 'sweep-report':
            pass
        elif name in ('archive-uncertain-freeze', 'archive-partial-listing', 'post-finalize-review'):
            command.add_argument('--reason', required=True)
        elif name == 'partition':
            command.add_argument('--batch-size', type=int, required=True)
            command.add_argument('--prefix', required=True)
        elif name == 'deduplicate-queue':
            command.add_argument('--source-run-id', required=True)
            command.add_argument('--against-run-id', action='append')
            command.add_argument('--against-dataset', action='store_true')
            command.add_argument('--limit', type=int, help='Select at most this many novel IDs in listing order')
        elif name == 'supplemental':
            command.add_argument('--parent-run-id', required=True)
            command.add_argument('--observation', required=True, help='Listing JSON filename in parent diagnostics')
            command.add_argument('--evidence', required=True, help='Filter evidence JSON filename in parent diagnostics')
        elif name == 'continuation':
            command.add_argument('--parent-run-id', required=True)
        elif name == 'partial-suffix':
            command.add_argument('--source-run-id', required=True)
        elif name == 'freeze':
            command.add_argument('--retry-reason')
            command.add_argument('--ui-next', action='store_true', help='Click observed Next in explicit tab and verify landed page')
            command.add_argument('--resume-route-authorization-id', help='Recorded human browser authorization for switching an unfinished freeze to UI Next')
            command.add_argument('--branch-only', action='store_true', help='Freeze listing evidence for a sweep branch; never capture this run')
            command.add_argument('--max-pages', type=int, help='Finite page cap for --branch-only, at most 30')
            command.add_argument('--resume', action='store_true', help='Recover an interrupted freeze with identical inputs')
            command.add_argument('--accept-saved', action='store_true', help='Offline: accept the incomplete saved listing with explicit human instruction')
            command.add_argument('--human-instruction', help='Explicit instruction to accept exactly the saved listing')
            command.add_argument('--agent-model', default=None, help='Actual browser agent model, not a model switch')
            command.add_argument('--url', required=True, help='Observed listing URL after filter confirmation')
            size = command.add_mutually_exclusive_group(required=True)
            size.add_argument('--limit', type=int, help='Total unique vacancies; no fixed page/card ceiling')
            size.add_argument('--all-pages', action='store_true', help='Follow observed Next links until the listing ends')
            command.add_argument('--selection-criterion', choices=['remote', 'hours_11_12'], default='remote')
            command.add_argument('--candidate-limit', type=int, help='Select first N exact remote parameter matches from this many observed candidates')
            command.add_argument('--geography', required=True)
            command.add_argument('--profession', default='', help='Optional assertion; profession is read from selected listing filters')
            command.add_argument('--profession-scope', choices=['top20', 'all'], help='Selection already applied in the UI; do not recheck collapsed controls')
            command.add_argument('--segments', default='')
            command.add_argument('--filter-evidence', required=True, help='What was visibly confirmed in the listing')
        elif name == 'diagnostic-pending-capture':
            command.add_argument('--id', required=True)
            command.add_argument('--human-instruction', required=True)
            command.add_argument('--authorization-id', required=True)
            command.add_argument('--connection-evidence', required=True)
        elif name == 'capture':
            command.add_argument('--retry-reason')
            command.add_argument('--batch-size', type=int, default=30)
        elif name == 'finalize':
            command.add_argument('--partial-prefix', action='store_true', help='Export a verified stopped ordinary or partition queue prefix, preserving remaining IDs')
            command.add_argument('--partial-instruction', help='Exact authorization to publish the verified prefix offline')
            command.add_argument('--city-anomaly-policy', choices=['blank','search_filter'], default='blank')
            command.add_argument('--profession-from-observed-controls', action='store_true')
            command.add_argument('--profession-correction-reason')
            command.add_argument('--retry-reason')
            command.add_argument('--review-model', default=None, help='Actual reviewer model, not a model switch')
            command.add_argument('--placements', type=Path, help='Reviewed ID -> city,region,evidence JSON for unknown places')
        elif name == 'publish':
            command.add_argument('--retry-reason')
            command.add_argument('--summary', required=True)
            command.add_argument('--title', required=True)
            command.add_argument('--prepare-only', action='store_true')
        elif name == 'salary-review-policy':
            command.add_argument('--id', required=True)
            command.add_argument('--authorization-id', required=True)
            command.add_argument('--standing-technical', action='store_true', help='Apply an explicit standing instruction to fix technical collection issues and continue')
        elif name == 'review-salary-line':
            command.add_argument('--id', required=True)
            command.add_argument('--authorization-id', required=True)
            command.add_argument('--source', help='Previously recorded recurring request policy source')
        elif name == 'standing-unavailable-policy':
            command.add_argument('--authorization-id', default=STANDING_UNAVAILABLE_ID)
            command.add_argument('--human-instruction', default=STANDING_UNAVAILABLE_INSTRUCTION)
        elif name == 'blocked-listing-policy':
            command.add_argument('--authorization-id', required=True)
        elif name == 'review-policy':
            command.add_argument('--authorization-id', required=True)
        elif name == 'closed-policy':
            command.add_argument('--authorization-id', required=True)
            command.add_argument('--authorization-question', help='Exact preceding question when the saved answer confirms it')
        elif name == 'mark-unavailable':
            command.add_argument('--offline-standing', action='store_true')
            command.add_argument('--standing-rejected', action='store_true', help='Apply the recorded project unavailable policy to an evidenced rejected 404 using the previous marker set')
            command.add_argument('--id', required=True)
            command.add_argument('--authorization-id', required=True)
        elif name == 'upgrade-unresolved-extractor':
            pass
        elif name == 'unresolved-policy':
            command.add_argument('--authorization-id', required=True)
            command.add_argument('--authorization-question', required=True)
        elif name == 'mark-unresolved':
            command.add_argument('--id', required=True)
            command.add_argument('--authorization-id', required=True)
        elif name == 'preserve-stopped-observation':
            command.add_argument('--authorization-id', required=True)
        elif name == 'retry-redirect-observation':
            command.add_argument('--id', required=True)
            command.add_argument('--observed-url', required=True)
            command.add_argument('--authorization-id', required=True)
        elif name == 'mark-redirect-unavailable':
            command.add_argument('--id', required=True)
            command.add_argument('--observed-url', required=True)
            command.add_argument('--authorization-id', required=True)
        elif name == 'retry-incomplete-observation':
            command.add_argument('--id', required=True)
            command.add_argument('--authorization-id', required=True)
            command.add_argument('--standing-incomplete', action='store_true', help='Use an explicit standing human instruction permitting incomplete-shell technical retries')
        elif name == 'reselect':
            command.add_argument('--reason', required=True)
        elif name == 'record-error':
            for flag in ['stage', 'message', 'action']:
                command.add_argument('--' + flag, required=True)
            command.add_argument('--url')
            command.add_argument('--agent')
            command.add_argument('--immediate', action='store_true')
        elif name in ('check-stage', 'stage-ok'):
            command.add_argument('--stage', required=True)
            command.add_argument('--retry-reason' if name == 'check-stage' else '--evidence', required=name == 'stage-ok')
        elif name in ('resume-offline', 'resume-browser', 'diagnostic-authorization', 'external-threshold-authorization'):
            command.add_argument('--human-instruction', required=True)
            if name in ('resume-browser', 'diagnostic-authorization', 'external-threshold-authorization'):
                command.add_argument('--authorization-id', required=True, help='Stable ID for this fresh human instruction; reuse on CLI retries')
            if name == 'resume-browser': command.add_argument('--pending-id-no-budget', help='Authorize one pending ID without replenishing an exhausted retry budget')
            if name == 'resume-browser': command.add_argument('--detail-hydration', action='store_true', help='Limit pending authorization to an evidenced incomplete DETAIL hydration race')
            if name == 'resume-browser': command.add_argument('--unbounded-retries', action='store_true', help='Record explicit authorization for unbounded ordinary technical retries in this run')
            if name == 'resume-browser': command.add_argument('--correct-record-error', action='store_true', help='Audit the narrowly identified pre-freeze administrative record-error misclassification')
            if name == 'resume-browser': command.add_argument('--retain-unbounded', action='store_true', help='Retain an existing unbounded technical retry authorization after a fresh human resume')
        else:
            for flag in ['file', 'reason', 'validation']:
                command.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    for target in ('host', 'account', 'profile'):
        value = getattr(args, target, None); key = 'TL_ASIDE_'+target.upper()
        if value:
            if os.environ.get(key) and os.environ[key] != value: parser.error(f'--{target} differs from {key}')
            os.environ[key] = value
    if args.command in ('freeze', 'capture', 'diagnostic-pending-capture'):
        target_file = paths(ROOT, args.run_id)[2]/'aside-target.local.json'
        if target_file.exists():
            pinned = read(target_file)
            for target in ('host', 'account', 'profile'):
                value = getattr(args, target, None)
                if value and value != pinned.get('TL_ASIDE_'+target.upper()): parser.error(f'--{target} differs from the pinned run target')
    if args.command == 'capture' and args.batch_size < 1:
        parser.error('batch-size must be positive')
    with ExitStack() as stack:
        try:
            if args.command == 'register-standing-run':
                stack.enter_context(operation_locked(paths(ROOT, args.run_id)[2]))
            if args.command in ('redact-finalized-provenance', 'archive-partial-pending', 'recapture-cleared-restriction', 'user-detail-skip-policy', 'user-skip-policy', 'mark-policy-skipped', 'mark-user-skipped', 'public-provenance', 'review-address-button', 'recurring-reviewed-continuation', 'recurring-unavailable-continuation', 'recurring-incomplete-policy', 'recurring-incomplete-attempt', 'recurring-ip-after-preflight', 'recurring-ip-policy', 'recurring-ip-attempt', 'archive-uncertain-freeze', 'archive-partial-listing', 'partition', 'partial-suffix', 'deduplicate-queue', 'continuation', 'supplemental', 'freeze', 'capture', 'finalize', 'post-finalize-review', 'publish', 'reselect', 'standing-unavailable-policy', 'closed-policy', 'review-policy', 'blocked-listing-policy', 'salary-review-policy', 'review-salary-line', 'mark-unavailable', 'mark-redirect-unavailable', 'unresolved-policy', 'upgrade-unresolved-extractor', 'mark-unresolved', 'preserve-stopped-observation', 'retry-redirect-observation', 'retry-incomplete-observation', 'record-error', 'stage-ok', 'resume-browser', 'resume-offline', 'diagnostic-authorization', 'external-threshold-authorization', 'diagnostic-pending-capture'):
                lock_run_id = args.parent_run_id if args.command in ('continuation', 'supplemental') else args.source_run_id if args.command in ('partial-suffix', 'deduplicate-queue') else args.run_id
                stack.enter_context(operation_locked(paths(ROOT, lock_run_id)[2]))
            globals()[args.command.replace('-', '_')](args)
            if args.command in ('freeze', 'capture', 'finalize', 'publish') and not (paths(ROOT, args.run_id)[0]/'dedup-empty.json').exists():
                _, _, diag = paths(ROOT, args.run_id)
                succeeded(diag, args.command, 'Command completed successfully')
        except HumanRequired as error:
            parser.exit(2, f'{error}\n')
        except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
            if args.command in ('freeze', 'capture', 'finalize', 'publish'):
                _, _, diag = paths(ROOT, args.run_id)
                event = record_failure(diag, args.command, str(error), args.command,
                    url=getattr(args,'url',None), agent=getattr(args,'agent_model',None),
                    immediate=getattr(error,'immediate',False))
                print(json.dumps(event, ensure_ascii=False), file=sys.stderr)
            parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
