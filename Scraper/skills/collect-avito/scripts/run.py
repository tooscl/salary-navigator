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
from threading import Thread
from urllib.parse import urlsplit

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

VERSION = '1.7.26'
AUTOMATION_FILES = ['Scraper/skills/collect-avito/references/top-professions.json', 'utils/title_flags.py', 'Scraper/extract.js', 'Scraper/skills/collect-avito/SKILL.md'] + [
    'Scraper/skills/collect-avito/scripts/' + name for name in ('run.py', 'browser_batch.js', 'check_capture.py', 'failure_log.py', 'preflight.py')]


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


def validate_plan_profession(plan, listing_path):
    if 'profession' not in plan:
        raise ValueError('Plan profession is missing')
    if plan.get('listing_sha256') and sha(listing_path) != plan['listing_sha256']:
        raise ValueError('Frozen listing changed after freeze; preserve the original observation')
    observed = selection_profession(read(listing_path), plan.get('profession_scope'))
    if norm(plan.get('profession') or '') != observed:
        raise ValueError('Plan profession differs from frozen listing filters; do not export inconsistent metadata')
    for index, page in enumerate(plan.get('listing_pages', [])):
        path = listing_path.parent / page['file']
        if sha(path) != page['sha256']:
            raise ValueError('Frozen listing page changed after freeze')
        if index:
            validate_page_filters(read(listing_path), read(path))



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



def remote_decisions(queue, out, target):
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
        values = [value for name, value in fields if name == 'Формат работы']
        if len(values) > 1:
            raise ValueError('Duplicate format parameter requires review')
        match = values == ['Удалённо']
        decisions.append(dict(id=item['id'], listing_rank=item['listing_rank'],
            accepted=match, parameter='Формат работы', observed_value=values[0] if values else None,
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
    for name in ['listing.json', 'checkpoint.json', 'timing.json', 'problem.snapshot.txt', 'browser-tab.json']:
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


def pipe_chunks(stream):
    """Read subprocess stdout without select(), which only accepts sockets on Windows."""
    chunks = Queue()

    def read_chunks():
        try:
            while True:
                chunk = stream.read1(65536)
                if not chunk:
                    break
                chunks.put(chunk)
        except (OSError, ValueError):
            pass  # EOF/error is reported by the waiting caller as disconnect.
        finally:
            chunks.put(None)

    reader = Thread(target=read_chunks, daemon=True)
    reader.start()
    return chunks, reader


def retain_repl(process, diag, sandbox, job):
    """Keep our REPL pipes alive until an explicit local release marker is created."""
    release = diag / f'release-browser-session-{process.pid}'
    if release.exists():
        raise RuntimeError('Release marker collision; inspect diagnostics')
    code = """import os,select,sys,time
inp,out,release=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
while not os.path.exists(release):
    ready,_,_=select.select([out],[],[],1)
    if ready and not os.read(out,65536): break
os.close(inp); os.close(out)
"""
    keeper = subprocess.Popen([sys.executable, '-c', code, str(process.stdin.fileno()),
        str(process.stdout.fileno()), str(release)], pass_fds=(process.stdin.fileno(), process.stdout.fileno()),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    tab = read(sandbox/'browser-tab.json') if sandbox and (sandbox/'browser-tab.json').exists() else {}
    write(diag/'browser-session.local.json', dict(repl_pid=process.pid, keeper_pid=keeper.pid,
        sandbox=str(sandbox) if sandbox else None, tab=tab, release_marker=str(release)))


def release_browser(diag):
    path = diag/'browser-session.local.json'
    if path.exists():
        session = read(path)
        Path(session['release_marker']).touch()
        path.rename(diag/'browser-session-released.local.json')


def browser(job, out, diag):
    # One-shot Aside buffers stdout until completion. An interactive handshake
    # records its sandbox BEFORE collection, so interruption cannot orphan raw.
    executable = find_aside()
    if not executable:
        raise RuntimeError('Aside CLI not found; follow collect-avito/references/FIRST_RUN.md')
    process = subprocess.Popen(aside_command(executable) + ['repl'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, cwd=ROOT)
    incoming, reader = pipe_chunks(process.stdout)
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
        sandbox = Path(until(r'TL_SESSION=([^\r\n]+)\r?\n', 20)[1].strip())
        write(diag / 'sandbox.json', {'path': str(sandbox)})
        # One REPL RPC per navigation avoids Aside's per-evaluation 120s budget
        # becoming a timeout on the entire queue. No model calls between cards.
        jobs = [dict(job, close_tab=False)] if job['mode'] == 'freeze' else [
            dict(job, queue=[item], close_tab=False)
            for index,item in enumerate(job['queue'])]
        done = list(job['completed_ids'])
        for part in jobs:
            part['completed_ids'] = done
            script = (SCRIPTS / 'browser_batch.js').read_text()
            script = script.replace('__JOB__', json.dumps(part, ensure_ascii=False), 1)
            script = script.replace('__EXTRACTOR__', json.dumps((ROOT / 'Scraper/extract.js').read_text()), 1)
            expression = 'await eval(' + json.dumps('(async()=>{' + script + '})()') + ')\n'
            offset = len(chunks)
            process.stdin.write(expression.encode()); process.stdin.flush()
            until(r'TL_FINISHED\r?\n', 140, offset)
            import_sandbox(sandbox, out, diag)
            cp = read(diag/'checkpoint.json')
            cp['capture_seconds'] = job.get('previous_capture_seconds', 0) + time.perf_counter()-started
            write(diag/'checkpoint.json', cp)
            done = cp['completed_ids']
            if cp['state'] not in ('READY', 'CAPTURED'):
                break
            if job.get('selection_queue') is not None:
                decisions = remote_decisions(job['selection_queue'], out, job['selection_target'])
                if sum(d['accepted'] for d in decisions) >= job['selection_target']:
                    break
    finally:
        if preserve and process.poll() is None and process.stdin and process.stdout:
            retain_repl(process, diag, sandbox, job)
        if process.stdin:
            process.stdin.close()
        if not preserve:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate(); process.wait(timeout=3)
        if not preserve:
            process.stdout.close()
            reader.join(timeout=1)
        (diag / f'aside-{stamp}.local.log').write_text(chunks.decode('utf-8', errors='replace'))
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
        raise ValueError('A superseded freeze already exists; inspect it before proceeding')
    if not (service/'plan.json').is_file() or not (out/'queue.json').is_file():
        raise ValueError('A complete frozen plan and queue are required')
    archive.mkdir(parents=True)
    shutil.copytree(out, archive/'data')
    shutil.copy2(service/'plan.json', archive/'plan.json')
    write(archive/'reason.json', {'reason': args.reason, 'at': datetime.now(timezone.utc).isoformat()})
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
    pages = [read(parent_out/p['file']) for p in plan['listing_pages']]
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
        target = out/page['file']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_out/page['file'], target)
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
        raise ValueError('Run already exists; capture resumes it, new collection needs a new run_id')
    if (service / 'manifest.json').exists() or (service / 'plan.json').exists():
        raise ValueError('Queue already frozen; use capture, not freeze')
    avito_url(args.url)
    candidate_limit = getattr(args, 'candidate_limit', None)
    all_pages = getattr(args, 'all_pages', False)
    if (all_pages and (args.limit is not None or candidate_limit is not None)) or (not all_pages and (args.limit is None or args.limit < 1)) or (candidate_limit is not None and candidate_limit < args.limit):
        raise ValueError('Use a positive --limit or --all-pages; candidate limit requires a numeric target')
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
    if (out/'listing_pages').exists() and not (diag/'freeze_request.json').exists():
        raise ValueError('Frozen pages lack their original request; cannot safely resume')
    if existing and (diag/'freeze_request.json').exists() and read(diag / 'freeze_request.json') != request:
        raise ValueError('Resume must preserve the original filter request')
    if not (diag / 'freeze_request.json').exists():
        write(diag / 'freeze_request.json', request)
    page_dir = out / 'listing_pages'
    page_paths = sorted(page_dir.glob('*.json'), key=lambda p: int(p.stem)) if page_dir.exists() else []
    if [p.name for p in page_paths] != [f'{i:04d}.json' for i in range(1, len(page_paths)+1)]:
        raise ValueError('Listing page sequence is incomplete')
    pages = [read(path) for path in page_paths]
    for index, page in enumerate(pages):
        expected_url = args.url if index == 0 else pages[index-1].get('pagination', {}).get('next_url')
        if not expected_url or not same_observed_url(expected_url, page.get('source_url', '')):
            raise ValueError('Frozen listing page does not match the observed Next link')
        if index:
            validate_page_filters(pages[0], page)
    pending = diag / 'pending_listing.json'
    if pages and pending.exists() and read(pending) == {'url': pages[-1]['source_url']}:
        pending.unlink()
    cp = read(diag/'checkpoint.json') if (diag/'checkpoint.json').exists() else dict(state='READY', completed_ids=[])
    target = candidate_limit or args.limit
    listing_end_reason = None
    if accept_saved:
        if candidate_limit is not None or all_pages or not pages or (diag/'pending_listing.json').exists() or any((out/'raw').glob('*')):
            raise ValueError('Saved-page acceptance requires complete saved pages and a numeric ordinary request')
        target = len(queue_from_pages(pages, None))
        if not target or target >= args.limit:
            raise ValueError('Saved pages must contain a nonempty incomplete selection')
    while not pages or target is None or len(queue_from_pages(pages, target, False)) < target:
        url = args.url if not pages else pages[-1].get('pagination', {}).get('next_url')
        if not url:
            listing_end_reason = 'confirmed_end' if pages[-1].get('pagination', {}).get('end_confirmed') else 'no_next_link'
            break
        avito_url(url)
        if any(same_observed_url(url, p['source_url']) for p in pages):
            listing_end_reason = 'repeated_next_link'
            break
        recovering = pending.exists() and read(pending) == {'url': url}
        if pending.exists() and not recovering:
            raise ValueError('Pending listing does not match the observed Next link')
        if recovering and (diag / 'sandbox.json').exists():
            import_sandbox(Path(read(diag / 'sandbox.json')['path']), out, diag)
        if recovering and (diag/'checkpoint.json').exists() and read(diag/'checkpoint.json')['state'] == 'READY' and (diag/'listing.json').exists():
            cp = read(diag/'checkpoint.json')
        else:
            for name in ('sandbox.json', 'listing.json', 'checkpoint.json'):
                (diag/name).unlink(missing_ok=True)
            write(pending, {'url': url})
            cp, _ = browser(dict(mode='freeze', url=url, completed_ids=[], tab_id=getattr(args, 'tab_id', None), allow_navigation=bool(pages)), out, diag)
        if cp['state'] != 'READY':
            raise CollectionStopped(f'Stopped: {cp["state"]}; inspect local diagnostics', cp['state'])
        page = read(diag / 'listing.json')
        if not same_observed_url(url, page.get('source_url', '')):
            raise ValueError('Loaded listing URL differs from observed link')
        if pages:
            validate_page_filters(pages[0], page)
        else:
            selection_profession(page, getattr(args, 'profession_scope', None), args.profession)
            if candidate_limit is not None and not any(c.get('label') == 'Удалённо' and 'params[172815]' in (c.get('marker') or '') for c in page.get('selected_controls', [])):
                raise ValueError('Remote selection requires the observed Удалённо filter')
        queue_from_pages([page], target, False)  # validate every observed card before persisting
        path = page_dir / f'{len(pages)+1:04d}.json'
        write(path, page)
        pages.append(page); page_paths.append(path)
        pending.unlink(missing_ok=True)
        release_browser(diag)
    listing = pages[0]
    profession = selection_profession(listing, getattr(args, 'profession_scope', None), args.profession)
    if candidate_limit is not None and not any(c.get('label') == 'Удалённо' and 'params[172815]' in (c.get('marker') or '') for c in listing.get('selected_controls', [])):
        raise ValueError('Remote selection requires the observed Удалённо filter')
    if candidate_limit is not None and candidate_limit < args.limit:
        raise ValueError('Candidate limit must be at least the target limit')
    queue = queue_from_pages(pages, target, require_full=False)
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
                available_listing_exhausted=listing_end_reason == 'confirmed_end', listing_pages=[dict(file=str(p.relative_to(out)), sha256=sha(p)) for p in page_paths]))
            print(json.dumps({'state': 'EMPTY', 'listing_end_reason': listing_end_reason}))
            return
        raise ValueError('No vacancies were observed; cannot create an empty capture queue')
    write(out / 'listing.json', listing)
    write(out / 'queue.json', queue)
    write(service / 'plan.json', dict(run_id=args.run_id, geography=args.geography,
        profession=profession, segments=args.segments, filter_evidence=args.filter_evidence,
        listing_url=listing['source_url'], listing_request_url=args.url, selected_controls=listing.get('selected_controls', []),
        listing_sha256=sha(out / 'listing.json'),
        profession_scope=getattr(args, 'profession_scope', None),
        profession_source='selected_ui_context' if getattr(args, 'profession_scope', None) else 'observed_controls',
        listing_pages=[dict(file=str(path.relative_to(out)), sha256=sha(path), source_url=page['source_url'])
                       for path, page in zip(page_paths, pages)],
        selection_mode='saved_incomplete_listing' if accept_saved else ('first_N_remote_parameter_matches' if candidate_limit is not None else 'first_N_unique_on_frozen_listing'), limit=args.limit if candidate_limit is not None else len(queue), requested_all_pages=all_pages,
        saved_selection_authorization={'instruction': instruction, 'at': datetime.now(timezone.utc).isoformat(), 'original_request': request} if accept_saved else None,
        requested_limit=args.limit, available_unique=len(queue),
        listing_end_reason=listing_end_reason, available_listing_exhausted=listing_end_reason == 'confirmed_end', continuation=continuation_info,
        candidate_limit=candidate_limit,
        created_at=datetime.now(timezone.utc).isoformat(), instruction_version=VERSION,
        extractor_sha256=sha(ROOT / 'Scraper/extract.js'), instruction_sha256=sha(SCRIPTS.parent / 'SKILL.md'),
        code_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        setup_seconds=time.perf_counter() - started, live_route_verified=False,
        execution_model=args.agent_model,
        automation_sha256={file: sha(ROOT/file) for file in AUTOMATION_FILES},
        uncommitted_automation_files=dirty_automation(ROOT)))
    cp.update(next_id=queue[0]['id'], capture_seconds=0)
    write(diag / 'checkpoint.json', cp)
    print(json.dumps({'state': 'READY', 'queued': len(queue), 'heading': listing.get('heading'),
                      'selected_controls': listing.get('selected_controls'),
                      'requested_limit': args.limit, 'listing_end_reason': listing_end_reason,
                      'message': 'Конец выдачи подтверждён' if listing_end_reason == 'confirmed_end' else 'Обход остановлен с ограничением' if listing_end_reason else 'Запрошенная очередь готова'}, ensure_ascii=False))


CLOSED_MARKERS = ('Объявление закрыто.', 'Объявление снято с публикации.')
REVIEW_MARKER = 'Оно ещё на проверке.'
BLOCKED_LISTING_MARKER = 'Оно заблокировано после проверки.'
DELETED_MARKER = 'Объявление удалено.'
NO_LONGER_AVAILABLE_MARKER = 'Объявление больше недоступно.'
UNAVAILABLE_TO_VIEW_MARKER = 'Объявление недоступно для просмотра.'
OTHER_UNAVAILABLE_MARKERS = (DELETED_MARKER, NO_LONGER_AVAILABLE_MARKER, UNAVAILABLE_TO_VIEW_MARKER)
ALL_UNAVAILABLE_MARKERS = (*CLOSED_MARKERS, REVIEW_MARKER, BLOCKED_LISTING_MARKER, *OTHER_UNAVAILABLE_MARKERS)
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
        if not closed_raw_matches(data, identity, record['url'], record.get('evidence')) or not record.get('authorization_id'):
            raise ValueError('Invalid unavailable observation')
        ids.add(identity)
    return records


def mark_unavailable(args):
    out, service, diag = paths(ROOT, args.run_id)
    if (service/'manifest.json').exists():
        raise ValueError('Completed run is immutable')
    ledger = load_failures(diag)
    if ledger['state'] == 'NEEDS_HUMAN' or (ledger['state'] == 'RETRY_STARTED' and not getattr(args, 'automatic', False)):
        raise HumanRequired('Resume browser authorization is required after the stop')
    events = ledger['events']
    auth = [i for i, event in enumerate(events) if event.get('event') in ('human_browser_authorization', 'project_unavailable_policy') and event.get('authorization_id') == args.authorization_id]
    latest_auth = max((i for i, event in enumerate(events) if event.get('event') in ('human_browser_authorization', 'project_unavailable_policy')), default=-1)
    if not auth or latest_auth < 0 or any(event.get('state') == 'NEEDS_HUMAN' for event in events[latest_auth+1:]):
        raise HumanRequired('Current human authorization required')
    instruction = events[auth[-1]]['instruction']
    policies = [read(path) for path in (out/'closed_policy.json', out/'review_policy.json', out/'blocked_listing_policy.json', out/'standing_unavailable_policy.json') if path.exists()]
    policy = next((p for p in policies if p.get('authorization_id') == args.authorization_id and p.get('human_instruction') == instruction and p.get('markers') in (list(CLOSED_MARKERS), [REVIEW_MARKER], [BLOCKED_LISTING_MARKER], list(ALL_UNAVAILABLE_MARKERS))), None)
    if args.id not in instruction and not policy:
        raise ValueError('Human authorization must identify the unavailable ID or authorize its policy')
    cp = read(diag/'checkpoint.json')
    queue = read(out/'queue.json')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != args.id or args.id in cp.get('completed_ids', []):
        raise ValueError('Expected the reviewed pending checkpoint ID')
    if read(service/'plan.json').get('selection_mode') == 'first_N_remote_parameter_matches':
        raise ValueError('Unavailable handling is not supported for remote candidate selection')
    matches = [item for item in queue if item['id'] == args.id]
    prior_excluded = {record['id'] for record in unavailable_records(out, queue)}
    pending_index = next((i for i, item in enumerate(queue) if item['id'] == args.id), None)
    if pending_index is None or cp.get('completed_ids') != [item['id'] for item in queue[:pending_index] if item['id'] not in prior_excluded]:
        raise ValueError('Checkpoint completed IDs must match the reviewed queue prefix')
    if len(matches) != 1:
        raise ValueError('Expected one frozen queue item')
    raw = out/'raw'/f'{args.id}.json'; data = read(raw)
    snap = diag/'problem.snapshot.txt'; snapshot = snap.read_text(encoding='utf-8')
    url = matches[0]['url']
    markers = [marker for marker in ALL_UNAVAILABLE_MARKERS if any(norm(line) == f'- text: "{marker}"' for line in snapshot.splitlines())]
    if len(markers) != 1 or f'[url={url}]' not in snapshot or not closed_raw_matches(data, args.id, url, markers[0]):
        raise ValueError('Missing exact closed-page evidence for the pending URL')
    if policy and markers[0] not in policy['markers']:
        raise ValueError('Observed marker is outside this authorization')
    records = unavailable_records(out, queue)
    if any(record['id'] == args.id for record in records):
        raise ValueError('Unavailable ID already recorded')
    pointer = diag/'sandbox.json'
    if pointer.exists():
        sandbox = Path(read(pointer)['path'])
        if (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or read(sandbox/'checkpoint.json').get('next_id') != args.id:
            raise ValueError('Sandbox differs from stopped observation')
    records.append(dict(id=args.id, url=url, status=('unavailable_under_review' if markers[0] == REVIEW_MARKER else 'unavailable_blocked_after_review' if markers[0] == BLOCKED_LISTING_MARKER else 'unavailable_deleted' if markers[0] == DELETED_MARKER else 'unavailable_other' if markers[0] in OTHER_UNAVAILABLE_MARKERS else 'unavailable_closed'), evidence=markers[0],
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
        if review.get('raw_sha256') != sha(path) or review.get('method') != 'joined_salary_line_v1' or not review.get('authorization_id'):
            raise ValueError('Offline review provenance mismatch')
        data = salary_line_review(data, item['id'], item['url'])
    return data


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
    direct = 'исправ' in lower and 'продолж' in lower
    dialogue = re.search(r'вопрос пользователю:\s*«([^»]+)»\s*ответ пользователя:\s*«([^»]+)»', lower)
    contextual = False
    if dialogue:
        question, answer = dialogue.groups()
        contextual = (all(term in question for term in ('разреш', 'обработ', 'карточк', 'проверк', 'последующ', 'аналогич', 'продолж'))
                      and re.match(r'^\s*да(?:\s|[,!.]|$)', answer)
                      and 'всех таких случа' in answer
                      and not re.search(r'\b(?:нет|не|кроме|только)\b', answer))
    if args.id not in instruction or not (direct or contextual):
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
    if ledger['state'] == 'NEEDS_HUMAN' or (ledger['state'] == 'RETRY_STARTED' and not getattr(args, 'automatic', False)): raise HumanRequired('Current human authorization required')
    auth = [i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization' and event.get('authorization_id') == args.authorization_id]
    latest = max((i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization'), default=-1)
    if not auth or latest < 0 or any(event.get('state') == 'NEEDS_HUMAN' for event in events[latest+1:]):
        raise HumanRequired('Current browser authorization required')
    policy_path = out/'salary_review_policy.json'
    policy = read(policy_path) if policy_path.exists() else {}
    policy_valid = policy.get('authorization_id') == args.authorization_id and policy.get('human_instruction') == events[auth[-1]]['instruction'] and policy.get('method') == 'joined_salary_line_v1'
    if args.id not in events[auth[-1]]['instruction'] and not policy_valid:
        raise HumanRequired('Authorization must identify this ID or cover this salary review')
    cp = read(diag/'checkpoint.json'); queue = read(out/'queue.json')
    if cp.get('state') != 'REVIEW_REQUIRED' or cp.get('stop_reason') != 'identity_or_template_review' or cp.get('next_id') != args.id:
        raise ValueError('Expected the pending review checkpoint')
    matches = [item for item in queue if item['id'] == args.id]
    if len(matches) != 1: raise ValueError('Expected one queue item')
    excluded = {record['id'] for record in unavailable_records(out, queue)}
    index = next(i for i, item in enumerate(queue) if item['id'] == args.id)
    if cp.get('completed_ids') != [item['id'] for item in queue[:index] if item['id'] not in excluded]:
        raise ValueError('Checkpoint prefix mismatch')
    raw = out/'raw'/f'{args.id}.json'; salary_line_review(read(raw), args.id, matches[0]['url'])
    pointer = diag/'sandbox.json'
    if pointer.exists():
        sandbox = Path(read(pointer)['path'])
        if (sandbox/'raw'/raw.name).read_bytes() != raw.read_bytes() or read(sandbox/'checkpoint.json').get('next_id') != args.id:
            raise ValueError('Sandbox differs from stopped observation')
    dest = out/'offline_reviews'/f'{args.id}.json'
    if dest.exists(): raise ValueError('Offline review already exists')
    write(dest, dict(id=args.id, raw_sha256=sha(raw), method='joined_salary_line_v1', authorization_id=args.authorization_id,
        human_instruction=events[auth[-1]]['instruction'], reviewed_at=datetime.now(timezone.utc).isoformat()))
    cp['completed_ids'].append(args.id)
    cp.update(state='READY', next_id=queue[index+1]['id'] if index+1 < len(queue) else None, stop_reason=None)
    write(diag/'checkpoint.json', cp); pointer.unlink(missing_ok=True)
    print(json.dumps({'reviewed': args.id, 'next_id': cp['next_id']}))


def validate_partition(out, service):
    plan = read(service/'plan.json')
    info = plan.get('partition')
    if not info:
        continuation_info = plan.get('continuation')
        if continuation_info:
            provenance = out/'continuation-provenance'
            if sha(provenance/'parent-plan.json') != continuation_info['parent_plan_sha256'] or sha(provenance/'parent-queue.json') != continuation_info['parent_queue_sha256']:
                raise ValueError('Continuation parent provenance changed')
            parent = read(provenance/'parent-plan.json')
            original = read(provenance/'parent-queue.json')
            pages = [read(out/p['file']) for p in plan['listing_pages']]
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
    pages = [read(out/item['file']) for item in parent['listing_pages']]
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


def partition(args):
    parent_out, parent_service, parent_diag = paths(ROOT, args.run_id)
    ensure_allowed(parent_diag, 'partition')
    if args.batch_size < 1:
        raise ValueError('Partition size must be positive')
    plan = read(parent_service/'plan.json')
    if plan.get('selection_mode') != 'first_N_unique_on_frozen_listing' or plan.get('partition') or plan.get('continuation'):
        raise ValueError('Partition supports an ordinary frozen parent queue only')
    if (parent_service/'manifest.json').exists() or (parent_service/'partition.json').exists() or any((parent_out/'raw').glob('*.json')):
        raise ValueError('Partition requires an unpartitioned, uncaptured parent')
    validate_plan_profession(plan, parent_out/'listing.json')
    queue = read(parent_out/'queue.json')
    if not queue or not plan.get('listing_pages') or queue_from_pages([read(parent_out/p['file']) for p in plan['listing_pages']], len(queue)) != queue:
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
            target = out/page['file']; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent_out/page['file'], target)
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


def capture(args):
    out, service, diag = paths(ROOT, args.run_id)
    validate_partition(out, service)
    ensure_allowed(diag, 'capture', getattr(args, 'retry_reason', None), action=True)
    if (service / 'manifest.json').exists():
        raise ValueError('Completed run is immutable')
    plan = read(service / 'plan.json')
    validate_plan_profession(plan, out / 'listing.json')
    preflight(getattr(args, 'config', None), diag)
    changed = {file for file, value in plan['automation_sha256'].items() if sha(ROOT/file) != value}
    changes = read(diag/'automation_changes.json') if (diag/'automation_changes.json').exists() else []
    if changed - {'Scraper/skills/collect-avito/scripts/run.py', 'Scraper/skills/collect-avito/SKILL.md'}:
        raise ValueError('Unapproved collector source changed; preserve old run')
    validate_automation_changes(changes, changed)
    standing_unavailable_policy(argparse.Namespace(run_id=args.run_id, authorization_id=STANDING_UNAVAILABLE_ID,
        human_instruction=STANDING_UNAVAILABLE_INSTRUCTION))
    cp, queue = read(diag / 'checkpoint.json'), read(out / 'queue.json')
    # Recover an interrupted transfer before considering another navigation.
    if (diag / 'sandbox.json').exists():
        import_sandbox(Path(read(diag / 'sandbox.json')['path']), out, diag)
    if plan.get('selection_mode') == 'first_N_remote_parameter_matches':
        decisions = remote_decisions(queue, out, plan['limit'])
        accepted = sum(d['accepted'] for d in decisions)
        if accepted == plan['limit']:
            print(json.dumps(dict(state='CAPTURED', captured=len(decisions), accepted=accepted, remaining=0)))
            return
        if len(decisions) == len(queue):
            raise CollectionStopped('Candidate limit exhausted before target; preserve run without export', 'REVIEW_REQUIRED')
        pending = queue[len(decisions):len(decisions)+args.batch_size]
        cp2, elapsed = browser(dict(mode='capture', queue=pending, completed_ids=[d['id'] for d in decisions],
            selection_queue=queue, selection_target=plan['limit'], tab_id=getattr(args, 'tab_id', None),
            previous_capture_seconds=cp.get('capture_seconds', 0)), out, diag)
        cp2['capture_seconds'] = cp.get('capture_seconds', 0) + elapsed
        write(diag / 'checkpoint.json', cp2)
        if cp2['state'] != 'CAPTURED':
            raise CollectionStopped('Stopped; inspect diagnostics', cp2['state'])
        decisions = remote_decisions(queue, out, plan['limit'])
        if cp2['completed_ids'] != [d['id'] for d in decisions]:
            raise ValueError('Checkpoint does not match the saved candidate prefix')
        accepted = sum(d['accepted'] for d in decisions)
        if accepted < plan['limit'] and len(decisions) == len(queue):
            raise CollectionStopped('Candidate limit exhausted before target; preserve run without export', 'REVIEW_REQUIRED')
        release_browser(diag)
        print(json.dumps(dict(state='CAPTURED' if accepted == plan['limit'] else 'IN_PROGRESS',
            captured=len(decisions), accepted=accepted, remaining=max(0, plan['limit']-accepted))))
        return
    unavailable = {record['id'] for record in unavailable_records(out, queue)}
    pending = []
    done = []
    for item in queue:
        if item['id'] in unavailable:
            continue
        path = out / 'raw' / (item['id'] + '.json')
        if path.exists():
            data = reviewed_capture(out, item)
            if not check(data, item['id'])['structural_check_passed'] or not data.get('parameter_audit', {}).get('rendered_complete'):
                raise ValueError(f'Saved {item["id"]} needs review; never overwrite or silently skip it')
            done.append(item['id'])
        else:
            pending.append(item)
    if not pending:
        print(json.dumps({'state': 'CAPTURED', 'captured': len(done), 'remaining': 0}))
        return
    cp2, elapsed = browser(dict(mode='capture', queue=pending[:args.batch_size], completed_ids=done, tab_id=getattr(args, 'tab_id', None),
                               previous_capture_seconds=cp.get('capture_seconds', 0)), out, diag)
    cp2['capture_seconds'] = cp.get('capture_seconds', 0) + elapsed
    write(diag / 'checkpoint.json', cp2)
    print(json.dumps(dict(cp2, captured=len(cp2['completed_ids']), remaining=len(queue)-len(unavailable)-len(cp2['completed_ids']))))
    if cp2['state'] != 'CAPTURED':
        if cp2['state'] == 'REVIEW_REQUIRED':
            for policy_path in (out/'closed_policy.json', out/'review_policy.json', out/'blocked_listing_policy.json', out/'standing_unavailable_policy.json'):
                if not policy_path.exists(): continue
                policy = read(policy_path)
                try:
                    mark_unavailable(argparse.Namespace(run_id=args.run_id, id=cp2['next_id'], authorization_id=policy['authorization_id'], automatic=True))
                except (ValueError, HumanRequired, OSError, KeyError):
                    continue
                print(json.dumps({'state': 'IN_PROGRESS', 'unavailable': cp2['next_id']}))
                return
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
    # Parameter labels may be new, but cannot replace identity or provenance.
    if set(fields) & set(row):
        raise ValueError('Parameter collides with metadata')
    row.update(fields)
    review = dict(id=data['id'], structural_check=result, parameter_audit=data['parameter_audit'],
        review_method='deterministic_rendered_template_v1', semantic_review_performed=False,
        city=city, region=region, location_evidence=location, city_source=city_source,
        city_anomaly=anomaly, analysis_status='validated')
    return row, review, region


def finalize(args):
    out, service, diag = paths(ROOT, args.run_id)
    validate_partition(out, service)
    ensure_allowed(diag, 'finalize', getattr(args, 'retry_reason', None), action=True)
    if (service / 'manifest.json').exists():
        raise ValueError('Finalized run exists; do not overwrite')
    started = time.perf_counter()
    queue, plan = read(out / 'queue.json'), read(service / 'plan.json')
    validate_plan_profession(plan, out / 'listing.json')
    plan = dict(plan, city_anomaly_policy=getattr(args,'city_anomaly_policy','blank'))
    changes = read(diag / 'automation_changes.json') if (diag / 'automation_changes.json').exists() else []
    changed = {file for file,value in plan.get('automation_sha256', {}).items() if sha(ROOT/file) != value}
    reviewed_changes = {change['file'] for change in changes}
    if changed - (reviewed_changes & {'Scraper/skills/collect-avito/scripts/run.py',
            'Scraper/skills/collect-avito/scripts/failure_log.py', 'Scraper/skills/collect-avito/SKILL.md'}):
        raise ValueError('Unregistered collector source changed after capture; use a new run')
    registry_path = ROOT / 'Data/dataset/sources.json'
    registry = read(registry_path)
    registered = [e for e in registry['exports'] if e['run_id'] == args.run_id]
    validate_automation_changes(changes, set(plan.get('uncommitted_automation_files', [])) | changed)
    regions = {}
    for entry in registry['exports']:
        for city, region in entry['placement_regions'].items():
            if city in regions and regions[city] != region:
                raise ValueError(f'Ambiguous city {city}; require disambiguated placement mapping')
            regions[city] = region
    overrides = read(args.placements) if args.placements else {}
    decisions = None
    selected_queue = queue
    if plan.get('selection_mode') == 'first_N_remote_parameter_matches':
        decisions = remote_decisions(queue, out, plan['limit'])
        if sum(d['accepted'] for d in decisions) != plan['limit']:
            raise ValueError('Target not reached; incomplete candidate selection cannot be exported')
        selected_ids = {d['id'] for d in decisions if d['accepted']}
        selected_queue = [item for item in queue if item['id'] in selected_ids]
    unavailable = unavailable_records(out, queue)
    if unavailable and decisions is not None:
        raise ValueError('Unavailable handling is not supported for remote candidate selection')
    excluded = {record['id'] for record in unavailable}
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
    with (exports / 'vacancies_flat.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    write(exports / 'vacancies.json', rows)
    if decisions is not None:
        write(exports / 'selection.json', dict(criterion={'field': 'Формат работы', 'value': 'Удалённо'},
            candidate_limit=len(queue), target=plan['limit'], decisions=decisions))
    for review in reviews:
        write(out / 'reviewed' / (review['id'] + '.json'), review)
    write(exports / 'validation.json', dict(planned=len(queue), accepted=len(rows),
        unique_ids=len({r['id'] for r in rows}), unavailable=unavailable, errors=[], review_method='deterministic_rendered_template_v1'))
    entry = dict(run_id=args.run_id, source='avito',
        path=str((exports / 'vacancies_flat.csv').relative_to(ROOT)), sha256=sha(exports / 'vacancies_flat.csv'),
        placement_regions=mapping, enrichment_status='not_run')
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
    (ROOT / 'Data/dataset/vacancies.csv').write_bytes(csv_bytes)
    (ROOT / 'Data/dataset/manifest.json').write_bytes(manifest_bytes)
    manifest = dict(run_id=args.run_id, dataset_id=args.run_id, status='complete',
        schema=dict(columns=columns, unknown_policy='empty_not_zero'), collection=plan,
        coverage=dict(planned=len(queue), accepted=len(rows), unavailable=unavailable, unique_ids=[r['id'] for r in rows], errors=[],
            limitations=['Visible rendered parameters only; enrichment not run; placement on site; live route unverified']),
        timing=dict(setup_seconds=plan['setup_seconds'], capture_seconds=read(diag/'checkpoint.json').get('capture_seconds'),
            review_export_seconds=time.perf_counter()-started, publication_seconds=None),
        review_model=args.review_model, automation_changes=changes,
        post_capture_automation_sha256={file: sha(ROOT/file) for file in sorted(changed)}, files={})
    for path in sorted(list(out.rglob('*')) + [service / 'plan.json']):
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
    if (service / 'manifest.json').exists():
        raise ValueError('Record automation fixes before finalizing')
    relative = Path(args.file)
    if relative.is_absolute() or '..' in relative.parts or not (ROOT / relative).is_file():
        raise ValueError('Change must reference a reusable file inside the repository')
    path = diag / 'automation_changes.json'
    changes = read(path) if path.exists() else []
    changes.append(dict(file=str(relative), reason=args.reason, validation=args.validation))
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


def preserve_stopped_observation(args):
    """Preserve a blocked observation before retrying exactly the checkpoint ID."""
    out, service, diag = paths(ROOT, args.run_id)
    ledger = load_failures(diag)
    if ledger['state'] in ('NEEDS_HUMAN', 'RETRY_STARTED'):
        raise HumanRequired('Resolve the current stop or active retry before preserving an observation')
    events = ledger['events']
    authorizations = [i for i, event in enumerate(events) if event.get('event') == 'human_browser_authorization' and event.get('authorization_id') == args.authorization_id]
    if not authorizations or any(event.get('state') == 'NEEDS_HUMAN' for event in events[authorizations[-1]+1:]):
        raise HumanRequired('Current browser authorization required after the stop')
    cp = read(diag/'checkpoint.json')
    identity = cp.get('next_id')
    if cp.get('state') != 'NEEDS_HUMAN' or cp.get('stop_reason') != 'access_restriction_or_captcha' or not identity or identity in cp.get('completed_ids', []):
        raise ValueError('Expected a stopped CAPTCHA observation at the pending checkpoint ID')
    queue = read(out/'queue.json')
    if len([item for item in queue if item['id'] == identity]) != 1:
        raise ValueError('Checkpoint ID must occur once in the frozen queue')
    source = out/'raw'/f'{identity}.json'
    data = read(source)
    if data.get('state') != 'NEEDS_HUMAN' or data.get('reason') != 'access_restriction_or_captcha' or data.get('source_url') != next(item['url'] for item in queue if item['id'] == identity):
        raise ValueError('Raw observation does not match the stopped queue item')
    sandbox_pointer = diag/'sandbox.json'
    if sandbox_pointer.exists():
        sandbox = Path(read(sandbox_pointer)['path'])
        other = sandbox/'raw'/source.name
        if not other.exists() or other.read_bytes() != source.read_bytes():
            raise ValueError('Sandbox raw differs; inspect interrupted transfer')
        sandbox_cp = read(sandbox/'checkpoint.json')
        if sandbox_cp.get('next_id') != identity or sandbox_cp.get('state') != 'NEEDS_HUMAN':
            raise ValueError('Sandbox checkpoint differs; inspect interrupted transfer')
    archive = out/'observations'/identity
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive/(sha(source) + '.json')
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError('Observation archive collision')
    if not destination.exists():
        shutil.copy2(source, destination)
    write(archive/(sha(source) + '.provenance.json'), dict(authorization_id=args.authorization_id,
        original_file=str(source.relative_to(out)), sha256=sha(source), observed_at=data.get('observed_at'),
        preserved_at=datetime.now(timezone.utc).isoformat(), reason=data['reason']))
    # The old sandbox remains on disk; removing only its pointer prevents reimport of
    # the already archived stopped observation into the pending raw slot.
    sandbox_pointer.unlink(missing_ok=True)
    source.unlink()
    print(json.dumps({'pending_id': identity, 'preserved': str(destination.relative_to(out))}))


def resume_browser(args):
    _, _, diag = paths(ROOT, args.run_id)
    authorize_browser(diag, args.human_instruction, args.authorization_id)
    print('Human browser authorization recorded; history retained, bounded retry budget active')


def publish(args):
    out, service, diag = paths(ROOT, args.run_id)
    validate_partition(out, service)
    ensure_allowed(diag, 'publish', getattr(args, 'retry_reason', None), action=True)
    manifest = read(service / 'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Only completed exports can be published')
    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=ROOT, text=True).strip()
    git('fetch', 'origin', 'main')
    git('merge-base', '--is-ancestor', 'origin/main', 'HEAD')
    if git('branch', '--show-current') == 'main':
        raise ValueError('Publish from a codex/<run_id> branch')
    for path, expected in manifest['files'].items():
        if sha(ROOT / path) != expected['sha256']:
            raise ValueError(f'Published artifact changed: {path}')
    expected_automation = dict(manifest['collection'].get('automation_sha256', {}))
    expected_automation.update(manifest.get('post_capture_automation_sha256', {}))
    if any(sha(ROOT/file) != value for file,value in expected_automation.items()):
        raise ValueError('Collector source changed after collection; document and resolve before publication')
    # Existing fast batch validations suffice for data-only runs; changed code needs its own tests.
    subprocess.run([sys.executable, 'utils/build_dataset.py', '--check'], check=True, cwd=ROOT)
    files = [str(out.relative_to(ROOT)), str((service/'plan.json').relative_to(ROOT)),
        str((service/'manifest.json').relative_to(ROOT)), 'Data/dataset/sources.json',
        'Data/dataset/vacancies.csv', 'Data/dataset/manifest.json', 'Data/README.md', 'Data/dataset/README.md']
    if (service/'superseded-freeze').exists():
        files.append(str((service/'superseded-freeze').relative_to(ROOT)))
    files += [change['file'] for change in manifest.get('automation_changes', [])]
    staged = git('diff', '--cached', '--name-only').splitlines()
    if any(not any(path == own or path.startswith(own + '/') for own in files) for path in staged):
        raise ValueError('Stage contains unrelated changes; inspect before publication')
    git('add', '--', *files)
    body = diag / 'pr.md'
    subprocess.run([sys.executable, 'utils/dataset_pr.py', '--run-id', args.run_id,
        '--summary', args.summary, '--output', str(body)], check=True, cwd=ROOT)
    git('diff', '--cached', '--check')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['archive-partial-listing', 'partition', 'continuation', 'freeze', 'capture', 'finalize', 'publish', 'reselect', 'standing-unavailable-policy', 'closed-policy', 'review-policy', 'blocked-listing-policy', 'salary-review-policy', 'review-salary-line', 'mark-unavailable', 'preserve-stopped-observation', 'note-change', 'record-error', 'check-stage', 'stage-ok', 'resume-offline', 'resume-browser']:
        command = commands.add_parser(name)
        command.add_argument('--run-id', required=True)
        if name in ('freeze', 'capture'):
            command.add_argument('--tab-id', help='Explicitly authorized Aside browser targetId')
            command.add_argument('--config', type=Path, help='Local config also used by setup_workspace/preflight')
        if name == 'archive-partial-listing':
            command.add_argument('--reason', required=True)
        elif name == 'partition':
            command.add_argument('--batch-size', type=int, required=True)
            command.add_argument('--prefix', required=True)
        elif name == 'continuation':
            command.add_argument('--parent-run-id', required=True)
        elif name == 'freeze':
            command.add_argument('--retry-reason')
            command.add_argument('--resume', action='store_true', help='Recover an interrupted freeze with identical inputs')
            command.add_argument('--accept-saved', action='store_true', help='Offline: accept the incomplete saved listing with explicit human instruction')
            command.add_argument('--human-instruction', help='Explicit instruction to accept exactly the saved listing')
            command.add_argument('--agent-model', default=None, help='Actual browser agent model, not a model switch')
            command.add_argument('--url', required=True, help='Observed listing URL after filter confirmation')
            size = command.add_mutually_exclusive_group(required=True)
            size.add_argument('--limit', type=int, help='Total unique vacancies; no fixed page/card ceiling')
            size.add_argument('--all-pages', action='store_true', help='Follow observed Next links until the listing ends')
            command.add_argument('--candidate-limit', type=int, help='Select first N exact remote parameter matches from this many observed candidates')
            command.add_argument('--geography', required=True)
            command.add_argument('--profession', default='', help='Optional assertion; profession is read from selected listing filters')
            command.add_argument('--profession-scope', choices=['top20', 'all'], help='Selection already applied in the UI; do not recheck collapsed controls')
            command.add_argument('--segments', default='')
            command.add_argument('--filter-evidence', required=True, help='What was visibly confirmed in the listing')
        elif name == 'capture':
            command.add_argument('--retry-reason')
            command.add_argument('--batch-size', type=int, default=30)
        elif name == 'finalize':
            command.add_argument('--city-anomaly-policy', choices=['blank','search_filter'], default='blank')
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
        elif name == 'review-salary-line':
            command.add_argument('--id', required=True)
            command.add_argument('--authorization-id', required=True)
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
            command.add_argument('--id', required=True)
            command.add_argument('--authorization-id', required=True)
        elif name == 'preserve-stopped-observation':
            command.add_argument('--authorization-id', required=True)
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
        elif name in ('resume-offline', 'resume-browser'):
            command.add_argument('--human-instruction', required=True)
            if name == 'resume-browser':
                command.add_argument('--authorization-id', required=True, help='Stable ID for this fresh human instruction; reuse on CLI retries')
        else:
            for flag in ['file', 'reason', 'validation']:
                command.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    if args.command == 'capture' and args.batch_size < 1:
        parser.error('batch-size must be positive')
    with ExitStack() as stack:
        try:
            if args.command in ('archive-partial-listing', 'partition', 'continuation', 'freeze', 'capture', 'finalize', 'publish', 'reselect', 'standing-unavailable-policy', 'closed-policy', 'review-policy', 'blocked-listing-policy', 'salary-review-policy', 'review-salary-line', 'mark-unavailable', 'preserve-stopped-observation', 'record-error', 'stage-ok', 'resume-browser', 'resume-offline'):
                lock_run_id = args.parent_run_id if args.command == 'continuation' else args.run_id
                stack.enter_context(operation_locked(paths(ROOT, lock_run_id)[2]))
            globals()[args.command.replace('-', '_')](args)
            if args.command in ('freeze', 'capture', 'finalize', 'publish'):
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
