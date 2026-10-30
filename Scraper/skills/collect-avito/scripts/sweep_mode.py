"""Offline orchestration for branch-by-branch Avito listing sweeps.

Browser work remains in the existing freeze/capture runner.  This module only
validates immutable frozen branches, builds a globally deduplicated queue, and
records every discovery membership used by reporting/export.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


TERMINAL_BRANCH_STATES = {'completed', 'partial', 'unavailable', 'skipped'}
ENTRY_MODES = {'city_profession', 'profession_regions', 'region_professions', 'constraint_regions'}


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_paths(root: Path, run_id: str):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', run_id):
        raise ValueError('Use simple run and sweep IDs')
    return root / 'Data/avito_runs' / run_id, root / 'Scraper/runs' / run_id


def _sweep_paths(root: Path, sweep_id: str):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', sweep_id):
        raise ValueError('Use a simple sweep ID')
    return root / 'Data/avito_sweeps' / sweep_id, root / 'Scraper/sweeps' / sweep_id


def _control_key(control):
    return tuple(str(control.get(key) or '').strip() for key in ('marker', 'value', 'label'))


def _controls(controls):
    return sorted(set(_control_key(control) for control in controls))


def _validate_spec(spec):
    if spec.get('schema_version') != 1 or spec.get('entry_mode') not in ENTRY_MODES:
        raise ValueError('Sweep spec requires schema_version=1 and a supported entry_mode')
    if not isinstance(spec.get('base'), dict) or not spec['base'].get('geography') and not spec['base'].get('profession'):
        raise ValueError('Sweep base must preserve an explicit geography or profession')
    if not isinstance(spec.get('max_pages', 30), int) or not 1 <= spec.get('max_pages', 30) <= 30:
        raise ValueError('Sweep max_pages must be between 1 and 30')
    branches = spec.get('branches')
    if not isinstance(branches, list) or not branches or branches[0].get('kind') != 'base':
        raise ValueError('Sweep requires a base branch first')
    ids = [branch.get('id') for branch in branches]
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value or '') for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('Sweep branch IDs must be unique and path-safe')
    for branch in branches:
        if branch.get('kind') not in ('base', 'route', 'additional'):
            raise ValueError('Unknown branch kind')
        if not isinstance(branch.get('expected_selected_controls'), list):
            raise ValueError('Every branch requires an exact expected_selected_controls list')
        extra = branch.get('additional_filter')
        if branch['kind'] == 'additional':
            if not isinstance(extra, dict) or not extra.get('name') or not extra.get('value'):
                raise ValueError('Additional branch requires exactly one filter and value')
        elif extra:
            raise ValueError('Only additional branches may set additional_filter')
        if branch['kind'] == 'route' and (branch.get('route_dimension') not in ('geography','profession') or not branch.get('route_value')):
            raise ValueError('Route branches require route_dimension and route_value')
        parents = branch.get('parent_context', [])
        if not isinstance(parents, list) or any(not isinstance(value, dict) for value in parents):
            raise ValueError('parent_context must be a finite list')
        if not set(_controls(parents)).issubset(set(_controls(branch['expected_selected_controls']))):
            raise ValueError('Dependent parent_context must be present in expected controls')
        interval = branch.get('numeric_range')
        if interval:
            if set(interval) - {'minimum', 'maximum', 'unit'} or not all(isinstance(interval.get(key), (int, float)) for key in ('minimum', 'maximum')):
                raise ValueError('Numeric ranges require finite minimum and maximum')
            if interval['minimum'] > interval['maximum']:
                raise ValueError('Numeric range minimum exceeds maximum')
    routing = spec.get('routing', {})
    if spec['entry_mode'] == 'profession_regions':
        if not spec['base'].get('profession') or not routing.get('regions'):
            raise ValueError('profession_regions requires a supplied region list')
    if spec['entry_mode'] == 'region_professions':
        if not spec['base'].get('geography') or not routing.get('professions'):
            raise ValueError('region_professions requires a supplied profession list')
    if spec['entry_mode'] == 'constraint_regions':
        if not spec['base'].get('constraints') or not routing.get('regions'):
            raise ValueError('constraint_regions requires fixed constraints and a supplied region list')
    if spec['entry_mode'] == 'city_profession' and not (spec['base'].get('geography') and spec['base'].get('profession')):
        raise ValueError('city_profession requires both city and profession')
    catalog = spec.get('filter_catalog', [])
    if not isinstance(catalog, list):
        raise ValueError('filter_catalog must be a list')
    branch_ids = set(ids)
    planned_branch_ids = set()
    for filter_record in catalog:
        if not filter_record.get('name') or not isinstance(filter_record.get('values'), list):
            raise ValueError('Catalog filters require names and values')
        for value in filter_record['values']:
            status = value.get('status', 'planned')
            if status == 'planned' and value.get('branch_id') not in branch_ids:
                raise ValueError('Every planned filter value must map to a branch')
            if status == 'planned': planned_branch_ids.add(value['branch_id'])
            if status in ('unavailable', 'skipped') and not value.get('reason'):
                raise ValueError('Unavailable/skipped filter values require a reason')
            if status not in ('planned', 'unavailable', 'skipped'):
                raise ValueError('Unknown catalog value status')
    if spec['entry_mode'] == 'city_profession':
        additional_ids = {branch['id'] for branch in branches if branch['kind'] == 'additional'}
        if additional_ids != planned_branch_ids:
            raise ValueError('Every additional branch must map exactly once to the filter catalog')
    route_values = {branch.get('route_value') for branch in branches if branch['kind'] == 'route'}
    if spec['entry_mode'] in ('profession_regions','constraint_regions'):
        expected = set(routing['regions']) - {spec['base'].get('geography')}
        if route_values != expected:
            raise ValueError('Region route branches must exactly match the supplied list')
    if spec['entry_mode'] == 'region_professions':
        expected = set(routing['professions']) - {spec['base'].get('profession')}
        if route_values != expected:
            raise ValueError('Profession route branches must exactly match the supplied list')
    return spec


def init_sweep(root: Path, sweep_id: str, spec_path: Path, metadata: dict):
    data, service = _sweep_paths(root, sweep_id)
    if data.exists() or service.exists():
        raise ValueError('Sweep already exists; use its checkpoint')
    spec = _validate_spec(_read(spec_path))
    plan = dict(spec, sweep_id=sweep_id, created_at=datetime.now(timezone.utc).isoformat(), **metadata)
    states = {}
    catalog_status = {}
    for branch in spec['branches']:
        states[branch['id']] = {'status': 'pending', 'reason': None, 'run_id': None}
    for record in spec.get('filter_catalog', []):
        for value in record['values']:
            if value.get('status') in ('unavailable', 'skipped'):
                catalog_status[value.get('branch_id') or f"{record['name']}::{value.get('value')}"] = {
                    'status': value['status'], 'reason': value['reason']}
    _write(service / 'plan.json', plan)
    _write(service / 'checkpoint.json', {'state': 'READY', 'branches': states, 'catalog_status': catalog_status,
        'next_branch_id': spec['branches'][0]['id'], 'materialized_run_id': None})
    _write(data / 'unique_queue.json', [])
    _write(data / 'memberships.json', {})
    _write(data / 'coverage.json', {'sweep_id': sweep_id, 'branches': [], 'unique_ids': 0, 'partial': False})
    render_report(root, sweep_id)
    return {'state': 'READY', 'sweep_id': sweep_id, 'branches': len(spec['branches']), 'next_branch_id': spec['branches'][0]['id']}


def _branch_spec(plan, branch_id):
    matches = [branch for branch in plan['branches'] if branch['id'] == branch_id]
    if len(matches) != 1:
        raise ValueError('Unknown sweep branch')
    return matches[0]


def _first_unique_observations(run_out: Path, run_plan: dict, branch: dict):
    observed, seen = [], set()
    pages = run_plan.get('listing_pages', [])
    for page_index, record in enumerate(pages, 1):
        page_path = run_out / record['file']
        if not page_path.exists() or record.get('sha256') != _sha(page_path):
            raise ValueError('Branch listing page hash differs')
        page = _read(page_path)
        if page.get('state') == 'EMPTY_RESULT':
            if page_index != 1 or page.get('items'):
                raise ValueError('Invalid empty listing evidence')
            continue
        if page.get('state') != 'LISTING':
            raise ValueError('Branch page is not a successful listing')
        for position, item in enumerate(page.get('items', []), 1):
            identity = str(item.get('id') or '')
            if not identity.isdigit() or not item.get('url'):
                raise ValueError('Branch item lacks ID or URL')
            if identity in seen:
                continue
            seen.add(identity)
            observed.append((dict(item), {
                'branch_id': branch['id'], 'branch_kind': branch['kind'],
                'filter': (branch.get('additional_filter') or {}).get('name') or branch.get('route_dimension'),
                'value': (branch.get('additional_filter') or {}).get('value') or branch.get('route_value'),
                'parent_context': branch.get('parent_context', []),
                'numeric_range': branch.get('numeric_range'),
                'page': page_index, 'position': position, 'url': item['url'],
                'observed_at': page.get('observed_at'), 'promoted': item.get('promoted')
            }))
    return observed


def _verify_reset(plan, branch, reset_path: Path | None):
    if branch['kind'] == 'base':
        return None
    if reset_path is None or not reset_path.exists():
        raise ValueError('A non-base branch requires observed reset evidence')
    reset = _read(reset_path)
    if reset.get('state') not in ('LISTING', 'EMPTY_RESULT'):
        raise ValueError('Reset evidence must be a successfully observed listing')
    base = plan['branches'][0]
    if _controls(reset.get('selected_controls', [])) != _controls(base['expected_selected_controls']):
        raise ValueError('Additional filter was not removed back to the base slice')
    return reset


def register_branch(root: Path, sweep_id: str, branch_id: str, branch_run_id: str, reset_path: Path | None):
    data, service = _sweep_paths(root, sweep_id)
    plan, checkpoint = _read(service / 'plan.json'), _read(service / 'checkpoint.json')
    _validate_spec(plan)
    branch = _branch_spec(plan, branch_id)
    state = checkpoint['branches'][branch_id]
    if state['status'] != 'pending':
        raise ValueError('Branch already has a terminal status')
    run_out, run_service = _run_paths(root, branch_run_id)
    run_plan_path, queue_path = run_service / 'plan.json', run_out / 'queue.json'
    if not run_plan_path.exists() or not queue_path.exists():
        raise ValueError('Register only a successfully frozen branch; errors are not empty results')
    run_plan, queue = _read(run_plan_path), _read(queue_path)
    if len(run_plan.get('listing_pages', [])) > plan.get('max_pages', 30):
        raise ValueError('Branch exceeded the configured page cap')
    if _controls(run_plan.get('selected_controls', [])) != _controls(branch['expected_selected_controls']):
        raise ValueError('Branch controls differ from the exact planned slice')
    expected = branch.get('expected_slice', {})
    for key in ('geography', 'profession', 'profession_scope', 'segments'):
        if key in expected and run_plan.get(key) != expected[key]:
            raise ValueError(f'Branch changed the planned {key}')
    observations = _first_unique_observations(run_out, run_plan, branch)
    observed_ids = [item['id'] for item, _ in observations]
    if [str(item['id']) for item in queue] != observed_ids:
        raise ValueError('Branch queue differs from first unique listing observations')
    reset = _verify_reset(plan, branch, reset_path)
    end_reason = run_plan.get('listing_end_reason')
    if not queue and end_reason != 'empty_result':
        raise ValueError('Zero results require explicit EMPTY_RESULT evidence')
    branch_status = 'partial' if end_reason in ('no_next_link', 'repeated_next_link') else 'completed'
    checkpoint['branches'][branch_id] = {'status': branch_status, 'reason': end_reason, 'run_id': branch_run_id,
        'plan_sha256': _sha(run_plan_path), 'queue_sha256': _sha(queue_path),
        'reset_sha256': _sha(reset_path) if reset is not None else None}
    branch_store = service / 'branches' / branch_id
    branch_store.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run_plan_path, branch_store / 'plan.json')
    shutil.copy2(queue_path, branch_store / 'queue.json')
    if reset is not None:
        shutil.copy2(reset_path, branch_store / 'reset.json')
    _write(branch_store / 'observations.json', [{'item': item, 'membership': membership} for item, membership in observations])
    _write(service / 'checkpoint.json', checkpoint)
    recompute(root, sweep_id)
    return checkpoint['branches'][branch_id]


def set_branch_status(root: Path, sweep_id: str, branch_id: str, status: str, reason: str):
    if status not in ('unavailable', 'skipped', 'error') or not reason.strip():
        raise ValueError('Use unavailable, skipped or error with a reason')
    _, service = _sweep_paths(root, sweep_id)
    plan, checkpoint = _read(service / 'plan.json'), _read(service / 'checkpoint.json')
    _branch_spec(plan, branch_id)
    if checkpoint['branches'][branch_id]['status'] != 'pending':
        raise ValueError('Branch already has a terminal status')
    checkpoint['branches'][branch_id] = {'status': status, 'reason': reason, 'run_id': None}
    _write(service / 'checkpoint.json', checkpoint)
    recompute(root, sweep_id)
    return checkpoint['branches'][branch_id]


def recompute(root: Path, sweep_id: str):
    data, service = _sweep_paths(root, sweep_id)
    plan, checkpoint = _read(service / 'plan.json'), _read(service / 'checkpoint.json')
    memberships, first_items, coverage, seen, order = {}, {}, [], set(), []
    for branch in plan['branches']:
        state = checkpoint['branches'][branch['id']]
        row = {'branch_id': branch['id'], 'kind': branch['kind'],
            'filter': (branch.get('additional_filter') or {}).get('name') or branch.get('route_dimension'),
            'value': (branch.get('additional_filter') or {}).get('value') or branch.get('route_value'),
            'parent_context': branch.get('parent_context', []), 'numeric_range': branch.get('numeric_range'),
            'status': state['status'], 'reason': state.get('reason'), 'run_id': state.get('run_id'),
            'pages': 0, 'observed_ids': 0, 'new_unique': 0, 'duplicates': 0}
        path = service / 'branches' / branch['id'] / 'observations.json'
        if path.exists():
            observations = _read(path)
            row['observed_ids'] = len(observations)
            row['pages'] = max((record['membership']['page'] for record in observations), default=0)
            for record in observations:
                identity = str(record['item']['id'])
                membership = record['membership']
                memberships.setdefault(identity, []).append(membership)
                if identity not in seen:
                    seen.add(identity); order.append(identity); first_items[identity] = record['item']; row['new_unique'] += 1
                else:
                    row['duplicates'] += 1
        coverage.append(row)
    queue = []
    for rank, identity in enumerate(order, 1):
        item = dict(first_items[identity], listing_rank=rank, sweep_memberships=memberships[identity])
        queue.append(item)
    terminal = {branch_id: state['status'] for branch_id, state in checkpoint['branches'].items()}
    pending = [branch['id'] for branch in plan['branches'] if terminal[branch['id']] == 'pending']
    checkpoint['next_branch_id'] = pending[0] if pending else None
    if any(status == 'error' for status in terminal.values()):
        checkpoint['state'] = 'PARTIAL_ERROR'
    elif pending:
        checkpoint['state'] = 'IN_PROGRESS'
    else:
        checkpoint['state'] = 'LISTINGS_COMPLETE'
    _write(service / 'checkpoint.json', checkpoint)
    _write(data / 'unique_queue.json', queue)
    _write(data / 'memberships.json', memberships)
    _write(data / 'coverage.json', {'sweep_id': sweep_id, 'branches': coverage,
        'unique_ids': len(queue), 'partial': checkpoint['state'] == 'PARTIAL_ERROR'})
    render_report(root, sweep_id)
    return queue


def render_report(root: Path, sweep_id: str):
    data, service = _sweep_paths(root, sweep_id)
    plan, checkpoint = _read(service / 'plan.json'), _read(service / 'checkpoint.json')
    coverage = _read(data / 'coverage.json')
    rows = coverage['branches']
    with (data / 'coverage.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields=['branch_id','kind','filter','value','parent_context','numeric_range','status','reason','run_id','pages','observed_ids','new_unique','duplicates']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row,parent_context=json.dumps(row.get('parent_context',[]),ensure_ascii=False,separators=(',',':')),
                numeric_range=json.dumps(row.get('numeric_range'),ensure_ascii=False,separators=(',',':')) if row.get('numeric_range') else ''))
    lines = [f'# Sweep {sweep_id}', '', f'- Режим входа: `{plan["entry_mode"]}`',
        f'- Состояние: `{checkpoint["state"]}`', f'- Уникальных ID: **{coverage["unique_ids"]}**',
        f'- Максимум страниц на ветку: **{plan.get("max_pages", 30)}**', '',
        '| Ветка | Фильтр | Значение | Контекст/интервал | Статус | Страниц | Найдено | Новых | Пересечений | Причина |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        context_value = row.get('numeric_range') or row.get('parent_context')
        context = (json.dumps(context_value, ensure_ascii=False, separators=(',', ':'))
                   if context_value else '—')
        values = [row['branch_id'], row.get('filter') or 'база', row.get('value') or '', context, row['status'],
            str(row['pages']), str(row['observed_ids']), str(row['new_unique']), str(row['duplicates']), row.get('reason') or '']
        lines.append('| ' + ' | '.join(str(value).replace('|', '\\|') for value in values) + ' |')
    (data / 'coverage.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return coverage


def materialize(root: Path, sweep_id: str, capture_run_id: str, metadata: dict, allow_partial=False):
    data, service = _sweep_paths(root, sweep_id)
    plan, checkpoint = _read(service / 'plan.json'), _read(service / 'checkpoint.json')
    states = [checkpoint['branches'][branch['id']]['status'] for branch in plan['branches']]
    if any(state == 'pending' for state in states):
        raise ValueError('Complete or explicitly classify every branch before materialization')
    if any(state == 'error' for state in states) and not allow_partial:
        raise ValueError('Error branches require --allow-partial and remain explicit in the report')
    if checkpoint.get('materialized_run_id'):
        raise ValueError('Sweep already has a materialized capture run')
    queue = recompute(root, sweep_id)
    if not queue:
        raise ValueError('Sweep has no unique vacancies to capture')
    out, run_service = _run_paths(root, capture_run_id)
    if out.exists() or run_service.exists():
        raise ValueError('Capture run already exists')
    base_state = checkpoint['branches'][plan['branches'][0]['id']]
    if not base_state.get('run_id'):
        raise ValueError('Base branch must be completed')
    base_out, base_service = _run_paths(root, base_state['run_id'])
    base_plan = _read(base_service / 'plan.json')
    out.mkdir(parents=True); (run_service / 'diagnostics').mkdir(parents=True)
    shutil.copy2(base_out / 'listing.json', out / 'listing.json')
    _write(out / 'queue.json', queue)
    provenance = out / 'sweep-provenance'; provenance.mkdir()
    for name, source in (('sweep-plan.json', service/'plan.json'), ('sweep-checkpoint.json', service/'checkpoint.json'),
                         ('coverage.json', data/'coverage.json'), ('memberships.json', data/'memberships.json')):
        shutil.copy2(source, provenance / name)
    branch_hashes = []
    for branch in plan['branches']:
        state = checkpoint['branches'][branch['id']]
        if not state.get('run_id'):
            continue
        branch_dir = service / 'branches' / branch['id']
        target = provenance / 'branches' / branch['id']; target.mkdir(parents=True)
        names = ['plan.json','queue.json','observations.json']
        if (branch_dir/'reset.json').exists(): names.append('reset.json')
        for name in names:
            shutil.copy2(branch_dir/name, target/name)
        branch_hashes.append({'branch_id': branch['id'], 'run_id': state['run_id'],
            'plan_sha256': _sha(branch_dir/'plan.json'), 'queue_sha256': _sha(branch_dir/'queue.json'),
            'observations_sha256': _sha(branch_dir/'observations.json'),
            'reset_sha256': _sha(branch_dir/'reset.json') if (branch_dir/'reset.json').exists() else None})
    sweep_info = {'sweep_id': sweep_id, 'sweep_plan_sha256': _sha(service/'plan.json'),
        'sweep_checkpoint_sha256': _sha(service/'checkpoint.json'), 'coverage_sha256': _sha(data/'coverage.json'),
        'memberships_sha256': _sha(data/'memberships.json'), 'branches': branch_hashes,
        'unique_ids': len(queue), 'partial': any(state in ('partial','error') for state in states)}
    capture_plan = dict(base_plan, run_id=capture_run_id, selection_mode='sweep_unique_queue',
        limit=len(queue), requested_limit=len(queue), available_unique=len(queue), sweep=sweep_info, **metadata)
    _write(run_service / 'plan.json', capture_plan)
    _write(run_service / 'diagnostics/checkpoint.json', {'state': 'READY', 'completed_ids': [],
        'next_id': queue[0]['id'], 'capture_seconds': 0})
    checkpoint['materialized_run_id'] = capture_run_id
    checkpoint['state'] = 'CAPTURE_READY'
    _write(service / 'checkpoint.json', checkpoint)
    validate_materialized(root, out, run_service)
    return {'state': 'READY', 'run_id': capture_run_id, 'unique_ids': len(queue), 'partial': sweep_info['partial']}


def validate_materialized(root: Path, out: Path, service: Path):
    plan = _read(service / 'plan.json')
    info = plan.get('sweep')
    if not info:
        return None
    data, sweep_service = _sweep_paths(root, info['sweep_id'])
    provenance = out / 'sweep-provenance'
    originals = [('sweep-plan.json', sweep_service/'plan.json', info['sweep_plan_sha256']),
        ('coverage.json', data/'coverage.json', info['coverage_sha256']),
        ('memberships.json', data/'memberships.json', info['memberships_sha256'])]
    for name, original, digest in originals:
        if _sha(provenance/name) != digest or _sha(original) != digest:
            raise ValueError('Sweep provenance changed')
    for branch in info['branches']:
        source = sweep_service/'branches'/branch['branch_id']
        target = provenance/'branches'/branch['branch_id']
        for name, key in (('plan.json','plan_sha256'),('queue.json','queue_sha256'),('observations.json','observations_sha256')):
            if _sha(source/name) != branch[key] or _sha(target/name) != branch[key]:
                raise ValueError('Sweep branch provenance changed')
        if branch.get('reset_sha256'):
            if _sha(source/'reset.json') != branch['reset_sha256'] or _sha(target/'reset.json') != branch['reset_sha256']:
                raise ValueError('Sweep reset provenance changed')
    expected = _read(data/'unique_queue.json')
    if _read(out/'queue.json') != expected or len(expected) != info['unique_ids']:
        raise ValueError('Sweep materialized queue changed')
    ids = [str(item['id']) for item in expected]
    if len(ids) != len(set(ids)) or any(not item.get('sweep_memberships') for item in expected):
        raise ValueError('Sweep queue requires unique IDs with memberships')
    return info


def demo_export(root: Path, sweep_id: str, source_run_id: str, limit: int | None = None):
    if limit is not None and limit < 1:
        raise ValueError('Demo limit must be positive')
    data, _ = _sweep_paths(root, sweep_id)
    source_out, _ = _run_paths(root, source_run_id)
    source = source_out / 'exports/vacancies_flat.csv'
    if not source.exists():
        raise ValueError('Demo source must be a finalized compatible CSV')
    queue = {str(item['id']): item for item in _read(data/'unique_queue.json')}
    rows = []
    with source.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            item = queue.get(str(row.get('id')))
            if not item:
                continue
            memberships = item['sweep_memberships']
            row.update(discovery_slices='; '.join(record['branch_id'] for record in memberships),
                discovery_branch_count=str(len(memberships)),
                discovery_memberships_json=json.dumps(memberships, ensure_ascii=False, separators=(',',':')))
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError('Finalized source has no IDs in the sweep')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    target = data / 'demo_vacancies.csv'
    with target.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    _write(data/'demo_validation.json', {'source_run_id': source_run_id, 'sample_limit': limit,
        'rows': len(rows),
        'unique_ids': len({row['id'] for row in rows}), 'sha256': _sha(target), 'columns': columns})
    return {'path': str(target.relative_to(root)), 'rows': len(rows), 'sha256': _sha(target)}
