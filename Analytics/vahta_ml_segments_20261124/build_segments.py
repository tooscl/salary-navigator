"""Pilot ML datasets for declared net pay per shift; never infer a pay period."""
import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Analytics/vahta_extraction_20261123'))
from utils.salary_normalization import _parse_salary
from kostya_rules import clean_description, normalize_fragment

SOURCE = ROOT / 'Analytics/vahta_extraction_20261123'
HERE = Path(__file__).resolve().parent
ROLES = {'loading', 'packing', 'picking', 'warehouse_general', 'sorting', 'warehouse_equipment'}
BLOCKERS = {'card_body_conflict', 'rotation_period_conflict', 'multiple_rate_options', 'piecework_context'}
ROLE_NAMES = {'loading': 'Погрузка', 'packing': 'Упаковка', 'picking': 'Комплектация', 'warehouse_pool': 'Склад: общая модель с признаком роли'}
BASE_FEATURES = [
    'common__schedule_patterns', 'common__shift_hours', 'common__time_of_day',
    'common__meals_support', 'loader__work_environment', 'loader__equipment',
    'loader__max_load_kg', 'loader__picking', 'picker__picking_technology',
    'picker__work_environment', 'picker__product_categories', 'picker__packing',
]

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)

def label_decision(row, military=False, upper_cap=15000):
    parsed, _ = _parse_salary(row['salary_after_candidate'])
    if military:
        return None, 'military_or_title_review_flag'
    if not parsed or parsed['unit'] != 'shift':
        return None, 'not_explicit_shift_rate'
    if row['after_tax_status'] != 'net' or parsed['tax'] != 'net':
        return None, 'not_explicit_net'
    if set(row['after_reasons'].split('|')) & BLOCKERS:
        return None, 'text_conflict_alternatives_or_piecework'
    if parsed['shape'] not in {'exact', 'range', 'from'}:
        return None, 'not_observed_floor'
    lo = parsed['low']
    hi = parsed['high'] if math.isfinite(parsed['high']) else None
    if lo < 1000 or lo > upper_cap or (hi is not None and hi > upper_cap):
        return None, 'outside_declared_pilot_scope_1000_cap'
    if hi is not None and hi / lo > 1.5:
        return None, 'closed_range_wider_than_50_percent'
    # The lower endpoint is the observed target, including for an open offer.
    return {'y_floor_net_shift_rub': lo, 'y_upper_net_shift_rub': hi,
            'y_midpoint_closed_net_shift_rub': (lo + hi) / 2 if hi is not None else None,
            'label_shape': parsed['shape'], 'closed_label': hi is not None}, 'eligible'

def source_groups(decisions, raw):
    """Union original dependence proxies, digit-masked exact templates and precise sites."""
    ids = sorted(decisions.id)
    parent = {i: i for i in ids}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[root(a)] = root(b)
    buckets = collections.defaultdict(list)
    for x in decisions.to_dict('records'):
        i = x['id']
        buckets[('original', x['source_dependence_group'])].append(i)
        text = normalize_fragment(clean_description(raw.loc[i, 'description']))
        text = re.sub(r'https?://\S+|[\w.+-]+@[\w.-]+\.[a-z]+', ' CONTACT ', text)
        text = re.sub(r'\d+(?:[.,]\d+)*', ' NUM ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            buckets[('masked_template', hashlib.sha256(text.encode()).hexdigest())].append(i)
        site = x['workplace_candidate']
        if x['workplace_status'] == 'resolved' and re.search(r'\d', site) and len(site) >= 20:
            buckets[('precise_site_candidate', site)].append(i)
    for ii in buckets.values():
        for i in ii[1:]:
            union(ii[0], i)
    groups = collections.defaultdict(list)
    for i in ids:
        groups[root(i)].append(i)
    return {i: 'G' + hashlib.sha256('|'.join(sorted(ii)).encode()).hexdigest()[:16]
            for ii in groups.values() for i in ii}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-csv', required=True)
    ap.add_argument('--out', default=str(HERE / 'results'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((SOURCE / 'provenance.json').read_text())
    assert digest(args.raw_csv) == provenance['input_sha256']['current32097'], 'Pinned raw CSV mismatch'
    original_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for entry in original_manifest['files']:
        assert digest(SOURCE / entry['path']) == entry['sha256'], entry['path']
    decisions = read(SOURCE / 'results/vacancies_decisions.csv')
    features = read(SOURCE / 'results/kostya_features7193.csv').set_index('id')
    inventory = read(SOURCE / 'inputs/inventory7189.csv').set_index('id')
    raw = read(args.raw_csv).set_index('id')
    groups = source_groups(decisions, raw)
    records, audit, sensitivities = [], [], []
    for row in decisions.to_dict('records'):
        i = row['id']
        military = raw.loc[i, 'military'].lower() in {'true', '1', 'да'} or (i in inventory.index and inventory.loc[i, 'military_title_review_flag'] == 'True')
        label, reason = label_decision(row, military)
        pure_role = row['roles'] in ROLES
        audit.append({'id': i, 'role_flags': row['roles'], 'label_eligibility': reason,
                      'warehouse_pure_role': pure_role, 'included': bool(label and pure_role)})
        for cap in [12000, 15000, 20000]:
            lab, _ = label_decision(row, military, cap)
            if lab and pure_role:
                sensitivities.append({'cap_rub': cap, 'role': row['roles'], 'id': i, 'split_group': groups[i], 'closed_label': lab['closed_label']})
        if label is None or not pure_role:
            continue
        x = {'id': i, 'cohort': row['cohort'], 'segment_role': row['roles'],
             'observed_at': raw.loc[i, 'observed_at'],
             'source_dependence_group': row['source_dependence_group'], 'split_group': groups[i],
             'body_rate_linked': row['text_rate_period_link_confirmed_by_pattern'] == 'True',
             'label_source': 'description_recovered' if row['text_rate_recovered'] == 'True' else 'card',
             'assumption_period_from_card': row['text_rate_period_link_confirmed_by_pattern'] != 'True',
             'salary_source_text': row['salary_after_candidate'], **label,
             'x__role': row['roles'], 'x__publication_city_NOT_workplace': row['publication_city_NOT_workplace'],
             'x__hours_card': row['hours_card'] or 'not_stated',
             'x__work_duration': row['vahta_work_duration'] or 'not_stated',
             'x__work_duration_status': row['vahta_work_status'],
             'x__rest_duration': row['rest_duration'] or 'not_stated',
             'x__rest_status': row['rest_status'],
             'x__paid_hours': row['paid_hours'] or 'not_stated',
             'x__paid_hours_status': row['paid_hours_status']}
        for key in BASE_FEATURES:
            value = features.loc[i, 'extracted__' + key]
            status = features.loc[i, 'extracted__' + key + '__status']
            x['x__' + key] = value if status in {'resolved', 'explicit_negative'} and value else 'not_stated'
            x['x__' + key + '__status'] = status
        records.append(x)
    df = pd.DataFrame(records).sort_values('id').reset_index(drop=True)
    df['cv_fold'] = -1
    for fold, (_, test) in enumerate(GroupKFold(n_splits=3).split(df, groups=df.split_group)):
        df.loc[test, 'cv_fold'] = fold
    df['source_weight_pool'] = 1 / df.groupby('split_group').id.transform('size')
    df['source_weight_role'] = 1 / df.groupby(['segment_role', 'split_group']).id.transform('size')
    assert df.id.is_unique and df.groupby('split_group').cv_fold.nunique().max() == 1
    registry = []
    for role in ['loading', 'packing', 'picking', 'warehouse_pool']:
        z = (df if role == 'warehouse_pool' else df[df.segment_role == role]).copy()
        z['training_weight'] = 1 / z.groupby('split_group').id.transform('size')
        z.to_csv(out / (role + '.csv'), index=False)
        closed = z[z.closed_label].copy()
        closed['training_weight'] = 1 / closed.groupby('split_group').id.transform('size')
        closed.to_csv(out / (role + '_point.csv'), index=False)
        if role == 'warehouse_pool':
            anchor = closed[closed.body_rate_linked].copy()
            anchor['training_weight'] = 1 / anchor.groupby('split_group').id.transform('size')
            anchor.to_csv(out / 'warehouse_text_linked_point.csv', index=False)
        fold_counts = z.groupby('cv_fold').size().to_dict()
        groups_n = z.split_group.nunique()
        ready = len(z) >= 60 and groups_n >= 20 and len(fold_counts) == 3 and min(fold_counts.values()) >= 10
        point_folds = closed.groupby('cv_fold').size().to_dict()
        point_ready = len(closed) >= 60 and closed.split_group.nunique() >= 20 and len(point_folds) == 3 and min(point_folds.values()) >= 10
        registry.append({'segment': role, 'label_ru': ROLE_NAMES[role], 'rows': len(z),
                         'split_groups': groups_n, 'original_source_proxy_groups': z.source_dependence_group.nunique(),
                         'closed_rows': len(closed), 'from_rows': int((z.label_shape == 'from').sum()),
                         'closed_split_groups': closed.split_group.nunique(), 'point_fold_rows': json.dumps(point_folds),
                         'body_linked_rows': int(z.body_rate_linked.sum()),
                         'fold_rows': json.dumps(fold_counts),
                         'median_floor_rub': float(z.y_floor_net_shift_rub.median()),
                         'unique_floor_values': z.y_floor_net_shift_rub.nunique(),
                         'pilot_ready': ready, 'point_pilot_ready': point_ready, 'production_validated': False,
                         'target': 'declared_net_shift_floor',
                         'relationship': 'parent_view_do_not_sum_with_children' if role == 'warehouse_pool' else 'mutually_exclusive_pure_keyword_role',
                         'assumptions': 'card_period_if_not_contradicted;title_role;net_explicit;1000_15000_scope;range_width_le_1.5;source_proxies'})
    pd.DataFrame(registry).to_csv(out / 'segment_registry.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'all7193_selection_audit.csv', index=False)
    ss = pd.DataFrame(sensitivities)
    sensitivity = ss.groupby(['cap_rub', 'role']).agg(rows=('id', 'size'), split_groups=('split_group', 'nunique'), closed_rows=('closed_label', 'sum')).reset_index()
    sensitivity.to_csv(out / 'scope_sensitivity.csv', index=False)
    allowlist = [x for x in df.columns if x.startswith('x__')]
    (out / 'feature_allowlist.json').write_text(json.dumps(allowlist, ensure_ascii=False, indent=2) + '\n')
    metadata = {'input_commit': '34cff91b3f9a3af6ec78cfc159cdd013e64b8546',
                'raw_sha256': digest(args.raw_csv), 'raw_rows': len(raw), 'cohort_rows': len(decisions),
                'selected_rows': len(df), 'folds': 3, 'split': 'original_source_proxy OR digit_masked_exact_template OR precise_text_site;global_before_filter',
                'label': 'observed advertised lower endpoint in RUB net per shift; not realized income or estimated market median',
                'midpoint': 'only closed exact/range; missing for open lower bounds',
                'operational_pilot_bar': '>=60 ads,>=20 conservative groups,>=10 held-out rows in each of3 folds; not universal statistical sufficiency',
                'feature_allowlist': allowlist,
                'selection_counts': dict(collections.Counter(x['label_eligibility'] for x in audit)),
                'source_input_sha256': {name: digest(SOURCE / 'results' / name) for name in ['vacancies_decisions.csv', 'kostya_features7193.csv']}}
    (out / 'dataset_contract.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(pd.DataFrame(registry).to_string(index=False))

if __name__ == '__main__':
    main()
