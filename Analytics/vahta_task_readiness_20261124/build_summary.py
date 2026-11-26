"""Criterion-oriented vahta inventory and evidence; no model training or prediction."""
import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Analytics/vahta_extraction_20261123'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))
from kostya_rules import clean_description, normalize_fragment
from vahta_extensions import extract
from utils.salary_normalization import _parse_salary, normalize_salaries, SalaryConfig

def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

REDACTIONS = 0
def quote(text):
    """Only free-text evidence; never touch IDs, salary values or numeric columns."""
    global REDACTIONS
    result = re.sub(r'(?<!\d)(?:\+7|8)\s*\(?\d{3}\)?[ -]*\d{3}[ -]*\d{2}[ -]*\d{2}(?!\d)', '[телефон]', text)
    result = re.sub(r'[\w.+-]+@[\w.-]+\.[a-z]+|https?://\S+', '[контактная ссылка]', result, flags=re.I)
    REDACTIONS += result != text
    return result

def encode(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, default=lambda value: value.item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-csv', required=True)
    args = ap.parse_args()
    provenance = json.loads((HERE / 'provenance.json').read_text())
    selection_path = HERE / 'inputs/warehouse_selected453.csv'
    if sha(selection_path) != provenance['selection_projection_sha256']:
        raise ValueError('Pinned warehouse selection projection SHA mismatch')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for entry in source_manifest['files']:
        if sha(SOURCE / entry['path']) != entry['sha256']:
            raise ValueError('Pinned extraction source SHA mismatch: ' + entry['path'])
    review_path = HERE / 'inputs/working_day_review36.csv'
    if sha(review_path) != provenance['working_day_review_sha256']:
        raise ValueError('Pinned working-day review SHA mismatch')
    out = HERE / 'results'
    out.mkdir(parents=True, exist_ok=True)
    p = json.loads((SOURCE / 'provenance.json').read_text())
    assert sha(args.raw_csv) == p['input_sha256']['current32097']
    d = read(SOURCE / 'results/vacancies_decisions.csv')
    f = read(SOURCE / 'results/kostya_features7193.csv').set_index('id')
    raw = read(args.raw_csv).set_index('id')
    inv = read(SOURCE / 'inputs/inventory7189.csv')
    passports = read(SOURCE / 'results/passports28.csv')
    selected = read(selection_path)
    reviewed = read(review_path)
    rows = []
    extraction = {}
    for x in d.to_dict('records'):
        i = x['id']
        e = extract(raw.loc[i, 'description'])
        extraction[i] = e
        text = clean_description(raw.loc[i, 'description'])
        cycle_lines = [line for line in text.splitlines() if re.search(r'за\s+цикл\s+вахт', normalize_fragment(line))]
        working_day_pattern = r'(?:рабоч\w*\s+день|продолжительность\s+работы)\s*[:—–-]?\s*(?:по\s+)?(\d{1,2}(?:[.,]\d+)?)\s*час'
        working_day_lines = [line for line in text.splitlines() if re.search(working_day_pattern, normalize_fragment(line))]
        before, _ = _parse_salary(x['salary_before'])
        after, _ = _parse_salary(x['salary_after_candidate'])
        r = {k: x[k] for k in ['id', 'source_url', 'cohort', 'roles', 'employer_observed', 'employer_profile_url',
                               'source_dependence_group', 'publication_city_NOT_workplace', 'salary_before',
                               'salary_after_candidate', 'after_tax_status', 'after_reasons', 'hours_card']}
        r.update(independent_legal_employer='not_established',
                 pay_period_card=before['unit'] if before else 'unparsed',
                 pay_period_candidate=after['unit'] if after else 'unparsed',
                 explicit_cycle_period=bool(cycle_lines), cycle_period_quotes=quote(' || '.join(cycle_lines)),
                 card_cycle_unit_conflict=bool(cycle_lines and before and before['unit'] in {'shift', 'day', 'hour'}),
                 working_day_candidate_quotes=quote(' || '.join(working_day_lines)),
                 working_day_candidate_requires_review=bool(working_day_lines and f.loc[i, 'extracted__common__shift_hours__status'] == 'not_stated'),
                 rate_period_quotes=quote(x['rate_period_evidence']),
                 pavel_technical_before=x['strict_candidate_before'], pavel_technical_after=x['technical_pavel_after'],
                 pavel_with_direct_rate_text=x['text_supported_calculation_candidate'],
                 pavel_with_direct_text_and_paid_hours=x['paid_hours_supported_calculation_candidate'])
        for name, key in [('shift_hours', 'common__shift_hours'), ('load_kg', 'loader__max_load_kg'),
                          ('loading_equipment', 'loader__equipment'), ('picking_technology', 'picker__picking_technology'),
                          ('mixed_loading_picking', 'loader__picking')]:
            r[name] = f.loc[i, 'extracted__' + key]
            r[name + '_status'] = f.loc[i, 'extracted__' + key + '__status']
            r[name + '_quotes'] = quote(f.loc[i, 'extracted__' + key + '__evidence'])
        for name, key in [('paid_hours', 'paid_hours'), ('work_duration', 'vahta_work_duration'),
                          ('rest_duration', 'rest_duration'), ('workplace', 'workplace')]:
            z = e[key]
            r[name] = z['value'] if z['value'] is not None else ''
            r[name + '_status'] = z['status']
            r[name + '_reported_variants'] = encode([a['value'] for a in z['candidates']])
            r[name + '_quotes'] = quote(' || '.join(z['evidence']))
        # Source parity: evidence attachment must not silently change the original study.
        for target, origin in [('paid_hours', 'paid_hours'), ('work_duration', 'vahta_work_duration'),
                               ('rest_duration', 'rest_duration'), ('workplace', 'workplace_candidate')]:
            assert str(r[target]) == x[origin], (i, target, r[target], x[origin])
        rows.append(r)
    conditions = pd.DataFrame(rows)
    assert conditions.id.is_unique and len(conditions) == 7193
    assert not ((conditions.explicit_cycle_period) & (conditions.pavel_technical_after == 'True')).any()
    conditions.to_csv(out / 'vacancy_conditions7193.csv', index=False)
    conditions[conditions.working_day_candidate_requires_review].to_csv(out / 'working_day_candidates_review.csv', index=False)
    conditions[conditions.cohort == 'new4'].to_csv(out / 'new4_conditions.csv', index=False)

    summaries, collection = [], []
    for x in passports.to_dict('records'):
        role = x['role']
        ids = {r['id'] for r in inv.to_dict('records') if role in set(r['title_keyword_flags'].split('|') + r['profession_keyword_flags'].split('|'))}
        z = conditions[conditions.id.isin(ids)]
        r = {k: x[k] for k in ['role', 'label', 'filter', 'ads', 'technical_rates_before', 'technical_rates_after',
                               'text_supported_rates_after', 'paid_hours_supported_rates_after', 'observed_names_before',
                               'observed_names_after', 'observed_names_text_supported_after', 'dependence_proxy_groups_before',
                               'dependence_proxy_groups_after', 'dependence_proxy_groups_text_after', 'kostya_candidate_schemas',
                               'kostya_role_features', 'kostya_role_coverage']}
        r['independent_legal_employers'] = 'not_established'
        for name in ['shift_hours', 'paid_hours', 'work_duration', 'rest_duration', 'workplace', 'load_kg']:
            r[name + '_resolved_ads'] = int((z[name + '_status'] == 'resolved').sum())
            r[name + '_status_distribution'] = encode(dict(z[name + '_status'].value_counts()))
        for name in ['pay_period_card', 'pay_period_candidate', 'work_duration', 'rest_duration', 'workplace']:
            r[name + '_distribution'] = encode(dict(z[name].replace('', 'not_detected').value_counts()))
        r['explicit_cycle_ads'] = int(z.explicit_cycle_period.sum())
        r['actual_workplace_verified'] = False
        r['candidate_conditions'] = x['candidate_features']
        r['missing_parameters'] = 'period_and_amount_link;tax_basis;paid_hours;work_and_rest_units;actual_site;source_identity;role_duties'
        summaries.append(r)
        near = z.assign(n_gaps=z.after_reasons.str.count(r'\|') + z.after_reasons.ne('').astype(int)).sort_values(['n_gaps', 'id'])
        gaps = {name: int((z[name + '_status'] != 'resolved').sum()) for name in ['shift_hours', 'paid_hours', 'work_duration', 'rest_duration', 'workplace']}
        collection.append({'role': role, 'label': x['label'], 'ads': len(z),
                           'pavel_candidates': x['technical_rates_after'], 'gaps_by_field': encode(gaps),
                           'ids_to_check_first': '|'.join(near.id.head(5)),
                           'parameters_to_collect': r['missing_parameters'],
                           'source_check': 'link profile/company/site; do not count observed names as independent legal employers',
                           'volume_quota': 'not_assigned_without_confirmed_observation_unit'})
    pd.DataFrame(summaries).to_csv(out / 'subsegments28_criteria.csv', index=False)
    pd.DataFrame(collection).to_csv(out / 'addressed_collection28.csv', index=False)

    prepared = []
    for role in ['loading', 'packing', 'picking', 'warehouse_pool']:
        ss = selected if role == 'warehouse_pool' else selected[selected.segment_role == role]
        z = conditions[conditions.id.isin(ss.id)]
        prep = {'segment': role, 'ads': len(z), 'closed_or_exact': int((ss.closed_label == 'True').sum()),
                'open_lower_bound': int((ss.label_shape == 'from').sum()),
                'observed_names': z.employer_observed.replace('', pd.NA).dropna().str.casefold().nunique(),
                'source_proxy_groups_prepared': ss.split_group.nunique(),
                'source_proxy_groups_closed': ss[ss.closed_label == 'True'].split_group.nunique(),
                'independent_legal_employers': 'not_established',
                'pavel_candidates_in_prepared_selection': int((z.pavel_technical_after == 'True').sum()),
                'body_linked_rates': int((ss.body_rate_linked == 'True').sum()),
                'pay_period': 'declared_shift', 'tax_basis': 'explicit_net',
                'scope_assumptions': 'card_period_unless_contradicted;pure_keyword_role;1000_15000_RUB;closed_width_le_50_percent',
                'relationship': 'parent_includes_children' if role == 'warehouse_pool' else 'disjoint_child_role'}
        for name in ['shift_hours', 'paid_hours', 'work_duration', 'rest_duration', 'workplace', 'load_kg']:
            prep[name + '_resolved_ads'] = int((z[name + '_status'] == 'resolved').sum())
        prepared.append(prep)
    pd.DataFrame(prepared).to_csv(out / 'prepared_segments4.csv', index=False)

    case_patterns = {
        '8367177511': ('Смена, оплачиваемые часы, нагрузка и объект', ['оплата', 'оплачиваемых часов', '35 кг', 'логистическая']),
        '4450631921': ('Смена карточки против цикла в тексте', ['за цикл вахты']),
        '8034544281': ('Сумма за цикл и фактический объект', ['325 000', 'место работы']),
        '8482273205': ('Разные единицы работы/отдыха и конфликт периода', ['60 смен', 'рабочий день', '140 000']),
        '8347662104': ('Работа в месяцах и отдых отдельно', ['2,5 месяца']),
        '3795144835': ('Необязательная подработка не обычная смена', ['дневная смена', 'продолжительность смены', '18 часов оплачивается', 'по желанию']),
        '8517572336': ('Доплата не основная ставка', ['доплата за заправку']),
        '3641970493': ('Список вариантов не последнее число', ['20/33/45/90']),
        '8124352013': ('Рабочий день и неоплачиваемая межвахта', ['рабочий день', 'фактическое место работы', 'межвахта не оплачивается']),
        '8404796280': ('Месячная сумма и оплачиваемый отдых не доход цикла', ['394', 'отдых'])}
    examples = []
    for i, (topic, pats) in case_patterns.items():
        text = clean_description(raw.loc[i, 'description'])
        qs = []
        for line in text.splitlines():
            n = normalize_fragment(line)
            if any(pat in n for pat in pats):
                qs.append(quote(line))
        assert qs, i
        z = conditions[conditions.id == i].iloc[0]
        examples.append({'id': i, 'topic': topic, 'source_url': z.source_url,
                         'salary_card': z.salary_before, 'quotes': ' || '.join(qs),
                         'pavel_candidate': z.pavel_technical_after, 'exclusion_or_limits': z.after_reasons,
                         'source_group': z.source_dependence_group})
    pd.DataFrame(examples).to_csv(out / 'examples10_by_criteria.csv', index=False)
    # Separate sensitivity: explicit working-day hours stand in for shift hours.
    # Never infer paid hours, erase tax/period conflicts, or change the original comparison.
    sensitivity = conditions.merge(reviewed, on='id', validate='one_to_one')
    sensitivity = sensitivity[(sensitivity.pavel_technical_after == 'False') &
                              (sensitivity.after_reasons == 'no_single_shift_hours') &
                              (sensitivity.review_status == 'explicit_text_hours')].copy()
    trial = raw.loc[sensitivity.id].reset_index().copy()
    mapping = sensitivity.set_index('id')
    trial['salary_raw'] = trial.id.map(mapping.salary_after_candidate)
    trial['Количество рабочих часов в день'] = trial.id.map(mapping.reviewed_text_working_day_hours)
    trial_result = normalize_salaries(trial, SalaryConfig(unspecified_as_month=False, split_range_by_hours=False, impute_missing_hours=False), print_stats=False, return_details=True)
    trial_result = trial_result[['id','salary_month_gross_normalized','salary_hour_gross_normalized','salary_unit','salary_shape','salary_tax']].merge(
        sensitivity[['id','reviewed_text_working_day_hours','evidence_quote']], on='id', validate='one_to_one')
    trial_result['assumption'] = 'explicit_working_day_hours_as_shift_hours_not_paid_hours'
    trial_result.to_csv(out / 'pavel_working_day_sensitivity.csv', index=False)

    old = conditions[conditions.cohort == 'baseline7189']
    def n(field):
        return int((old[field] == 'resolved').sum())
    stats = {'base_rows': 7189, 'new_rows': 4, 'pavel_before': int((old.pavel_technical_before == 'True').sum()),
             'pavel_after': int((old.pavel_technical_after == 'True').sum()),
             'pavel_with_direct_text': int((old.pavel_with_direct_rate_text == 'True').sum()),
             'pavel_with_paid_hours': int(((old.pavel_technical_after == 'True') & (old.paid_hours_status == 'resolved')).sum()),
             'pavel_direct_text_and_paid_hours': int((old.pavel_with_direct_text_and_paid_hours == 'True').sum()),
             'extra_working_day_candidates_require_review': int(old.working_day_candidate_requires_review.sum()), 'shift_hours': n('shift_hours_status'), 'paid_hours': n('paid_hours_status'),
             'work_duration': n('work_duration_status'), 'rest_duration': n('rest_duration_status'),
             'workplace_text': n('workplace_status'), 'explicit_cycle_ads': int(old.explicit_cycle_period.sum()),
             'work_duration_units': dict(old[old.work_duration_status == 'resolved'].work_duration.str.split().str[-1].value_counts()),
             'rest_duration_units': dict(old[old.rest_duration_status == 'resolved'].rest_duration.str.split().str[-1].value_counts()),
             'independent_legal_employers': 'not_established', 'redacted_evidence_cells': REDACTIONS,
             'additional_candidates_with_working_day_assumption': len(trial_result), 'scope': 'segment preparation and extraction/readiness; no ML task or model execution'}
    (out / 'summary.json').write_text(encode(stats) + '\n')
    print(encode(stats))

if __name__ == '__main__':
    main()
