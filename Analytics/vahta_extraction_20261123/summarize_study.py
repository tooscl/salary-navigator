"""Additional evidence tiers; never equate a computable formula with verified income."""
import csv,json,re,sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from utils.salary_normalization import _parse_salary
from vahta_extensions import extract
from kostya_rules import clean_description,normalize_fragment

def read(p):
 with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,r):
 with open(p,'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]),lineterminator='\n');w.writeheader();w.writerows(r)
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--base',required=True);ap.add_argument('--current',required=True);ap.add_argument('--inventory',required=True);ap.add_argument('--dictionary',required=True);args=ap.parse_args();out=Path(args.study)
 r=read(out/'vacancies_before_after.csv');raw={x['id']:x for x in read(args.base)};raw.update({x['id']:x for x in read(args.current) if x['id'] not in raw});inv={x['id']:x for x in read(args.inventory)};norm={x['id']:x for x in read(out/'pavel_after_strict.csv')}
 for x in r:
  e=extract(raw[x['id']]['description']);rate,_=_parse_salary(x['salary_after_candidate']);confirmed=[]
  if rate:
   confirmed=[z['evidence'] for z in e['rate_options']['candidates'] if z['value']['unit']==rate['unit'] and z['value']['low']==rate['low'] and not z['value']['conditional'] and not z['value']['bonus_context']]
  x['text_rate_period_link_confirmed_by_pattern']=bool(confirmed);x['rate_period_evidence']=' || '.join(confirmed)
  x['technical_pavel_after']=x['id'] in norm
  x['text_supported_calculation_candidate']=x['id'] in norm and bool(confirmed)
  x['paid_hours_supported_calculation_candidate']=x['text_supported_calculation_candidate'] and x['paid_hours_status']=='resolved'
  x['rate_requires_manual_period_check']=x['id'] in norm and not confirmed
  x['actual_site_and_source_independence_verified']='False'
  x['comparison_status']='technical_candidate_with_direct_rate_text' if x['text_supported_calculation_candidate'] else 'requires_semantic_confirmation_or_excluded'
 for x in r:
  blockers={'unknown_tax','hours_conflict','card_body_conflict','rotation_period_conflict','multiple_rate_options','hour_range_not_single','piecework_context'}
  x['before_under_common_text_guards']=x['strict_candidate_before']=='True' and not(set(x['after_reasons'].split('|')) & blockers)
 write(out/'vacancies_decisions.csv',r)
 sub=[]
 for d in read(args.dictionary):
  key=d['role'];ids={i for i,x in inv.items() if key in set(x['title_keyword_flags'].split('|')+x['profession_keyword_flags'].split('|'))};rr=[x for x in r if x['id'] in ids];before={x['id'] for x in rr if x['strict_candidate_before']=='True'};after={x['id'] for x in rr if x['technical_pavel_after']};text={x['id'] for x in rr if x['text_supported_calculation_candidate']}
  observed=lambda ids:len({re.sub(r'\s+',' ',raw[i]['employer_name_observed'].strip().casefold()) for i in ids if raw[i]['employer_name_observed'].strip()})
  sub.append({'role':key,'label':d['label_ru'],'filter':'keyword in title or original profession; overlapping groups','ads':len(rr),'technical_rates_before':len(before),'technical_rates_after':len(after),'gained':len(after-before),'lost':len(before-after),'text_supported_rates_after':len(text),'paid_hours_supported_rates_after':sum(x['paid_hours_supported_calculation_candidate'] for x in rr),'observed_names_before':observed(before),'observed_names_after':observed(after),'observed_names_text_supported_after':observed(text),'independent_legal_employers':'not_established','verified_legal_source_observations':'not_assessed','actual_profile_urls_among_candidates':len({raw[i]['employer_profile_url'] for i in text if raw[i]['employer_profile_url']}),'next_collection':'|'.join(f"{x['id']}:{x['after_reasons']}" for x in sorted(rr,key=lambda x:(not x['text_rate_recovered'],x['id']))[:5])})
 write(out/'subsegments28_before_after.csv',sub)
 # Contrastive examples from actual failures: preserve citation and no silent repair.
 notes={'7376145621':'Карточка120–179тыс/смена; описание8–11часов. Не считать диапазон одной точной длительностью.', '8430921626':'Карточка130–140тыс/смена, текст за15дней. Исключить конфликт периода.', '8482273205':'Карточкаот280тыс/смена, текст140тыс/месяц;межвахта1месяц. Не делить280тыс на10часов.', '8502628032':'Карточка121500–231000/смена, текст3500за11часовуюсмену;вахта35/45/66. Не смешивать сумму завахту и ставку.', '8515435280':'515000завахту3месяца, карточка за смену. Исключить из сменного таргета.', '8102436215':'Карточка85–120тыс/смена, в тексте нет подтверждения периода. Математический пересчёт возможен, сопоставимость не установлена.', '8175674058':'Карточкаот80000/смена, текст80000без периода и30/30. Не угадывать период из правдоподобия суммы.', '8404796280':'394584вмесяцgrossявно;оплачиваемаямежвахта42дня. Стандарт176часов не отражает доход за цикл.'}
 examples=[]
 for i,note in notes.items():
  x=next(z for z in r if z['id']==i);e=extract(raw[i]['description']);q=[]
  for name in ['rate_options','non_shift_amounts','vahta_work_duration','rest_duration','paid_hours','workplace']:q.extend(e[name]['evidence'])
  if not q:q=[raw[i]['description'][:800]]
  examples.append({'id':i,'url':x['source_url'],'card_salary':x['salary_before'],'decision':note,'after_reasons':x['after_reasons'],'text_supported_candidate':x['text_supported_calculation_candidate'],'quotes':' || '.join(dict.fromkeys(q))[:2400]})
 write(out/'discrepancy_examples8.csv',examples)
 corrections=[]
 correction_ids={'3795144835':'paid_hours','8517572336':'rate_options','3641970493':'vahta_work_duration','3768203928':'vahta_work_duration','8400783230':'vahta_work_duration','8302300159':'vahta_work_duration','8346358436':'vahta_work_duration','8350194279':'vahta_work_duration','8347662104':'vahta_work_duration','8370152087':'paid_hours','8305155700':'paid_hours'}
 for i,feature in correction_ids.items():
  e=extract(raw[i]['description']);x=next(z for z in r if z['id']==i);z=e[feature]
  evidence=z['evidence']+(e['paid_hours_rejected']['evidence'] if feature=='paid_hours' else [])
  if not evidence:evidence=[line for line in clean_description(raw[i]['description']).splitlines() if 'час' in normalize_fragment(line) and 'оплач' in normalize_fragment(line)]
  corrections.append({'id':i,'feature':feature,'corrected_value':json.dumps(z['value'],ensure_ascii=False),'corrected_status':z['status'],'technical_candidate':x['technical_pavel_after'],'direct_text_candidate':x['text_supported_calculation_candidate'],'evidence':' || '.join(evidence)[:2400],'reason':'reviewed_v1.1_optional_hours_supplement_or_duration_options;not_population_accuracy'})
 write(out/'review_corrections.csv',corrections)
 summaries=[]
 for feature in ['driver__vehicle_arrangement','picker__picking_technology','loader__max_load_kg']:
  rr=[x for x in read(out/'nlp_comparison.csv') if x['feature']==feature];app=[x for x in rr if x['in_relevant_keyword_subsegment']=='True'];summaries.append({'feature':feature,'all_ads_compared':len(rr),'all_disagreements':sum(x['same']=='False' for x in rr),'relevant_keyword_ads':len(app),'relevant_disagreements':sum(x['same']=='False' for x in app),'base_resolved_relevant':sum(x['base_status']=='resolved' for x in app),'nlp_resolved_relevant':sum(x['nlp_status']=='resolved' for x in app),'interpretation':'coverage_and_disagreement_not_accuracy'})
 write(out/'nlp_summary.csv',summaries)
 old=[x for x in r if x['cohort']=='baseline7189'];new=[x for x in r if x['cohort']=='new4'];summary={'baseline':len(old),'new':len(new),'technical_before':sum(x['strict_candidate_before']=='True' for x in old),'technical_after':sum(x['technical_pavel_after'] for x in old),'text_supported_after':sum(x['text_supported_calculation_candidate'] for x in old),'paid_hours_supported_after':sum(x['paid_hours_supported_calculation_candidate'] for x in old),'rate_period_requires_manual_check':sum(x['rate_requires_manual_period_check'] for x in old),'base_hours_resolved':sum(x['shift_hours_text_status']=='resolved' for x in old),'paid_hours_resolved':sum(x['paid_hours_status']=='resolved' for x in old),'work_duration_resolved':sum(x['vahta_work_status']=='resolved' for x in old),'rest_duration_resolved':sum(x['rest_status']=='resolved' for x in old),'workplace_anchor_resolved':sum(x['workplace_status']=='resolved' for x in old),'independent_legal_employers':'not_established','new_technical_after':sum(x['technical_pavel_after'] for x in new),'new_text_supported_after':sum(x['text_supported_calculation_candidate'] for x in new)}
 (out/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2));print(summaries)
 assert sum(len([x for x in r if x['cohort']==c]) for c in ['baseline7189','new4'])==7193
if __name__=='__main__':main()
