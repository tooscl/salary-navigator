"""Reproducible study; original rules remain separate from vahta additions and NLP."""
import argparse,csv,json,re,hashlib,time,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
from utils.salary_normalization import normalize_salaries,NOTEBOOK_CONFIG,SalaryConfig,_parse_salary,_parse_hours
from kostya_rules import extract_description,clean_description,normalize_fragment,ALL_FEATURES,flatten_features
from vahta_extensions import extract as extra,VERSION as EXTENSION_VERSION,format_rate_amount
from kostya_nlp import extract_vehicle_arrangement_nlp,extract_picking_technology_nlp,extract_manual_load_nlp

def read(p):
 with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,r):
 if not r:return
 with open(p,'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]),lineterminator='\n');w.writeheader();w.writerows(r)
def compact(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',required=True);a.add_argument('--current',required=True);a.add_argument('--inventory',required=True);a.add_argument('--out',required=True);args=a.parse_args();out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
 expected=json.loads((Path(__file__).with_name('provenance.json')).read_text())['input_sha256']
 for name,path in [('base31610',args.base),('current32097',args.current),('portable_inventory7189',args.inventory)]:
  actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  if actual!=expected[name]:raise ValueError(f'Pinned input mismatch: {name}: {actual}')
 base={x['id']:x for x in read(args.base)};latest={x['id']:x for x in read(args.current)};inv=read(args.inventory);invby={x['id']:x for x in inv};oldids=set(invby)
 isva=lambda x:bool(re.search('вахт',x['График']+' '+x['title'],re.I)) and x['military'].lower() not in ('true','1','да','yes')
 newids={i for i,x in latest.items() if isva(x)}-oldids
 assert len(oldids)==7189 and len(newids)==4
 rowsraw=[{**base[i],'cohort':'baseline7189'} for i in sorted(oldids)]+[{**latest[i],'cohort':'new4'} for i in sorted(newids)]
 states=[];features={};additions={};quotes=[]
 for pos,x in enumerate(rowsraw):
  i=x['id'];f=extract_description(x['description']);e=extra(x['description']);features[i]=f;additions[i]=e
  if pos%1000==0:print('extract',pos,flush=True)
  shift=f['common__shift_hours'];hours=str(shift['value']) if shift['status']=='resolved' else ''
  orig=x['Количество рабочих часов в день'];parsed,_=_parse_hours(orig);strict_before_h=orig if re.fullmatch(r'\d+(?:\.\d+)?',orig.strip()) else ''
  hour_range=bool(re.search(r'\d+(?:[.,]\d+)?\s*[-–—]\s*\d+(?:[.,]\d+)?\s*час',normalize_fragment(clean_description(x['description']))))
  after_h=hours or strict_before_h
  if hour_range and e['paid_hours']['status']!='resolved':after_h=''
  if e['paid_hours']['status']=='resolved':after_h=str(e['paid_hours']['value'])
  hour_conflict=bool(hours and strict_before_h and float(hours)!=float(strict_before_h))
  card,_=_parse_salary(x['salary_raw']);card_tax='net' if re.search(r'на\s+руки|net',x['salary_raw'],re.I) else 'gross' if re.search(r'до\s+вычета|gross',x['salary_raw'],re.I) else 'unknown'
  rates=[z for z in e['rate_options']['candidates'] if not z['value']['conditional'] and not z['value']['bonus_context'] and z['value']['shape']!='to' and z['value']['unit'] in ('shift','hour','day','month')]
  unique={compact(z['value']):z for z in rates};chosen=next(iter(unique.values())) if len(unique)==1 else None
  after_salary=x['salary_raw'];chosen_tax=card_tax;pay_conflict=False;recovered=False
  rotation_conflict=bool(card and card['unit'] in ('shift','day','hour') and (any(z['value']['total_rub'] in (card['low'],card['high']) for z in e['rotation_totals']['candidates']) or any(z['value'] in (card['low'],card['high']) for z in e['non_shift_amounts']['candidates'])))
  if card and card['unit']!='unspecified' and rates and not any(z['value']['unit']==card['unit'] and z['value']['low']==card['low'] and (pd.isna(card['high']) or z['value']['high']==card['high']) for z in rates):pay_conflict=True
  if chosen:
   v=chosen['value'];pay_conflict=pay_conflict or bool(card and card['unit']!='unspecified' and (card['unit']!=v['unit'] or card['low']!=v['low'] or (pd.notna(card['high']) and card['high']!=v['high'])))
   if card is None or card['unit']=='unspecified':
    unit={'shift':'за смену','hour':'за час','day':'за день','month':'в месяц'}[v['unit']];amount=format_rate_amount(v);chosen_tax=v['tax'];tax='на руки' if chosen_tax=='net' else 'до вычета налогов' if chosen_tax=='gross' else ''
    after_salary=f'{amount} ₽ {unit} {tax}'.strip();recovered=True
  # Broad semantic guards intentionally only exclude; a marker never proves a model.
  n=normalize_fragment(clean_description(x['description']));piece=bool(re.search(r'сдельн|\bсделк|при выполнении\s+норм|по выработ',re.sub(r'не\s+сдельн\w*|не\s+сделка|без\s+сделки','',n)))
  reasons=[]
  for label,bad in [('unknown_tax',chosen_tax=='unknown'),('hours_conflict',hour_conflict),('card_body_conflict',pay_conflict),('rotation_period_conflict',rotation_conflict),('multiple_rate_options',len(unique)>1),('hour_range_not_single',hour_range and e['paid_hours']['status']!='resolved'),('piecework_context',piece)]:
   if bad:reasons.append(label)
  before_ok=bool(card and card['unit']!='unspecified' and card_tax!='unknown' and (card['unit'] not in ('shift','day') or strict_before_h))
  after_card,_=_parse_salary(after_salary);after_ok=bool(after_card and after_card['unit']!='unspecified' and not reasons and (after_card['unit'] not in ('shift','day') or after_h))
  if not after_h:reasons.append('no_single_shift_hours')
  if not after_card or after_card['unit']=='unspecified':reasons.append('no_linked_pay_period')
  info=invby.get(i,{})
  if not info:
   mapping={'Водитель грузового транспорта':('transport','driver'),'Кассир':('retail','retail'),'Продавец':('retail','retail')}
   family,role=mapping.get(x['profession'],('new_unclassified',''))
   info={'direction_keyword_group':family,'role_keyword_flags':role}
  states.append({'id':i,'source_url':x['source_url'],'cohort':x['cohort'],'direction':info.get('keyword_direction',info.get('direction_keyword_group','new_unclassified')),'roles':info.get('role_keyword_flags',''),'employer_observed':x['employer_name_observed'],'employer_profile_url':x['employer_profile_url'],'publication_city_NOT_workplace':x['city'],'salary_before':x['salary_raw'],'salary_after_candidate':after_salary,'text_rate_recovered':recovered,'after_tax_status':chosen_tax,'hours_card':orig,'shift_hours_text':hours,'shift_hours_text_status':shift['status'],'shift_hours_for_pavel_strict':after_h,'rotation_salary_candidates':compact(e['rotation_totals']['candidates']),'paid_hours':e['paid_hours']['value'],'paid_hours_status':e['paid_hours']['status'],'paid_hours_rejected_candidates':compact(e['paid_hours_rejected']['candidates']),'vahta_work_duration':e['vahta_work_duration']['value'],'vahta_work_status':e['vahta_work_duration']['status'],'rest_duration':e['rest_duration']['value'],'rest_status':e['rest_duration']['status'],'workplace_candidate':e['workplace']['value'],'workplace_status':e['workplace']['status'],'strict_candidate_before':before_ok,'strict_candidate_after':after_ok,'after_reasons':'|'.join(reasons),'independent_legal_employer':'not_verified','actual_income_verified':'False'})
  for k,z in {**f,**{'vahta__'+k:v for k,v in e.items() if isinstance(v,dict)}}.items():
   for q in z.get('evidence',[])[:3]:
    q=re.sub(r'(?:\+?\d[\s()\-]*){10,13}','[phone_redacted]',q);q=re.sub(r'[\w.+-]+@[\w.-]+\.[a-z]+|https?://\S+','[contact_redacted]',q,flags=re.I)
    quotes.append({'id':i,'feature':k,'status':z['status'],'value':compact(z['value']),'evidence':q})
 write(out/'kostya_features7193.csv',[{'id':i,**flatten_features(f)} for i,f in features.items()])
 write(out/'kostya_coverage.csv',[{'feature':key,**{status:sum(features[i][key]['status']==status for i in oldids) for status in ['resolved','not_stated','explicit_negative','ambiguous']}} for key in ALL_FEATURES])
 write(out/'vacancies_before_after.csv',states);write(out/'evidence.csv',quotes)
 # Apply Pavel unchanged as a documented scenario, not evidence for unknown period.
 configs={'faithful':NOTEBOOK_CONFIG,'strict':SalaryConfig(unspecified_as_month=False,split_range_by_hours=False,impute_missing_hours=False)}
 rawdf=pd.DataFrame(rowsraw);normsets={};normstats=[]
 for stage in ['before','after']:
  df=rawdf.copy()
  if stage=='after':
   df['salary_raw']=[x['salary_after_candidate'] for x in states]
   df['Количество рабочих часов в день']=[x['shift_hours_for_pavel_strict'] or x['hours_card'] for x in states]
  for cfg,config in configs.items():
   used=df.copy()
   if cfg=='strict':
    used['Количество рабочих часов в день']=[(x['shift_hours_for_pavel_strict'] if stage=='after' else (x['hours_card'] if re.fullmatch(r'\d+(?:\.\d+)?',x['hours_card'].strip()) else '')) for x in states]
    allow={x['id'] for x in states if x['strict_candidate_'+stage]};used=used[used.id.isin(allow)]
   normalized=normalize_salaries(used,config,print_stats=False,return_details=True);normsets[stage+'_'+cfg]=set(normalized['id'])
   columns=['id','cohort','salary_month_gross_normalized','salary_hour_gross_normalized','salary_unit','salary_shape','salary_tax','hours_variant','hours_variant_count']
   normalized[columns].to_csv(out/f'pavel_{stage}_{cfg}.csv',index=False,encoding='utf-8-sig')
   for cohort in ['baseline7189','new4']:
    z=normalized[normalized.cohort==cohort];normstats.append({'stage':stage,'scenario':cfg,'cohort':cohort,'ads':z.id.nunique(),'variant_rows':len(z),'interpretation':'176h_standard_equivalent_not_real_rotation_income'})
 write(out/'pavel_coverage.csv',normstats)
 groups=collections.defaultdict(list)
 for x in states:groups[(x['cohort'],x['direction'])].append(x)
 summary=[]
 for (cohort,direction),ss in groups.items():
  b=[x for x in ss if x['id'] in normsets['before_strict']];a=[x for x in ss if x['id'] in normsets['after_strict']]
  names=lambda r:len({re.sub(r'\s+',' ',x['employer_observed'].lower()).strip() for x in r if x['employer_observed'].strip()})
  summary.append({'cohort':cohort,'direction':direction,'ads':len(ss),'strict_rates_before':len(b),'strict_rates_after':len(a),'added':len({x['id'] for x in a}-{x['id'] for x in b}),'lost':len({x['id'] for x in b}-{x['id'] for x in a}),'distinct_observed_names_before':names(b),'distinct_observed_names_after':names(a),'independent_legal_employers':'not_established','shift_hours_resolved':sum(bool(x['shift_hours_text']) for x in ss),'paid_hours_resolved':sum(x['paid_hours_status']=='resolved' for x in ss),'work_duration_resolved':sum(x['vahta_work_status']=='resolved' for x in ss),'rest_resolved':sum(x['rest_status']=='resolved' for x in ss),'workplace_resolved':sum(x['workplace_status']=='resolved' for x in ss),'next_collection':'Confirm role, actual site, rate period/tax and paid hours; group dependent sources before quota'})
 write(out/'subsegments_before_after.csv',summary)
 (out/'coverage_summary.json').write_text(json.dumps({'rows':len(states),'baseline':7189,'new':4,'normalization':normstats,'base_rules':'rules-context-1.0','extension_rules':EXTENSION_VERSION},ensure_ascii=False,indent=2)+'\n')
 # Unique normalized texts parsed once; all source IDs retained.
 import spacy
 nlp=spacy.load('ru_core_news_sm',disable=['ner']);pipeline={'driver__vehicle_arrangement':extract_vehicle_arrangement_nlp,'picker__picking_technology':extract_picking_technology_nlp,'loader__max_load_kg':extract_manual_load_nlp}
 texts={};ids=collections.defaultdict(list)
 for x in rowsraw:
  t=normalize_fragment(clean_description(x['description']));h=hashlib.sha256(t.encode()).hexdigest();texts[h]=t;ids[h].append(x['id'])
 cachefile=out/'nlp_by_text.jsonl';cache={}
 if cachefile.exists():
  for line in cachefile.read_text().splitlines():
   z=json.loads(line);cache[z['text_sha256']]=z['features']
 todo=[h for h in texts if h not in cache]
 with cachefile.open('a') as sink:
  for j,(h,doc) in enumerate(zip(todo,nlp.pipe((texts[h] for h in todo),batch_size=16))):
   z={k:fn(doc) for k,fn in pipeline.items()};cache[h]=z;sink.write(compact({'text_sha256':h,'features':z})+'\n');sink.flush()
   if j%100==0:print('nlp',j,'/',len(todo),flush=True)
 comparisons=[]
 for h,ii in ids.items():
  for i in ii:
   for key,r in cache[h].items():
    original=features[i][key];same=(str(original['value']),original['status'])==(str(r['value']),r['status'])
    comparisons.append({'id':i,'feature':key,'base_value':compact(original['value']),'base_status':original['status'],'nlp_value':compact(r['value']),'nlp_status':r['status'],'same':same,'nlp_evidence':' || '.join(r['evidence']),'in_relevant_keyword_subsegment':({'driver__vehicle_arrangement':'driver','picker__picking_technology':'picking','loader__max_load_kg':'loading'}[key] in (set(invby[i]['title_keyword_flags'].split('|')+invby[i]['profession_keyword_flags'].split('|')) if i in invby else set(next(x['roles'].split('|') for x in states if x['id']==i)))),'manual_adjudication':'not_adjudicated_ad_level;separate36fragment_audit'})
 write(out/'nlp_comparison.csv',comparisons);write(out/'nlp_disagreements.csv',[x for x in comparisons if not x['same']])
 (out/'nlp_runtime.json').write_text(json.dumps({'spacy_version':spacy.__version__,'model_meta':{k:nlp.meta[k] for k in ['name','version','lang']},'unique_texts':len(texts),'ads':len(rowsraw),'comparisons':len(comparisons),'disagreements':sum(not x['same'] for x in comparisons)},ensure_ascii=False,indent=2)+'\n')
 print('DONE',len(states),'rows;',len(comparisons),'NLP comparisons',flush=True)
if __name__=='__main__':main()
