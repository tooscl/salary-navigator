"""Conservative source groups for dependence control, not legal employer identities."""
import csv,re,json,hashlib,argparse
from pathlib import Path
from collections import defaultdict
from kostya_rules import clean_description,normalize_fragment

def read(p):
 with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,r):
 with open(p,'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]),lineterminator='\n');w.writeheader();w.writerows(r)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--base',required=True);ap.add_argument('--current',required=True);ap.add_argument('--inventory',required=True);args=ap.parse_args();out=Path(args.study);r=read(out/'vacancies_decisions.csv');raw={x['id']:x for x in read(args.base)};raw.update({x['id']:x for x in read(args.current) if x['id'] not in raw});inv={x['id']:x for x in read(args.inventory)};parent={x['id']:x['id'] for x in r}
 def root(i):
  if parent[i]!=i:parent[i]=root(parent[i])
  return parent[i]
 def union(i,j):parent[root(i)]=root(j)
 buckets=defaultdict(list)
 for x in r:
  i=x['id'];a=raw[i];name=re.sub(r'\s+',' ',normalize_fragment(a['employer_name_observed'])).strip();profile=a['employer_profile_url'].split('?')[0].rstrip('/')
  if name:buckets[('name',name)].append(i)
  if profile:buckets[('profile',profile)].append(i)
  text=re.sub(r'\s+',' ',normalize_fragment(clean_description(a['description']))).strip()
  if text:buckets[('exact_normalized_text',hashlib.sha256(text.encode()).hexdigest())].append(i)
 for ids in buckets.values():
  for i in ids[1:]:union(ids[0],i)
 groups=defaultdict(list)
 for i in parent:groups[root(i)].append(i)
 names={i:f'S{j:04d}' for j,ids in enumerate(sorted(groups.values(),key=lambda z:min(z)),1) for i in ids};groupmap=[]
 for x in r:
  i=x['id'];x['source_dependence_group']=names[i];x['source_group_method']='connected_observed_name_or_profile_or_exact_text;not_verified_employer'
  groupmap.append({'id':i,'source_group':names[i],'employer_observed':raw[i]['employer_name_observed'],'profile_url':raw[i]['employer_profile_url'],'group_size':len(groups[root(i)]),'meaning':'dependence_proxy_not_independence_proof'})
 write(out/'vacancies_decisions.csv',r);write(out/'source_dependence_groups.csv',groupmap)
 write(out/'new4_separate.csv',[x for x in r if x['cohort']=='new4'])
 for name in ['subsegments_before_after.csv','subsegments28_before_after.csv']:
  s=read(out/name)
  for q in s:
   if 'role' in q:ids={i for i,v in inv.items() if q['role'] in set(v['title_keyword_flags'].split('|')+v['profession_keyword_flags'].split('|'))}
   else:ids={x['id'] for x in r if x['cohort']==q['cohort'] and x['direction']==q['direction']}
   ss=[x for x in r if x['id'] in ids]
   for stage,field in [('before','strict_candidate_before'),('after','technical_pavel_after'),('text_after','text_supported_calculation_candidate')]:
    q['dependence_proxy_groups_'+stage]=len({names[x['id']] for x in ss if x[field]=='True'})
   q['independence_status']='analytical_proxy_only;not_verified_independent_employers'
  write(out/name,s)
 cells=defaultdict(list)
 for x in r:
  if x['text_supported_calculation_candidate']=='True' and x['workplace_status']=='resolved':
   key=(x['cohort'],x['roles'],x['workplace_candidate'],x['shift_hours_for_pavel_strict'],x['paid_hours'] if x['paid_hours_status']=='resolved' else 'unknown',x['salary_after_candidate'])
   cells[key].append(x)
 cellrows=[]
 for j,(key,rr) in enumerate(sorted(cells.items()),1):
  cohort,roles,site,hours,paid,rate=key
  cellrows.append({'cell':j,'cohort':cohort,'keyword_roles':roles,'workplace_text':site,'declared_shift_hours_for_formula':hours,'paid_hours':paid,'rate_original_or_recovered':rate,'ads':len(rr),'observed_names':len({x['employer_observed'] for x in rr}),'dependence_proxy_groups':len({names[x['id']] for x in rr}),'ids':'|'.join(x['id'] for x in rr),'status':'paid_hours_and_site_stated_not_verified_independence' if paid!='unknown' else 'paid_hours_missing_not_full_comparability','ml_sufficient':'not_established'})
 if cellrows:write(out/'condition_cells.csv',cellrows)
 info={'ads':len(r),'dependence_groups':len(groups),'grouping':'connected observed normalized name OR profile URL OR exact normalized description','profile_url_ads':sum(bool(raw[x['id']]['employer_profile_url']) for x in r),'meaning':'controls dependence conservatively; does not prove legal identities or independence'}
 (out/'source_group_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps(info,ensure_ascii=False))
if __name__=='__main__':main()
