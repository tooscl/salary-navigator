"""Enrich the 28 inherited, overlapping role groups; not ML-ready clusters."""
import argparse,csv,json,collections,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from utils.salary_normalization import _parse_salary
from kostya_rules import FINAL_FEATURES

def read(p):
 with open(p,encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--results",required=True);ap.add_argument("--inventory",required=True);a=ap.parse_args();p=Path(a.results);rows=read(p/"vacancies_decisions.csv");inv=read(a.inventory);by={x["id"]:x for x in rows};out=[]
 for s in read(p/"subsegments28_before_after.csv"):
  ids={x["id"] for x in inv if s["role"] in set(x["title_keyword_flags"].split("|")+x["profession_keyword_flags"].split("|"))};rr=[by[i] for i in sorted(ids)]
  def distribution(field):return json.dumps(dict(collections.Counter(x[field] or "not_stated" for x in rr)),ensure_ascii=False,sort_keys=True)
  schema_names={"picking":["Комплектовщик"],"loading":["Грузчик"],"driver":["Водитель грузового транспорта"],"retail":["Кассир","Продавец"]}.get(s["role"],[])
  schema_features=sorted({feature for name in schema_names for feature in FINAL_FEATURES[name]})
  def periods(field):
   counts=collections.Counter()
   for x in rr:
    parsed,_=_parse_salary(x[field]);counts[parsed["unit"] if parsed else "not_parsed"]+=1
   return json.dumps(dict(counts),sort_keys=True)
  s.update({"declared_card_period_distribution":periods("salary_before"),"candidate_period_distribution":periods("salary_after_candidate"),"kostya_candidate_schemas":"|".join(schema_names),"kostya_role_features":"|".join(schema_features),"kostya_role_coverage":"schema_candidate_role_requires_confirmation" if schema_names else "no_specific_role_schema_among_five;common_and_vahta_features_only"})
  s.update({"place_candidates_distribution":distribution("workplace_candidate"),"work_duration_distribution":distribution("vahta_work_duration"),"rest_distribution":distribution("rest_duration"),"salary_period_evidence_ads":sum(x["text_rate_period_link_confirmed_by_pattern"]=="True" for x in rr),"paid_hours_resolved_ads":sum(x["paid_hours_status"]=="resolved" for x in rr),"before_common_text_guards":sum(x["before_under_common_text_guards"]=="True" for x in rr),"exclusion_reason_counts":json.dumps(dict(collections.Counter(z for x in rr for z in x["after_reasons"].split("|") if z)),sort_keys=True),"ml_use":"feature extraction and annotation queue; salary target not approved", "candidate_features":"shift_hours;paid_hours;work_duration_value_and_unit;rest_value_and_unit;workplace_text;pay_period;tax_basis;piecework_flag;source_dependence_group;Kostya role features where schema applies", "missing_for_target":"linked guaranteed rate and tax;paid hours;actual site;work/rest cycle;role purity;verified source independence", "targeted_followup_ids":"|".join(x["id"] for x in sorted(rr,key=lambda x:(x["text_supported_calculation_candidate"]!="True",x["id"]))[:5]),"followup_action":"check these source ads, link rate/tax/paid hours/site; then collect comparable independent sources for same conditions; no fixed quota inferred"})
  out.append(s)
 with (p/"passports28.csv").open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print("passports",len(out))
if __name__=="__main__":main()
