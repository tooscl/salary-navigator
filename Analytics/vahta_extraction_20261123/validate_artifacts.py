"""Integrity checks for the delivered study, not extraction quality estimates."""
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent

def read(name):
 with (P/"results"/name).open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def main():
 r=read("vacancies_decisions.csv");ids={x["id"] for x in r};assert len(r)==len(ids)==7193
 old=[x for x in r if x["cohort"]=="baseline7189"];new=[x for x in r if x["cohort"]=="new4"];assert (len(old),len(new))==(7189,4)
 select=lambda field:{x["id"] for x in old if x[field]=="True"}
 before=select("strict_candidate_before");after=select("technical_pavel_after");text=select("text_supported_calculation_candidate");paid=select("paid_hours_supported_calculation_candidate")
 assert (len(before),len(after),len(text),len(paid))==(17,199,107,2)
 assert paid<=text<=after and len(after-before)==190 and len(before-after)==8
 assert len(select("before_under_common_text_guards"))==9
 assert not any(x["technical_pavel_after"]=="True" for x in new)
 by_id={x["id"]:x for x in r}
 for i in ["3795144835","8517572336"]:assert by_id[i]["technical_pavel_after"]=="False"
 assert by_id["3795144835"]["paid_hours_status"]=="not_stated"
 assert by_id["8517572336"]["salary_after_candidate"]==by_id["8517572336"]["salary_before"]
 for i in ["3641970493","3768203928","8400783230","8302300159","8346358436","8350194279"]:assert by_id[i]["vahta_work_status"]=="ambiguous"
 assert by_id["8347662104"]["vahta_work_duration"]=="2.5 months" and by_id["8347662104"]["rest_duration"]=="1 months"
 assert len(read("new4_separate.csv"))==4 and read("new4_separate.csv")==new
 coverage=read("kostya_coverage.csv");assert len(coverage)==27
 for feature in coverage:assert sum(int(feature[k]) for k in ["resolved","not_stated","explicit_negative","ambiguous"])==7189
 corrections=read("review_corrections.csv");assert len(corrections)==11 and all(x["evidence"] for x in corrections)
 assert paid=={"8367177511","8557342995"}
 assert len({x["source_dependence_group"] for x in old if x["id"] in paid})==1
 assert len(read("subsegments28_before_after.csv"))==len(read("passports28.csv"))==28
 features=read("kostya_features7193.csv");assert len(features)==7193 and len(features[0])==163 and {x["id"] for x in features}==ids
 for passport in read("passports28.csv"):
  for field in ["declared_card_period_distribution","candidate_period_distribution"]:assert sum(json.loads(passport[field]).values())==int(passport["ads"])
  assert passport["kostya_role_coverage"]
  if passport["kostya_candidate_schemas"]:assert passport["kostya_role_features"]
  else:assert not passport["kostya_role_features"]
 assert len(read("condition_cells.csv"))==13
 assert len(read("nlp_disagreements.csv"))==625
 gold=read("manual_snippet_evaluation36.csv");assert len(gold)==36
 assert sum(x["base_correct"]=="True" for x in gold)==26 and sum(x["nlp_correct"]=="True" for x in gold)==13
 for name in ["pavel_before_strict.csv","pavel_after_strict.csv"]:
  rows=read(name);assert {x["id"] for x in rows}==(before if "before" in name else after)
  for x in rows:
   for k in ["salary_month_gross_normalized","salary_hour_gross_normalized"]:assert float(x[k])>0 and "redacted" not in x[k]
 normalized_by_id={x["id"]:x for x in read("pavel_after_strict.csv")}
 for i in ["8287683019","8388323517","8404796280","8420819428","8479338678"]:
  assert normalized_by_id[i]["salary_shape"]=="from" and by_id[i]["salary_after_candidate"].startswith("от ")
 examples=read("discrepancy_examples8.csv");assert len(examples)==8 and all(x["id"] in ids and x["quotes"] for x in examples)
 assert all(x["independent_legal_employer"]=="not_verified" for x in r)
 print(json.dumps({"artifact_integrity":"pass","ads":len(r),"original":len(old),"new":len(new),"before":len(before),"common_guard_before":9,"after":len(after),"direct_text":len(text),"paid_hours":len(paid),"nlp_disagreements":625,"manual_fragments":36,"quality_accuracy_claim":"not_estimated"},ensure_ascii=False))
if __name__=="__main__":main()
