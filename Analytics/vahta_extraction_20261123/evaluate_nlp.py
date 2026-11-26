import sys,json,csv,collections
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from kostya_rules import extract_description,clean_description,normalize_fragment
from kostya_nlp import extract_vehicle_arrangement_nlp,extract_picking_technology_nlp,extract_manual_load_nlp
import spacy
nlp=spacy.load('ru_core_news_sm',disable=['ner']);ref=json.loads((P/'manual_snippet_reference36.json').read_text());fn={'driver__vehicle_arrangement':extract_vehicle_arrangement_nlp,'picker__picking_technology':extract_picking_technology_nlp,'loader__max_load_kg':extract_manual_load_nlp};out=[]
for x,doc in zip(ref,nlp.pipe([normalize_fragment(clean_description(x['text'])) for x in ref],batch_size=16)):
 b=extract_description(x['text'])[x['feature']];n=fn[x['feature']](doc);expected=(x['expected_value'],x['expected_status']);out.append({**x,'base_value':b['value'],'base_status':b['status'],'nlp_value':n['value'],'nlp_status':n['status'],'base_correct':(b['value'],b['status'])==expected,'nlp_correct':(n['value'],n['status'])==expected})
with (P/'results/manual_snippet_evaluation36.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
for k in fn:
 r=[x for x in out if x['feature']==k];print(k,len(r),sum(x['base_correct'] for x in r),sum(x['nlp_correct'] for x in r))
