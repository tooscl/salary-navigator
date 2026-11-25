"""Compare per-unit specialists without changing the frozen test folds."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from remote_data import HERE, ROOT, META
from remote_train import calibrated, context, fit, measure, quantiles, ridge_predict, ridge_setup, widen


def save_unit_ridges(cache,out):
    labels,features,text,card=pd.read_pickle(cache/"features.local.pkl")
    x=context(features,labels)
    cols=card+META
    for unit in ("month","shift"):
        d=labels.loc[labels.strict&labels.unit.eq(unit)]
        b=ridge_setup(x.loc[d.index],text,d,cols,100)
        joblib.dump((b,cols),out/f"specialist_ridge_{unit}.joblib",compress=3)


def run(cache,out):
    labels,features,text,card=pd.read_pickle(cache/"features.local.pkl")
    x=context(features,labels)
    out.mkdir(parents=True,exist_ok=True)
    pieces=[]
    for unit in ("month","shift"):
        d=labels.loc[labels.strict&labels.unit.eq(unit)].copy()
        cols=card+META
        for fold in sorted(d.fold.unique()):
            t=d.loc[d.fold.ne(fold)]
            e=d.loc[d.fold.eq(fold)]
            assert set(t.group).isdisjoint(e.group)
            model=fit(x.loc[t.index],t.y,cols)
            pp={"unit_card":quantiles(model,x.loc[e.index],cols)}
            model,radius=calibrated(x,t,cols,42+int(fold))
            pp["unit_card_calibrated"]=widen(quantiles(model,x.loc[e.index],cols),radius)
            bundle=ridge_setup(x.loc[t.index],text,t,cols,100+int(fold))
            pp["unit_ridge_tfidf"]=ridge_predict(bundle,x.loc[e.index],text.loc[e.index],cols,e.unit)
            for variant,p in pp.items():
                q=e[["id","group","fold","unit","mode","quote_kind","role","y"]].copy()
                q[["p25","p50","p75"]]=p
                q["variant"]=variant
                pieces.append(q)
        model,radius=calibrated(x,d,cols,42)
        name=f"specialist_strict_{unit}"
        model.save_model(str(out/f"{name}.cbm"))
        (out/f"{name}.json").write_text(json.dumps({"columns":cols,"log_radius":radius,"unit":unit},
            ensure_ascii=False,indent=2),encoding="utf8")
    oof=pd.concat(pieces,ignore_index=True)
    oof.to_csv(out/"specialists_strict_oof.csv",index=False)
    metrics=[]
    for (u,v),d in oof.groupby(["unit","variant"]):
        for key,s in [("all",d),*d.groupby("quote_kind")]:
            metrics.append(dict(unit=u,variant=v,quote_kind=key,**measure(s)))
    table=pd.DataFrame(metrics)
    table.to_csv(out/"specialists_strict_metrics.csv",index=False)
    save_unit_ridges(cache,out)
    print(table.loc[table.quote_kind.eq("all"),["unit","variant","n","mape","median_ape","coverage","pinball_log"]]
          .round(4).to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=ROOT/"ML/.cache/remote-20261126")
    p.add_argument("--out",type=Path,default=HERE/"results")
    a=p.parse_args()
    run(a.cache,a.out)
