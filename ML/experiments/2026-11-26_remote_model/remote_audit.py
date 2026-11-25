"""Check the saved out-of-fold predictions and quantify uncertainty by source group."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from remote_data import HERE
from remote_train import measure


def paired_bounds(variant, baseline, draws=5000):
    base=baseline[["id","group","y","p50"]].rename(columns={"p50":"p50_base"})
    d=variant.merge(base,on=["id","group","y"],validate="one_to_one")
    d=d.dropna(subset=["p50","p50_base"])
    d["delta"]=(abs(d.p50-d.y)-abs(d.p50_base-d.y))/d.y
    g=d.groupby("group").delta.agg(["sum","count"])
    rng=np.random.default_rng(42)
    index=rng.integers(len(g),size=(draws,len(g)))
    vals=g["sum"].to_numpy()[index].sum(axis=1)/g["count"].to_numpy()[index].sum(axis=1)
    return {"n":len(d),"groups":len(g),"mape_change_pp":float(100*d.delta.mean()),
            "ci95_pp":(100*np.quantile(vals,[.025,.975])).tolist()}


def audit(out):
    labels=pd.read_csv(out/"labels.csv",dtype={"id":str})
    p=pd.read_csv(out/"predictions_oof.csv",dtype={"id":str})
    c=pd.read_csv(out/"specialists_strict_oof.csv",dtype={"id":str})
    f=pd.read_csv(out/"format_oof.csv",dtype={"id":str})
    stats=pd.read_csv(out/"metrics.csv")
    assert labels.id.is_unique and labels.groupby("group").fold.nunique().max()==1
    assert not p.duplicated(["id","cohort","variant"]).any()
    assert not c.duplicated(["id","variant"]).any()
    ids=labels.set_index("id")
    for d in (p,c,f):
        assert d.group.eq(d.id.map(ids.group)).all()
        assert d.fold.eq(d.id.map(ids.fold)).all()
    for d in (p,c):
        q=d.dropna(subset=["p50"])
        assert (q.p25.gt(0)&q.p25.le(q.p50)&q.p50.le(q.p75)).all()
    for s in stats.loc[stats.dimension.eq("all")].itertuples():
        d=p.loc[p.cohort.eq(s.cohort)&p.variant.eq(s.variant)]
        values=measure(d)
        assert values["n"]==s.n and np.isclose(values["mape"],s.mape)
        assert np.isclose(values["coverage"],s.coverage)
    assert f.id.is_unique and set(f.id)==set(labels.id)
    groups={}
    for k in ("month","shift","hour"):
        d=p.loc[p.cohort.eq("strict")&p.unit.eq(k)]
        groups[k]=paired_bounds(d.loc[d.variant.eq("rules_calibrated")],d.loc[d.variant.eq("empirical")])
    out_data={"strict_total":int(labels.strict.sum()),"full_numeric":int(labels.y.notna().sum()),
              "forecast_rows":len(p),"specialist_forecast_rows":len(c),
              "bootstrap":"paired resampling of source groups; 5000 draws; exploratory",
              "calibrated_vs_empirical":groups}
    (out/"evaluation_audit.json").write_text(json.dumps(out_data,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(out_data,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    audit(p.parse_args().out)
