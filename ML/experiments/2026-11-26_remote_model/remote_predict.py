"""Load the calibrated CatBoost and forecast the requested native pay quote."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor

from remote_data import BENEFITS, CARD, HERE, META, make_features, normalized, role_of
from remote_train import context, grouped_base, quantiles, ridge_predict, widen


def infer(raw, unit, tax, quote_kind, folder=HERE/"results", variant="rules_calibrated"):
    if raw.empty:
        raise ValueError("Empty request")
    for required in ["title","description","Формат работы"]:
        if required not in raw:
            raise ValueError(f"Missing field: {required}")
    if not raw["Формат работы"].map(normalized).map(lambda s:"удаленно" in {p.strip() for p in s.split(",")}).all():
        raise ValueError("Use remote-only or remote mixed-format rows")
    raw=raw.copy()
    for name in [*CARD,BENEFITS]:
        if name not in raw:
            raw[name]=""
    if "id" not in raw:
        raw["id"]=range(1,len(raw)+1)
    x,text,_=make_features(raw)
    request=pd.DataFrame({"unit":unit,"tax":tax,"quote_kind":quote_kind,
                          "role":raw.title.map(role_of)},index=raw.index)
    labels=pd.read_csv(folder/"labels.csv",dtype={"id":str})
    known=labels.loc[labels.strict]
    support=known.groupby(META).size()
    count=int(support.get((unit,tax,quote_kind),0))
    if not count:
        raise ValueError("No strict training offers for the requested unit, tax and quote shape")
    if variant=="empirical":
        p=grouped_base(known,request)
    elif variant == "unit_ridge":
        if unit not in {"month","shift"}:
            raise ValueError("Strict unit specialists exist only for month and shift")
        b,cols=joblib.load(folder/f"specialist_ridge_{unit}.joblib")
        p=ridge_predict(b,context(x,request),text,cols,request.unit)
    else:
        if variant == "unit_calibrated" and unit not in {"month","shift"}:
            raise ValueError("Strict unit specialists exist only for month and shift")
        filename = f"specialist_strict_{unit}" if variant == "unit_calibrated" else "rules_calibrated"
        meta=json.loads((folder/f"{filename}.json").read_text(encoding="utf8"))
        model=CatBoostRegressor()
        model.load_model(str(folder/f"{filename}.cbm"))
        p=widen(quantiles(model,context(x,request),meta["columns"]),meta["log_radius"])
    out=raw[["id"]].join(request)
    out["references"]=count
    out["low_support"]=count<20
    out[["p25","p50","p75"]]=p
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--unit",choices=["month","shift","hour","week","piece"],required=True)
    p.add_argument("--tax",choices=["net","gross","unknown"],required=True)
    p.add_argument("--quote-kind",choices=["central","floor","ceiling"],required=True)
    p.add_argument("--variant",choices=["rules_calibrated","empirical","unit_calibrated","unit_ridge"],default="rules_calibrated")
    p.add_argument("--models",type=Path,default=HERE/"results")
    a=p.parse_args()
    infer(pd.read_csv(a.input,dtype=str,keep_default_na=False),a.unit,a.tax,a.quote_kind,a.models,a.variant).to_csv(
        a.output,index=False)


if __name__=="__main__":
    main()
