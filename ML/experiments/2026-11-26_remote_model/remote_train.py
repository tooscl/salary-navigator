"""Train and compare remote pay models on fixed grouped folds."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from remote_data import HERE, ROOT, META

ALPHAS=np.array([.25,.5,.75])
BROAD=("empirical","card","rules","card_char","rules_tfidf","ridge","specialists","blend")
STRICT=("empirical","card","rules","card_char","rules_tfidf","rules_calibrated","augmented_month")
CATS=("card","rules","card_char","rules_tfidf")


def context(x,y):
    q=x.copy()
    for c in META:
        q[c]=y[c].astype(object)
    return q


def fit(x,y,cols):
    model=CatBoostRegressor(loss_function="MultiQuantile:alpha=0.25,0.5,0.75",iterations=120,depth=4,
                            learning_rate=.08,l2_leaf_reg=6,border_count=24,one_hot_max_size=24,
                            thread_count=4,allow_writing_files=False,verbose=False,random_seed=42)
    model.fit(x[cols],np.log(y),cat_features=[c for c in cols if x[c].dtype==object])
    return model


def quantiles(model,x,cols):
    return np.exp(np.sort(model.predict(x[cols]),axis=1))


def grouped_base(train,test):
    levels=[["unit","tax","quote_kind","role"],["unit","quote_kind","role"],META,
            ["unit","quote_kind"],["unit"]]
    tables=[]
    for cols in levels:
        tables.append({k: np.quantile(z.y,ALPHAS) for k,z in train.groupby(cols)
                       if len(cols)==1 or (len(z)>=9 and z.group.nunique()>=4)})
    return np.array([next((t[key] for cols,t in zip(levels,tables,strict=True)
                          if (key:=tuple(row[c] for c in cols)) in t),
                          np.full(3,np.nan))
                     for row in test.to_dict("records")])


def text_fit(text):
    options=dict(min_df=3,sublinear_tf=True,dtype=np.float32)
    vectorizers=[TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=32000,**options),
                 TfidfVectorizer(ngram_range=(1,2),max_features=18000,**options)]
    return vectorizers,[v.fit_transform(text) for v in vectorizers]


def text_transform(text,v):
    return [m.transform(text) for m in v]


def dense_fit(mats,index):
    values={}
    projections=[]
    with threadpool_limits(limits=3):
        for mat,cap,name in zip(mats,[32,24],["char","word"],strict=True):
            k=min(cap,min(mat.shape)-1)
            model=TruncatedSVD(n_components=k,n_iter=5,random_state=42)
            cols=[f"svd_{name}_{j}" for j in range(k)]
            values.update(zip(cols,model.fit_transform(mat).T,strict=True))
            projections.append((model,cols))
    return pd.DataFrame(values,index=index),projections


def dense_apply(mats,index,projections):
    values={}
    for mat,(model,cols) in zip(mats,projections,strict=True):
        values.update(zip(cols,model.transform(mat).T,strict=True))
    return pd.DataFrame(values,index=index)


def dense_columns(variant,base,projection):
    suffix=[c for c in projection if c.startswith("svd_char_")] if variant=="card_char" else (
        list(projection) if variant=="rules_tfidf" else [])
    return base+META+suffix


def sparse(x,mats,cols,encoder=None):
    cat=[c for c in cols if x[c].dtype==object]
    num=[c for c in cols if c not in cat]
    if encoder is None:
        encoder=OneHotEncoder(handle_unknown="ignore",min_frequency=3,dtype=np.float32)
        encoded=encoder.fit_transform(x[cat])
    else:
        encoded=encoder.transform(x[cat])
    vals=x[num].fillna(0).to_numpy(dtype=np.float32)
    return hstack([encoded,csr_matrix(np.sign(vals)*np.log1p(np.abs(vals))),*mats],format="csr"),encoder


def ridge_setup(x,text,labels,cols,seed):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed).split(labels,groups=labels.group))
    t,c=labels.iloc[a],labels.iloc[b]
    v,mat=text_fit(text.loc[t.index])
    ax,ohe=sparse(x.loc[t.index],mat,cols)
    bx,_=sparse(x.loc[c.index],text_transform(text.loc[c.index],v),cols,ohe)
    model=Ridge(alpha=8,solver="lsqr").fit(ax,np.log(t.y))
    res=pd.Series(np.log(c.y.to_numpy())-model.predict(bx),index=c.index)
    offsets={"*":np.quantile(res,[.25,.75])}
    for unit,z in res.groupby(c.unit):
        if len(z)>=25:
            offsets[unit]=np.quantile(z,[.25,.75])
    return model,v,ohe,offsets


def ridge_predict(bundle,x,text,cols,unit):
    reg,v,ohe,offsets=bundle
    mat,_=sparse(x,text_transform(text,v),cols,ohe)
    mid=reg.predict(mat)
    band=np.vstack([offsets.get(u,offsets["*"]) for u in unit])
    return np.exp(np.column_stack([mid+np.minimum(band[:,0],0),mid,mid+np.maximum(band[:,1],0)]))


def calibrated(x,rows,cols,seed):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(rows,groups=rows.group))
    train,cal=rows.iloc[a],rows.iloc[b]
    model=fit(x.loc[train.index],train.y,cols)
    p=np.log(quantiles(model,x.loc[cal.index],cols))
    y=np.log(cal.y.to_numpy())
    err=np.maximum(p[:,0]-y,y-p[:,2])
    k=min(len(err)-1,int(np.ceil((len(err)+1)/2))-1)
    return model,max(0.,float(np.sort(err)[k]))


def widen(p,r):
    return p*np.exp([-r,0,r])


def measure(d):
    y=d.y.to_numpy()
    p=d[["p25","p50","p75"]].to_numpy()
    finite=np.isfinite(p).all(axis=1)
    possible=len(d)
    d=d.loc[finite]
    y,p=y[finite],p[finite]
    if not len(y):
        return dict(n=0,possible=possible,groups=0)
    ae=np.abs(y-p[:,1])
    ape=ae/y
    diff=np.log(y[:,None])-np.log(p)
    loss=np.maximum(diff*ALPHAS,diff*(ALPHAS-1)).mean(axis=1)
    inside=(p[:,0]<=y)&(y<=p[:,2])
    by_group=pd.DataFrame({"group":d.group.to_numpy(),"ape":ape,"loss":loss}).groupby("group").mean()
    return dict(n=len(y),possible=possible,groups=d.group.nunique(),mape=ape.mean(),median_ape=np.median(ape),
                mae=ae.mean(),coverage=inside.mean(),median_width=np.median((p[:,2]-p[:,0])/p[:,1]),
                pinball_log=loss.mean(),equal_group_mape=by_group.ape.mean(),equal_group_pinball=by_group.loss.mean(),
                within_20pct=(ape<=.2).mean())


def summaries(pred):
    records=[]
    for (cohort,variant),d in pred.groupby(["cohort","variant"],sort=False):
        for dim,v in {"all":pd.Series("all",index=d.index),"unit":d.unit,"mode":d["mode"],
                  "role":d.role,"quote_kind":d.quote_kind,"unit_mode":d.unit+"|"+d["mode"]}.items():
            for value,s in d.groupby(v,sort=False):
                records.append(dict(cohort=cohort,variant=variant,dimension=dim,key=value,**measure(s)))
    return pd.DataFrame(records)


def format_kind(d):
    return d.unit.where(d.unit.isin({"unspecified","shift","month","hour","piece","no_amount"}),"other")


def format_fit(a,b,y):
    m=SGDClassifier(loss="log_loss",class_weight="balanced",alpha=.0008,max_iter=1000,random_state=42)
    m.fit(a,y)
    return m,m.predict(b)


def evaluate(labels,features,text,card):
    x=context(features,labels)
    all_columns=list(features)
    specs={"card":card,"rules":all_columns,"card_char":card,"rules_tfidf":all_columns}
    broad=labels[labels.y.notna()].copy()
    strict=labels[labels.strict].copy()
    format_label=format_kind(labels)
    formats=labels[["id","group","fold"]].copy()
    formats["actual"]=format_label
    formats["majority"]=""
    formats["text_classifier"]=""
    rows=[]
    for fold in sorted(labels.fold.unique()):
        source_train=labels[labels.fold.ne(fold)]
        source_test=labels[labels.fold.eq(fold)]
        assert set(source_train.group).isdisjoint(source_test.group)
        enc,mats=text_fit(text.loc[source_train.index])
        test_mats=text_transform(text.loc[source_test.index],enc)
        a_dense,proj=dense_fit(mats,source_train.index)
        b_dense=dense_apply(test_mats,source_test.index,proj)
        a=x.loc[a_dense.index].join(a_dense)
        b=x.loc[b_dense.index].join(b_dense)
        for scope,d in [("broad",broad),("strict",strict)]:
            t=d[d.fold.ne(fold)]
            v=d[d.fold.eq(fold)]
            expected=BROAD if scope=="broad" else STRICT
            estimates={"empirical":grouped_base(t,v)}
            for key,base in specs.items():
                col=dense_columns(key,base,a_dense)
                m=fit(a.loc[t.index],t.y,col)
                estimates[key]=quantiles(m,b.loc[v.index],col)
            rule_cols=all_columns+META
            if scope=="broad":
                ridge=ridge_setup(x.loc[t.index],text,t,rule_cols,100+int(fold))
                estimates["ridge"]=ridge_predict(ridge,b.loc[v.index],text.loc[v.index],rule_cols,v.unit)
                estimates["blend"]=np.sqrt(estimates["rules_tfidf"]*estimates["ridge"])
                specialist=estimates["rules"].copy()
                for unit,u in t.groupby("unit"):
                    mask=v.unit.eq(unit).to_numpy()
                    if len(u)>=40 and u.group.nunique()>=12 and mask.any():
                        m=fit(a.loc[u.index],u.y,rule_cols)
                        specialist[mask]=quantiles(m,b.loc[v.loc[mask].index],rule_cols)
                estimates["specialists"]=specialist
            else:
                model,radius=calibrated(a,t,rule_cols,42+int(fold))
                estimates["rules_calibrated"]=widen(quantiles(model,b.loc[v.index],rule_cols),radius)
                weak=source_train.loc[source_train.heuristic_month].copy()
                weak["unit"]="month"
                aug_x=pd.concat([a.loc[t.index],context(a.loc[weak.index],weak)])
                aug_y=pd.concat([t.y,weak.y])
                m=fit(aug_x,aug_y,rule_cols)
                estimates["augmented_month"]=quantiles(m,b.loc[v.index],rule_cols)

            for variant in expected:
                oof=v[["id","fold","group","unit","tax","quote_kind","role","mode","source_kind","y"]].copy()
                oof[["p25","p50","p75"]]=estimates[variant]
                oof["cohort"],oof["variant"]=scope,variant
                rows.append(oof)
        ax,ohe=sparse(features.loc[a_dense.index],mats,card)
        bx,_=sparse(features.loc[b_dense.index],test_mats,card,ohe)
        _,p=format_fit(ax,bx,format_label.loc[a_dense.index])
        formats.loc[b_dense.index,"text_classifier"]=p
        formats.loc[b_dense.index,"majority"]=format_label.loc[a_dense.index].mode().iat[0]
        print(f"Fold {fold}/5: {len(source_test)} cards, {sum(source_test.strict)} strict targets",flush=True)
    pred=pd.concat(rows,ignore_index=True)
    assert not pred.duplicated(["id","cohort","variant"]).any()
    assert pred.groupby("group").fold.nunique().max()==1
    return pred,formats


def save_models(out,labels,features,text,card):
    x=context(features,labels)
    all_columns=list(features)
    rule_cols=all_columns+META
    s=labels[labels.strict]
    model,radius=calibrated(x,s,rule_cols,42)
    model.save_model(str(out/"rules_calibrated.cbm"))
    conf=dict(columns=rule_cols,log_radius=radius,cohort="strict")
    (out/"rules_calibrated.json").write_text(json.dumps(conf,ensure_ascii=False,indent=2),encoding="utf8")
    enc,mats=text_fit(text)
    dense,proj=dense_fit(mats,labels.index)
    x=x.join(dense)
    broad=labels[labels.y.notna()]
    joblib.dump((enc,proj),out/"text_encoders.joblib",compress=3)
    for key,cols in {"card":card,"rules":all_columns,"card_char":card,"rules_tfidf":all_columns}.items():
        col=dense_columns(key,cols,dense)
        model=fit(x.loc[broad.index],broad.y,col)
        model.save_model(str(out/f"broad_{key}.cbm"))
        (out/f"broad_{key}.json").write_text(json.dumps({"columns":col},ensure_ascii=False),encoding="utf8")
    ridge=ridge_setup(x.loc[broad.index],text,broad,rule_cols,101)
    joblib.dump((ridge,rule_cols),out/"ridge_bundle.joblib",compress=3)
    f,one=sparse(features,mats,card)
    model,_=format_fit(f,f,format_kind(labels))
    joblib.dump((model,one,card),out/"format_model.joblib",compress=3)
    for unit,part in broad.groupby("unit"):
        if len(part)>=40 and part.group.nunique()>=12:
            model=fit(x.loc[part.index],part.y,rule_cols)
            model.save_model(str(out/f"specialist_{unit}.cbm"))
    model=CatBoostRegressor()
    model.load_model(str(out/"rules_calibrated.cbm"))
    pd.Series(model.get_feature_importance(),index=rule_cols,name="importance").sort_values(ascending=False).to_csv(
        out/"feature_importance.csv",index_label="feature")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=ROOT/"ML/.cache/remote-20261126")
    p.add_argument("--out",type=Path,default=HERE/"results")
    args=p.parse_args()
    labels,x,text,card=pd.read_pickle(args.cache/"features.local.pkl")
    args.out.mkdir(parents=True,exist_ok=True)
    pred,formats=evaluate(labels,x,text,card)
    pred.to_csv(args.out/"predictions_oof.csv",index=False)
    table=summaries(pred)
    table.to_csv(args.out/"metrics.csv",index=False)
    formats.to_csv(args.out/"format_oof.csv",index=False)
    pd.DataFrame([{"variant":v,"n":len(formats),"accuracy":accuracy_score(formats.actual,formats[v]),
                    "balanced_accuracy":balanced_accuracy_score(formats.actual,formats[v]),
                    "macro_f1":f1_score(formats.actual,formats[v],average="macro")}
                  for v in ["majority","text_classifier"]]).to_csv(args.out/"format_metrics.csv",index=False)
    print(table.loc[table.dimension.eq("all"),["cohort","variant","n","mape","median_ape","coverage","pinball_log"]]
          .round(4).to_string(index=False),flush=True)
    save_models(args.out,labels,x,text,card)
    print("Saved predictions, metrics and inference bundles.",flush=True)


if __name__=="__main__":
    main()
