"""Build salary labels, reproducible text features and leakage-aware groups for remote offers."""
from pathlib import Path
import argparse
from collections import defaultdict
import hashlib
import html
import json
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ML.text_features.conditions import condition_features
from ML.text_features.redaction import normalize_homoglyphs, strict_redact
from utils.salary_normalization import _clean_text, _parse_salary

VERSION = "remote-rate-evidence-v1"
CARD = ["profession_resolved", "region", "График", "Занятость", "Способ оформления",
        "Количество рабочих часов в день", "Количество рабочих дней в неделю",
        "Частота выплат", "Сфера деятельности компании", "Смены", "Опыт работы",
        "Режим работы", "Тип работ", "Тип продаж", "Модель продаж", "Общение с клиентами",
        "Канал коммуникации", "Тип клиентов", "Специализация", "Грейд"]
META = ["unit", "tax", "quote_kind"]
BENEFITS = "Что получают работники"
LIMITS = {"month": (10_000, 1_000_000), "hour": (80, 10_000), "lesson": (100, 30_000),
          "shift": (300, 50_000), "day": (300, 50_000), "week": (1_000, 300_000),
          "piece": (50, 500_000), "project": (100, 1_000_000)}
ROLE_RULES = [
    ("Взыскание", r"взыскан|коллектор|просроч"),
    ("Операторы", r"колл|call|кол-центр|контакт\w*[ -]центр|оператор|поддержк|лини[июя]\b|\bчат"),
    ("Рекрутинг", r"рекрут|подбор|\bhr\b|эйчар|кадр"),
    ("Преподавание", r"репетитор|преподава|учител|педагог|наставник|куратор|тренер|логопед"),
    ("Маркетплейсы", r"маркетплейс|ozon|озон|wildberries|\bwb\b|\bвб\b|авитолог|карточ|контент"),
    ("Агенты", r"\bагент(?!ств)|турагент|риелт|риэлт|недвижим|страхов|брокер|банковск\w* продукт"),
    ("Продажи", r"продаж|продаве|привлечен|развити\w* бизнес|сделок|торгов\w* представ|b2b|работе с клиент|клиентск|аккаунт"),
    ("Маркетинг", r"маркетолог|маркетинг|smm|смм|таргет|реклам|продвиж|блогер|трафик|\bpr\b|seo"),
    ("Дизайн", r"дизайн|иллюстрат|3d|визуализ|монтаж\w* видео|видеомонтаж|моушн"),
    ("Ассистенты", r"ассистент|помощник|секретар|администратор|офис[ -]менеджер|модератор|ввод\w* данн"),
    ("Финансы", r"бухгалт|экономист|финанс|кассир-операционист"),
    ("Юристы", r"юрист|юрисконсульт|правов|арбитраж|адвокат"),
    ("Редактура", r"копирайт|редактор|корректор|автор\b|переводчик|рерайт"),
    ("Логистика", r"логист|диспетчер|закуп"),
    ("IT и инженеры", r"разработ|программ|developer|python|java|тестиров|\bqa\b|аналитик|инженер|проектировщ|конструктор|сметчик|\b1с\b|1c|сисадмин|devops|frontend|backend"),
    ("Надомная работа", r"надом|швея|сборщ|упаков|вязал|рукодел"),
]
REMOTE_FLAGS = {
    "fixed": r"(?<!без )\bоклад|фиксированн\w* (?:ставк|оплат)|гарантированн\w* доход",
    "no_fixed": r"без (?:оклада|фикса)|только (?:процент|комиссия)",
    "percent": r"%\s*от|процент\w*\s+от|от выручки|от продаж|\bkpi\b",
    "bonus": r"преми|бонус",
    "inbound": r"входящ|прием обращен|чат\w* поддержк",
    "outbound": r"исходящ|холодн|обзвон|телемаркет",
    "no_sales": r"без продаж|продавать не нужно",
    "b2b": r"\bb2b\b|юр\w*\s+лиц|корпоративн\w*\s+клиент",
    "equipment": r"компьютер|ноутбук|\bпк\b|гарнитур|наушник|микрофон|стабильн\w+\s+интернет",
    "moscow_time": r"по мск|по московскому времени",
    "paid_training": r"оплачиваем\w*\s+(?:обуч|стажир)|(?:обучени|стажировк)\w*\s+оплачива",
    "part_time": r"подработ|частичн\w*\s+занят|неполн\w*\s+(?:день|занят)",
    "exams": r"\bегэ\b|\bогэ\b",
    "language": r"английск|немецк|китайск|русск\w* язык",
    "tech_support": r"техническ\w*\s+поддержк|сервис деск",
    "marketplace": r"\bwb\b|wildberries|озон|\bozon\b|яндекс маркет",
}
PERIODS = {
    "month": r"(?:в|за)\s+мес(?:яц\w*)?\.?|/\s*мес(?:яц)?\b|ежемесячно",
    "hour": r"(?:в|за)\s+(?:1\s+)?час\b|/\s*(?:ч|час)\b",
    "lesson": r"за\s+(?:1\s+)?(?:урок|занятие)\b|/\s*урок\b",
    "shift": r"(?:в|за)\s+смену\b|/\s*смену\b",
    "day": r"за\s+(?:рабочий\s+)?день\b|/\s*день\b",
    "week": r"(?:в|за)\s+неделю\b|/\s*недел[юя]\b",
    "piece": r"за\s+(?:1\s+)?(?:звонок|заявку|лид|сделку|карточку|заказ|кандидата|анкету|встречу|выход)\b",
    "project": r"за\s+(?:1\s+)?проект\b",
}
PERIOD_RE = {k: re.compile(v) for k, v in PERIODS.items()}
DIGITS = r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d{1,7})(?:[,.]\d{1,2})?"
SCALE = r"(?:тыс\.?|[кk]\b)"
CURRENCY = r"(?:₽|руб\w*\.?|\bр\.?(?![а-я]))"
MONEY = re.compile(
    rf"(?<![\d.])(?P<bound>\bот\s+|\bдо\s+)?(?P<lo>{DIGITS})\s*(?P<scale_lo>{SCALE})?"
    rf"(?:\s*(?:-|до)\s*(?P<hi>{DIGITS})\s*(?P<scale_hi>{SCALE})?)?\s*"
    rf"(?P<currency>{CURRENCY})?(?![\d])"
)
PAY = re.compile(r"зарплат|заработ|з\s*/\s*п|оклад|вознаграж|оплат|почасов|ставк|доход")
FREQ = re.compile(r"\b(?:(?:\d+|два|три|четыре)\s+(?:раз\w*|выплат\w*)|раз|дважды|трижды)\s+(?:в|за)\s+(?:мес(?:яц)?|недел[юя])\b")
SUPPLEMENT = re.compile(r"преми|бонус|компенсац|штраф|доплат|реферал|чаев")
MONTH_EXCL = re.compile(r"партнерск\w*\s+(?:программ|предложен)|сетев\w*\s+маркетинг")
RISK = re.compile(r"(?:платн\w*\s+обучени|(?:необходим\w*|требу\w*)\s+вложени|вступительн\w*\s+взнос)")
STOP = re.compile(r"\b(?:удаленно|удаленная|работа|на дому|подработка|без опыта|требуется|в|на|с|по)\b")


def normalized(value):
    s=html.unescape(unicodedata.normalize("NFKC",str(value or "")))
    s=re.sub(r"<[^>]+>"," ",s)
    s=normalize_homoglyphs(s).lower().replace("ё","е")
    return re.sub(r"[‐‑‒–—−]","-",s)


def cleaned(value):
    return strict_redact(normalized(value))


def numeric(value):
    return float(re.sub(r"\s","",value).replace(",","."))


def tax_basis(s):
    net=bool(re.search(r"на руки|чистыми|\bnet\b",s))
    gross=bool(re.search(r"до вычета|до налог|\bgross\b",s))
    return "conflict" if net and gross else "net" if net else "gross" if gross else "unknown"


def card_quote(raw):
    s = _clean_text(raw)
    ceiling=bool(re.match(r"^до\s+",s))
    body=re.sub(r"^до\s+","",s,count=1) if ceiling else s
    original=body
    unit=None
    for key in ("lesson","piece","project"):
        if PERIOD_RE[key].search(body):
            body=PERIOD_RE[key].sub("",body).strip()
            body=re.sub(r"\s+"," ",body)
            unit=key
            break
    card,_=_parse_salary(body)
    if card is None and re.search(r"\bсдельно$",body):
        card,_=_parse_salary(re.sub(r"\s+сдельно$","",body))
        unit="piece"
    if card is None or (ceiling and card["shape"]!="exact"):
        return None
    card=dict(card, tax=tax_basis(original))
    if unit:
        card["unit"]=unit
    if ceiling:
        card.update(low=np.nan,high=card["low"],shape="to")
    return card


def description_rates(description):
    text=normalized(description)
    fragments=re.split(r"\n+|[;!?]+|(?<!\d)\.\s+|(?<=\d)\.\s+(?=[а-я])",text)
    out=[]
    for part in fragments:
        part=FREQ.sub(" ",part)
        hits=[(m.start(),m.end(),key) for key,p in PERIOD_RE.items() for m in p.finditer(part)]
        if not hits:
            continue
        for m in MONEY.finditer(part):
            lo,hi=m["lo"],m["hi"]
            scale_lo,scale_hi=m["scale_lo"],m["scale_hi"]
            if not m["currency"] and not (scale_lo or scale_hi):
                context=part[max(0,m.start()-22):m.start()]
                if not PAY.search(context) or (len(re.sub(r"\D","",lo)) < 3):
                    continue
            if re.match(r"\s*(?:%|раз\b|час\w*\s+(?:в|за)\s+(?:день|недел))", part[m.end():]):
                continue
            prefix=part[max(0,m.start()-42):m.start()]
            supp=list(SUPPLEMENT.finditer(prefix))
            main=list(PAY.finditer(prefix))
            bonus=bool(supp and (not main or supp[-1].start()>main[-1].start()))
            nearest=[]
            for a,b,u in hits:
                gap=max(a-m.end(),m.start()-b,0)
                if gap<=63:
                    nearest.append((gap,u))
            if not nearest:
                continue
            best=min(g for g,u in nearest)
            units={u for g,u in nearest if g==best}
            if len(units)!=1:
                continue
            unit=next(iter(units))
            a=numeric(lo)*(1000 if scale_lo or (scale_hi and numeric(lo)<1000) else 1)
            b=numeric(hi)*(1000 if scale_hi else 1) if hi else a
            if (not 0<a<=b<=10_000_000) or bonus:
                continue
            bound=(m["bound"] or "").strip()
            shape="range" if hi else "from" if bound=="от" else "to" if bound=="до" else "exact"
            nearby=part[max(0,m.start()-55):min(len(part),m.end()+60)]
            out.append({"low": np.nan if shape=="to" else a, "high": np.nan if shape=="from" else b,
                        "shape":shape, "unit":unit, "tax":tax_basis(nearby)})
    return out


def safe_value(a,b):
    return (pd.isna(a) and pd.isna(b)) or a==b


def same_amount(a,b):
    if a["shape"] != b["shape"]:
        return False
    return safe_value(a["low"],b["low"]) and safe_value(a["high"],b["high"])


def rate_target(raw, description):
    card=card_quote(raw)
    rates=description_rates(description)
    if card is not None:
        same=[r for r in rates if same_amount(r,card)]
        candidates={r["unit"] for r in same}
        unit=card["unit"]
        provenance="card_explicit" if unit!="unspecified" else "card_unknown_period"
        if len(candidates)>1 or (candidates and unit!="unspecified" and unit not in candidates):
            unit,provenance="conflict","card_text_conflict"
        elif unit=="unspecified" and len(candidates)==1:
            unit,provenance=next(iter(candidates)),"matched_description"
        card["unit"]=unit
        taxes={r["tax"] for r in same if r["unit"]==unit and r["tax"]!="unknown"}
        if len(taxes)>1 or (taxes and card["tax"] not in taxes|{"unknown"}):
            card["tax"]="conflict"
        elif card["tax"]=="unknown" and len(taxes)==1:
            card["tax"]=next(iter(taxes))
        if unit=="unspecified":
            # Separate exploratory heuristic. Never used for the recommended target.
            plain=normalized(description)
            month=bool(re.search(PERIODS["month"],FREQ.sub(" ",plain)) and PAY.search(plain))
            competing=any(PERIOD_RE[u].search(plain) for u in ("hour","lesson","piece","project","week","shift"))
            weak_month=month and not competing and min(v for v in (card["low"],card["high"]) if pd.notna(v))>=10000
        else:
            weak_month=False
        return card,provenance,weak_month

    if _clean_text(raw) in {"","зарплата не указана","не указана"}:
        eligible={}
        for q in rates:
            low,high=LIMITS[q["unit"]]
            vals=[v for v in (q["low"],q["high"]) if pd.notna(v)]
            if vals and min(vals)>=low and max(vals)<=high:
                key=(q["unit"],q["shape"],q["tax"],*(None if pd.isna(q[k]) else q[k] for k in ("low","high")))
                eligible[key]=q
        if len(eligible)==1:
            # Useful diagnostic, but not admitted to the recommended cohort.
            return next(iter(eligible.values())),"description_only",False
    return {"low":np.nan,"high":np.nan,"shape":"missing","unit":"no_amount","tax":"unknown"},"no_amount",False


def role_of(title):
    s = normalized(title)
    if re.search(r"оператор\s+(?:пк|ввода данных)",s):
        return "Ассистенты"
    return next((role for role,pat in ROLE_RULES if re.search(pat,s)), "Прочее")


def make_features(raw):
    title=raw["title"].map(cleaned)
    desc=raw["description"].map(cleaned)
    benefits=raw[BENEFITS].map(cleaned)
    joined=title+"\n"+desc
    x=raw[CARD].fillna("").replace("","_missing").astype(object)
    x["remote_mode"]=raw["Формат работы"].str.contains(",",regex=False).map({False:"remote_only",True:"mixed"}).astype(object)
    x["remote_role"]=title.map(role_of).astype(object)
    conditions=condition_features(title,desc,benefits)
    remote=pd.DataFrame({f"remote_{k}":joined.str.contains(v,regex=True).astype("int8")
                         for k,v in REMOTE_FLAGS.items()},index=raw.index)
    def hours(text, period):
        m=re.search(rf"(?<!\d)(\d{{1,2}})\s*час\w*\s+(?:в|за)\s+{period}",text)
        return float(m[1]) if m and 0<int(m[1])<=(24 if period=="день" else 80) else np.nan
    remote["remote_hours_day"]=joined.map(lambda t:hours(t,"день"))
    remote["remote_hours_week"]=joined.map(lambda t:hours(t,"неделю"))
    text=(title+"\n"+desc+"\n"+benefits).str.replace(r"\d+(?:[ ,\.]\d+)*"," ",regex=True)
    text=text.str.replace(r"https?://\S+|\S+@\S+\.\S+|(?:t\.me|wa\.me)/\S+"," ",regex=True)
    return pd.concat([x,conditions,remote],axis=1),text,list(x.columns)


def group_sources(raw, labels):
    parent=list(range(len(raw)))
    seen={}
    templates=defaultdict(list)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def link(a,b):
        parent[root(a)]=root(b)
    for i,row in raw.iterrows():
        name=" ".join(normalized(row.employer_name_observed).split())
        if name in {"пользователь","частное лицо","работодатель","-"}:
            name=""
        if "voxys" in name or "воксис" in name:
            name="voxys"
        profile=row.employer_profile_url.split("?")[0].rstrip("/")
        description=" ".join(normalized(row.description).split())
        key=hashlib.sha256(description.encode()).hexdigest() if len(description)>=60 else ""
        prefix=hashlib.sha256(description[:400].encode()).hexdigest() if len(description)>=400 else ""
        for kind,value in enumerate((name,profile,key,prefix)):
            if value:
                link(i,seen.setdefault((kind,value),i))
        stem=" ".join(STOP.sub(" ",re.sub(r"[^а-яa-z ]"," ",normalized(row.title))).split()[:4])
        low=labels.at[i,"low"]
        if len(stem)>=12 and pd.notna(low):
            templates[(stem,low)].append((i,name or profile or f"unknown:{i}"))
    for v in templates.values():
        if len({name for _,name in v})>=3:
            for i,_ in v[1:]:
                link(i,v[0][0])
    return [root(i) for i in range(len(raw))]


def prepare(data,cache,out):
    all_rows=pd.read_csv(data,dtype=str,keep_default_na=False)
    is_remote=all_rows["Формат работы"].map(normalized).str.split(",").map(lambda x:"удаленно" in {t.strip() for t in x})
    d=all_rows.loc[is_remote].reset_index(drop=True)
    rows=[]
    for row in d.itertuples(index=False):
        q,src,weak=rate_target(row.salary_raw,row.description)
        text=normalized(row.description)
        low=q["low"]
        partner=bool(MONTH_EXCL.search(text) or (
             pd.notna(low) and low>=300_000 and re.search(r"партнер|доход от",text)))
        if re.search(r"\bedler\b|\bэдлер\b",normalized(row.employer_name_observed)):
            partner=True
        rows.append(dict(q,source_kind=src,heuristic_month=weak,partner_risk=partner,
                         payment_risk=bool(RISK.search(text))))
    labels=pd.concat([d[["id","military"]],pd.DataFrame(rows)],axis=1)
    labels["group"]=group_sources(d,labels)
    labels["fold"]=-1
    for fold,(_,test) in enumerate(GroupKFold(n_splits=5).split(labels,groups=labels.group),1):
        labels.loc[test,"fold"]=fold
    labels["military"]=labels.military.str.casefold().isin({"true","1","yes","да"})
    labels["mode"]=np.where(d["Формат работы"].str.contains(",",regex=False),"mixed","remote_only")
    labels["role"]=d.title.map(role_of)
    labels["quote_kind"]=labels["shape"].map(
        {"range":"central","exact":"central","from":"floor","to":"ceiling","missing":"no_amount"})
    labels["y"]=np.select([labels["shape"].eq("range"),labels["shape"].eq("to")],
                           [(labels.low+labels.high)/2,labels.high],default=labels.low)
    okay=~(labels.military|labels.partner_risk|labels.payment_risk|labels.tax.eq("conflict"))
    in_bounds=pd.Series(False,index=labels.index)
    for unit,(low,high) in LIMITS.items():
        in_bounds |= labels.unit.eq(unit)&labels[["low","high"]].min(axis=1).ge(low)&labels[["low","high"]].max(axis=1).le(high)
    labels["strict"]=okay & in_bounds & labels.source_kind.isin({"card_explicit","matched_description"})
    labels["heuristic_month"]=okay & labels.heuristic_month & labels.y.between(*LIMITS["month"])
    assert labels.id.is_unique and labels.groupby("group").fold.nunique().max()==1
    out.mkdir(parents=True,exist_ok=True)
    cache.mkdir(parents=True,exist_ok=True)
    labels.to_csv(out/"labels.csv",index=False)
    meta=dict(version=VERSION,snapshot_sha256=hashlib.file_digest(data.open("rb"),"sha256").hexdigest(),
      dataset_count=len(all_rows),remote_cards=len(labels),groups=int(labels.group.nunique()),
      only=int(labels["mode"].eq("remote_only").sum()),mixed=int(labels["mode"].eq("mixed").sum()),
      military=int(labels.military.sum()),partner=int(labels.partner_risk.sum()),payment_risk=int(labels.payment_risk.sum()),
      numeric=int(labels.y.notna().sum()),strict=int(labels.strict.sum()),heuristic_month=int(labels.heuristic_month.sum()),
      origins=labels.source_kind.value_counts().to_dict(), units=labels.unit.value_counts().to_dict(),
      strict_by_unit=labels.loc[labels.strict,"unit"].value_counts().to_dict(),
      strict_by_mode=labels.loc[labels.strict,"mode"].value_counts().to_dict())
    (out/"dataset.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    x,text,card=make_features(d)
    pd.to_pickle((labels,x,text,card),cache/"features.local.pkl")
    print(f"Prepared {len(x)} feature rows; {len(x.columns)} columns before pay context",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"Data/dataset/vacancies.csv")
    p.add_argument("--cache",type=Path,default=ROOT/"ML/.cache/remote-20261126")
    p.add_argument("--out",type=Path,default=HERE/"results")
    a=p.parse_args()
    prepare(a.data,a.cache,a.out)
