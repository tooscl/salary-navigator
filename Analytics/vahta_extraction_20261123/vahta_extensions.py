"""Vahta additions v1: conservative textual candidates, separate from Kostya.
No shift/rest/address fact is inferred from a publication city or clock span.
"""
import re
from kostya_rules import clean_description,normalize_fragment
VERSION='vahta-additions-1.1'
def format_rate_amount(value):
    if value['low']!=value['high']:return f"{value['low']:g} — {value['high']:g}"
    prefix={'from':'от ','to':'до '}.get(value['shape'],'')
    return prefix+f"{value['low']:g}"
def result(candidates):
    unique=[]
    for c in candidates:
        if c not in unique:unique.append(c)
    values={str(c['value']) for c in unique}
    return {'value':unique[0]['value'] if len(values)==1 else None,'status':'resolved' if len(values)==1 else 'ambiguous' if unique else 'not_stated','candidates':unique,'evidence':list(dict.fromkeys(c['evidence'] for c in unique))}
def extract(description):
    clean=clean_description(description);lines=[x.strip() for x in clean.splitlines() if x.strip()];n=normalize_fragment(clean)
    paid=[];paid_rejected=[];work=[];rest=[];site=[];rates=[];rotation=[];period_conflicts=[]
    # Work/rest pairs require explicit words; slash options remain alternatives.
    for line_index,raw in enumerate(lines):
        t=normalize_fragment(raw)
        for pat in [r'(\d{1,2}(?:[.,]\d+)?)\s*(?:оплачиваем\w*\s+час\w*|час\w*\s+(?:к\s+оплате|оплачива(?:ются|ется)\b))',r'(?:к\s+оплате|оплачиваем\w*)\s*[:—-]?\s*(\d{1,2}(?:[.,]\d+)?)\s*час\w*']:
            for m in re.finditer(pat,t):
                v=float(m[1].replace(',','.'))
                context=normalize_fragment(lines[max(0,line_index-1)])+' '+t
                optional=bool(re.search(r'переработ|подработ|по\s+желанию|сверхуроч',context))
                break_only=bool(re.match(r'\s*(?:оплачиваем\w*\s+)?(?:обед|перерыв)',t[m.end():]))
                if 0<v<=24:
                    candidate={'value':v,'evidence':raw,'rule':'paid_hours.optional_or_break_rejected' if optional or break_only else 'paid_hours.explicit'}
                    (paid_rejected if optional or break_only else paid).append(candidate)
        if re.search(r'вахт|рабоч\w*\s+смен',t):
            for m in re.finditer(r'(?<![\d/.,])(\d{1,3}(?:\s*(?:[/–—,;.-]|\bили\b|\bдо\b)\s*\d{1,3})*)\s*(?:рабоч\w*\s+)?(дн\w*|смен\w*|сут\w*|месяц\w*)',t):
                if re.match(r'\s*(?:на\s+)?отдых',t[m.end():]) or re.search(r'(?:межвахт\w*|отдых\w*)\s*[:—-]?\s*$',t[:m.start()]):continue
                decimal_month=m[2].startswith('месяц') and bool(re.fullmatch(r'\d{1,3}[.,]\d{1,2}',m[1]))
                options=[m[1].replace(',','.')] if decimal_month else re.split(r'\s*(?:[/–—,;.-]|\bили\b|\bдо\b)\s*',m[1])
                for option in options:
                    v=float(option)
                    if 1<=v<=180:work.append({'value':f'{v:g} '+('work_shifts' if m[2].startswith('смен') else 'months' if m[2].startswith('месяц') else 'days_unspecified_basis'),'evidence':raw,'rule':'vahta.options_or_pair_unassigned' if len(options)>1 else 'vahta.explicit_duration'})
        for pat in [r'(?<![\d.,])(\d{1,3})\s*(дн\w*|сут\w*|недел\w*|месяц\w*)\s+(?:на\s+)?отдых\w*',r'отдых\w*\s*[:—-]?\s*(\d{1,3})\s*(дн\w*|сут\w*|недел\w*|месяц\w*)',r'межвахт\w*\s*[:—-]?\s*(\d{1,3})\s*(дн\w*|сут\w*|недел\w*|месяц\w*)']:
            for m in re.finditer(pat,t):
                if 0<int(m[1])<=180:rest.append({'value':f'{m[1]} '+('weeks' if m[2].startswith('недел') else 'months' if m[2].startswith('месяц') else 'days'),'evidence':raw,'rule':'rest.explicit_number'})
        m=re.search(r'(?:адрес\s+(?:работ\w*|объект\w*|склад\w*)|место\s+(?:работы|вахты)|работать\s+предстоит|где\s+работаем)\s*[:—-]\s*(.{4,})',t)
        if m:site.append({'value':m[1].strip(),'evidence':raw,'rule':'workplace.anchor'})
        if re.search(r'(?:место\s+(?:работы|вахты)|адрес\s+(?:работы|объекта|склада))\s*:\s*$',t) and line_index+1<len(lines):
            value=normalize_fragment(lines[line_index+1]);site.append({'value':value,'evidence':raw+' '+lines[line_index+1],'rule':'workplace.heading_next_line'})
        amount=r"(\d{1,3}(?:[ .\u00a0]\d{3})*|\d{4,6})"
        for pattern,reverse in [(amount+r"\s*(?:₽|руб\w*\.?|р\.)\s*за\s*(\d{1,3})\s*(?:рабоч\w*\s+)?смен\w*",False),(r"за\s*(\d{1,3})\s*(?:рабоч\w*\s+)?смен\w*\s*[-:—]?\s*"+amount+r"\s*(?:₽|руб\w*\.?|р\.)",True)]:
            for m in re.finditer(pattern,t):
                cash,shifts=(m[2],m[1]) if reverse else (m[1],m[2]);cash=float(re.sub(r"[ .\u00a0]","",cash));shifts=int(shifts)
                if 0<cash<=10000000 and 1<=shifts<=180:rotation.append({'value':{'total_rub':cash,'work_shifts':shifts},'evidence':raw,'rule':'rotation_total.explicit_shifts'})
        money=r'(\d{1,3}(?:[ .\u00a0]\d{3})*|\d{4,6})(?:\s*(?:—|–|-|до)\s*(\d[\d .\u00a0]*?))?\s*(?:₽|руб\w*\.?|[рp]\.?)\s*(?:/|за\s+|в\s+)(?:\d{1,2}[- ]?часов\w*\s+)?(смен\w*|час\w*|день|сут\w*|мес\w*\.?|вахт\w*)'
        for m in re.finditer(money,t):
            lo=float(re.sub(r'[ .\u00a0]','',m[1]));hi=float(re.sub(r'[ .\u00a0]','',m[2])) if m[2] else lo
            if not 0<lo<=hi<=10000000:continue
            left=t[max(0,m.start()-12):m.start()];near=t[max(0,m.start()-60):min(len(t),m.end()+65)]
            unit=next((v for k,v in [('смен','shift'),('час','hour'),('день','day'),('сут','duty_day'),('мес','month'),('вахт','rotation')] if m[3].startswith(k)),None)
            shape='to' if re.search(r'до\s*$',left) else 'from' if re.search(r'от\s*$',left) else 'range' if lo!=hi else 'exact'
            tax='net' if re.search(r'на\s+руки|чистыми',near) else 'gross' if re.search(r'до\s+вычета|gross',near) else 'unknown'
            conditional=bool(re.search(r'сдельн|сделк|выработ|норм[ауы]|стажир|обучени|первые?\s+\d+|при\s+услов',t))
            clause_prefix=t[:m.start()]
            supplement=list(re.finditer(r'бонус|аванс|преми|компенсац|доплат|надбавк|суточн',clause_prefix))
            primary=list(re.finditer(r'(?<![а-я])(?:зарплат\w*|заработ\w*|ставк\w*|оплат\w*|з/п)',clause_prefix))
            bonus=bool(supplement and (not primary or supplement[-1].start()>primary[-1].start()))
            rates.append({'value':{'low':lo,'high':hi,'unit':unit,'shape':shape,'tax':tax,'conditional':conditional,'bonus_context':bonus},'evidence':raw,'rule':'rate.explicit_money_period'})
        if re.search(r"(?:зп|з/п|зарплат\w*|заработ\w*|оплат\w*|за\s+вахт\w*)",t):
            for m in re.finditer(r"(?:за\s+вахт\w*\s*(?:\d+\s*мес\w*)?|за\s+\d+\s*(?:дн\w*|мес\w*))[^\n]{0,30}?(\d{1,3}(?:[ .]\d{3})+|\d{4,6})",t):
                amount=float(re.sub(r"[ .]","",m[1]));period_conflicts.append({'value':amount,'evidence':raw,'rule':'salary.explicit_multi_day_period_amount'})
            for m in re.finditer(r"(\d{1,3}(?:[ .]\d{3})+|\d{4,6})\s*(?:₽|руб\w*\.?|р\.)?\s*(?:на\s+руки\s*)?за\s+\d+\s*(?:дн\w*|мес\w*)",t):
                period_conflicts.append({'value':float(re.sub(r"[ .]","",m[1])),'evidence':raw,'rule':'salary.amount_before_multi_day_period'})
    work_result=result(work)
    if any(c['rule']=='vahta.options_or_pair_unassigned' for c in work):
        work_result.update(value=None,status='ambiguous')
    return {'version':VERSION,'paid_hours':result(paid),'paid_hours_rejected':result(paid_rejected),'vahta_work_duration':work_result,'rest_duration':result(rest),'workplace':result(site),'rate_options':result(rates),'rotation_totals':result(rotation),'non_shift_amounts':result(period_conflicts)}
