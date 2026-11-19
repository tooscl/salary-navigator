"""Job conditions extracted from vacancy text into a closed set of features.

The aim is to keep WHAT the ad says about the job (pay structure without
amounts, schedule, shift length, perks, requirements, equipment, vehicle)
and drop HOW and BY WHOM it is written. Every feature is a concept, so the
same condition worded differently by two employers gives the same value, and
an employer's unique template phrasing has nothing to attach to.

Not extracted on purpose: brand and employer names, marketing / "about us"
text, layout (length, emoji, bullets), Latin look-alike obfuscation.
Input must be salary-redacted (redaction.strict_redact): no amounts here.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# concept -> regex (lower-case text, ё -> е, homoglyphs already normalized)
CONCEPTS = {
    # --- pay structure (never amounts)
    "pay_daily": r"ежедневн\w* (?:выплат|оплат|расчет)|выплат\w* (?:каждый день|ежедневно)|оплата каждый день|расчет (?:в конце|после) (?:смены|дня)",
    "pay_weekly": r"еженедельн\w* (?:выплат|оплат)|выплат\w* (?:раз в неделю|еженедельно|каждую неделю)|оплата раз в неделю",
    "pay_twice_month": r"(?:2|два|дважды) раза в месяц|аванс и (?:зарплат|расчет)",
    "pay_advance": r"аванс",
    "pay_bonus": r"преми|бонус|надбавк",
    "pay_kpi_percent": r"\bkpi\b|\bкпи\b|процент\w* от|% от|от выручк|от продаж",
    "pay_piecework": r"сдельн|за (?:каждый |1 )?(?:заказ|рейс|точк|адрес|посылк|коробк|паллет|единиц)|за выполненн",
    "pay_tips": r"чаев",
    "pay_overtime": r"переработк\w* оплачива|сверхурочн",
    "pay_fixed": r"оклад|фиксированн\w* (?:оплат|зарплат|ставк|доход)|гарантированн\w* (?:оплат|доход|минимум)",
    "pay_no_delay": r"без задерж|вовремя|стабильн\w* (?:выплат|оплат)",
    "pay_card_or_cash": r"на карт|наличн",
    "pay_referral": r"приведи друга|за друга|реферальн",
    "pay_welcome_bonus": r"приветственн|бонус при (?:трудоустр|выходе)|подъемные",
    "pay_indexation": r"индексаци",
    # --- employment form
    "emp_official": r"по тк\b|трудов\w* (?:кодекс|договор)|официальн\w* (?:трудоустр|оформл|работ)|белая",
    "emp_selfemployed": r"самозанят",
    "emp_gph": r"\bгпх\b|договор\w* (?:гпх|подряд|оказани)",
    "emp_agency": r"аутстафф|аутсорс|кадров\w* агентств",
    "emp_probation": r"испытательн",
    "emp_fast_hiring": r"(?:в день|день в день|в течени\w* дня)\s+(?:обращени|собеседовани)|выход (?:завтра|сегодня|на следующий день)|быстр\w* (?:оформлени|трудоустр|выход)",
    "emp_urgent": r"срочно",
    "emp_medbook_paid": r"медкнижк\w* (?:за счет|оплачива|бесплатн)|медицинск\w* книжк\w* (?:за счет|оплачива|бесплатн)",
    "emp_medbook_required": r"медкнижк|медицинск\w* книжк",
    # --- schedule
    "sch_2_2": r"\b2\s*/\s*2\b|2 через 2|два через два",
    "sch_5_2": r"\b5\s*/\s*2\b|пятидневк|5 через 2",
    "sch_6_1": r"\b6\s*/\s*1\b|6 через 1|шестидневк",
    "sch_3_3": r"\b3\s*/\s*3\b|3 через 3",
    "sch_1_3_or_1_2": r"\b1\s*/\s*[23]\b|сутки через (?:двое|трое)",
    "sch_vahta": r"вахт",
    "sch_night": r"ночн",
    "sch_day_night": r"день\s*/\s*ночь|день-ночь|дневн\w* и ночн",
    "sch_flexible": r"гибк\w* график|свободн\w* график|сам\w* выбира\w* (?:график|смен|время)",
    "sch_part_time": r"подработ|неполн\w* (?:рабоч|день|недел)|частичн",
    "sch_weekends": r"выходн\w* дн\w* (?:сб|суб)|только (?:в )?выходн|по выходным",
    "sch_no_days_off": r"без выходн",
    # --- perks
    "perk_housing": r"проживани|общежити|жиль[еоя]|хостел|комнат\w* для прожив",
    "perk_housing_compensation": r"компенсац\w* (?:аренд|жиль|проживани)",
    "perk_food": r"питани|обед|столов",
    "perk_transport": r"развоз|корпоративн\w* транспорт|доставк\w* (?:до|к) (?:места )?работ|компенсац\w* (?:проезд|транспорт)",
    "perk_fuel": r"\bгсм\b|топлив\w* (?:за счет|компенс|оплачива)|компенсац\w* (?:топлив|бензин)",
    "perk_uniform": r"спецодежд|униформ|форм\w* (?:выдает|бесплатн|за счет)",
    "perk_dms": r"\bдмс\b|медицинск\w* страхов|телемедицин",
    "perk_discount": r"скидк|кэшбек|кешбек",
    "perk_training_paid": r"оплачиваем\w* (?:обучени|стажировк)|обучени\w* (?:за счет|бесплатн)",
    "perk_training": r"обучени|обучим|научим|наставник",
    "perk_career": r"карьерн|рост до|повышени|перспектив",
    "perk_rest_room": r"комнат\w* отдыха|зон\w* отдыха",
    "perk_sport": r"фитнес|\bспорт",
    "perk_corporate": r"корпоратив|подарк",
    "perk_relocation_help": r"билет\w* (?:за счет|оплачива|компенс)|оплат\w* (?:проезд\w* до|дорог)",
    # --- candidate requirements
    "req_no_experience": r"без опыта|опыт не (?:обязателен|требуется|нужен|важен)|можно без опыта",
    "req_experience": r"опыт\w* (?:работы )?от|с опытом|опыт обязателен|требуется опыт",
    "req_students": r"студент",
    "req_pensioners": r"пенсионер",
    "req_foreigners": r"иностран|\bснг\b|патент|\bрвп\b|\bвнж\b|мигрант",
    "req_citizens_only": r"только (?:граждан\w* )?рф|гражданств\w* рф",
    "req_men": r"мужчин",
    "req_women": r"женщин|девушк",
    "req_record_ok": r"судимост",
    "req_physical": r"тяжест|физическ\w* (?:нагрузк|выносл|работ)|грузоподъем",
    "req_pc": r"\bпк\b|компьютер|\b1с\b",
    # --- equipment and place
    "eq_scanner": r"\bтсд\b|терминал сбора|сканер|штрих",
    "eq_forklift": r"погрузчик|штабелер|рохл|электротележк|ричтрак",
    "place_marketplace": r"маркетплейс|\bпвз\b|пункт\w* выдачи",
    "place_dc": r"распределительн\w* центр|\bрц\b|логистическ\w* (?:центр|комплекс)|фулфилмент",
    "place_cold": r"холодн\w* (?:склад|камер)|морозильн|рефрижератор|минусов\w* температур",
    "place_store": r"магазин|торгов\w* зал|супермаркет|гипермаркет",
    "place_cafe": r"кафе|ресторан|кофейн|пекарн|фастфуд|\bбар\b",
    "place_outside_city": r"за город|област\w* (?:склад|площадк)|промзон",
    "task_cash": r"кассир|касс[аеуы]\b|за кассой",
    "task_display": r"выкладк|мерчендайз",
    "task_receiving": r"приемк|приемк\w* товар|разгрузк",
    "task_picking": r"сборк\w* заказ|комплектаци|подбор заказ",
    "task_packing": r"упаковк|маркировк|фасовк",
    "task_inventory": r"инвентаризац",
    "task_senior": r"старш|бригадир|руковод|администратор|заведующ",
    # --- transport (drivers and couriers)
    "veh_own_car": r"(?:на )?(?:сво[еи]м|личн\w*) (?:авто|автомобил|транспорт|машин)|собственн\w* (?:авто|автомобил|транспорт)",
    "veh_company_car": r"служебн\w* (?:авто|автомобил|транспорт|машин)|(?:авто|автомобил|машин)\w* компании|предоставля\w* (?:авто|автомобил|машин|транспорт)",
    "veh_gazelle": r"газел",
    "veh_truck_heavy": r"\bфур|тягач|полуприцеп|еврофур|\bкамаз|самосвал|манипулятор|длинномер",
    "veh_reefer": r"рефрижератор|\bреф\b",
    "veh_bicycle": r"велосипед|\bвело|самокат|электровело",
    "veh_on_foot": r"пеш\w* курьер|пешком|пеший",
    "lic_b": r"категори\w* [вb]\b|\bкат\.?\s*[вb]\b|права [вb]\b",
    "lic_c": r"категори\w* [сc]\b|\bкат\.?\s*[сc]\b|права [сc]\b",
    "lic_ce": r"категори\w* [сc]\s*[еe]\b|\b[сc]\s*[еe]\b|\bкат\.?\s*[сc]?\s*[еe]\b|категори\w* [еe]\b",
    "lic_d": r"категори\w* [дd]\b",
    "lic_tachograph": r"тахограф|карт\w* водител",
    "lic_adr": r"\bдопог\b|\badr\b|опасн\w* груз",
    "route_intercity": r"межгород|междугород|по россии|дальнобой|командировк|рейс\w* (?:по|в) (?:регион|област)",
    "route_city": r"по городу|по москв|по санкт|по спб|по екатеринбург|городск\w* (?:рейс|маршрут)",
    "route_international": r"международн|в казахстан|в беларус|снг рейс",
    "courier_thermo_bag": r"термо(?:сумк|короб|рюкзак)",
    "courier_near_home": r"рядом с дом|возле дома|в своем район",
}
_COMPILED = {name: re.compile(pattern) for name, pattern in CONCEPTS.items()}

# Small numbers that describe the job (amounts are already removed).
_SHIFT_HOURS = re.compile(r"(?:смен\w*|рабоч\w* день)\D{0,15}?(\d{1,2})\s*(?:-|–|до)?\s*(\d{1,2})?\s*час|(\d{1,2})[\s-]*(?:часов\w*|ч\b)\s*(?:смен|рабоч)")
_TIME_RANGE = re.compile(r"\bс\s*(\d{1,2})(?:[:.]\d{2})?\s*до\s*(\d{1,2})(?:[:.]\d{2})?\b")
_VAHTA_DAYS = re.compile(r"вахт\w*\D{0,20}?(\d{2,3})\s*/\s*(\d{2,3})|(\d{2,3})\s*/\s*(\d{2,3})\s*вахт|вахт\w*\D{0,15}?(?:от\s*)?(\d{2,3})\s*(?:дн|смен|суток)")
_AGE = re.compile(r"(?:от|с)\s*(1[6-9]|[2-5]\d)\s*(?:лет|годов)|(?:до)\s*([3-7]\d)\s*лет")
_EXPERIENCE_YEARS = re.compile(r"опыт\w*[^.\n]{0,25}?от\s*(\d{1,2})\s*(?:лет|год|года)")
_TONNAGE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*(?:т\b|тонн)")
_WEIGHT_KG = re.compile(r"(?:до|от|свыше)?\s*(\d{1,3})\s*кг")


def _first_float(match, groups) -> float:
    for g in groups:
        value = match.group(g)
        if value:
            return float(value.replace(",", "."))
    return np.nan


def _numbers(text: str) -> dict[str, float]:
    out = {"num_shift_hours": np.nan, "num_vahta_days": np.nan, "num_age_min": np.nan, "num_age_max": np.nan,
           "num_experience_years": np.nan, "num_tonnage": np.nan, "num_weight_kg": np.nan}
    m = _SHIFT_HOURS.search(text)
    if m:
        hours = _first_float(m, (2, 1, 3))
        out["num_shift_hours"] = hours if 3 <= hours <= 24 else np.nan
    if np.isnan(out["num_shift_hours"]):
        m = _TIME_RANGE.search(text)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            hours = (end - start) % 24
            out["num_shift_hours"] = float(hours) if 3 <= hours <= 24 else np.nan
    m = _VAHTA_DAYS.search(text)
    if m:
        days = _first_float(m, (1, 3, 5))
        out["num_vahta_days"] = days if 7 <= days <= 120 else np.nan
    ages_min = [int(a) for a, _ in _AGE.findall(text) if a]
    ages_max = [int(b) for _, b in _AGE.findall(text) if b]
    out["num_age_min"] = float(min(ages_min)) if ages_min else np.nan
    out["num_age_max"] = float(max(ages_max)) if ages_max else np.nan
    m = _EXPERIENCE_YEARS.search(text)
    if m and int(m.group(1)) <= 15:
        out["num_experience_years"] = float(m.group(1))
    tons = [float(t.replace(",", ".")) for t in _TONNAGE.findall(text)]
    tons = [t for t in tons if 0.5 <= t <= 60]
    out["num_tonnage"] = max(tons) if tons else np.nan
    kgs = [int(k) for k in _WEIGHT_KG.findall(text) if 5 <= int(k) <= 200]
    out["num_weight_kg"] = float(max(kgs)) if kgs else np.nan
    return out


def condition_features(title: pd.Series, description: pd.Series, benefits: pd.Series) -> pd.DataFrame:
    """One row per vacancy: 0/1 concept flags, a few job numbers, perk count.

    title/description must be strictly redacted; benefits is the structured
    Avito perk list, used as text here (its items are concepts already)."""
    text = (title.fillna("") + "\n" + description.fillna("") + "\n" + benefits.fillna("")).str.lower().str.replace("ё", "е")
    out = {f"cond_{name}": text.str.contains(pattern).astype(np.int8) for name, pattern in _COMPILED.items()}
    numbers = pd.DataFrame([_numbers(t) for t in text], index=text.index)
    items = benefits.fillna("").astype(str).str.split(r"\s*,\s*")
    out["cond_perk_list_count"] = items.map(lambda xs: sum(1 for x in xs if x.strip()))
    return pd.concat([pd.DataFrame(out, index=text.index), numbers], axis=1)
