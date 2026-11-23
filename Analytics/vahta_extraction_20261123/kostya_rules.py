"""Rule-based vacancy feature extraction. Standard library only.
Values describe statements in a vacancy, not independently verified job conditions.
"""
import re
import html
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser

VERSION = "rules-context-1.0"

FINAL_FEATURES = {
    "Кассир": ["common__schedule_patterns", "common__shift_hours", "common__time_of_day", "cashier__retail_format", "cashier__stocking", "cashier__food_preparation", "common__meals_support"],
    "Продавец": ["common__schedule_patterns", "common__shift_hours", "common__time_of_day", "seller__cash_operations", "seller__stocking", "seller__consulting", "seller__product_categories", "seller__retail_format"],
    "Грузчик": ["common__schedule_patterns", "common__shift_hours", "common__time_of_day", "loader__work_environment", "loader__equipment", "loader__max_load_kg", "loader__picking", "common__meals_support"],
    "Комплектовщик": ["common__schedule_patterns", "common__shift_hours", "common__time_of_day", "picker__picking_technology", "picker__work_environment", "picker__product_categories", "picker__packing", "common__meals_support"],
    "Водитель грузового транспорта": ["driver__license_categories", "driver__vehicle_types", "driver__vehicle_arrangement", "driver__capacity_tons", "driver__body_types", "driver__route_scope", "common__experience_required"],
}
ALL_FEATURES = list(dict.fromkeys(f for fs in FINAL_FEATURES.values() for f in fs))
FEATURE_SEGMENTS = {f: [p for p, fs in FINAL_FEATURES.items() if f in fs] for f in ALL_FEATURES}


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}: self.hidden += 1
        elif tag in {"br", "p", "div", "li", "ul", "ol", "h1", "h2", "h3"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style"}: self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "li", "h1", "h2", "h3"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)

_HOMO = str.maketrans("ABCEHKMOPTXYabceopxy", "АВСЕНКМОРТХУавсеорху")

def clean_description(value):
    if value is None: return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\\n", "\n")
    if re.search(r"</?[a-zA-Z][^>]*>", text):
        parser = _HTMLText(); parser.feed(text); text = "".join(parser.parts)
    text = html.unescape(text)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()

def normalize_fragment(text):
    def fix(m):
        word = m.group()
        return word.translate(_HOMO) if re.search(r"[а-яА-ЯёЁ]", word) and re.search(r"[a-zA-Z]", word) else word
    text = re.sub(r"[a-zA-Zа-яА-ЯёЁ]+", fix, text)
    return text.lower().replace("ё", "е")

@dataclass(frozen=True)
class Fragment:
    raw: str
    text: str
    section: str = "unknown"

SECTION_RE = re.compile(r"^\s*(обязанности|задачи|ваши задачи|твои задачи|условия|мы предлагаем|требования|пожелания|проживание|общежитие)\s*:\s*(.*)$", re.I)

def fragments_from_clean(clean):
    result = []; section = "unknown"
    for line in clean.splitlines():
        line = line.strip(" \t•▪●*-✅📋")
        if not line: continue
        heading = SECTION_RE.match(line)
        if heading:
            section = normalize_fragment(heading[1]); line = heading[2]
        for part in re.split(r";|(?<!\d)\.(?=\s|$)|[!?](?:\s+|$)", line):
            part = part.strip()
            if part: result.append(Fragment(part, normalize_fragment(part), section))
    return result


# Conditions express uncertainty, not feature absence.
CONDITIONAL = re.compile(r"\b(?:возможн[аоы]?|по желанию|по согласованию|при необходимости|при условии|желательно|приветствуется|будет преимуществом)\b")

def is_negated(text, match):
    left = text[max(0, match.start()-55):match.start()]
    if re.search(r"не\s+(?:только|просто)\s+$", left): return False
    if re.search(r"(?:\bбез|\bне)\s+(?:(?:использования|применения|работы\s+с|работы\s+на|обязательного)\s+)?$", left): return True
    right = text[match.end():match.end()+65]
    return bool(re.match(r"\s*(?:[-,:]\s*)?(?:не\s+(?:нуж\w*|треб\w*|использ\w*|предостав\w*|предусмотр\w*)|отсутств\w*)", right))

def make_result(value=None, status="not_stated", evidence=(), rules=(), candidates=(), negatives=()):
    return {"value": value, "status": status,
            "evidence": list(dict.fromkeys(evidence)),
            "rule_ids": list(dict.fromkeys(rules)),
            "candidates": sorted(set(candidates), key=str),
            "negated_values": sorted(set(negatives), key=str)}

def aggregate(hits, mode="set"):
    # hit: (value, positive/negative/conditional, original fragment, rule id)
    positive = {v for v, state, _, _ in hits if state == "positive"}
    negative = {v for v, state, _, _ in hits if state == "negative"}
    conditional = {v for v, state, _, _ in hits if state == "conditional"}
    evidence = [e for _, _, e, _ in hits]; rules = [r for _, _, _, r in hits]
    kwargs = dict(evidence=evidence, rules=rules, candidates=positive|conditional, negatives=negative)
    if positive & negative or conditional - positive:
        return make_result(status="ambiguous", **kwargs)
    if not positive:
        return make_result(status="explicit_negative" if negative else "not_stated", **kwargs)
    if mode == "max": value = max(positive)
    elif mode == "one":
        if len(positive) != 1: return make_result(status="ambiguous", **kwargs)
        value = next(iter(positive))
    else: value = " | ".join(sorted(map(str, positive)))
    return make_result(value, "resolved", **kwargs)

def add_matches(hits, fragment, label, pattern, rule_id, use_conditions=True):
    for m in re.finditer(pattern, fragment.text):
        state = "negative" if is_negated(fragment.text, m) else "conditional" if use_conditions and CONDITIONAL.search(fragment.text) else "positive"
        hits.append((label, state, fragment.raw, rule_id))


CATEGORY_PATTERNS = {
 "time_of_day": {
    "day": r"\bдневн\w*\s+(?:смен\w*|работ\w*|врем\w*)|\bдневн\w*\s+и\s+ночн\w*\s+смен\w*",
    "night": r"\bночн\w*\s+(?:смен\w*|работ\w*|врем\w*)",
    "evening": r"\bвечерн\w*\s+(?:смен\w*|работ\w*|врем\w*)"},
 "retail_format": {
    "supermarket": r"\b(?:супермаркет|гипермаркет)\w*",
    "fuel_station": r"\bазс\b|\bавтозаправ\w*",
    "restaurant": r"\b(?:ресторан|кафе|фастфуд)\w*|\bбыстр\w*\s+питани\w*",
    "specialty_store": r"\bспециализированн\w*\s+магазин\w*|\bвинотек\w*",
    "convenience_store": r"\bмагазин\w*\s+у\s+дома\b"},
 "product_categories": {
    "food": r"\bпродукт\w*\s+питани\w*|\bпродуктов\w*\s+(?:магазин|сет|склад)\w*|\bпродовольствен\w*|\bмолочн\w*\s+продукц\w*",
    "clothing_textile": r"\bодежд\w*|\bобув\w*|\bтекстил\w*|\bбель(?:е|я|ю|ем)\b",
    "electronics": r"\bэлектроник\w*|\bбытов\w*\s+техник\w*|\bсмартфон\w*",
    "building_materials": r"\bстройматериал\w*|\bстроительн\w*\s+материал\w*",
    "furniture": r"\bмебел\w*",
    "alcohol": r"\bалкогол\w*|\bвинотека\w*|\bвино[- ]водоч\w*",
    "cosmetics": r"\bкосметик\w*|\bпарфюмер\w*",
    "auto_parts": r"\bавтозапчаст\w*|\bзапчаст\w*"},
 "work_environment": {
    "freezer": r"\bморозильн\w*\s+(?:камер|склад|зон)\w*|\bзамороженн\w*\s+продукц\w*",
    "refrigerated": r"\bхолодильн\w*\s+(?:камер|склад|зон)\w*|\bохлажденн\w*\s+продукц\w*|\bхолодн\w*\s+склад\w*",
    "heated": r"\bтепл\w*\s+склад\w*|\bотапливаем\w*\s+(?:склад|помещени)\w*",
    "outdoors": r"\bна\s+улице\b|\bоткрыт\w*\s+(?:площадк|воздух)\w*",
    "production": r"\bна\s+производств\w*|\bпроизводственн\w*\s+цех\w*"},
 "equipment": {
    "pallet_jack": r"\b(?:рохл|рокл)\w*|\bгидравлическ\w*\s+тележ\w*",
    "forklift": r"\b(?:погрузчик|штабелер)\w*",
    "conveyor": r"\bконвейер\w*",
    "electric_cart": r"\b(?:электротележ|электророхл)\w*"},
 "picking_technology": {
    "tsd": r"\bтсд\b|\bтерминал\w*\s+сбор\w*\s+данн\w*",
    "voice": r"\bголосов\w*\s+(?:отбор|сбор|комплектац)\w*",
    "paper": r"\bбумажн\w*\s+(?:спис|накладн)\w*|\bпо\s+накладн\w*",
    "app": r"\bприложени\w*\s+(?:на|в)\s+телефон\w*|\bсборк\w*.{0,65}приложени\w*"},
 "vehicle_types": {
    "van": r"\bфургон\w*|\bларгус\w*|\blargus\b|\bгазел\w*",
    "tractor": r"\bтягач\w*", "dump_truck": r"\bсамосвал\w*",
    "truck": r"\bгрузовик\w*|\bгрузов\w*\s+автомобил\w*",
    "light_commercial": r"\bмалотоннаж\w*|\bкаблук\w*",
    "truck_crane": r"\bавтокран\w*|\bманипулятор\w*"},
 "body_types": {
    "tent": r"\bтент\w*", "refrigerated": r"\bрефрижератор\w*",
    "flatbed": r"\bбортов\w*", "tank": r"\bцистерн\w*",
    "container": r"\bконтейнеровоз\w*"},
}

BOOLEAN_PATTERNS = {
 "stocking": (r"\b(?:выкладк\w*|выкладыва\w*).{0,65}(?:товар|продукц)\w*", r"\bбез\s+выкладк\w*|\bвыкладк\w*.{0,25}не\s+(?:нужн|треб)\w*"),
 "food_preparation": (r"\b(?:приготовлени\w*|готовить|доготовк\w*|доготавлива\w*).{0,45}(?:блюд\w*|ед[ауы]|напит\w*|кофе)|\bвыпек\w*.{0,35}(?:хлеб|булоч|выпеч)\w*", r"\bбез\s+приготовлени\w*|\bготовить.{0,25}не\s+нужно"),
 "cash_operations": (r"\b(?:работ\w*|обслужив\w*|расчет\w*|рассчитыва\w*).{0,45}(?:на\s+касс\w*|за\s+касс\w*|с\s+касс\w*)|\bрасчет\w*\s+(?:с\s+)?покупател\w*", r"\bбез\s+(?:работы\s+)?(?:на\s+)?кассы\b|\bна\s+кассе\s+работать\s+не\s+нужно"),
 "consulting": (r"\b(?:консультир\w*|консультаци\w*).{0,80}(?:покупател|клиент|абонент)\w*", r"\bбез\s+консультир\w*"),
 # Contract: order picking, not every movement/placement of goods.
 "picking": (r"\b(?:сборк\w*|собира\w*|комплектаци\w*|комплектова\w*).{0,50}заказ\w*", r"\bбез\s+(?:сборки|комплектации)\s+заказов"),
 "packing": (r"\b(?:упаковк\w*|упаковыва\w*).{0,65}(?:заказ|товар|продукц)\w*", r"\bбез\s+упаковк\w*"),
}


def extract_boolean(parts, kind):
    positive_re, negative_re = BOOLEAN_PATTERNS[kind]; hits = []
    for f in parts:
        negs = list(re.finditer(negative_re, f.text))
        for m in negs: hits.append(("no", "positive", f.raw, kind+".explicit_no"))
        for m in re.finditer(positive_re, f.text):
            if any(m.start() < n.end() and n.start() < m.end() for n in negs): continue
            if is_negated(f.text, m): value, state = "no", "positive"
            else: value, state = "yes", "conditional" if CONDITIONAL.search(f.text) else "positive"
            hits.append((value, state, f.raw, kind+".action_object"))
    return aggregate(hits, "one")

def extract_experience(parts):
    hits = []
    for f in parts:
        t = f.text
        if not re.search(r"\bопыт\w*", t): continue
        if re.search(r"опыт\w*\s+(?:службы|военной|армейской)", t) and not re.search(r"опыт\w*\s+(?:работы|вождения|водителем)", t): continue
        no = re.search(r"\bбез\s+опыта\b|\bопыт\w*.{0,25}не\s+(?:нужен|требуется|обязателен)", t)
        yes = re.search(r"\bопыт\w*.{0,65}(?:от\s+\d|не\s+менее\s+\d|обязателен|требуется)|\b(?:требуется|необходим)\s+опыт\w*|\bс\s+опытом\s+работы", t)
        if no and yes and no.start() < yes.end() and yes.start() < no.end() and not re.search(r"от\s+\d|не\s+менее\s+\d", yes.group()): yes = None
        if no: hits.append(("no", "positive", f.raw, "experience.no"))
        if yes: hits.append(("yes", "positive", f.raw, "experience.required"))
        if not no and not yes and CONDITIONAL.search(t): hits.append(("yes", "conditional", f.raw, "experience.optional"))
    return aggregate(hits, "one")

def product_context(f):
    t = f.text
    if f.section in {"проживание", "общежитие"}: return False
    if re.search(r"общежити|засел|спецодежд|униформ|одежд\w*\s+для\s+работ|бесплатн\w*\s+одежд|выда\w*.{0,20}одежд|скидк", t): return False
    return bool(re.search(r"товар|продукц|продукт|ассортимент|склад|комплект|упаков|перевоз|магазин|прода\w*|торгов", t))

def extract_categories(parts, kind):
    hits = []
    for f in parts:
        if kind == "product_categories" and not product_context(f): continue
        if kind == "retail_format" and re.search(r"карьерн\w*\s+рост|скидк|партнерск\w*\s+ресторан", f.text): continue
        for label, pattern in CATEGORY_PATTERNS[kind].items():
            add_matches(hits, f, label, pattern, kind+"."+label)
    return aggregate(hits)


FOOD = r"(?:питани\w*|обед\w*|ужин\w*|завтрак\w*)"

def extract_meals(parts):
    hits = []
    for f in parts:
        for text in re.split(r",|\|", f.text):
            if not re.search(r"\b"+FOOD, text): continue
            if re.search(r"перерыв|час\w*\s+(?:на\s+)?обед", text) and not re.search(r"бесплат|предостав|компенсац", text): continue
            label = None
            if re.search(r"без\s+питания|"+FOOD+r".{0,20}не\s+предостав", text): label = "not_provided"
            elif re.search(r"скидк|частичн\w*\s+компенсац", text): label = "discount_or_partial"
            elif re.search(r"бесплат|за\s+(?:наш\s+)?счет\s+(?:компании|работодателя|предприятия)|за\s+наш\s+счет|100\s*%", text): label = "free_or_employer_paid"
            elif re.search(r"за\s+свой\s+счет|удержан|вычет|(?<!бес)платн\w*\s+"+FOOD, text): label = "paid_by_worker_or_deduction"
            elif re.search(r"компенсац|оплачива", text):
                if re.search(r"не\s+оплачива", text):
                    hits.append(("payment_not_specified", "conditional", f.raw, "meals.negated_payment")); continue
                label = "compensation_terms_unspecified"
            elif re.search(r"предостав|выда\w*|обеспеч|питани\w*\s+в\s+столов", text): label = "provided_terms_unspecified"
            if label:
                state = "conditional" if CONDITIONAL.search(text) else "positive"
                hits.append((label, state, f.raw, "meals."+label))
    concrete = {v for v, s, _, _ in hits if s == "positive" and v != "provided_terms_unspecified"}
    if concrete:
        hits = [h for h in hits if h[0] != "provided_terms_unspecified"]
    return aggregate(hits, "one")

NUMBER = r"\d+(?:[,.]\d+)?"

def extract_numeric(parts, kind):
    hits = []
    for f in parts:
        t = f.text
        if kind == "shift_hours":
            patterns = [
                r"\bсмен\w*\s*(?:по|длительностью|продолжительностью|до|[-:])?\s*("+NUMBER+r")\s*(?:час\w*|ч\b)",
                r"\b("+NUMBER+r")\s*[- ]?часов\w*\s+смен\w*",
                r"\b(?:работ\w*|смен\w*).{0,20}\bпо\s+("+NUMBER+r")\s*час\w*",
            ]
            for pattern in patterns:
                for m in re.finditer(pattern, t):
                    v = float(m[1].replace(",", "."))
                    if 2 <= v <= 24:
                        state = "negative" if is_negated(t, m) else "conditional" if CONDITIONAL.search(t) else "positive"
                        hits.append((v, state, f.raw, "hours.explicit"))
        else:
            unit = r"(?:кг\b|килограмм\w*)" if kind == "max_load_kg" else r"(?:тонн\w*|т\b|кг\b|килограмм\w*)"
            pattern = r"(?:(?:от\s*)?("+NUMBER+r")\s*(?:-|до)\s*)?("+NUMBER+r")\s*("+unit+r")"
            for m in re.finditer(pattern, t):
                context = t[max(0, m.start()-75):m.end()+35]
                v = float(m[2].replace(",", "."))
                if kind == "max_load_kg":
                    if not re.search(r"перенос|подъем|подним|тяжест|вес|груз|короб|рулон", context): continue
                    if re.search(r"грузоподъем|тоннаж", context): continue
                    if not 0 < v <= 200: continue
                else:
                    if not re.search(r"грузоподъем|тоннаж", context): continue
                    if re.match(r"кг|килограмм", m[3]): v /= 1000
                    if not 0 < v <= 100: continue
                state = "negative" if is_negated(t, m) else "conditional" if CONDITIONAL.search(t) else "positive"
                hits.append((v, state, f.raw, kind+".explicit_units"))
    return aggregate(hits, "max" if kind == "max_load_kg" else "one")


def extract_schedule(parts):
    hits = []
    for f in parts:
        for m in re.finditer(r"(?<![\d/])([1-7])\s*[/\\хx]\s*([0-7])(?![\d/])", f.text):
            v = m[1]+"/"+m[2]
            # Do not derive schedule from dates, e.g. 5/2/2026.
            state = "negative" if is_negated(f.text, m) else "conditional" if CONDITIONAL.search(f.text) else "positive"
            hits.append((v, state, f.raw, "schedule.formula"))
    return aggregate(hits)

def extract_license(parts):
    hits = []; translation = str.maketrans("АВСДЕ", "ABCDE")
    token = re.compile(r"[abcdeавсде](?:1)?(?:[eе])?(?![a-zа-я0-9])", re.I)
    separator = re.compile(r"(?:\s*[«»\"'()]*\s*[,/+;]\s*[«»\"']*\s*|\s*[«»\"']*\s+(?:и|или)\s+[«»\"']*\s*|[«»\"']*\s+)", re.I)
    valid = {"A", "B", "C", "D", "E", "BE", "CE", "DE", "C1", "C1E", "D1", "D1E"}
    for f in parts:
        for marker in re.finditer(r"\b(?:категори\w*|кат\.)\s*[:\-]?\s*[«\"']?", f.text):
            pos = marker.end()
            while True:
                m = token.match(f.text, pos)
                if not m: break
                value = m.group().upper().translate(translation)
                if value in valid:
                    state = "conditional" if CONDITIONAL.search(f.text) else "positive"
                    hits.append((value, state, f.raw, "license.explicit"))
                sep = separator.match(f.text, m.end())
                if not sep: break
                pos = sep.end()
    return aggregate(hits)

def extract_arrangement(parts):
    hits = []
    patterns = {
      "company": r"\b(?:авто|автомобиль|автомобили|автомобиле|машина|машины|машине)\s+(?:компании|работодателя)\b|\bкорпоративн\w*\s+(?:авто\b|автомобил\w*|машин\w*)|\bпредостав\w*.{0,25}\b(?:авто|автомобиль|машину)\b",
      "own": r"\b(?:сво\w*|личн\w*|собственн\w*)\s+(?:авто\b|автомобил\w*|машин\w*)"}
    for f in parts:
        for label, pattern in patterns.items(): add_matches(hits, f, label, pattern, "vehicle_arrangement."+label)
    return aggregate(hits)

def extract_routes(parts):
    hits = []
    patterns = {
      "city": r"\bпо\s+городу\b|\bгородск\w*\s+(?:маршрут|достав|перевоз)\w*",
      "suburban": r"\bпригородн\w*\s+(?:маршрут|рейс|достав)\w*|\b(?:рейс|достав|перевоз)\w*.{0,30}по\s+области\b",
      "intercity": r"\bмежгород\w*|\bмеждугород\w*|\b(?:рейс|маршрут|перевоз|достав)\w*.{0,35}(?:по\s+городам|по\s+россии|по\s+рф)\b",
      "international": r"\bмеждународн\w*\s+(?:рейс|маршрут|перевоз|достав)\w*|\b(?:рейс|маршрут|перевоз|достав)\w*.{0,40}(?:снг|европ\w*|за\s+рубеж)"}
    for f in parts:
        if re.search(r"граждан|внж", f.text) and not re.search(r"рейс|маршрут|перевоз|достав", f.text): continue
        for label, pattern in patterns.items(): add_matches(hits, f, label, pattern, "route."+label)
    return aggregate(hits)

def extract_feature(parts, feature):
    kind = feature.split("__", 1)[1]
    if kind == "schedule_patterns": return extract_schedule(parts)
    if kind in {"shift_hours", "max_load_kg", "capacity_tons"}: return extract_numeric(parts, kind)
    if kind == "license_categories": return extract_license(parts)
    if kind == "vehicle_arrangement": return extract_arrangement(parts)
    if kind == "route_scope": return extract_routes(parts)
    if kind == "meals_support": return extract_meals(parts)
    if kind == "experience_required": return extract_experience(parts)
    if kind in BOOLEAN_PATTERNS: return extract_boolean(parts, kind)
    if kind in CATEGORY_PATTERNS: return extract_categories(parts, kind)
    raise KeyError("Unsupported feature: "+feature)

def extract_from_clean(clean, profession=None):
    if profession is not None and profession not in FINAL_FEATURES: raise ValueError("Unknown profession: "+str(profession))
    parts = fragments_from_clean(clean)
    features = ALL_FEATURES if profession is None else FINAL_FEATURES[profession]
    return {f: extract_feature(parts, f) for f in features}

def extract_description(description, profession=None):
    return extract_from_clean(clean_description(description), profession)

def flatten_features(results, prefix="extracted__"):
    row = {}
    for feature, r in results.items():
        col = prefix+feature
        row[col] = r["value"]
        row[col+"__status"] = r["status"]
        row[col+"__evidence"] = " || ".join(r["evidence"]) or None
        row[col+"__rule_ids"] = " | ".join(r["rule_ids"]) or None
        row[col+"__candidates"] = " | ".join(map(str, r["candidates"])) or None
        row[col+"__negated_values"] = " | ".join(map(str, r["negated_values"])) or None
    return row

