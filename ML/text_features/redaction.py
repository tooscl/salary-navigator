"""Salary redaction levels compared in the research.

Avito descriptions are obfuscated with Latin look-alike letters inside Russian
words ("зaрплaтa oт 110 000 pублeй"). utils.delete_salary matches Cyrillic
words only, so such fragments survive it. Levels:

raw        no redaction (leakage ceiling, never for a real model)
prod       utils.delete_salary, as in ML.salary_model
homoglyph  look-alike Latin letters -> Cyrillic, then delete_salary
strict     homoglyph + drop any remaining segment with a money-like number
clean      strict + no decoration: emoji, bullet glyphs, repeated punctuation and
           line structure removed, so TF-IDF sees words, not the ad's layout
"""

from __future__ import annotations

import re

from utils.delete_salary import _SEGMENT_RE, delete_salary

# Latin letters that look like Cyrillic ones. Applied only inside words that
# already mix both scripts, so English words and brands stay intact.
_LOOKALIKE = str.maketrans({
    "a": "а", "A": "А", "c": "с", "C": "С", "e": "е", "E": "Е", "o": "о", "O": "О",
    "p": "р", "P": "Р", "x": "х", "X": "Х", "y": "у", "Y": "У", "k": "к", "K": "К",
    "M": "М", "T": "Т", "B": "В", "H": "Н",
})
_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[a-zA-Z]")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_MIXED_WORD = re.compile(r"(?=[A-Za-zА-Яа-яЁё]*[а-яёА-ЯЁ])(?=[A-Za-zА-Яа-яЁё]*[a-zA-Z])[A-Za-zА-Яа-яЁё]+")

# Any amount that could be money: 1 000+, 4-7 digits, "55 тыс", "60к".
MONEY_LIKE = re.compile(
    r"\d{1,3}(?:[\s .,’']\d{3})+|\d{4,7}|\d{1,3}(?:[.,]\d)?\s*(?:тыс|т\.?\s*р\b|k\b|к\b)",
    re.IGNORECASE,
)


def _fix_word(match: re.Match) -> str:
    word = match.group(0)
    if _CYRILLIC.search(word) and _LATIN.search(word):
        return word.translate(_LOOKALIKE)
    return word


def normalize_homoglyphs(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = _WORD.sub(_fix_word, text)
    # "3 500 p." - a lone Latin p after a number is a ruble sign.
    return re.sub(r"(?<=\d)\s*p\.", " р.", text)


def mixed_script_share(text: str) -> float:
    """Share of words written with both scripts (obfuscation style)."""
    if not isinstance(text, str) or not text:
        return 0.0
    words = _WORD.findall(text)
    return len(_MIXED_WORD.findall(text)) / len(words) if words else 0.0


def strict_redact(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    lines = []
    for line in delete_salary(normalize_homoglyphs(text)).split("\n"):
        kept = [s for s in _SEGMENT_RE.split(line) if s.strip() and not MONEY_LIKE.search(s)]
        line = "".join(kept).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


_DECORATION = re.compile(
    "[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u23FF\u25A0-\u27BF\u2900-\u2BFF\uFE0F\u200D\u20E3"
    "•●▪■◆◇★☆✓✔➤►▶·]+"
)
_PUNCT_RUN = re.compile(r"([!?.,:;\-–—_=*~#|/\\])\1+")


def strip_decoration(text: str) -> str:
    """Emoji, bullet glyphs, repeated punctuation and line structure removed."""
    text = _DECORATION.sub(" ", text or "")
    text = _PUNCT_RUN.sub(r"\1", text)
    text = re.sub(r"(?m)^\s*[-–—*]\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Strictly redacted text without layout and decoration (one line, single spaces)."""
    return strip_decoration(strict_redact(text))


# --- "content": what the ad says about the job, not who wrote it -----------
# Employer templates: a sentence that one employer repeats in >= 3 of its ads
# and that is >= 80 % that employer's (other employers rarely write it this
# way) is the employer's template wording ("Пятёрочка приглашает...", its
# standard perk list). Generic sentences many employers use stay. Templates are
# learnt from texts and employer names only, never from salaries.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
TEMPLATE_MIN_ADS = 3
TEMPLATE_MIN_SHARE = 0.8

# Chains and platforms frequent in the data; names are identity, not content.
BRANDS = [
    "пятерочк", "пятёрочк", "магнит", "перекрест", "перекрёст", "дикси", "лента", "ашан", "вкусвилл", "бристоль",
    "fix price", "фикс прайс", "светофор", "монетк", "верный", "spar", "спар", "глобус", "globus", "мария-ра",
    "красное&белое", "красное и белое", "озон", "ozon", "wildberries", "вайлдберриз", "валдберриз", "яндекс", "yandex",
    "самокат", "сбермаркет", "купер", "kuper", "x5", "х5", "rostic", "ростикс", "kfc", "вкусно и точка", "бургер кинг",
    "burger king", "додо", "теремок", "леруа", "сдэк", "cdek", "деловые линии", "почта росси", "достависта",
    "delivery club", "магнолия", "чижик", "окей", "o'key", "metro c&c", "auchan", "dixy",
]


def sentence_key(sentence: str) -> str:
    key = sentence.lower().replace("ё", "е")
    key = re.sub(r"[^\w\s/]", " ", key)
    key = re.sub(r"\d+", "0", key)
    return re.sub(r"\s+", " ", key).strip()


def employer_templates(texts: list[str], employers: list[str]) -> set[str]:
    """Sentence keys that are some employer's template wording."""
    ads, by_employer = {}, {}
    for row, (text, employer) in enumerate(zip(texts, employers)):
        for key in {sentence_key(s) for s in _SENTENCE_SPLIT.split(text or "")}:
            if len(key) < 15:
                continue
            ads[key] = ads.get(key, 0) + 1
            by_employer[(key, employer)] = by_employer.get((key, employer), 0) + 1
    best = {}
    for (key, employer), count in by_employer.items():
        if employer and count > best.get(key, 0):
            best[key] = count
    return {k for k, c in best.items() if c >= TEMPLATE_MIN_ADS and c / ads[k] >= TEMPLATE_MIN_SHARE}


def remove_names(text: str, employer: str) -> str:
    """Brand names and the words of this ad's employer name."""
    words = [w for w in re.findall(r"[a-zа-я0-9]{4,}", (employer or "").lower().replace("ё", "е"))]
    for name in BRANDS + words:
        text = re.sub(re.escape(name) + r"\w*", " ", text, flags=re.IGNORECASE)
    return text


def content_text(strict_text: str, templates: set[str], employer: str) -> str:
    """Strictly redacted text without employer templates, names and decoration."""
    kept = [s for s in _SENTENCE_SPLIT.split(strict_text or "") if sentence_key(s) not in templates]
    return strip_decoration(remove_names(" ".join(kept), employer))


def redact(text: str, level: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    if level == "raw":
        return text
    if level == "prod":
        return delete_salary(text)
    if level == "homoglyph":
        return delete_salary(normalize_homoglyphs(text))
    if level == "strict":
        return strict_redact(text)
    if level == "clean":
        return clean_text(text)
    raise ValueError(f"Unknown redaction level: {level}")


def amounts_in_text(text: str) -> set[int]:
    """Rubles mentioned in text, for the leakage audit."""
    found = set()
    for match in re.finditer(r"\d{1,3}(?:[\s .,’']\d{3})+|\d{4,7}", text or ""):
        found.add(int(re.sub(r"\D", "", match.group(0))))
    for match in re.finditer(r"(\d{1,3}(?:[.,]\d)?)\s*(?:тыс|т\.?\s*р|k\b|к\b)", text or "", re.IGNORECASE):
        found.add(int(float(match.group(1).replace(",", ".")) * 1000))
    return found


def target_in_text(text: str, bounds: list[float], tolerance: float = 0.01) -> bool:
    """True when an observed salary bound (within 1 %) is written in the text."""
    values = [b for b in bounds if b == b and b is not None]
    return any(abs(a - b) <= tolerance * b for a in amounts_in_text(text) for b in values)
