"""Remove salary disclosures from Russian vacancy text.

The module is deliberately conservative: a sentence, bullet, or table line that
looks like a compensation disclosure is removed in full.  Replacing only a
number is unsafe because surrounding wording (for example, ``оклад от ...``)
still leaks target information to a model.
"""

from __future__ import annotations

import re
import unicodedata


_SPACE_RE = re.compile(r"[\t\f\v \u00a0\u202f]+")
_SEGMENT_RE = re.compile(r"(?<=[.!?;])(?=\s+)")

# Common ways a Russian vacancy names compensation.  ``ставка`` is included
# only when a number is nearby, which avoids removing unrelated phrases such as
# "ставка на развитие".
_PAY_WORD = r"(?:зар(?:аботн(?:ая|ой)?|плат[аыье]?)|з[\s./-]*п|доход(?:а|ом)?|заработ(?:ок|ка)|оклад(?:а|ом)?|оплат[аыье]|ставк[аиу]|тариф(?:а|ом)?|вознаграждени[ея]|гонорар(?:а|ом)?|компенсаци[яи])"
_CURRENCY = r"(?:₽|руб(?:\.|л(?:ей|я)?|\b)?|р\.|rur|rub|\$|usd|€|eur)"
_PERIOD = r"(?:в|за|/)?\s*(?:час(?:[ао]в?)?|смен[ауые]|день|дн[ьяе]|сут(?:ки|ок)?|недел[юьи]|месяц(?:[аев]?|\.?)|вахт[ауые])"
_NUMBER = r"(?:\d{1,3}(?:[\s,.'’]\d{3})+|\d{4,7}|\d{1,3}\s*(?:тыс\.?|т\.?|к\b))"
_WORD_NUMBER = r"(?:ноль|один(?:надцать)?|дв(?:а|е|енадцать)|три(?:надцать)?|четыр(?:е|надцать)|пят(?:ь|надцать)|шест(?:ь|надцать)|сем(?:ь|надцать)|восем(?:ь|надцать)|девят(?:ь|надцать)|десят(?:ь|ь?)|сто|тысяч[аеи]?|миллион[аов]?)"

_MONEY_WITH_CURRENCY_RE = re.compile(
    rf"{_NUMBER}\s*{_CURRENCY}|{_CURRENCY}\s*{_NUMBER}", re.IGNORECASE
)
_PAY_AND_NUMBER_RE = re.compile(
    rf"{_PAY_WORD}.{{0,60}}{_NUMBER}|{_NUMBER}.{{0,60}}{_PAY_WORD}", re.IGNORECASE
)
_PAY_AND_WORD_NUMBER_RE = re.compile(
    rf"{_PAY_WORD}.{{0,60}}{_WORD_NUMBER}|{_WORD_NUMBER}.{{0,60}}{_PAY_WORD}", re.IGNORECASE
)
_RATE_RE = re.compile(rf"{_NUMBER}\s*(?:{_CURRENCY}\s*)?{_PERIOD}\b", re.IGNORECASE)
_ON_HAND_RE = re.compile(rf"{_NUMBER}.{{0,30}}(?:на\s+руки|до\s+вычета|gross|net)\b", re.IGNORECASE)
_LONG_RANGE_RE = re.compile(
    r"\b(?:от\s+)?\d{4,7}\s*(?:[-–—]|до)\s*\d{4,7}\b", re.IGNORECASE
)
_BONUS_RE = re.compile(
    rf"(?:преми[яи]|бонус(?:ы|а)?|kpi|процент(?:ы|ов)?|сдельн\w*).{{0,60}}(?:{_NUMBER}|\d+\s*%)|(?:{_NUMBER}|\d+\s*%).{{0,60}}(?:преми[яи]|бонус(?:ы|а)?|kpi|процент(?:ы|ов)?|сдельн\w*)",
    re.IGNORECASE,
)


def _normalise(value: str) -> str:
    """Make typographic variants searchable while retaining readable text."""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _SPACE_RE.sub(" ", value)
    # A period in these abbreviations is not a sentence boundary.  This is
    # important for "5 000 руб. за смену": its rate unit must be removed with
    # the amount rather than left as a separate fragment.
    value = re.sub(r"(?i)\b(руб|тыс|мес|ч)\.", r"\1", value)
    return value.strip()


def _is_salary_segment(segment: str) -> bool:
    """Return whether *segment* contains an explicit or strongly implied pay."""
    return any(
        pattern.search(segment)
        for pattern in (
            _MONEY_WITH_CURRENCY_RE,
            _PAY_AND_NUMBER_RE,
            _PAY_AND_WORD_NUMBER_RE,
            _RATE_RE,
            _ON_HAND_RE,
            _LONG_RANGE_RE,
            _BONUS_RE,
        )
    )


def delete_salary(text: str | None) -> str:
    """Return vacancy text with salary-bearing fragments removed.

    ``None`` and an empty string yield ``""``.  Paragraph breaks between the
    retained fragments are preserved.  The function adds no replacement token:
    a token such as ``[salary removed]`` would itself be a model feature.
    """
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for line in _normalise(str(text)).split("\n"):
        segments = _SEGMENT_RE.split(line)
        kept = [segment for segment in segments if segment.strip() and not _is_salary_segment(segment)]
        cleaned_line = "".join(kept).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)
