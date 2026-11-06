"""Profession of a vacancy from its title.

The scraper labels a vacancy with the search query that found it, so the label
is missing for vacancies from catch-all runs ("all vacancies in Moscow",
remote, watch) and is wrong whenever Avito's search returns neighbouring jobs
(the "кассир" query also returns "Продавец-консультант", the "грузчик" query
returns couriers). The title says what the job is, so it wins when it names
one of the six professions; the query label is only a fallback.

    profession_from_title("Продавец-кассир в магазин у дома")  -> "Продавец"
    resolve_profession("Курьер на своём авто", "Грузчик")      -> ("Курьер", "title")
    resolve_profession("Сотрудник склада", "Грузчик")          -> ("Грузчик", "query")
    resolve_profession("Бариста", "")                          -> ("", "none")

When a title names several professions ("Грузчик-комплектовщик"), the first one
mentioned is taken. Run `python utils/professions.py "title" ...` to try titles.
"""

from __future__ import annotations

import re
import sys

COURIER = "Курьер"
CASHIER = "Кассир"
SELLER = "Продавец"
PICKER = "Комплектовщик"
LOADER = "Грузчик"
TRUCK_DRIVER = "Водитель грузового транспорта"
PROFESSIONS = (COURIER, CASHIER, SELLER, PICKER, LOADER, TRUCK_DRIVER)

_W = r"(?<![а-яa-z0-9])"  # start of a word

PROFESSION_PATTERNS: dict[str, re.Pattern[str]] = {
    COURIER: re.compile(_W + r"(?:курьер|велокурьер|автокурьер|доставщик)"),
    CASHIER: re.compile(_W + r"кассир"),
    SELLER: re.compile(_W + r"(?:продав[еця]|(?:сотрудник|работник|специалист)\w* торгового зала)"),
    # Warehouse order picking and packing. A bare "сборщик" is usually factory
    # assembly (furniture, drones), a bare "кладовщик" is a storekeeper: both
    # are left to the query label.
    PICKER: re.compile(
        _W + r"(?:комплектовщ|упаковщ|сортировщ|стикеровщ|отборщ"
        r"|сборщик\w*[\s\-/]+(?:\w+[\s\-/]+)?(?:заказ|товар|на склад|склад|интернет|онлайн))"
    ),
    # "погрузчик" contains "грузчик": the word-start guard keeps forklift
    # operators out.
    LOADER: re.compile(_W + r"(?:грузчик|такелажник)"),
}

# A driver counts as a truck driver only with an explicit truck marker in the
# title: licence category C/E, a truck type or a tonnage. "Водитель категории В",
# taxi, bus and "водитель-экспедитор на своём авто" fall back to the query label.
DRIVER_RE = re.compile(_W + r"(?:водител|дальнобойщик|шофер|шофёр)")
TRUCK_MARKER_RE = re.compile(
    r"(?:кат\w*\.?\s*[«\"]?\s*(?:[ве]\s*[,/и+]?\s*)?[сcе]\b|\b[сc]\s*[,/+]\s*е\b|\bсе\b|\bce\b"
    r"|грузов|газел|фур[аеуыо]?\b|тягач|самосвал|манипулятор|рефрижератор|\bреф\b|бортов|еврофур|дальнобо"
    r"|миксер|бетоновоз|лесовоз|эвакуатор|\d+(?:[.,]\d+)?\s*(?:т\b|тонн|-?ти\s*тонн))"
)
NOT_TRUCK_RE = re.compile(r"погрузчик|штабел|электрокар|такси|автобус|трамва|троллейбус|личн\w* водител|персональн\w* водител")


def _normalise(title: str) -> str:
    text = (title or "").casefold().replace("ё", "е")
    # Avito titles often mix Latin look-alikes into Cyrillic words to dodge
    # moderation ("Kурьер", "Пятёpoчкa"); map them back.
    return text.translate(str.maketrans("aceopxyk", "асеорхук"))


def _truck_driver_position(text: str) -> int | None:
    match = DRIVER_RE.search(text)
    if match is None or NOT_TRUCK_RE.search(text) or not TRUCK_MARKER_RE.search(text):
        return None
    return match.start()


def profession_from_title(title: str | None) -> str:
    """One of PROFESSIONS, or '' when the title names none of them."""
    text = _normalise(title or "")
    if not text:
        return ""
    found: list[tuple[int, str]] = []
    for profession, pattern in PROFESSION_PATTERNS.items():
        match = pattern.search(text)
        if match:
            found.append((match.start(), profession))
    position = _truck_driver_position(text)
    if position is not None:
        found.append((position, TRUCK_DRIVER))
    return min(found)[1] if found else ""


def resolve_profession(title: str | None, query_label: str | None) -> tuple[str, str]:
    """(profession, profession_source): 'title' when the title names one of the
    six professions, 'query' when only the scraper's search label is there,
    'none' when neither."""
    from_title = profession_from_title(title)
    if from_title:
        return from_title, "title"
    label = (query_label or "").strip()
    if label:
        return label, "query"
    return "", "none"


if __name__ == "__main__":
    for argument in sys.argv[1:] or [line.strip() for line in sys.stdin if line.strip()]:
        print(f"{profession_from_title(argument) or '—':32s} {argument}")
