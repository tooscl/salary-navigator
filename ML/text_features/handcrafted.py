"""Interpretable text features: keyword flags and simple text statistics.

Computed on strictly redacted text (no amounts), so a flag can say "weekly
payouts" or "housing provided", but never how much.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .redaction import mixed_script_share

FLAGS = {
    "kw_housing": r"проживани|общежити|жиль[ёео]|хостел",
    "kw_weekly_pay": r"еженедельн\w* (?:выплат|оплат)|выплат\w* (?:раз в неделю|еженедельно|каждую неделю)",
    "kw_daily_pay": r"ежедневн\w* (?:выплат|оплат)|выплат\w* (?:каждый день|ежедневно)|оплата каждый день",
    "kw_advance": r"аванс",
    "kw_tips": r"чаев",
    "kw_bonus": r"преми|бонус",
    "kw_percent_sales": r"процент\w* от|% от (?:продаж|выручк)",
    "kw_piecework": r"сдельн",
    "kw_official": r"по тк\b|трудов\w* кодекс|официальн\w* (?:трудоустр|оформл)",
    "kw_selfemployed": r"самозанят",
    "kw_no_experience": r"без опыта|опыт не (?:обязателен|требуется|нужен)",
    "kw_training": r"обучени|обучим",
    "kw_night": r"ночн",
    "kw_12h": r"12[\s-]*час|по 12\b",
    "kw_2_2": r"2/2|2 через 2|два через два",
    "kw_5_2": r"5/2|пятидневк",
    "kw_6_1": r"6/1|6 через 1",
    "kw_food": r"питани|обед",
    "kw_transport": r"развоз|корпоративн\w* транспорт|доставк\w* до (?:места )?работ",
    "kw_uniform": r"спецодежд|униформ",
    "kw_dms": r"\bдмс\b|медицинск\w* страхов",
    "kw_discount": r"скидк",
    "kw_own_car": r"(?:сво[её]м|личн\w*) (?:авто|автомобил|транспорт)",
    "kw_category_ce": r"категори\w* [сc]?\s?е\b|\bce\b|\bсе\b|кат\.? ?е\b",
    "kw_truck_big": r"\bфур|тягач|полуприцеп|еврофур",
    "kw_gazelle": r"газел",
    "kw_students": r"студент",
    "kw_foreign": r"иностран|\bснг\b|патент|мигрант",
    "kw_agency": r"аутстафф|аутсорс|кадров\w* агентств",
    "kw_urgent": r"срочно",
    "kw_big_company": r"крупн\w* (?:компан|сет)|федеральн\w* сет",
    "kw_part_time": r"подработ|частичн",
    "kw_senior": r"старш|бригадир|руковод|наставн",
    "kw_scanner": r"\bтсд\b|терминал сбора|штрих|\bwms\b",
    "kw_marketplace": r"озон|ozon|wildberries|вайлдберриз|яндекс",
    "kw_grocery_chain": r"пят[её]рочк|перекр[её]ст|магнит|\bx5\b|дикси|вкусвилл",
}
_COMPILED = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in FLAGS.items()}
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅]")
_BULLET = re.compile(r"^\s*(?:[-•*—–]|\d+[.)]|[\U0001F300-\U0001FAFF☀-➿✅])", re.MULTILINE)


def benefit_vocabulary(benefits: pd.Series, top_k: int = 15) -> list[str]:
    """Most common perks of the structured list "Что получают работники"."""
    items = benefits.dropna().astype(str).str.split(r"\s*,\s*").explode().str.strip()
    return items[items != ""].value_counts().head(top_k).index.tolist()


def handcrafted_features(
    title: pd.Series, description: pd.Series, raw_description: pd.Series, benefits: pd.Series, vocabulary: list[str]
) -> pd.DataFrame:
    """One row per vacancy. title/description must already be redacted."""
    text = (title.fillna("") + "\n" + description.fillna("")).str.lower()
    out = {name: text.str.contains(pattern).astype(np.int8) for name, pattern in _COMPILED.items()}
    out["txt_len_log"] = np.log1p(description.fillna("").str.len())
    out["txt_lines"] = description.fillna("").str.count("\n") + 1
    out["txt_bullets"] = description.fillna("").map(lambda t: len(_BULLET.findall(t)))
    out["txt_emoji"] = description.fillna("").map(lambda t: len(_EMOJI.findall(t)))
    out["txt_exclamations"] = description.fillna("").str.count("!")
    out["txt_mixed_script_share"] = raw_description.fillna("").map(mixed_script_share)
    items = benefits.fillna("").astype(str).str.split(r"\s*,\s*")
    out["benefit_count"] = items.map(lambda xs: sum(1 for x in xs if x.strip()))
    for i, perk in enumerate(vocabulary):
        out[f"benefit_{i:02d}"] = items.map(lambda xs, p=perk: int(p in [x.strip() for x in xs])).astype(np.int8)
    return pd.DataFrame(out, index=title.index)
