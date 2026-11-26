import re

def lemma(token):
    return token.lemma_.lower().replace("ё", "е")


def dependency_path(child, parent, max_steps=4):
    """
    Путь от зависимого слова к предполагаемому
    действию/главному слову.

    None означает, что подходящего пути нет.
    """
    path = []
    current = child

    for _ in range(max_steps):
        if current == parent:
            return path

        if current.head == current:
            return None

        # Не перескакиваем между независимыми
        # частями сложного предложения.
        if current.dep_ in {
            "parataxis",
            "acl:relcl",
            "advcl",
        }:
            return None

        path.append(current.dep_)
        current = current.head

    return path if current == parent else None


def negated_action(action):
    """
    Проверяем отрицание действия, в том числе:
    'поднимать не надо'.

    Это ещё ограниченная модель области отрицания,
    а не универсальный разбор всех конструкций.
    """
    current = action

    for _ in range(3):
        children = list(current.children)

        if any(
            token.lower_ in {"не", "ни"}
            for token in children
        ):
            return True

        parent = current.head

        if parent == current:
            break

        modal_parent = lemma(parent) in {
            "надо",
            "нужно",
            "нужный",
            "необходимо",
            "необходимый",
            "требоваться",
            "должен",
            "должный",
        }

        if modal_parent:
            current = parent
        else:
            break

    return False


def local_modifiers(token):
    """Ближайшие зависимые слова объекта."""
    result = list(token.children)

    for child in list(result):
        if child.dep_ in {"amod", "det", "nmod"}:
            result.extend(child.children)

    return result


def semantic_result(
    value=None,
    status="not_stated",
    evidence=None,
    relation=None,
):
    return {
        "value": value,
        "status": status,
        "evidence": evidence or [],
        "relation": relation or [],
        "method": "dependency_frames",
    }


def resolve_semantic_hits(hits):
    """
    hits:
        value, is_negated, sentence, relation
    """
    positive = {
        value
        for value, negative, _, _ in hits
        if not negative
    }

    negative = {
        value
        for value, is_negative, _, _ in hits
        if is_negative
    }

    evidence = list(dict.fromkeys(
        sentence
        for _, _, sentence, _ in hits
    ))

    relations = list(dict.fromkeys(
        relation
        for _, _, _, relation in hits
    ))

    if positive & negative:
        return semantic_result(
            status="ambiguous",
            evidence=evidence,
            relation=relations,
        )

    if positive:
        return semantic_result(
            value=" | ".join(sorted(positive)),
            status="resolved",
            evidence=evidence,
            relation=relations,
        )

    if negative:
        return semantic_result(
            status="explicit_negative",
            evidence=evidence,
            relation=relations,
        )

    return semantic_result()

WORK_ACTIONS = {
    "работать",
    "работа",
    "доставлять",
    "доставить",
    "доставка",
    "перевозить",
    "перевозка",
    "ездить",
    "рейс",
}

PROVIDE_ACTIONS = {
    "предоставлять",
    "предоставить",
    "выдавать",
    "выдать",
}

CAR_OBJECTS = {
    "авто",
    "автомобиль",
    "машина",
    "грузовик",
}

OWN_MARKERS = {
    "свой",
    "собственный",
    "личный",
}

COMPANY_MARKERS = {
    "компания",
    "работодатель",
    "корпоративный",
}

BENEFIT_ANCHORS = {
    "стоянка",
    "парковка",
    "парковать",
}


def extract_vehicle_arrangement_nlp(doc):
    hits = []

    for sentence in doc.sents:
        actions = [
            token
            for token in sentence
            if lemma(token) in (
                WORK_ACTIONS | PROVIDE_ACTIONS
            )
        ]

        cars = [
            token
            for token in sentence
            if lemma(token) in CAR_OBJECTS
        ]

        for car in cars:
            # Личный автомобиль в контексте стоянки
            # не является рабочим транспортом.
            nearby_ancestors = list(car.ancestors)[:3]

            if any(
                lemma(token) in BENEFIT_ANCHORS
                for token in nearby_ancestors
            ):
                continue

            modifiers = {
                lemma(token)
                for token in local_modifiers(car)
            }

            for action in actions:
                path = dependency_path(
                    car,
                    action,
                    max_steps=3,
                )

                if path is None:
                    continue

                value = None

                if modifiers & OWN_MARKERS:
                    value = "own"

                elif modifiers & COMPANY_MARKERS:
                    value = "company"

                elif lemma(action) in PROVIDE_ACTIONS:
                    value = "company"

                if value is not None:
                    hits.append((
                        value,
                        negated_action(action),
                        sentence.text,
                        (
                            f"{lemma(action)}"
                            f" → {lemma(car)}"
                            f" → {value}"
                        ),
                    ))

    return resolve_semantic_hits(hits)

PICKING_ACTIONS = {
    "собирать",
    "собрать",
    "сборка",
    "комплектовать",
    "комплектация",
    "отбирать",
    "отбор",
}

PICKING_TOOLS = {
    "приложение": "app",
    "тсд": "tsd",
    "накладная": "paper",
}


def extract_picking_technology_nlp(doc):
    hits = []

    for sentence in doc.sents:
        actions = [
            token
            for token in sentence
            if lemma(token) in PICKING_ACTIONS
        ]

        tools = [
            token
            for token in sentence
            if lemma(token) in PICKING_TOOLS
        ]

        for tool in tools:
            for action in actions:
                path = dependency_path(
                    tool,
                    action,
                    max_steps=4,
                )

                if path is None:
                    continue

                # Для этой первой версии учитываем
                # прямое "без ТСД/приложения".
                without_tool = any(
                    child.lower_ == "без"
                    for child in tool.children
                )

                hits.append((
                    PICKING_TOOLS[lemma(tool)],
                    negated_action(action) or without_tool,
                    sentence.text,
                    (
                        f"{lemma(action)}"
                        f" → {lemma(tool)}"
                    ),
                ))

    return resolve_semantic_hits(hits)

MANUAL_LOAD_ACTIONS = {
    "поднимать",
    "поднять",
    "переносить",
    "перенести",
    "подъем",
    "перенос",
}

WEIGHT_PATTERN = re.compile(
    r"(\d+(?:[,.]\d+)?)\s*"
    r"(?:кг\b|килограмм\w*)",
    re.I,
)


def extract_manual_load_nlp(doc):
    positive = []
    negative = []
    evidence = []
    relations = []

    for sentence in doc.sents:
        actions = [
            token
            for token in sentence
            if lemma(token) in MANUAL_LOAD_ACTIONS
        ]

        for match in WEIGHT_PATTERN.finditer(
            sentence.text
        ):
            start = sentence.start_char + match.start()
            end = sentence.start_char + match.end()

            span = doc.char_span(
                start,
                end,
                alignment_mode="expand",
            )

            if span is None:
                continue

            value = float(
                match[1].replace(",", ".")
            )

            for action in actions:
                linked = any(
                    dependency_path(
                        token,
                        action,
                        max_steps=4,
                    ) is not None
                    for token in span
                )

                if not linked:
                    continue

                is_negative = negated_action(action)

                if is_negative:
                    negative.append(value)
                else:
                    positive.append(value)

                evidence.append(sentence.text)
                relations.append(
                    f"{lemma(action)} → {value} кг"
                    f" → negated={is_negative}"
                )

    if positive:
        return semantic_result(
            value=max(positive),
            status="resolved",
            evidence=list(dict.fromkeys(evidence)),
            relation=list(dict.fromkeys(relations)),
        )

    if negative:
        # Важно: отрицание подъёма 100 кг
        # не сообщает положительный допустимый максимум.
        return semantic_result(
            status="explicit_negative",
            evidence=list(dict.fromkeys(evidence)),
            relation=list(dict.fromkeys(relations)),
        )

    return semantic_result()
