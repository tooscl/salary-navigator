# Добор через независимые ветки фильтров

Sweep-режим обходит ограничение наблюдаемой выдачи, не пытаясь строить скрытые
URL и не перебирая декартово произведение фильтров. Каждая ветка начинается от
одного подтверждённого базового среза, добавляет ровно одно значение, сохраняет
до 30 страниц, после чего UI возвращается к базе. Карточки читаются один раз —
после объединения всех веток по `(source, id)`.

## Режимы входа

- `profession_regions`: профессия и прочие исходные ограничения фиксированы;
  список регионов передаёт участник.
- `region_professions`: регион и прочие исходные ограничения фиксированы;
  список профессий передаёт участник.
- `constraint_regions`: профессия не задана, но формат работы или другой
  исходный constraint фиксирован; список регионов передаёт участник.
- `city_profession`: город и профессия фиксированы; сначала сохраняется база,
  затем все доступные дополнительные значения по одному. Зависимое значение
  содержит минимальный `parent_context`; независимые фильтры не накапливаются.

В основном сценарии каталог включает все реально показанные группы, в частности
смены, график и «Можно заканчивать в 17:00», если они доступны для базового
среза. Отсутствующее значение получает `unavailable` с наблюдаемой причиной, а
не добавляется по предположению.

Формат работы, поиск, зарплата и другие исходные ограничения входят в `base` и
повторяются в `expected_selected_controls` каждой ветки. Числовой фильтр должен
быть заранее разбит на конечные `numeric_range` с minimum/maximum/unit.

## Спецификация

Создайте локальный JSON. Списки регионов/профессий приходят из задания; runner
не выбирает их сам. Для `city_profession` каталог получают через `/browse`:
открывают панель фильтров, раскрывают группы и сохраняют snapshot. Недоступные и
пропущенные значения получают status и непустую reason.

```json
{
  "schema_version": 1,
  "entry_mode": "city_profession",
  "max_pages": 30,
  "base": {
    "geography": "Москва",
    "profession": "Курьер",
    "constraints": [{"name": "Формат работы", "value": "Удалённо"}]
  },
  "routing": {},
  "filter_catalog": [{
    "name": "Смена",
    "values": [
      {"value": "Дневная", "status": "planned", "branch_id": "shift_day"},
      {"value": "Ночная", "status": "unavailable", "reason": "не показана в базовом срезе"}
    ]
  }],
  "branches": [
    {
      "id": "base",
      "kind": "base",
      "expected_selected_controls": [],
      "expected_slice": {"geography": "Москва", "profession": "Курьер"}
    },
    {
      "id": "shift_day",
      "kind": "additional",
      "additional_filter": {"name": "Смена", "value": "Дневная"},
      "parent_context": [],
      "expected_selected_controls": [],
      "expected_slice": {"geography": "Москва", "profession": "Курьер"}
    }
  ]
}
```

В реальном плане `expected_selected_controls` — точные объекты
`label/marker/value` из snapshot, а не пустые списки примера. Для зависимого
значения перечислите только обязательные родительские controls в
`parent_context`; комбинации с другими независимыми фильтрами запрещены.

## Запуск

```powershell
python -X utf8 Scraper/skills/collect-avito/scripts/run.py sweep-init `
  --run-id <sweep_id> --spec <plan.json>

# Необязательный каталог/доказательство возврата: панель уже открыта в своей вкладке.
python -X utf8 Scraper/skills/collect-avito/scripts/run.py sweep-snapshot `
  --run-id <sweep_id> --name catalog --url '<observed URL>' `
  --tab-id <targetId> --config <collector.local.json>

# База или одна ветка. Фильтры применены и подтверждены через /browse.
python -X utf8 Scraper/skills/collect-avito/scripts/run.py freeze `
  --run-id <branch_run_id> --url '<observed URL>' --all-pages `
  --branch-only --max-pages 30 --geography '<география>' `
  --profession '<профессия>' --segments '<исходные ограничения>' `
  --filter-evidence '<видимые controls>' --agent-model gpt-6.1-terra

# После небазовой ветки снять значение, подтвердить базовые controls и сохранить snapshot reset_<branch>.
python -X utf8 Scraper/skills/collect-avito/scripts/run.py sweep-register-branch `
  --run-id <sweep_id> --branch-id <branch_id> --branch-run-id <branch_run_id> `
  --reset-evidence Scraper/sweeps/<sweep_id>/observations/reset_<branch>.json
```

Базовая ветка не требует `--reset-evidence`. Для недоступного/пропущенного
значения или ошибки запишите статус, не создавая фиктивную пустую выдачу:

```powershell
python -X utf8 Scraper/skills/collect-avito/scripts/run.py sweep-set-branch-status `
  --run-id <sweep_id> --branch-id <branch_id> `
  --status unavailable --reason '<наблюдаемая причина>'
```

`EMPTY_RESULT` принимается только по явному тексту пустой выдачи. Отсутствие
Next, timeout, CAPTCHA, 429 и ошибка навигации нулевым результатом не являются.
Повтор Next фиксируется как частичное покрытие. На 30-й странице ветка завершается
`page_cap_30`, даже если сайт показывает дальнейший переход.

Когда каждая ветка имеет терминальный статус:

```powershell
python -X utf8 Scraper/skills/collect-avito/scripts/run.py sweep-report --run-id <sweep_id>
python -X utf8 Scraper/skills/collect-avito/scripts/run.py sweep-materialize `
  --run-id <sweep_id> --capture-run-id <capture_run_id>
python -X utf8 Scraper/skills/collect-avito/scripts/run.py capture `
  --run-id <capture_run_id> --batch-size 30
```

`--allow-partial` у materialize нужен только при явно принятом частичном sweep с
ветками `error`; их причины остаются в отчёте. После capture действуют обычные
finalize/publish и все остановки collect-avito.

## Артефакты и resume

- `Scraper/sweeps/<sweep_id>/plan.json` — неизменный план;
- `Scraper/sweeps/<sweep_id>/checkpoint.json` — статус каждой ветки и следующая;
- `Data/avito_sweeps/<sweep_id>/coverage.{json,csv,md}` — страницы, найденные,
  новые ID, пересечения, причины конца и статусы;
- `Data/avito_sweeps/<sweep_id>/unique_queue.json` — одна строка на ID;
- `memberships.json` — все ветки обнаружения;
- `Data/avito_runs/<capture_run_id>/sweep-provenance/` — закреплённые хеши.

Повторный запуск читает checkpoint. Зарегистрированную ветку не перезаписывают;
ошибку продолжают в её branch run штатным `freeze --resume`. Материализованная
очередь имеет один capture checkpoint. В CSV добавляются `discovery_slices`,
`discovery_branch_count`, `discovery_memberships_json`, первый фильтр/значение,
страница и позиция. Исходные `source_url`, observed_at, promoted, просмотры,
описание и характеристики сохраняются обычным экспортом.

Для быстрой проверки схемы на уже финализированном источнике:

```powershell
python -X utf8 Scraper/skills/collect-avito/scripts/run.py sweep-demo-export `
  --run-id <sweep_id> --source-run-id <finalized_run_id> --limit 100
```

`--limit` необязателен: без него выгружаются все доступные строки, а с ним
создаётся детерминированный review-sized образец. Demo CSV содержит только уже
прочитанные карточки, чьи ID есть в sweep; он не подменяет полный capture и явно
ссылается на исходный run и размер образца в validation JSON.
