# Журнал ML-экспериментов

Одна строка соответствует одному запуску ноутбука. ID ведёт к паспорту прогона. Pinball указан на общих ID при сравнении двух вариантов, охват — по всем тестовым строкам. Время обучения суммировано по фолдам в порядке база / вариант.

[Пайплайн экспериментов](../salary_experiment_pipeline.ipynb) · [Хранение пакетов](../models/README.md)

Пересобрать таблицу из JSON можно из корня проекта командой `python ML/update_experiment_index.py`.

## Сопоставимые прогоны

| ID и паспорт | Изменение | Pinball, общий тест ↓ | Pinball, gold ↓ | Прогнозы на общем тесте | Время по часам, с | Время CPU, с | Пакеты прогона | Архивный бейзлайн |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| <a name="run-2026-11-25_salary_routing_weekly"></a>[`2026-11-25_salary_routing_weekly`](2026-11-25_salary_routing_weekly/experiment.json) | `text_features` → `True` | 0.8098 → 0.8207 | 0.3161 → 0.3035 | 3160/3170 → 3160/3170 | 37.0 / 65.5 | 99.0 / 137.7 | [Пакеты](../models/2026-11-25_salary_routing_weekly) | [Архивная база](../models/2026-11-18_salary_routing/global.joblib) |
| <a name="run-2026-11-25_salary_routing_text_features"></a>[`2026-11-25_salary_routing_text_features`](2026-11-25_salary_routing_text_features/experiment.json) | `text_features` → `True` | 0.8098 → 0.8207 | 0.3161 → 0.3035 | 3160/3170 → 3160/3170 | — | — | [Пакеты](../models/2026-11-25_salary_routing_text_features) | [Архивная база](../models/2026-11-18_salary_routing/global.joblib) |

## Архивные исследования

- [2026-11-18_salary_routing](2026-11-18_salary_routing/experiment.json)
- [2026-11-18_salary_hourly](2026-11-18_salary_hourly/experiment.json)
- [2026-11-18_priority_salary_gold](2026-11-18_priority_salary_gold/experiment.json)
- [2026-11-05_083314_tfidf_catboost_quantile](2026-11-05_083314_tfidf_catboost_quantile/experiment.json)
