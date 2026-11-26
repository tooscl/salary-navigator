# Вахта: итог по шести критериям

**24.11.2026. Подготовка сегментов с учётом извлечения Кости.**

[Задача Buildin](https://buildin.ai/83ea70d5-0451-4a60-8118-7fc0d7c838b7) · [Полный отчёт по критериям](REPORT.md) · [Паспорта и сегменты](SEGMENTS.md) · [Вопросы команде](TEAM_SUMMARY.md)

7 189 прежних + 4 новых объявления проверены раздельно. 199 технических кандидатов Павла, 107 с прямой текстовой связью суммы/периода, 9 объявлений с явно оплачиваемыми часами во всей прежней когорте. Независимость работодателей не установлена. Критерий 1 закрыт частично; условия по остальным размечены с явно описанными ограничениями.

| Артефакт | Содержание |
|---|---|
| [REPORT.md](REPORT.md) | Матрица шести критериев, 10 примеров, границы пересчёта |
| [SEGMENTS.md](SEGMENTS.md) | 28 групп и четыре складских среза с допущениями |
| [vacancy_conditions7193.csv](results/vacancy_conditions7193.csv) | ID, цитаты, условия, статусы, причины по всем объявлениям |
| [subsegments28_criteria.csv](results/subsegments28_criteria.csv) | Фильтр, объём, ставки, источники, признаки и условия по группе |
| [addressed_collection28.csv](results/addressed_collection28.csv) | Адресная очередь проверки и недостающие параметры |
| [working_day_candidates_review.csv](results/working_day_candidates_review.csv) | 36 кандидатов до проверки; итог ручной проверки — inputs/working_day_review36.csv |
| [prepared_segments4.csv](results/prepared_segments4.csv) | Количества и полнота условий четырёх подготовленных срезов |
| [working_day_review36.csv](inputs/working_day_review36.csv) | Проверенные цитаты часов: 35 без обнаруженного конфликта, 1 с конфликтом |
| [pavel_working_day_sensitivity.csv](results/pavel_working_day_sensitivity.csv) | Ещё 4 технических расчёта при допущении «рабочий день = смена», отдельно от исходных 199 |
| [examples10_by_criteria.csv](results/examples10_by_criteria.csv) | Проверочные примеры с ID и цитатами |
| [new4_conditions.csv](results/new4_conditions.csv) | Отдельная проверка новых объявлений |

Сборка: Python с pandas и зависимостями исходного пакета; из корня репозитория:

```sh
python Analytics/vahta_task_readiness_20261124/build_summary.py --raw-csv /absolute/path/to/pinned/vacancies.csv
```

Сборщик проверяет SHA полного снимка 32 097 строк и соответствие исходному извлечению. Источники закреплены в [provenance.json](provenance.json). Модели не запускаются. Артефакты предыдущего модельного эксперимента не служат доказательством выполнения этих критериев.
