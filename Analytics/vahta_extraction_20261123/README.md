# Вахта: правила Кости и сравнение NLP

Результат исследования на 23.11.2026. Основной вывод и границы: [REPORT.md](REPORT.md). Вопросы команде: [TEAM_SUMMARY.md](TEAM_SUMMARY.md). Проверка критериев: [CRITERIA.md](CRITERIA.md).

## Артефакты

- `results/vacancies_decisions.csv`: одна строка на каждое из 7 193 объявлений; прежние 7 189 и новые четыре разделены колонкой `cohort`. Условия, статусы, причины исключения, источник, три уровня доказательности ставки. `publication_city_NOT_workplace` не является местом работы.
- `results/kostya_features7193.csv`: все 27 исходных признаков, включая статусы и доказательства; 163 столбца вместе с ID.
- `results/subsegments28_before_after.csv` и `results/passports28.csv`: 28 пересекающихся групп предыдущего исследования; второй файл добавляет распределения условий, пробелы и ID для адресной проверки. Объёмы нельзя суммировать.
- `results/condition_cells.csv`: 13 условных ячеек с текстовым местом, часами и ставкой. Не подтверждённые ML-сегменты.
- `results/pavel_*_strict.csv`, `pavel_coverage.csv`: технические эквиваленты и сравнение сценариев нормализатора; не реальные выплаты.
- `results/discrepancy_examples8.csv`: восемь конфликтных случаев с ID и цитатами; `review_corrections.csv` показывает ещё 11 случаев, проверенных при исправлении дополнений 1.1.
- `results/nlp_summary.csv`, `nlp_disagreements.csv`: покрытие и 625 расхождений; `manual_snippet_evaluation36.csv` и `manual_snippet_reference36.json`: диагностическая разметка/сопоставление 36 фрагментов. Это не precision/recall всех объявлений.
- `results/source_dependence_groups.csv`: консервативные группы зависимости, не юридически независимые работодатели.
- `results/new4_separate.csv`: отдельная проверка четырёх новых строк.
- `provenance.json`, `source_preflight.json`, `manifest.json`: версии, SHA, контроль входов и выходов.

## Источники

Базовый снимок 31 610 строк: commit `7dc8e62cbc141972333c99fcc1ccb9f466139211`, SHA-256 `7831e53ccd38c3c85e33c7e0bb3b8a521db58292e90349ebaae9cc4b025f873f`.
Текущий закреплённый снимок 32 097 строк: commit `d1f371ca6`, SHA-256 `a56e81b2b7f524da57ca93ce4c6fe984dae950d67ec98b9f37429e5fde76983e`.
Предыдущий inventory и словарь: commit `bd5efad0372520481b33395411b8c5160f7be323`, каталог `Analytics/vahta_readiness_20261117/`. Этот commit используется явно: наличие PR178 не означает его слияния в main.

Правила: [исследование Кости](https://buildin.ai/b6a54df3-6d88-4456-b7dc-13d3344ba2ee), [исходный Colab](https://colab.research.google.com/drive/1XHjKX3oOjG7tIivBGhg81VAGOKMsWEG5?usp=sharing). SHA скачанного notebook: `f34a68d5aed1e117107447d111ec99415cfec77af7648abded8ff06bac379d92`. Базовые определения и NLP перенесены без изменения; вахтовые дополнения находятся отдельно.

## Воспроизведение

Python 3.13, версии зависимостей в `requirements.lock.txt`. Модель ru_core_news_sm 3.8.0 устанавливается из официального релиза spaCy models; `requirements.txt` фиксирует используемые версии. Исходные данные не переписываются. Для воспроизведения закреплённых входов создайте отдельные checkout нужных commit и выполните `python3 utils/build_dataset.py --materialize` в каждом. Для воспроизводимого запуска используйте поставляемые `inputs/inventory7189.csv` и `inputs/keyword_dictionary.csv`: это минимальная проекция исходного keyword-inventory с нужной схемой и словарь. Опубликованная ранее readiness-таблица имеет другую схему и не подставляется напрямую. Сверьте SHA входов из `provenance.json` до запуска.

Команды из корня репозитория; INVENTORY_CSV и DICTIONARY_CSV указывают на два файла inputs выше; остальные переменные BASE_CSV, CURRENT_CSV, INVENTORY_CSV, DICTIONARY_CSV указывают на закреплённые входы, STUDY_OUT на новый каталог результатов:

```sh
python -m unittest discover -s Analytics/vahta_extraction_20261123 -p test_rules.py -v
python Analytics/vahta_extraction_20261123/run_study.py --base "$BASE_CSV" --current "$CURRENT_CSV" --inventory "$INVENTORY_CSV" --out "$STUDY_OUT"
python Analytics/vahta_extraction_20261123/summarize_study.py --base "$BASE_CSV" --current "$CURRENT_CSV" --inventory "$INVENTORY_CSV" --dictionary "$DICTIONARY_CSV" --study "$STUDY_OUT"
python Analytics/vahta_extraction_20261123/source_groups.py --base "$BASE_CSV" --current "$CURRENT_CSV" --inventory "$INVENTORY_CSV" --study "$STUDY_OUT"
python Analytics/vahta_extraction_20261123/build_passports.py --results "$STUDY_OUT" --inventory "$INVENTORY_CSV"
python Analytics/vahta_extraction_20261123/evaluate_nlp.py
```

Последняя команда обновляет только поставляемую диагностику 36 фрагментов в этом каталоге; остальные пишут в STUDY_OUT. Полная NLP-таблица из 21 579 сравнений и локальный кэш создаются при запуске, в поставку включены сводка и расхождения. Кэш переиспользовать только при неизменных коде, нормализации и версии модели; для нового эксперимента используйте пустой каталог.

Проверка целостности поставки: `python3 Analytics/vahta_extraction_20261123/validate_artifacts.py`.

Проверены: исходные 59 случаев Кости, ещё 32 сценария дополнений (23 unittest-метода); целостность ID/когорт, количества и вложенность уровней 199/107/2, 625 расхождений, 36 фрагментов, совпадение числовых выходов с расчётными CSV. Ручная проверка всех описаний, подтверждение выплат, юрлиц и фактических объектов не выполнялись. Эти ограничения сохранены в таблицах.
