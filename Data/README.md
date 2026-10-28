# Данные

Основной файл для всей команды — [общий датасет](dataset/vacancies.csv).
Сейчас он собран из семи проверенных выгрузок: 267 вакансий, 73 колонки.
Новые принятые сборы добавляются через [реестр источников](dataset/sources.json)
и `python3 utils/build_dataset.py`. Дубли по площадке и ID дают одну строку
с последним наблюдением. Город и регион означают размещение на сайте.
[Схема, обогащение и обновление через PR](dataset/README.md).

Каждый сбор хранится в своей папке `avito_runs/<название_сбора>/`.
В `exports/` лежат общие таблицы и сводки, в `raw/` — отдельные исходные
вакансии этого сбора. Для начала анализа достаточно готового CSV.

## Каталог выгрузок

| Сбор | Что собрали | Таблица |
|---|---|---|
| [21 октября 2026](avito_runs/first_export_2026-10-21/) | 6 профессий, Москва/СПб/Екатеринбург; 238 уникальных вакансий | [CSV для pandas](avito_runs/first_export_2026-10-21/exports/vacancies_flat.csv) |
| [26 октября 2026](avito_runs/courier_first5_2026-10-26/) | Первые 5 уникальных курьерских вакансий Москвы, все продвинутые; эксперимент объединения | [CSV для pandas](avito_runs/courier_first5_2026-10-26/exports/vacancies_flat.csv) |
| [26 октября, второй батч](avito_runs/courier_next5_2026-10-26/) | Ещё 5 курьерских вакансий, ранее показанных локально; опубликованы 27 октября | [CSV](avito_runs/courier_next5_2026-10-26/exports/vacancies_flat.csv) |
| [27 октября, курьеры](avito_runs/courier_first5_2026-10-27/) | 5 новых ID Москвы; скрейпинг карточек 9,624 с | [CSV](avito_runs/courier_first5_2026-10-27/exports/vacancies_flat.csv) |
| [27 октября, общая выдача](avito_runs/all_professions_first5_2026-10-27/) | 5 вакансий Москвы без фильтра профессии; скрейпинг карточек 8,343 с | [CSV](avito_runs/all_professions_first5_2026-10-27/exports/vacancies_flat.csv) |
| [27 октября, вахта](avito_runs/vahta_first5_2026-10-27/) | Первые 5 по всем регионам, без фильтра профессии; все продвинутые | [CSV](avito_runs/vahta_first5_2026-10-27/exports/vacancies_flat.csv) |
| [27 октября, гибкий график](avito_runs/flexible_all_regions_first5_2026-10-27/) | Первые 5 по всем регионам, без фильтра профессии; Москва, Троицк и Тверь; все продвинутые | [CSV](avito_runs/flexible_all_regions_first5_2026-10-27/exports/vacancies_flat.csv) |

Новый сбор 27 октября: [первые 30 вакансий по всем регионам](avito_runs/all_regions_first30_2026-10-27/exports/vacancies_flat.csv), без фильтров, в порядке первой страницы; все продвинутые. 30 строк, 62 колонки.

Ссылка на новый набор и краткое описание добавляются агентом после сбора
по инструкции collect-avito. Отдельного генератора каталога пока нет.

## Открыть данные

Из корня репозитория:

```python
import pandas as pd

df = pd.read_csv(
    "Data/dataset/vacancies.csv",
    dtype={"id": "string"},
)
df.head()
```

Примеры: [Analytics](../Analytics/00_load_data.ipynb),
[ML](../ML/00_load_first_export.ipynb). Оба направления читают одни данные.

В исходном плоском CSV первой выгрузки одна строка — одна уникальная вакансия, 55 колонок.
Здесь есть ID, заголовок, ссылка, время наблюдения, город, профессия,
зарплатная строка, описание и 32 параметра карточек в отдельных колонках.
Пустое значение означает отсутствие сохранённых сведений, а не ноль.

В `profession` и `segments` может быть несколько значений через `; `:
две вакансии включены одновременно в «Продавец» и «Кассир».
Поэтому уникальных вакансий 238, а включений в срезы — 240.
Зарплаты сохранены с разными периодами оплаты; таблица не является
готовой разметкой рыночной зарплаты.

## Как устроена папка сбора

```text
avito_runs/<название_сбора>/
├── raw/             Каждая прочитанная вакансия отдельным JSON-объектом
├── cards/           Карточки после проверки и уточнения значений
└── exports/         Вакансии, объединённые в таблицы, и сводки по выборке
```

Снимки страниц и скриншоты, если сохранены, помогают проверить спорное значение.
Состав отдельных сборов может различаться.

## Что лежит в exports первой выгрузки

| Файл | Содержимое |
|---|---|
| `vacancies_flat.csv` | Готовая таблица для pandas: параметры в отдельных колонках |
| `vacancies.csv` | Подробная общая таблица сборщика; часть сложных значений хранится как JSON внутри ячеек |
| `vacancies.json` | Те же вакансии как объекты со вложенными полями; удобны для программной обработки |
| `segment-overview.csv` | Объём и сводка по каждому срезу; `fields_min/median/max` — минимум, медиана и максимум числа наблюдавшихся полей в карточке |
| `field-coverage.csv` | Какие поля встретились и были заполнены в каждом срезе |
| `condition-field-coverage.csv` | Наличие полей по профессии, графику и занятости |
| `dictionary-by-profession.json` | Наблюдавшиеся характеристики по профессиям; это словарь полей, а не ещё одна копия вакансий |
| `observations.csv` | Очередь отбора: ID, срез, время, ранг и результат обработки, включая резерв и технические записи |

`slice_id` — технический ключ среза для связи таблиц. Для просмотра используйте
названия профессии и города. Очередь observations не является дополнительной
обучающей выборкой.

Покрытие полей показывает заполненность **собранных карточек**, а не долю рынка
или точность модели. Эти сводки рассчитаны для первой выгрузки; автоматического
универсального расчёта для новых схем пока нет.

## Служебные файлы агента

Они находятся в [Scraper/runs/first_export_2026-10-21/](../Scraper/runs/first_export_2026-10-21/):
`plan.json` — что планировали собрать; `manifest.json` — описание результата,
ссылки на файлы и контрольные суммы. «Паспорт» в старых инструкциях означает
это описание набора. Для открытия CSV оно не требуется.

[Шаблон описания нового набора](../Scraper/templates/manifest.template.json)
лежит в Scraper/templates. Технические журналы и замеры остаются локальными
в Scraper/runs/<название_сбора>/diagnostics и не входят в общий экспорт.

Таблицы можно заново подготовить из сохранённых файлов через
[utils/flatten_export.py](../utils/flatten_export.py), без открытия сайта.
Пример полной проверки происхождения — [отдельный ноутбук](../ML/01_load_with_passport.ipynb).

- [truck_driver_ekaterinburg_first40_2026-10-27](avito_runs/truck_driver_ekaterinburg_first40_2026-10-27/exports/vacancies_flat.csv): 40 вакансий.

- [courier_ekaterinburg_first40_benchmark_2026-10-27](avito_runs/courier_ekaterinburg_first40_benchmark_2026-10-27/exports/vacancies_flat.csv): 40 вакансий.

- [first5_2026-10-28_01](avito_runs/first5_2026-10-28_01/exports/vacancies_flat.csv): 5 вакансий.

- [general_first5_2026-10-28](avito_runs/general_first5_2026-10-28/exports/vacancies_flat.csv): 5 вакансий.

- [flexible_first5_2026-10-28](avito_runs/flexible_first5_2026-10-28/exports/vacancies_flat.csv): 5 вакансий.

- [remote10-verified-20261028-demo4](avito_runs/remote10-verified-20261028-demo4/exports/vacancies_flat.csv): 10 вакансий.

- [moscow_all_200_2026-10-28](avito_runs/moscow_all_200_2026-10-28/exports/vacancies_flat.csv): 200 вакансий.

- [ekaterinburg_all_100_2026-10-28](avito_runs/ekaterinburg_all_100_2026-10-28/exports/vacancies_flat.csv): 100 вакансий.

- [20261028-spb-50-all](avito_runs/20261028-spb-50-all/exports/vacancies_flat.csv): 50 вакансий.

- [moscow-all-10000-20261028](avito_runs/moscow-all-10000-20261028/exports/vacancies_flat.csv): 1397 вакансий.

- [spb_all_first50_2026-10-28](avito_runs/spb_all_first50_2026-10-28/exports/vacancies_flat.csv): 50 вакансий.

- [spb_all_first1500_2026-10-28](avito_runs/spb_all_first1500_2026-10-28/exports/vacancies_flat.csv): 1459 вакансий.

- [moscow_loader_20261028_batch_01](avito_runs/moscow_loader_20261028_batch_01/exports/vacancies_flat.csv): 295 вакансий.

- [moscow_loader_20261028_batch_02](avito_runs/moscow_loader_20261028_batch_02/exports/vacancies_flat.csv): 278 вакансий.

- [spb_gruzchik_20261028_b_01](avito_runs/spb_gruzchik_20261028_b_01/exports/vacancies_flat.csv): 299 вакансий.

- [spb_gruzchik_20261028_b_02](avito_runs/spb_gruzchik_20261028_b_02/exports/vacancies_flat.csv): 258 вакансий.

- [spb_gruzchik_20261028_tail_01](avito_runs/spb_gruzchik_20261028_tail_01/exports/vacancies_flat.csv): 30 вакансий.

- [ekb_loader_20261029_batch_01](avito_runs/ekb_loader_20261029_batch_01/exports/vacancies_flat.csv): 300 вакансий.

- [ekb_loader_20261029_batch_02](avito_runs/ekb_loader_20261029_batch_02/exports/vacancies_flat.csv): 52 вакансий.

- [moscow_picker_20261029_b_01](avito_runs/moscow_picker_20261029_b_01/exports/vacancies_flat.csv): 299 вакансий.

- [moscow_picker_20261029_b_02](avito_runs/moscow_picker_20261029_b_02/exports/vacancies_flat.csv): 293 вакансий.
