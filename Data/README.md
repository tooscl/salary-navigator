# Данные

Основной файл для всей команды — общий датасет `dataset/vacancies.csv`.
Восстановите его из сохранённых частей командой `python3 utils/build_dataset.py --materialize`
из корня проекта. Сейчас он содержит 31 543 вакансии и 131 колонку из 137 выгрузок.
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

- [moscow_picker_20261029_b_03](avito_runs/moscow_picker_20261029_b_03/exports/vacancies_flat.csv): 63 вакансий.

- [spb_picker_20261029_b_01](avito_runs/spb_picker_20261029_b_01/exports/vacancies_flat.csv): 300 вакансий.

- [spb_picker_20261029_b_02](avito_runs/spb_picker_20261029_b_02/exports/vacancies_flat.csv): 196 вакансий.

- [avito_f06_ekb_picker_20261029_b_01](avito_runs/avito_f06_ekb_picker_20261029_b_01/exports/vacancies_flat.csv): 298 вакансий.

- [avito_f07_moscow_prodavets_20261029_b_01](avito_runs/avito_f07_moscow_prodavets_20261029_b_01/exports/vacancies_flat.csv): 300 вакансий.

- [avito_f07_moscow_prodavets_20261029_b_02](avito_runs/avito_f07_moscow_prodavets_20261029_b_02/exports/vacancies_flat.csv): 300 вакансий.

- [avito_f07_moscow_prodavets_20261029_b_03](avito_runs/avito_f07_moscow_prodavets_20261029_b_03/exports/vacancies_flat.csv): 300 вакансий.

- [avito_f07_moscow_prodavets_20261029_b_04](avito_runs/avito_f07_moscow_prodavets_20261029_b_04/exports/vacancies_flat.csv): 45 вакансий.

- [f08_spb_seller_20261029_b_01](avito_runs/f08_spb_seller_20261029_b_01/exports/vacancies_flat.csv): 299 вакансий.

- [f08_spb_seller_20261029_b_02](avito_runs/f08_spb_seller_20261029_b_02/exports/vacancies_flat.csv): 298 вакансий.

- [f08_spb_seller_20261029_b_03](avito_runs/f08_spb_seller_20261029_b_03/exports/vacancies_flat.csv): 300 вакансий.

- [f08_spb_seller_20261029_b_04](avito_runs/f08_spb_seller_20261029_b_04/exports/vacancies_flat.csv): 65 вакансий.

- [filter09_ekaterinburg_prodavets_20261029_b_01](avito_runs/filter09_ekaterinburg_prodavets_20261029_b_01/exports/vacancies_flat.csv): 300 вакансий.

- [filter09_ekaterinburg_prodavets_20261029_b_02](avito_runs/filter09_ekaterinburg_prodavets_20261029_b_02/exports/vacancies_flat.csv): 218 вакансий.

- [avito_10_moscow_courier_20261029_01](avito_runs/avito_10_moscow_courier_20261029_01/exports/vacancies_flat.csv): 300 вакансий.

- [avito_10_moscow_courier_20261029_02](avito_runs/avito_10_moscow_courier_20261029_02/exports/vacancies_flat.csv): 299 вакансий.

- [avito_10_moscow_courier_20261029_03](avito_runs/avito_10_moscow_courier_20261029_03/exports/vacancies_flat.csv): 178 вакансий.

- [avito_f11_spb_courier_20261029_part_01](avito_runs/avito_f11_spb_courier_20261029_part_01/exports/vacancies_flat.csv): 300 вакансий.

- [avito_f11_spb_courier_20261029_part_02](avito_runs/avito_f11_spb_courier_20261029_part_02/exports/vacancies_flat.csv): 295 вакансий.

- [20261029_ekaterinburg_courier_part_01](avito_runs/20261029_ekaterinburg_courier_part_01/exports/vacancies_flat.csv): 299 вакансий.

- [20261029_ekaterinburg_courier_part_02](avito_runs/20261029_ekaterinburg_courier_part_02/exports/vacancies_flat.csv): 258 вакансий.

- [20261029_moscow_cashier_f13_01](avito_runs/20261029_moscow_cashier_f13_01/exports/vacancies_flat.csv): 299 вакансий.

- [20261029_moscow_cashier_f13_02](avito_runs/20261029_moscow_cashier_f13_02/exports/vacancies_flat.csv): 2 вакансий.

- [20261029_moscow_cashier_f13_02_resume](avito_runs/20261029_moscow_cashier_f13_02_resume/exports/vacancies_flat.csv): 296 вакансий.

- [20261029_moscow_cashier_f13_03](avito_runs/20261029_moscow_cashier_f13_03/exports/vacancies_flat.csv): 277 вакансий.

- [20261029_spb_cashier_part_01](avito_runs/20261029_spb_cashier_part_01/exports/vacancies_flat.csv): 300 вакансий.

- [20261029_spb_cashier_part_02](avito_runs/20261029_spb_cashier_part_02/exports/vacancies_flat.csv): 299 вакансий.

- [20261029_spb_cashier_part_03](avito_runs/20261029_spb_cashier_part_03/exports/vacancies_flat.csv): 300 вакансий.

- [20261029_spb_cashier_part_04](avito_runs/20261029_spb_cashier_part_04/exports/vacancies_flat.csv): 107 вакансий.

- [filter15_01](avito_runs/filter15_01/exports/vacancies_flat.csv): 299 вакансий.

- [filter15_02](avito_runs/filter15_02/exports/vacancies_flat.csv): 116 вакансий.

- [filter16_moscow_truck_driver_20261029_b_01](avito_runs/filter16_moscow_truck_driver_20261029_b_01/exports/vacancies_flat.csv): 299 вакансий.

- [filter16_moscow_truck_driver_20261029_b_02](avito_runs/filter16_moscow_truck_driver_20261029_b_02/exports/vacancies_flat.csv): 299 вакансий.

- [filter16_moscow_truck_driver_20261029_b_03](avito_runs/filter16_moscow_truck_driver_20261029_b_03/exports/vacancies_flat.csv): 300 вакансий.

- [filter16_moscow_truck_driver_20261029_b_04](avito_runs/filter16_moscow_truck_driver_20261029_b_04/exports/vacancies_flat.csv): 297 вакансий.

- [filter16_moscow_truck_driver_20261029_b_05](avito_runs/filter16_moscow_truck_driver_20261029_b_05/exports/vacancies_flat.csv): 66 вакансий.

- [20261029_sp_driver17_part_01](avito_runs/20261029_sp_driver17_part_01/exports/vacancies_flat.csv): 300 вакансий.

- [20261029_sp_driver17_part_02](avito_runs/20261029_sp_driver17_part_02/exports/vacancies_flat.csv): 87 вакансий.

- [20261029_sp_driver17_part_02_suffix](avito_runs/20261029_sp_driver17_part_02_suffix/exports/vacancies_flat.csv): 206 вакансий.

- [20261029_sp_driver17_part_03](avito_runs/20261029_sp_driver17_part_03/exports/vacancies_flat.csv): 294 вакансий.

- [20261029_sp_driver17_part_04](avito_runs/20261029_sp_driver17_part_04/exports/vacancies_flat.csv): 24 вакансий.

- [20261029_sp_driver17_supplement_p19](avito_runs/20261029_sp_driver17_supplement_p19/exports/vacancies_flat.csv): 24 вакансий.

- [filter18_ekaterinburg_truck_driver_20261029_part_01](avito_runs/filter18_ekaterinburg_truck_driver_20261029_part_01/exports/vacancies_flat.csv): 203 вакансий.

- [remote_1500_2026-10-29_shamil](avito_runs/remote_1500_2026-10-29_shamil/exports/vacancies_flat.csv): 1485 вакансий.

- [remote_moscow_2026-10-30_shamil_novel](avito_runs/remote_moscow_2026-10-30_shamil_novel/exports/vacancies_flat.csv): 490 вакансий.

- [remote_continuous_r000001_s000_new](avito_runs/remote_continuous_r000001_s000_new/exports/vacancies_flat.csv): 358 вакансий.

- [remote_continuous_r000001_s001_new](avito_runs/remote_continuous_r000001_s001_new/exports/vacancies_flat.csv): 78 вакансий.

- [remote_continuous_r000002_s000_new](avito_runs/remote_continuous_r000002_s000_new/exports/vacancies_flat.csv): 215 вакансий.

- [remote_continuous_r000002_s001_new](avito_runs/remote_continuous_r000002_s001_new/exports/vacancies_flat.csv): 24 вакансий.

- [remote_continuous_r000003_s000_new](avito_runs/remote_continuous_r000003_s000_new/exports/vacancies_flat.csv): 120 вакансий.

- [remote_continuous_r000003_s001_new](avito_runs/remote_continuous_r000003_s001_new/exports/vacancies_flat.csv): 14 вакансий.

- [remote_continuous_r000004_s000_new](avito_runs/remote_continuous_r000004_s000_new/exports/vacancies_flat.csv): 43 вакансий.

- [remote_continuous_r000004_s001_new](avito_runs/remote_continuous_r000004_s001_new/exports/vacancies_flat.csv): 14 вакансий.

- [remote_continuous_r000005_s000_new](avito_runs/remote_continuous_r000005_s000_new/exports/vacancies_flat.csv): 106 вакансий.

- [remote_continuous_r000005_s001_new](avito_runs/remote_continuous_r000005_s001_new/exports/vacancies_flat.csv): 27 вакансий.

- [remote_continuous_r000006_s000_new](avito_runs/remote_continuous_r000006_s000_new/exports/vacancies_flat.csv): 61 вакансий.

- [vahta_50_2026-11-02_sasha](avito_runs/vahta_50_2026-11-02_sasha/exports/vacancies_flat.csv): 50 вакансий.

- [vahta_1500_2026-11-02_sasha](avito_runs/vahta_1500_2026-11-02_sasha/exports/vacancies_flat.csv): 1257 вакансий.

- [avito_20261102_gruzchik_all_1500](avito_runs/avito_20261102_gruzchik_all_1500/exports/vacancies_flat.csv): 1389 вакансий.

- [samara_prodavets_first1200_20261106](avito_runs/samara_prodavets_first1200_20261106/exports/vacancies_flat.csv): 362 вакансий.

- [avito-yakutsk-20261106](avito_runs/avito-yakutsk-20261106/exports/vacancies_flat.csv): 1240 вакансий.

- [avito-amga-20261106](avito_runs/avito-amga-20261106/exports/vacancies_flat.csv): 1244 вакансий.

- [avito-kamchatskiy-kray-20261106](avito_runs/avito-kamchatskiy-kray-20261106/exports/vacancies_flat.csv): 1071 вакансий.

- [20261109-ekb-kassir-smenny](avito_runs/20261109-ekb-kassir-smenny/exports/vacancies_flat.csv): 335 вакансий.

- [20261109-ekb-kassir-bez-opyta](avito_runs/20261109-ekb-kassir-bez-opyta/exports/vacancies_flat.csv): 352 вакансий.

- [20261109-ekb-kassir-50000](avito_runs/20261109-ekb-kassir-50000/exports/vacancies_flat.csv): 318 вакансий.

- [20261109-krasnodar-all](avito_runs/20261109-krasnodar-all/exports/vacancies_flat.csv): 1112 вакансий.

- [20261109-krasnodar-gruzchik](avito_runs/20261109-krasnodar-gruzchik/exports/vacancies_flat.csv): 184 вакансий.

- [20261109-krasnodar-upakovshchik](avito_runs/20261109-krasnodar-upakovshchik/exports/vacancies_flat.csv): 132 вакансий.

- [20261109-krasnodar-kurer](avito_runs/20261109-krasnodar-kurer/exports/vacancies_flat.csv): 336 вакансий.

- [20261109-krasnodar-ohrannik](avito_runs/20261109-krasnodar-ohrannik/exports/vacancies_flat.csv): 536 вакансий.

- [20261109-krasnodar-sklad](avito_runs/20261109-krasnodar-sklad/exports/vacancies_flat.csv): 149 вакансий.

- [20261109-krasnodar-komplektovshchik](avito_runs/20261109-krasnodar-komplektovshchik/exports/vacancies_flat.csv): 131 вакансий.

- [20261109-krasnodar-prodavec](avito_runs/20261109-krasnodar-prodavec/exports/vacancies_flat.csv): 277 вакансий.

- [20261109-krasnodar-voditel-gruzovogo](avito_runs/20261109-krasnodar-voditel-gruzovogo/exports/vacancies_flat.csv): 388 вакансий.

- [20261109-krasnodar-fasovshchik](avito_runs/20261109-krasnodar-fasovshchik/exports/vacancies_flat.csv): 79 вакансий.

- [20261109-krasnodar-kladovshchik](avito_runs/20261109-krasnodar-kladovshchik/exports/vacancies_flat.csv): 107 вакансий.

- [20261110-ryazan-all](avito_runs/20261110-ryazan-all/exports/vacancies_flat.csv): 1199 вакансий.

- [20261110-ryazan-gruzchik](avito_runs/20261110-ryazan-gruzchik/exports/vacancies_flat.csv): 228 вакансий.

- [20261110-ryazan-upakovshchik](avito_runs/20261110-ryazan-upakovshchik/exports/vacancies_flat.csv): 200 вакансий.

- [20261110-ryazan-kurer](avito_runs/20261110-ryazan-kurer/exports/vacancies_flat.csv): 251 вакансий.

- [20261110-ryazan-ohrannik](avito_runs/20261110-ryazan-ohrannik/exports/vacancies_flat.csv): 407 вакансий.

- [20261110-ryazan-sklad](avito_runs/20261110-ryazan-sklad/exports/vacancies_flat.csv): 148 вакансий.

- [20261110-ryazan-komplektovshchik](avito_runs/20261110-ryazan-komplektovshchik/exports/vacancies_flat.csv): 211 вакансий.

- [20261110-ryazan-prodavec](avito_runs/20261110-ryazan-prodavec/exports/vacancies_flat.csv): исходный завершённый экспорт, 258 вакансий; [производное преобразование адресов](transformations/20261110-ryazan-prodavec-placement-v1/vacancies_flat.csv) зарегистрировано в общем датасете.

- [20261110-ryazan-voditel-gruzovogo](avito_runs/20261110-ryazan-voditel-gruzovogo/exports/vacancies_flat.csv): 233 вакансий.

- [20261110-ryazan-fasovshchik](avito_runs/20261110-ryazan-fasovshchik/exports/vacancies_flat.csv): 111 вакансий.

- [20261110-ryazan-kladovshchik](avito_runs/20261110-ryazan-kladovshchik/exports/vacancies_flat.csv): 110 вакансий.

- [20261110-krasnodar-remote-all](avito_runs/20261110-krasnodar-remote-all/exports/vacancies_flat.csv): региональный исходный архив, 147 вакансий; [подтверждённое предварительное подмножество Краснодара](transformations/20261110-krasnodar-remote-all-city-v1/vacancies_flat.csv) — 137 вакансий, полнота города не подтверждена.

- [20261110-krasnodar-remote-bez-opyta](avito_runs/20261110-krasnodar-remote-bez-opyta/exports/vacancies_flat.csv): региональный исходный архив, 73 вакансии; [предварительное подмножество Краснодара](transformations/20261110-krasnodar-remote-bez-opyta-city-v1/vacancies_flat.csv) — 69 вакансий, полнота города не подтверждена.

- [20261110-krasnodar-remote-50000](avito_runs/20261110-krasnodar-remote-50000/exports/vacancies_flat.csv): региональный исходный архив, 102 вакансии; [предварительное подмножество Краснодара](transformations/20261110-krasnodar-remote-50000-city-v1/vacancies_flat.csv) — 97 вакансий, полнота города не подтверждена.

- [20261110-ryazan-remote-all](avito_runs/20261110-ryazan-remote-all/exports/vacancies_flat.csv): региональный исходный архив, 634 вакансии; [предварительное подмножество Рязани](transformations/20261110-ryazan-remote-all-city-v1/vacancies_flat.csv) — 41 вакансия, полнота города не подтверждена.

- [20261110-ryazan-remote-bez-opyta](avito_runs/20261110-ryazan-remote-bez-opyta/exports/vacancies_flat.csv): региональный исходный архив, 575 вакансий; [предварительное подмножество Рязани](transformations/20261110-ryazan-remote-bez-opyta-city-v1/vacancies_flat.csv) — 27 вакансий, полнота города не подтверждена.

- [20261110-ryazan-remote-50000](avito_runs/20261110-ryazan-remote-50000/exports/vacancies_flat.csv): региональный исходный архив, 644 вакансии; [предварительное подмножество Рязани](transformations/20261110-ryazan-remote-50000-city-v1/vacancies_flat.csv) — 25 вакансий, полнота города не подтверждена.

- [20261110-remote-all-russia-batch_01](avito_runs/20261110-remote-all-russia-batch_01/exports/vacancies_flat.csv): исходный архив 199 прочитанных; предварительные адреса: [Краснодар](transformations/20261110-remote-all-russia-batch_01-cities-v1/city-01.csv), [Рязань](transformations/20261110-remote-all-russia-batch_01-cities-v1/city-02.csv). Полнота городов не подтверждена.

- [20261110-remote-all-russia-batch_02](avito_runs/20261110-remote-all-russia-batch_02/exports/vacancies_flat.csv): исходный архив 200 прочитанных; предварительные адреса: [Краснодар](transformations/20261110-remote-all-russia-batch_02-cities-v1/city-01.csv), [Рязань](transformations/20261110-remote-all-russia-batch_02-cities-v1/city-02.csv). Полнота городов не подтверждена.

- [20261110-remote-all-russia-batch_03](avito_runs/20261110-remote-all-russia-batch_03/exports/vacancies_flat.csv): исходный архив 199 прочитанных; предварительные адреса: [Краснодар](transformations/20261110-remote-all-russia-batch_03-cities-v1/city-01.csv), [Рязань](transformations/20261110-remote-all-russia-batch_03-cities-v1/city-02.csv). Полнота городов не подтверждена.

- [20261110-remote-all-russia-batch_04](avito_runs/20261110-remote-all-russia-batch_04/exports/vacancies_flat.csv): исходный архив 200 прочитанных; предварительные адреса: [Краснодар](transformations/20261110-remote-all-russia-batch_04-cities-v1/city-01.csv), [Рязань](transformations/20261110-remote-all-russia-batch_04-cities-v1/city-02.csv). Полнота городов не подтверждена.

- [20261110-remote-all-russia-batch_05](avito_runs/20261110-remote-all-russia-batch_05/exports/vacancies_flat.csv): исходный архив 155 прочитанных; предварительные адреса: [Краснодар](transformations/20261110-remote-all-russia-batch_05-cities-v1/city-01.csv), [Рязань](transformations/20261110-remote-all-russia-batch_05-cities-v1/city-02.csv). Полнота городов не подтверждена.

- [20261110-remote-bez-opyta-batch_01](avito_runs/20261110-remote-bez-opyta-batch_01/exports/vacancies_flat.csv): исходный архив 200 прочитанных; предварительные адреса: [Краснодар](transformations/20261110-remote-bez-opyta-batch_01-cities-v1/city-01.csv), [Рязань](transformations/20261110-remote-bez-opyta-batch_01-cities-v1/city-02.csv). Полнота городов не подтверждена.

- [20261110-remote-bez-opyta-batch_02](avito_runs/20261110-remote-bez-opyta-batch_02/exports/vacancies_flat.csv): исходный архив 200 прочитанных; предварительные адреса: [Краснодар](transformations/20261110-remote-bez-opyta-batch_02-cities-v1/city-01.csv), [Рязань](transformations/20261110-remote-bez-opyta-batch_02-cities-v1/city-02.csv). Полнота городов не подтверждена.

- [20261110-remote-bez-opyta-batch_03](avito_runs/20261110-remote-bez-opyta-batch_03/exports/vacancies_flat.csv): исходный архив 160 прочитанных; предварительные адреса: [Краснодар](transformations/20261110-remote-bez-opyta-batch_03-cities-v1/city-01.csv), [Рязань](transformations/20261110-remote-bez-opyta-batch_03-cities-v1/city-02.csv). Полнота городов не подтверждена.

- [20261110-remote-50000-batch_01](avito_runs/20261110-remote-50000-batch_01/exports/vacancies_flat.csv): 192 вакансий.

- [20261110-remote-50000-batch_02](avito_runs/20261110-remote-50000-batch_02/exports/vacancies_flat.csv): 196 вакансий.

- [20261110-remote-50000-batch_03](avito_runs/20261110-remote-50000-batch_03/exports/vacancies_flat.csv): 198 вакансий.

- [20261110-remote-50000-batch_04](avito_runs/20261110-remote-50000-batch_04/exports/vacancies_flat.csv): 193 вакансий.

- [20261110-remote-50000-batch_05](avito_runs/20261110-remote-50000-batch_05/exports/vacancies_flat.csv): 197 вакансий.

- [20261110-remote-50000-batch_06](avito_runs/20261110-remote-50000-batch_06/exports/vacancies_flat.csv): 185 вакансий.
