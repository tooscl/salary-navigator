# Общие утилиты данных

Эти функции доступны и аналитике, и ML. Они работают с сохранёнными файлами
и не запускают сбор вакансий.

| Утилита | Что делает |
|---|---|
| [flatten_export.py](flatten_export.py) | Превращает подробный CSV сборщика в таблицу с отдельными колонками характеристик; не перезаписывает существующий результат |
| [load_dataset.py](load_dataset.py) | Загружает подробный экспорт по описанию набора и проверяет контрольные суммы, колонки, ID и срезы |
| [build_dataset_splits.py](build_dataset_splits.py) | Строит проверяемые аналитические принадлежности общего датасета без копирования или изменения исходного CSV |
| [build_experiment_datasets.py](build_experiment_datasets.py) | Создаёт четыре чистых взаимно исключающих CSV для экспериментов из общего датасета; `locate_city` достаёт населённый пункт из адреса по справочнику `data/` |
| [professions.py](professions.py) | Профессия по названию вакансии: `profession_from_title(title)`, `resolve_profession(title, query_label)`; `python utils/professions.py "Продавец-кассир"` |
| [vacancy_labels.py](vacancy_labels.py) | Производные колонки общего CSV `profession_resolved`/`profession_source`, `city_resolved`/`city_resolved_source`; вызывается из `build_dataset.py` |

Обычный CSV можно читать напрямую через pandas. Эти утилиты нужны для
подготовки таблицы или подробной проверки её происхождения.

Для импорта из ноутбука добавьте корень репозитория в sys.path, затем используйте
`from utils.load_dataset import load_dataset`. Готовый пример —
[ML/01_load_with_passport.ipynb](../ML/01_load_with_passport.ipynb).
### `delete_salary.py`

Удаляет из `title` или `description` фрагменты, содержащие явное или сильно
подразумеваемое раскрытие оплаты. Функция `delete_salary(text)` не оставляет
маркера на месте удалённого фрагмента и подходит для подготовки текстовых
признаков перед обучением модели.

