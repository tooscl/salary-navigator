# Готовые пилотные ML-сегменты вахты, 24.11.2026

Запрос: «Нужны готовые сегменты для ML с возможными допущениями».
Источник: закреплённое исследование 23.11.2026, commit `34cff91b3f9a3af6ec78cfc159cdd013e64b8546`.
Задача: https://buildin.ai/83ea70d5-0451-4a60-8118-7fc0d7c838b7.

CSV уже содержат признаки, таргеты, группы, веса и три готовых фолда. [Выводы](REPORT.md), [допущения](ASSUMPTIONS.md), [реестр](results/segment_registry.csv).

| Выборка | Объявлений | Закрытых/точных | Групп: все / закрытые | Использование |
|---|---:|---:|---:|---|
| Погрузка | 162 | 74 | 96 / 47 | Отдельно оба таргета |
| Упаковка | 128 | 64 | 74 / 30 | Отдельно оба таргета |
| Комплектация | 109 | 37 | 67 / 23 | Отдельно минимум; середина в общей модели |
| Общая складская модель | 453 | 194 | 206 / 91 | Основной общий пилот с признаком роли |

Общая модель включает три роли выше и ещё 54 объявления смежных складских ролей. Её объём не складывается с дочерними. Дочерние роли не пересекаются. Группы контролируют зависимость, не подтверждают юридическую независимость компаний.

## Файлы и таргеты

- `results/loading.csv`, `packing.csv`, `picking.csv`, `warehouse_pool.csv`: **y_floor_net_shift_rub**, объявленная нижняя граница рублей на руки за смену. «От 4000» даёт наблюдаемый минимум 4000.
- Суффикс `_point.csv`: только точные суммы/закрытые вилки, **y_midpoint_closed_net_shift_rub**, середина объявленной вилки. Открытые «от» сюда не включены.
- `results/warehouse_text_linked_point.csv`: дополнительное ядро из 67 закрытых/точных предложений, у которых сумма и период прямо связаны в тексте; 37 групп. Это подмножество общей point-выборки, а не ещё 67 независимых строк.
- `results/feature_allowlist.json`: только 33 колонки `x__*` для X. Зарплатный текст, суммы, тип вилки, ID, группы, фолд и веса являются метаданными/таргетами.
- `cv_fold`: 0/1/2; один фолд в тесте, два в обучении. Группы источников/шаблонов/извлечённых точных объектов не пересекают фолды.
- `training_weight`: суммарный вес группы равен 1 в каждом конкретном CSV. Для point-файлов также брать именно эту колонку. Дополнительные source_weight_pool/role диагностические.
- `all7193_selection_audit.csv`: решение по каждому исходному ID; `scope_sensitivity.csv`: влияние ценового потолка; `pilot_metrics.csv` и `pilot_oof_predictions.csv`: выполненный эксперимент.

## Быстрый старт

```python
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

p = "Analytics/vahta_ml_segments_20261124/"
d = pd.read_csv(p + "results/warehouse_pool_point.csv", keep_default_na=False)
columns = json.load(open(p + "results/feature_allowlist.json"))
train, test = d[d.cv_fold != 0], d[d.cv_fold == 0]
assert not set(train.split_group) & set(test.split_group)
v = DictVectorizer()
X = v.fit_transform(train[columns].astype(str).to_dict("records"))
Xt = v.transform(test[columns].astype(str).to_dict("records"))
model = Ridge(alpha=10, solver="lsqr")
model.fit(X, np.log(train.y_midpoint_closed_net_shift_rub),
          sample_weight=train.training_weight)
prediction = np.exp(model.predict(Xt))
```

Это пример обучения, не утверждение о лучшей модели. Признаки представлены категориальными строками. Векторизатор обучается только на train.

## Воспроизведение

Python 3.13, версии в requirements.lock.txt. Из корня репозитория:

```sh
python -m pip install -r Analytics/vahta_ml_segments_20261124/requirements.lock.txt
python Analytics/vahta_ml_segments_20261124/build_segments.py --raw-csv "$RAW_CSV"
python -m unittest discover -s Analytics/vahta_ml_segments_20261124 -p test_segments.py -v
python Analytics/vahta_ml_segments_20261124/evaluate_pilot.py
```

RAW_CSV — снимок 32 097 строк, SHA проверяется автоматически. Он материализуется `python3 utils/build_dataset.py --materialize` на закреплённом commit. Полная пересборка требует репозитория с прежним исследованием; готовые CSV читаются независимо от сырого датасета. evaluate_pilot.py работает на готовом results, полное извлечение/NLP не запускает.

Статус: готовы к пилотному обучению по объявленным суммам. Фактические выплаты и качество на будущих данных не проверены.
