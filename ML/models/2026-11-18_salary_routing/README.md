# Готовые модели и статистический расчёт для маршрутизации

[Каталог](catalog.json) содержит одну общую модель, три модели по режимам,
статистические квартили и 13 пакетов для ячеек с достаточной опорой первого
фолда. В каталог записаны область и объём обучения каждого пакета.
Качество в сводке получено на трёх фолдах. Пакеты здесь воспроизводят
решения первого фолда на его отложенных строках.

- [Общий CatBoost](global.joblib)
- [Обычная работа](mode_ordinary.joblib) · [Вахта](mode_watch.joblib) · [Удалёнка](mode_remote.joblib)
- [Эмпирический расчёт](empirical.joblib)
- [Поддержанные сегменты и файлы](catalog.json)

## Применение

Установите [зависимости предыдущего эксперимента](../2026-11-18_priority_salary_gold/requirements.txt).
Запустите [исследовательский ноутбук](../../05_salary_routing.ipynb) или используйте функцию:

~~~python
import json
import sys
import pandas as pd

sys.path.insert(0, "ML")
import salary_routing as routing

folder = routing.EXPERIMENT_DIR
policy = json.loads((folder / "routing_policy.json").read_text(encoding="utf-8"))
vacancies = pd.read_csv(folder / "example_vacancies.csv", dtype=str, keep_default_na=False)
result = routing.predict_vacancies(vacancies, policy)
~~~

Входной CSV принимает mode (ordinary, watch, remote), unit (shift, hour,
week, day, month) и tax (net, gross). Поддержка конкретной комбинации
проверяется по каталогу. В текущих пакетах фактически представлены проверенные
контексты с базой net. Поля city_resolved и profession_resolved используются
для точного назначения. Дополнительные параметры карточки принимаются под
именами канонического датасета. Пропуски остаются отдельными значениями.
Поле зарплаты для прогноза не требуется.

[Файл соответствий](../../experiments/2026-11-18_salary_routing/routing_policy.json)
содержит точные условия, идентификатор подхода, обоснование и ссылку на
проверку. Для создания новой карты сохраните копию JSON и явно измените
одно или несколько правил. Резерв проверяет область применения указанного
пакета. Результат при отказе содержит причину и пустые квантили. Загружайте
joblib только из доверенного локального каталога.
