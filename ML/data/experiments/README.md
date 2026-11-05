# Experiment datasets

These four CSV files are mutually exclusive, reproducible views of
`Data/dataset/vacancies.csv`. Their union contains every canonical vacancy.

| File | Contents |
|---|---|
| `01_location_monthly.csv` | Non-remote, non-watch vacancies with a settlement and a plain amount (no period, no variable pay) treated as monthly. |
| `02_remote_monthly.csv` | Remote-only **and remote-capable** (`Удалённо, Гибрид`, `Удалённо, В офисе…`) non-watch vacancies with the same monthly assumption. |
| `03_watch.csv` | Every vacancy whose schedule is `Вахта`. |
| `04_other.csv` | Every remaining vacancy: explicit rates per shift/hour/week, variable pay, no amount, military contracts, likely rates and city-less rows. |

Rules (`rules_version` in `manifest.json`, currently `experiment_datasets_v2`):

- `city` is taken from the card; when the enrichment left it empty (an address
  without a reviewed placement entry, e.g. Колпино, Шушары, Видное) the
  settlement is recovered from `location_observed`. `city_source` says which:
  `card`, `location_observed` or `missing`.
- An amount without a period is treated as monthly only when its largest number
  is at least 10 000 rub; smaller ones (`2 800 — 3 480 ₽`, `от 4 000 ₽`) are
  almost always rates per shift and get `pay_mode=low_amount_likely_rate` in 04.
- Rows with the enrichment flag `military=True` (SVO / rear-service contracts
  with one-off sign-up payments) get `pay_mode=military_contract` in 04.

All CSV files use UTF-8 with BOM and the same English `snake_case` schema.
`salary_period_evidence=assumed_from_non_rate_listing` means that monthly pay
is an explicit experimental assumption, not an assertion about the raw source.
All non-empty source columns outside the compact schema are preserved in
`extra_attributes_json`; see `column_dictionary.csv`.

Rebuild from the repository root:

```sh
python utils/build_experiment_datasets.py
python utils/build_experiment_datasets.py --check
```
