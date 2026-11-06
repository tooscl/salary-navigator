# Reference data for utils

| File | Contents | Source / licence |
|---|---|---|
| `russian_cities.csv` | 1 117 Russian cities: `city`, `region`, `region_type`, `population` (Москва, Санкт-Петербург, Севастополь are listed with `city` = region) | [hflabs/city](https://github.com/hflabs/city) (`city.csv`, data from DaData, population from Wikipedia), reduced to four columns. Licence: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — keep this attribution and share changes to this file under the same licence. |
| `russian_cities_extra.csv` | Cities missing from hflabs/city (Одинцово, Клин, Истра, …), same columns, `population` unknown (0) | Added by hand for `locate_city`. |

Used by `locate_city` in `build_experiment_datasets.py` to recover the city from
`location_observed` when the card has none.
