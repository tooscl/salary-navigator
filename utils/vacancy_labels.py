"""Versioned derived labels for the shared CSV: profession and settlement.

They are added next to the observed columns and never replace them:
`profession` stays the search-query label of the collection and `city` stays the
placement city shown on the site (see Data/dataset/README.md).

- `profession_resolved`, `profession_source`: the profession named in the title
  (utils/professions.py) when it is one of the six collected professions,
  otherwise the query label; source is `title`, `query` or `none`.
- `city_resolved`, `city_resolved_source`: the placement city when present,
  otherwise the settlement parsed from `location_observed` against the
  gazetteer in utils/data (utils/build_experiment_datasets.py:locate_city);
  source is `card`, `location_observed`, `location_settlement`,
  `location_unlisted` or `missing`.
"""
from pathlib import Path

try:
    from .build_experiment_datasets import CITIES_EXTRA_PATH, CITIES_PATH, resolved_city
    from .professions import resolve_profession
except ImportError:  # Direct `python3 utils/build_dataset.py` entry point.
    from build_experiment_datasets import CITIES_EXTRA_PATH, CITIES_PATH, resolved_city
    from professions import resolve_profession

VERSION = 'vacancy_labels_v2'
COLUMNS = ('profession_resolved', 'profession_source', 'city_resolved', 'city_resolved_source',
           'labels_rule_version')
_HERE = Path(__file__).resolve().parent
# Everything the labels depend on; their hashes go to the manifest.
DEPENDENCIES = (Path(__file__).resolve(), _HERE / 'professions.py', _HERE / 'build_experiment_datasets.py',
                Path(CITIES_PATH), Path(CITIES_EXTRA_PATH))


def vacancy_labels(row):
    profession, profession_source = resolve_profession(row.get('title', ''), row.get('profession', ''))
    city, city_source = resolved_city(row)
    return dict(profession_resolved=profession, profession_source=profession_source,
                city_resolved=city, city_resolved_source=city_source, labels_rule_version=VERSION)
