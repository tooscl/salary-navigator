"""Contract checks for merging independent team exports without accessing Avito."""
import csv
import io
import tempfile
import unittest
from pathlib import Path

from utils.build_dataset import build, digest


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = {"schema_version": "1", "exports": []}

    def row(self, identity="1", **values):
        return dict(id=identity, source_url="https://example.org/" + identity,
                    observed_at="2026-10-21T10:00:00Z", city="Москва",
                    profession="Курьер", title="Курьер", description="Опыт не нужен",
                    salary_raw="100 000 ₽", analysis_status="validated",
                    export_eligibility="included", **values)

    def add(self, run, rows, source="avito"):
        path = self.root / (run + ".csv")
        names = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        self.registry["exports"].append(dict(run_id=run, source=source,
            path=path.name, sha256=digest(path.read_bytes()),
            placement_regions={"Москва": "Москва"}))
        return path

    def result(self):
        data, _ = build(self.root, self.registry)
        return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))

    def test_title_flags_recomputed_from_latest_title_without_modifying_sources(self):
        row = self.row(); row.update(title='Водитель БПЛА', military='False',
                                    military_keywords='', military_rule_version='older')
        path = self.add('old', [row]); original = path.read_bytes()
        first = self.result()[0]
        self.assertEqual(first['military'], 'True')
        self.assertEqual(first['military_keywords'], 'БПЛА')
        self.assertEqual(path.read_bytes(), original)
        row.update(observed_at='2026-10-22T10:00:00Z', title='Контракт-менеджер')
        self.add('new', [row])
        self.assertEqual(self.result()[0]['military'], 'False')

    def test_same_observation_different_derived_flags_does_not_conflict(self):
        row = self.row(); row['title'] = 'Военный водитель'
        self.add('old', [row])
        row.update(military='False', military_keywords='', military_rule_version='old')
        self.add('new', [row])
        result = self.result()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['military'], 'True')

    def test_one_export_keeps_flags_and_missing_values(self):
        row = self.row(**{"Медкнижка": ""})
        row["export_eligibility"] = "fit_review_required"
        self.add("one", [row])
        result = self.result()[0]
        self.assertEqual(result["Медкнижка"], "")
        self.assertEqual(result["export_eligibility"], "fit_review_required")
        self.assertEqual(result["region"], "Москва")

    def test_multiple_exports_latest_row_new_columns_and_history(self):
        self.add("old", [self.row(**{"Опыт": "3 года"})])
        newer = self.row(**{"Автомобиль": "Да"})
        newer.update(observed_at="2026-10-22T10:00:00Z", salary_raw="120 000 ₽")
        self.add("new", [newer, self.row("2", **{"Автомобиль": ""})])
        rows = self.result()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["salary_raw"], "120 000 ₽")
        self.assertEqual(rows[0]["Опыт"], "")  # No unobserved mix of old and new.
        self.assertEqual(rows[0]["first_seen_at"], "2026-10-21T10:00:00Z")
        self.assertEqual(rows[0]["last_seen_at"], "2026-10-22T10:00:00Z")
        self.assertEqual(rows[0]["run_ids"], "new; old")
        original = build(self.root, self.registry)
        self.registry["exports"].reverse()
        self.assertEqual(build(self.root, self.registry), original)

    def test_identity_includes_source(self):
        self.add("one", [self.row()])
        self.add("two", [self.row()], source="other")
        self.assertEqual(len(self.result()), 2)

    def test_duplicate_within_run_rejected(self):
        self.add("one", [self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            self.result()

    def test_changed_immutable_input_rejected(self):
        path = self.add("one", [self.row()])
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.result()

    def test_profession_correction_provenance_is_immutable(self):
        self.add('one', [self.row()])
        evidence = self.root/'correction.json'; evidence.write_text('{"reason":"observed"}')
        self.registry['exports'][0]['profession_correction'] = {'path': evidence.name, 'sha256': digest(evidence.read_bytes())}
        self.assertEqual(len(self.result()), 1)
        evidence.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Profession correction SHA-256'):
            self.result()

    def test_same_time_conflict_rejected(self):
        self.add("one", [self.row()])
        row = self.row()
        row["salary_raw"] = "200 000 ₽"
        self.add("two", [row])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.result()

    def test_identical_repeat_does_not_duplicate(self):
        self.add("one", [self.row()])
        self.add("two", [self.row()])
        self.assertEqual(len(self.result()), 1)

    def test_unreviewed_record_rejected(self):
        row = self.row()
        row["analysis_status"] = "pending"
        self.add("one", [row])
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            self.result()

    def test_enrichment_requires_exact_description_evidence(self):
        row = self.row(enrichment_status="completed", enriched_fields="Опыт",
                       enrichment_version="extract-v1", **{"Опыт": "Без опыта",
                       "Опыт__source": "description", "Опыт__evidence": "Опыт не нужен"})
        self.add("one", [row])
        self.assertEqual(self.result()[0]["enriched_fields"], "Опыт")
        row["Опыт__evidence"] = "придуманная цитата"
        self.registry["exports"].clear()
        self.add("one", [row])
        with self.assertRaisesRegex(ValueError, "quote"):
            self.result()

    def test_geography_never_comes_from_description(self):
        row = self.row()
        row["description"] = "Работа в Туле"
        self.add("one", [row])
        self.assertEqual(self.result()[0]["city"], "Москва")
        self.assertEqual(self.result()[0]["region"], "Москва")
        self.registry["exports"][0]["placement_regions"] = {}
        with self.assertRaisesRegex(ValueError, "region mapping"):
            self.result()

    def test_unknown_city_requires_id_specific_visible_placement_evidence(self):
        row = self.row(location_observed='Свердловская обл., Белоярский муниципальный округ')
        row['city'] = ''
        self.add('one', [row])
        with self.assertRaisesRegex(ValueError, 'unknown-city'):
            self.result()

        entry = self.registry['exports'][0]
        entry['unknown_city_placements'] = {'1': {'region': 'Свердловская область', 'evidence': 'Свердловская обл.'}}
        result = self.result()[0]
        self.assertEqual((result['city'], result['region']), ('', 'Свердловская область'))
        entry['unknown_city_placements']['1']['evidence'] = 'Москва'
        with self.assertRaisesRegex(ValueError, 'unknown-city'):
            self.result()

    def test_search_city_requires_explicit_source_and_registered_evidence(self):
        row = self.row(city__source='search_filter', city__evidence='https://www.avito.ru/moskva/vakansii',
                       city_anomaly='city_not_visible_in_card')
        self.add('one',[row])
        with self.assertRaisesRegex(ValueError,'search-filter provenance'):
            self.result()
        entry=self.registry['exports'][0]
        entry.update(search_city=row['city'],search_city_evidence=row['city__evidence'])
        self.assertEqual(self.result()[0]['city__source'],'search_filter')
        entry['search_city']='Екатеринбург'
        with self.assertRaisesRegex(ValueError,'search-filter provenance'):
            self.result()

    def test_missing_location_is_allowed_only_with_explicit_missing_status_and_registry(self):
        row = self.row(); row.update(city='', description='', location_observed='',
            read_status_location='missing', city__source='missing', city__evidence='',
            city_anomaly='Missing visible location')
        self.add('partial', [row])
        entry = self.registry['exports'][0]
        entry['unknown_city_placements'] = {'1': {'region':'', 'region_unknown':True,
                                                 'evidence':'', 'location_missing':True}}
        self.assertEqual(self.result()[0]['city'], '')
        self.assertEqual(self.result()[0]['description'], '')
        entry['unknown_city_placements']['1']['location_missing'] = False
        with self.assertRaisesRegex(ValueError, 'unknown-city'):
            self.result()

    def test_collection_without_profession_filter_preserves_empty_value(self):
        row = self.row()
        row["profession"] = ""
        row["segments"] = "moscow_all_professions"
        self.add("unfiltered", [row])
        self.assertEqual(self.result()[0]["profession"], "")
        self.assertEqual(self.result()[0]["segments"], "moscow_all_professions")


if __name__ == "__main__":
    unittest.main()
