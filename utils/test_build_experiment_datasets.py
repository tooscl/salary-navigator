import unittest

try:
    from build_experiment_datasets import (DATASETS, OUTPUT_DIRECTORY, ROOT, build, city_from_location, dataset_for,
                                           output_row, pay_mode)
except ModuleNotFoundError:
    from utils.build_experiment_datasets import (DATASETS, OUTPUT_DIRECTORY, ROOT, build, city_from_location,
                                                 dataset_for, output_row, pay_mode)


def row(**changes):
    base = {"source": "avito", "id": "1", "city": "Москва", "profession": "Курьер", "Формат работы": "",
            "График": "Сменный", "salary_raw": "80 000 — 100 000 ₽ на руки", "title": "Курьер"}
    base.update(changes)
    return base


class ExperimentDatasetTests(unittest.TestCase):
    def test_priority_and_assignment(self):
        self.assertEqual(dataset_for(row()), "01_location_monthly")
        self.assertEqual(dataset_for(row(city="", **{"Формат работы": "Удалённо"})), "02_remote_monthly")
        self.assertEqual(dataset_for(row(**{"График": "Вахта", "Формат работы": "Удалённо"})), "03_watch")
        self.assertEqual(dataset_for(row(salary_raw="4 000 ₽ за смену")), "04_other")

    def test_remote_capable_goes_to_remote(self):
        self.assertEqual(dataset_for(row(**{"Формат работы": "Удалённо, Гибрид"})), "02_remote_monthly")
        self.assertEqual(dataset_for(row(**{"Формат работы": "Удалённо, В офисе или на объекте"})), "02_remote_monthly")

    def test_city_recovered_from_address(self):
        no_city = row(city="", location_observed="Санкт-Петербург, Колпинский р-н, Колпино, Финляндская ул.")
        self.assertEqual(dataset_for(no_city), "01_location_monthly")
        clean = output_row(no_city, "01_location_monthly", list(no_city))
        self.assertEqual((clean["city"], clean["city_source"]), ("Санкт-Петербург", "location_observed"))
        self.assertEqual(output_row(row(), "01_location_monthly", list(row()))["city_source"], "card")
        self.assertEqual(dataset_for(row(city="", location_observed="")), "04_other")

    def test_city_from_location(self):
        self.assertEqual(city_from_location("Москва, Кленовый б-р Кленовый бульвар 6–10 мин."), "Москва")
        self.assertEqual(city_from_location("Свердловская обл., Екатеринбург, ул. Ленина"), "Екатеринбург")
        self.assertEqual(city_from_location("Московская обл., Ленинский г.о., Видное, Школьная ул."), "Видное")
        self.assertEqual(city_from_location("Московская обл., Одинцовский г.о., Одинцово Одинцово 16–20 мин."), "Одинцово")
        self.assertEqual(city_from_location("Республика Татарстан (Татарстан), Набережные Челны, ул."), "Набережные Челны")
        self.assertEqual(city_from_location(""), "")

    def test_low_amount_and_military_are_not_monthly(self):
        self.assertEqual(pay_mode(row(salary_raw="2 800 — 3 480 ₽ на руки")), "low_amount_likely_rate")
        self.assertEqual(dataset_for(row(salary_raw="от 4 000 ₽")), "04_other")
        self.assertEqual(dataset_for(row(salary_raw="от 10 000 ₽")), "01_location_monthly")
        self.assertEqual(pay_mode(row(military="True")), "military_contract")
        self.assertEqual(dataset_for(row(military="True")), "04_other")

    def test_clean_schema_and_month_assumption(self):
        clean = output_row(row(), "01_location_monthly", list(row()))
        self.assertEqual(clean["salary_period"], "month")
        self.assertEqual(clean["salary_period_evidence"], "assumed_from_non_rate_listing")
        self.assertEqual(clean["salary_lower_rub"], "80000")
        self.assertEqual(clean["salary_upper_rub"], "100000")
        self.assertNotIn(" ", clean["salary_raw"])

    def test_committed_outputs_match_current_dataset(self):
        outputs = build(ROOT)
        for name, _ in DATASETS:
            filename = f"{name}.csv"
            self.assertEqual((ROOT / OUTPUT_DIRECTORY / filename).read_bytes(), outputs[filename])
        self.assertEqual((ROOT / OUTPUT_DIRECTORY / "manifest.json").read_bytes(), outputs["manifest.json"])


if __name__ == "__main__":
    unittest.main()
