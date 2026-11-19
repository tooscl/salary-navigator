"""Examples from the rules of Analytics/salary_normalization_v1.ipynb."""

import unittest

import pandas as pd

from utils.salary_normalization import NOTEBOOK_CONFIG, normalize_salaries


def frame(*rows):
    columns = ["salary_raw", "Количество рабочих часов в день", "city_resolved", "profession_resolved", "График"]
    return pd.DataFrame([dict(zip(columns, row)) for row in rows])


class SalaryNormalizationTest(unittest.TestCase):
    def test_range_split_by_shift_lengths(self):
        # 3 000-6 000 per shift, 4-6 / 6-8 / 8-10 hours -> 3 500 / 4 500 / 5 500 at 5 / 7 / 9 hours
        out = normalize_salaries(frame(("3 000 — 6 000 ₽ за смену", "4–6, 6–8, 8–10", "Москва", "Курьер", "Сменный")),
                                 NOTEBOOK_CONFIG, print_stats=False, return_details=True)
        self.assertEqual(out["hours_variant"].tolist(), [5.0, 7.0, 9.0])
        expected = [3500 / 5 * 176, 4500 / 7 * 176, 5500 / 9 * 176]
        for got, want in zip(out["salary_month_gross_normalized"], expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_upper_bound_only_is_excluded_and_unspecified_is_month(self):
        out = normalize_salaries(frame(("до 80 000 ₽", None, "Москва", "Продавец", "Гибкий"),
                                       ("от 60 000 ₽", None, "Москва", "Продавец", "Гибкий")),
                                 NOTEBOOK_CONFIG, print_stats=False, return_details=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "salary_month_gross_normalized"], 60000)
        self.assertEqual(out.loc[0, "salary_unit"], "unspecified")

    def test_net_is_grossed_up(self):
        out = normalize_salaries(frame(("87 000 ₽ на руки", None, "Москва", "Продавец", "Гибкий")),
                                 NOTEBOOK_CONFIG, print_stats=False)
        self.assertAlmostEqual(out.loc[0, "salary_month_gross_normalized"], 100000, places=6)


if __name__ == "__main__":
    unittest.main()
