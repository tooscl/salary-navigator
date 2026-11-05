"""Regression tests for salary redaction."""

import unittest

from delete_salary import delete_salary


class DeleteSalaryTests(unittest.TestCase):
    def test_removes_salary_bullet_and_keeps_other_text(self) -> None:
        source = "Обязанности: доставка заказов.\nЗарплата от 120 000 ₽ на руки.\nГибкий график."
        self.assertEqual(delete_salary(source), "Обязанности: доставка заказов.\nГибкий график.")

    def test_removes_range_without_currency(self) -> None:
        self.assertEqual(delete_salary("Доход 80 000–120 000. Оформление по ТК."), "Оформление по ТК.")

    def test_removes_rate_and_bonus(self) -> None:
        source = "Оплата 5000 руб. за смену; премия до 20% за результат. Бесплатное питание."
        self.assertEqual(delete_salary(source), "Бесплатное питание.")

    def test_removes_spelled_out_amount(self) -> None:
        self.assertEqual(delete_salary("Оклад сто тысяч рублей. Работа рядом с домом."), "Работа рядом с домом.")

    def test_does_not_remove_non_salary_numbers(self) -> None:
        source = "Опыт от 3 лет. График 2/2, смена 12 часов. Адрес офиса: дом 15."
        self.assertEqual(delete_salary(source), source)

    def test_normalises_whitespace_and_handles_empty(self) -> None:
        self.assertEqual(delete_salary("  Вакансия\u00a0без\tсуммы  "), "Вакансия без суммы")
        self.assertEqual(delete_salary(None), "")


if __name__ == "__main__":
    unittest.main()
