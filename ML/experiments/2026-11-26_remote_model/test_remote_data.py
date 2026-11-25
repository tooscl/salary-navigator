"""Guard the evidence distinctions that define the remote salary target."""
import unittest

from remote_data import card_quote, description_rates, rate_target, role_of


class TargetTests(unittest.TestCase):
    def test_card_ceiling(self):
        out = card_quote("до 75 000 ₽ в месяц на руки")
        self.assertEqual((out["shape"], out["high"], out["unit"], out["tax"]), ("to", 75000, "month", "net"))

    def test_matching_bounds_and_period(self):
        out, why, weak = rate_target("50 000 — 75 000 ₽ на руки", "Оплата 50 000 — 75 000 ₽ в месяц на руки.")
        self.assertEqual((out["unit"], why, weak), ("month", "matched_description", False))

    def test_frequency_is_not_salary_period(self):
        out, why, weak = rate_target("75 000 ₽", "Оклад 75 000 ₽. Выплаты два раза в месяц.")
        self.assertEqual((out["unit"], why, weak), ("unspecified", "card_unknown_period", False))

    def test_amount_without_currency_inside_pay_clause(self):
        out, why, weak = rate_target("75 000 ₽", "Оклад 75 000 в месяц.")
        self.assertEqual((out["unit"], why, weak), ("month", "matched_description", False))

    def test_weak_period_cannot_confirm_different_amount(self):
        out, why, weak = rate_target("75 000 ₽", "Зарплата 85 000 ₽ в месяц.")
        self.assertEqual((out["unit"], why, weak), ("unspecified", "card_unknown_period", True))

    def test_conflicting_period(self):
        out, why, _ = rate_target("750 ₽ за час", "Зарплата 750 ₽ за урок.")
        self.assertEqual((out["unit"], why), ("conflict", "card_text_conflict"))

    def test_thousands_with_explicit_period(self):
        rates = description_rates("Оклад от 60 тыс ₽ в месяц.")
        self.assertTrue(any(r["low"] == 60000 and r["unit"] == "month" and r["shape"] == "from" for r in rates))

    def test_bonus_not_primary_pay(self):
        self.assertEqual(description_rates("Бонус 50 000 ₽ в месяц."), [])

    def test_description_only_is_separately_traced(self):
        out, why, _ = rate_target("Зарплата не указана", "Ставка 800 ₽ за урок.")
        self.assertEqual((out["unit"], why), ("lesson", "description_only"))

    def test_title_roles(self):
        self.assertEqual(role_of("Оператор ПК"), "Ассистенты")
        self.assertEqual(role_of("Оператор раннего взыскания"), "Взыскание")
