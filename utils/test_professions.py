import unittest

try:
    from professions import profession_from_title, resolve_profession
except ModuleNotFoundError:
    from utils.professions import profession_from_title, resolve_profession


class ProfessionTests(unittest.TestCase):
    def test_title_names_the_profession(self):
        cases = {
            "Курьер на своём авто с 18 лет": "Курьер",
            "Kурьер (латинская K)": "Курьер",
            "Автокурьер / Подключение к Яндекс Доставка": "Курьер",
            "Старший кассир в ночь": "Кассир",
            "Сотрудник торгового зала": "Продавец",
            "Продавец-консультант": "Продавец",
            "Сборщик заказов Яндекс Лавка": "Комплектовщик",
            "Кладовщик-комплектовщик": "Комплектовщик",
            "Упаковщик на склад чая": "Комплектовщик",
            "Грузчик на склад": "Грузчик",
            "Водитель категории С, Е": "Водитель грузового транспорта",
            "Водитель самосвала": "Водитель грузового транспорта",
            "Водитель газели межгород": "Водитель грузового транспорта",
            "Водитель с личным 5 тонником (реф)": "Водитель грузового транспорта",
        }
        for title, profession in cases.items():
            with self.subTest(title=title):
                self.assertEqual(profession_from_title(title), profession)

    def test_first_mentioned_profession_wins(self):
        self.assertEqual(profession_from_title("Продавец-кассир"), "Продавец")
        self.assertEqual(profession_from_title("Кассир-продавец"), "Кассир")
        self.assertEqual(profession_from_title("Грузчик-комплектовщик"), "Грузчик")
        self.assertEqual(profession_from_title("Водитель-курьер со своим грузовым автомобилем"),
                         "Водитель грузового транспорта")

    def test_not_one_of_the_six(self):
        for title in ("Водитель погрузчика", "Водитель штабелера", "Водитель категории В", "Водитель такси",
                      "Сборщик мебели", "Менеджер по продажам", "Бариста", "Водитель-экспедитор на своем автомобиле"):
            with self.subTest(title=title):
                self.assertEqual(profession_from_title(title), "")

    def test_query_label_is_the_fallback(self):
        self.assertEqual(resolve_profession("Курьер на своём авто", "Грузчик"), ("Курьер", "title"))
        self.assertEqual(resolve_profession("Сотрудник склада", "Грузчик"), ("Грузчик", "query"))
        self.assertEqual(resolve_profession("Бариста", ""), ("", "none"))


class SharedLabelColumnsTests(unittest.TestCase):
    def test_experiment_builder_skips_the_shared_label_columns(self):
        try:
            from build_experiment_datasets import SHARED_LABEL_COLUMNS
            from vacancy_labels import COLUMNS
        except ModuleNotFoundError:
            from utils.build_experiment_datasets import SHARED_LABEL_COLUMNS
            from utils.vacancy_labels import COLUMNS
        self.assertEqual(SHARED_LABEL_COLUMNS, set(COLUMNS))


if __name__ == "__main__":
    unittest.main()
