import unittest

from utils.dataset_pr import browser_evidence, render, validate_automation_changes


class PublicationTests(unittest.TestCase):
    def test_documentation_only_requires_notes_but_no_regression_test(self):
        path = 'Scraper/skills/collect-avito/SKILL.md'
        validate_automation_changes([dict(file=path, reason='Уточнение', validation='Проверено')], [path])
        with self.assertRaisesRegex(ValueError, 'every source'):
            validate_automation_changes([], [path])

    def test_automation_notes_cover_every_file_test_and_skill(self):
        def note(path):
            return dict(file=path, reason='Новый шаблон', validation='Тест пройден')
        files = ['Scraper/extract.js', 'Scraper/test_aside_runner.py', 'Scraper/skills/collect-avito/SKILL.md']
        validate_automation_changes([note(path) for path in files], files)
        for changes in [[], [note(files[0])], [note(files[0]), note(files[1])]]:
            with self.assertRaises(ValueError):
                validate_automation_changes(changes, files)
        with self.assertRaisesRegex(ValueError, 'regression'):
            validate_automation_changes([note(files[0]), note(files[2])], [files[0]])
    def test_new_and_repeated_observations_and_new_columns(self):
        before = (["source", "id"], [{"source": "avito", "id": "1"}])
        after = (["source", "id", "Цех"], [
            {"source": "avito", "id": "1", "Цех": ""},
            {"source": "avito", "id": "2", "Цех": "Горячий"}])
        result = render("Пять вакансий вахты.", before, after, [
            ("avito", [{"id": "1"}, {"id": "2"}])])
        self.assertIn("Новые ключи: 1; повторные наблюдения: 1", result)
        self.assertIn("1 → 2 строк; 2 → 3 колонок", result)
        self.assertIn("Новые характеристики: Цех", result)

    def test_missing_batch_rows_and_removed_keys_rejected(self):
        before = (["source", "id"], [{"source": "avito", "id": "1"}])
        with self.assertRaisesRegex(ValueError, "removes"):
            render("Сбор", before, (before[0], []), [("avito", [{"id": "2"}])])
        with self.assertRaisesRegex(ValueError, "absent"):
            render("Сбор", before, before, [("avito", [{"id": "2"}])])

    def test_browser_dumps_rejected_but_structured_data_allowed(self):
        for path in ["Data/avito_runs/run/evidence/a.jpg",
                     "Data/avito_runs/run/evidence/filter.txt",
                     "Data/avito_runs/run/1.snapshot.txt",
                     "Data/avito_runs/run/1.png",
                     "Data/avito_runs/run/snapshots/a.html"]:
            self.assertTrue(browser_evidence(path))
        for path in ["Data/avito_runs/run/raw/1.json",
                     "Data/avito_runs/run/evidence/queue.json",
                     "Data/avito_runs/run/exports/vacancies_flat.csv"]:
            self.assertFalse(browser_evidence(path))


if __name__ == "__main__":
    unittest.main()
