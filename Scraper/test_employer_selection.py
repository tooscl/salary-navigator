import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'skills/collect-avito/scripts'))
import employer_selection as selection


class EmployerSelectionTests(unittest.TestCase):
    def test_selection_rejects_unknown_leader_and_wrong_labels(self):
        rows = [dict(id=str(i), employer_name_observed=name) for i, name in enumerate(['', 'Работа есть', 'Другая', 'Еще'])]
        labels = dict(city_resolved='Екатеринбург', profession_resolved='Кассир', city_resolved_source='card', profession_source='query', labels_rule_version='test')
        with patch.object(selection, 'vacancy_labels', return_value=labels):
            result = selection.select(rows, 1, dict(city='Екатеринбург', profession='Кассир', leader='Работа есть'), {r['id']: 'sha' for r in rows})
            self.assertEqual([d['decision'] for d in result['decisions']], ['employer_unknown', 'excluded_leader_exact_match', 'accepted', 'after_target'])
            self.assertEqual(result['accepted_ids'], ['2'])
            with self.assertRaises(ValueError):
                selection.select(rows, 3, dict(city='Екатеринбург', profession='Кассир', leader='Работа есть'), {r['id']: 'sha' for r in rows})
            with self.assertRaises(ValueError):
                selection.select(rows + rows[:1], 1, dict(city='Екатеринбург', profession='Кассир', leader='Работа есть'), {r['id']: 'sha' for r in rows})

    def test_committed_baseline(self):
        root = Path(__file__).resolve().parents[1]
        result = selection.baseline(root, '7dc8e62cb', 'Екатеринбург', 'Кассир')
        self.assertEqual((result['cohort_count'], result['leader'], result['leader_count']), (385, 'Работа есть', 37))
