import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from build_segments import label_decision, source_groups
import evaluate_pilot

OUT = Path(__file__).resolve().parent / 'results'

class LabelContract(unittest.TestCase):
    def row(self, text, reasons='', tax='net'):
        return {'salary_after_candidate': text, 'after_reasons': reasons, 'after_tax_status': tax}

    def test_open_floor_never_becomes_midpoint(self):
        x, status = label_decision(self.row('от 4 000 ₽ за смену на руки'))
        self.assertEqual(status, 'eligible')
        self.assertEqual(x['y_floor_net_shift_rub'], 4000)
        self.assertIsNone(x['y_midpoint_closed_net_shift_rub'])
        self.assertFalse(x['closed_label'])

    def test_closed_interval_has_distinct_floor_and_center(self):
        x, _ = label_decision(self.row('4 000 — 6 000 ₽ за смену на руки'))
        self.assertEqual((x['y_floor_net_shift_rub'], x['y_midpoint_closed_net_shift_rub']), (4000, 5000))

    def test_unknown_period_tax_conflicts_and_cycle_amount_rejected(self):
        for row in [self.row('4000 ₽ на руки'), self.row('4000 ₽ за смену', tax='unknown'),
                    self.row('4000 ₽ за смену на руки', reasons='card_body_conflict'),
                    self.row('4000 ₽ за смену на руки', reasons='piecework_context'),
                    self.row('130000 ₽ за смену на руки'), self.row('3000 — 10000 ₽ за смену на руки')]:
            with self.subTest(row=row):
                self.assertIsNone(label_decision(row)[0])

    def test_military_rejected(self):
        self.assertIsNone(label_decision(self.row('4000 ₽ за смену на руки'), military=True)[0])

class DeliveredData(unittest.TestCase):
    def test_unique_rows_no_source_overlap_and_equal_group_weights(self):
        for name in ['loading', 'packing', 'picking', 'warehouse_pool']:
            for suffix in ['', '_point']:
                d = pd.read_csv(OUT / (name + suffix + '.csv'))
                self.assertTrue(d.id.is_unique)
                self.assertEqual(d.groupby('split_group').cv_fold.nunique().max(), 1)
                self.assertTrue(np.allclose(d.groupby('split_group').training_weight.sum(), 1))
                if suffix:
                    self.assertFalse((d.label_shape == 'from').any())
                    self.assertTrue(d.y_midpoint_closed_net_shift_rub.notna().all())

    def test_allowlist_has_no_money_text_or_identity(self):
        cols = json.loads((OUT / 'feature_allowlist.json').read_text())
        for col in cols:
            self.assertTrue(col.startswith('x__'))
            self.assertFalse(any(t in col.lower() for t in ['salary', 'evidence', 'employer', 'source_dependence', 'y_floor']))

    def test_role_children_disjoint_and_selection_audit_complete(self):
        ids = [set(pd.read_csv(OUT / (r + '.csv')).id) for r in ['loading', 'packing', 'picking']]
        self.assertEqual(sum(map(len, ids)), len(set.union(*ids)))
        pool = pd.read_csv(OUT / 'warehouse_pool.csv')
        self.assertTrue(set.union(*ids) <= set(pool.id))
        audit = pd.read_csv(OUT / 'all7193_selection_audit.csv')
        self.assertEqual((len(audit), audit.id.nunique()), (7193, 7193))
        self.assertEqual(int(audit.included.sum()), len(pool))
        anchor = pd.read_csv(OUT / 'warehouse_text_linked_point.csv')
        self.assertTrue(anchor.body_rate_linked.all())
        self.assertFalse((anchor.label_shape == 'from').any())
        self.assertTrue(set(anchor.id) <= set(pool.id))
        self.assertTrue(np.allclose(anchor.groupby('split_group').training_weight.sum(), 1))

class ReproductionContracts(unittest.TestCase):
    # Value: protects=label scope boundaries; fails_when=shape or inequalities change;
    # why_new=existing cases omit exact labels and boundary values; seam=none;
    # no_claim=labels remain advertised amounts, not verified income.
    def test_label_boundaries_and_upper_only(self):
        for amount, eligible in [(999, False), (1000, True), (15000, True), (15001, False)]:
            row = {'salary_after_candidate': f'{amount} ₽ за смену на руки',
                   'after_reasons': '', 'after_tax_status': 'net'}
            label, _ = label_decision(row)
            self.assertEqual(label is not None, eligible)
            if label:
                self.assertEqual(label['y_midpoint_closed_net_shift_rub'], amount)
        row['salary_after_candidate'] = 'до 4000 ₽ за смену на руки'
        self.assertIsNone(label_decision(row)[0])

    # Value: protects=transitive dependence grouping; fails_when=any linkage is omitted;
    # why_new=delivered CSV tests cannot detect missed links; seam=none;
    # no_claim=synthetic links do not establish legal employer identity.
    def test_transitive_groups_masked_templates_and_precise_sites(self):
        ids = ['A', 'B', 'C', 'D', 'E', 'F']
        site = 'Москва, Складская улица, дом 42'
        decisions = pd.DataFrame({
            'id': ids, 'source_dependence_group': ['ab', 'ab', 'c', 'd', 'e', 'f'],
            'workplace_candidate': ['', '', site, site, site, site],
            'workplace_status': ['unknown', 'unknown', 'resolved', 'resolved', 'unknown', 'unknown']})
        raw = pd.DataFrame({'id': ids, 'description': [
            'Отдельная работа А', 'Сборка товаров за 4000 рублей',
            'Сборка товаров за 5000 рублей', 'Другая работа Д', '', '']}).set_index('id')
        groups = source_groups(decisions, raw)
        self.assertEqual(len({groups[i] for i in 'ABCD'}), 1)
        self.assertEqual(len({groups['A'], groups['E'], groups['F']}), 3)
        self.assertEqual(groups, source_groups(decisions.iloc[::-1], raw))

    # Value: protects=pinned raw input; fails_when=SHA rejection is removed;
    # why_new=unit labels do not execute the CLI input boundary; seam=none;
    # no_claim=synthetic wrong bytes test rejection, not a complete extraction.
    def test_cli_rejects_unpinned_raw_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'wrong.csv'
            raw.write_text('id\nsynthetic\n')
            out = Path(tmp) / 'results'
            p = subprocess.run([sys.executable, str(OUT.parent / 'build_segments.py'),
                                '--raw-csv', str(raw), '--out', str(out)], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('Pinned raw CSV mismatch', p.stderr)
            self.assertEqual(list(out.iterdir()), [])

    # Value: protects=weighted comparator; fails_when=weights or ordering are ignored;
    # why_new=no existing test calls evaluator statistics; seam=none;
    # no_claim=small synthetic values verify the median convention only.
    def test_weighted_median_order_weights_and_tie(self):
        self.assertEqual(evaluate_pilot.weighted_median([30, 10, 20], [.1, .1, .8]), 20)
        self.assertEqual(evaluate_pilot.weighted_median([30, 10, 20], [.1, .8, .1]), 10)
        self.assertEqual(evaluate_pilot.weighted_median([20, 10], [1, 1]), 10)

    # Value: protects=published grouped OOF experiment reproducibility;
    # fails_when=folds, fitting, comparator or metrics drift; why_new=existing tests do not train;
    # seam=none; no_claim=reproducing silver-label results does not prove future accuracy.
    def test_pilot_reproduces_published_oof_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name in ['feature_allowlist.json'] + [r + s + '.csv'
                         for r in ['loading', 'packing', 'picking', 'warehouse_pool']
                         for s in ['', '_point']]:
                shutil.copyfile(OUT / name, target / name)
            with patch.object(evaluate_pilot, 'OUT', target), patch('builtins.print'):
                evaluate_pilot.main()
            for name in ['pilot_metrics.csv', 'pilot_oof_predictions.csv']:
                actual, expected = pd.read_csv(target / name), pd.read_csv(OUT / name)
                pd.testing.assert_frame_equal(actual, expected, rtol=1e-10, atol=1e-10)

if __name__ == '__main__':
    unittest.main()
