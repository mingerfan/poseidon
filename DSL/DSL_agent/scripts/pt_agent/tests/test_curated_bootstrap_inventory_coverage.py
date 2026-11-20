"""Metadata-only coverage checks, including a valid zero bootstrap count."""
from pathlib import Path
import unittest

from pt_agent.experiments import experiment_delivery_curated_refresh as subject


class CuratedBootstrapInventoryCoverageChecks(unittest.TestCase):
    @staticmethod
    def cell(name, count, *, primary=True):
        row = subject.pending_cell(Path('/synthetic-results') / name, {})
        row.update(primary_matrix_cell=primary, bootstrap_count=count)
        return row

    def check_coverage(self, rows, known, missing):
        delivery = dict(cells=rows, coverage=dict(cells_with_any_bootstrap_inventory=999))
        subject.coverage(delivery)
        coverage = delivery['coverage']
        self.assertEqual(coverage['cells_with_any_bootstrap_inventory'], known)
        self.assertEqual(coverage['cells_without_bootstrap_inventory'], missing)
        self.assertEqual(known + len(missing), coverage['primary_cells'])
        return delivery

    def test_zero_is_known_and_only_none_is_missing(self):
        zero = self.cell('zero', 0)
        absent = self.cell('unknown', None)
        positive = self.cell('positive', 12)
        self.check_coverage([zero, absent, positive], 2, [absent['cell_id']])

    def test_historical_supplements_do_not_enter_primary_inventory(self):
        absent = self.cell('unknown-primary', None)
        history = self.cell('old-success', 19, primary=False)
        history_missing = self.cell('old-failure', None, primary=False)
        delivery = self.check_coverage([absent, history, history_missing], 0, [absent['cell_id']])
        self.assertEqual(delivery['coverage']['supplemental_variant_rows'], 2)
        self.assertEqual(history['bootstrap_count'], 19)

    def test_new_primary_inventory_recomputes_previous_unknown_count(self):
        first = self.cell('retained-primary', 0)
        added = self.cell('new-primary', None)
        delivery = self.check_coverage([first, added], 1, [added['cell_id']])
        added['bootstrap_count'] = 12
        subject.coverage(delivery)
        self.assertEqual(delivery['coverage']['cells_with_any_bootstrap_inventory'], 2)
        self.assertEqual(delivery['coverage']['cells_without_bootstrap_inventory'], [])
        self.assertEqual(delivery['coverage']['primary_cells'], 2)

    def test_no_primary_cells_has_complementary_empty_inventory(self):
        self.check_coverage([self.cell('history-only', 0, primary=False)], 0, [])


if __name__ == '__main__':
    unittest.main()
