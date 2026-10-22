import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from interrupted_batch import load_remaining_source, select_remaining
from run_agent_batch import load_failed_source


class InterruptedBatchTests(unittest.TestCase):
    def fixture(self):
        catalog = [dict(id=str(i)) for i in range(4)]
        rows = [dict(descriptor=d, status='pending') for d in catalog]
        rows[0].update(status='passed', metrics=dict(passed=True))
        rows[1].update(status='failed', metrics=dict(passed=False))
        return catalog, dict(status='interrupted', cases=rows, interruption=dict(workers_stopped=True))

    def test_excludes_success_and_keeps_failures_and_unfinished_in_order(self):
        catalog, prior = self.fixture()
        original = copy.deepcopy(prior)
        self.assertEqual(select_remaining(catalog, prior), catalog[1:])
        self.assertEqual(prior, original)

    def test_rejects_live_and_inconsistent_snapshots(self):
        catalog, prior = self.fixture()
        for update in (dict(status='running'), dict(interruption={}), dict(cases=prior['cases'][::-1])):
            with self.assertRaises(ValueError):
                select_remaining(catalog, dict(prior, **update))
        for index, update in ((0, dict(status='failed')), (2, dict(metrics=dict(passed=True))),
                              (2, dict(status='unknown'))):
            changed = copy.deepcopy(prior)
            changed['cases'][index].update(update)
            with self.assertRaises(ValueError):
                select_remaining(catalog, changed)

    def test_hash_lineage_and_later_failure_retest(self):
        catalog, prior = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'report.json'
            original.write_text(json.dumps(dict(status='running', cases=prior['cases'])))
            prior['interruption'].update(source_report=str(original),
                source_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
            snapshot = root / 'interrupted-report.json'
            snapshot.write_text(json.dumps(prior))
            self.assertEqual(load_remaining_source(catalog, snapshot, root)[1], catalog[1:])
            child = root / 'child.json'
            child.write_text(json.dumps(dict(status='completed_with_failures',
                cases=[dict(descriptor=d, status='failed', metrics=dict(passed=False)) for d in catalog[1:]],
                selection=dict(kind='previous_failures_and_unfinished', source_report=str(snapshot),
                               source_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()))))
            self.assertEqual(load_failed_source(catalog, child, root)[1], catalog[1:])
            # A paused replacement preserves its own new successes as well.
            child_data = json.loads(child.read_text())
            child_data['status'] = 'running'
            child_data['cases'][0].update(status='passed', metrics=dict(passed=True))
            child_data['cases'][1] = dict(descriptor=catalog[2], status='pending')
            child.write_text(json.dumps(child_data))
            child_snapshot = root / 'child-interrupted.json'
            child_data.update(status='interrupted', interruption=dict(workers_stopped=True,
                source_report=str(child), source_sha256=hashlib.sha256(child.read_bytes()).hexdigest()))
            child_snapshot.write_text(json.dumps(child_data))
            self.assertEqual(load_remaining_source(catalog, child_snapshot, root)[1], catalog[2:])
            original.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                load_remaining_source(catalog, snapshot, root)


if __name__ == '__main__':
    unittest.main()
