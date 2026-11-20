"""An exact resource-rejection dependency never promotes rejected quality."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import autonomous_serial_batch as batch


class ResourceSuccessorStatus(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.producer = self.root / 'producer'
        self.producer.mkdir()
        self.target = self.root / 'successor'
        self.queue = self.root / 'queue'
        self.queue.mkdir()
        self.job = dict(id='conditional-resource-successor', out=str(self.target),
            argv=['python3', '-B', '-m',
                  batch.PREFIX + 'qwen_retained_component_replay',
                  '--out', str(self.target)], seconds=60,
            requires_case_statuses={str(self.producer): 'quality_rejected'})
        self.manifest = dict(schema_version=1, watch=dict(case=str(self.producer),
            pid=123, starttime=456, seconds=60), jobs=[self.job])

    def test_exact_known_rejection_status_is_admitted(self):
        self.assertIs(batch.validate(self.manifest, self.root), self.manifest)
        for status in ('rejected', 'resource_rejected', 'quality_rejected_any'):
            with self.subTest(status=status):
                value = copy.deepcopy(self.manifest)
                value['jobs'][0]['requires_case_statuses'][str(self.producer)] = status
                with self.assertRaises(ValueError):
                    batch.validate(value, self.root)

    def test_matching_dependency_runs_only_its_own_guarded_entry(self):
        (self.producer / 'result.json').write_text(json.dumps(dict(status='quality_rejected')))
        observed = []
        def runner(argv, **kwargs):
            observed.append(argv)
            self.target.mkdir()
            (self.target / 'result.json').write_text(json.dumps(dict(status='ok')))
            return dict(exit_code=0, timed_out=False, signal=None,
                        remaining_pids=[], signals_sent=[])
        rows = batch.execute_jobs([self.job], out=self.queue,
            results_root=self.root, runner=runner, free=lambda _: 20 * batch.GIB)
        self.assertEqual(observed, [self.job['argv']])
        self.assertEqual(rows[0]['status'], 'ok')
        self.assertEqual(rows[0]['experiment_status'], 'ok')
        self.assertFalse(batch.accepted_status(self.job['argv'], 'quality_rejected'))

    def test_other_statuses_skip_without_creating_output(self):
        for status in ('ok', 'error', 'prepared'):
            with self.subTest(status=status):
                (self.producer / 'result.json').write_text(json.dumps(dict(status=status)))
                def unexpected(*args, **kwargs):
                    self.fail('A different producer status launched a successor')
                rows = batch.execute_jobs([self.job], out=self.queue,
                    results_root=self.root, runner=unexpected,
                    free=lambda _: 20 * batch.GIB)
                self.assertEqual(rows[0]['status'], 'dependency_skipped')
                self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
