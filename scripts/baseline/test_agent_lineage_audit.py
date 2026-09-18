"""Offline lineage tamper tests; real saved-output checks are opt-in."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_agent_lineage as audit


class LineageAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.a, self.b = dict(id='a'), dict(id='b')
        self.catalog = [dict(descriptor=self.a), dict(descriptor=self.b)]
        self.mock = patch.object(audit, 'batch_plan', return_value=(self.catalog, None))
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def write(self, name, report):
        path = self.root/name
        path.write_text(json.dumps(report))
        return path

    def row(self, descriptor, passed):
        return dict(descriptor=descriptor, status='passed' if passed else 'failed',
                    metrics=dict(passed=passed))

    def chain(self):
        source = self.write('source.json', dict(status='completed_with_failures',
                            cases=[self.row(self.a, True), self.row(self.b, False)]))
        target = self.write('target.json', dict(status='passed', cases=[self.row(self.b, True)],
                            selection=dict(kind='previous_failures_only', source_report=str(source),
                                           source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
        return source, target

    def test_exact_failure_subset_is_preserved(self):
        _, target = self.chain()
        chain = audit.lineage(target, self.root)
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[-1][1]['cases'][0]['descriptor'], self.b)

    def test_changed_parent_hash_rejected(self):
        source, target = self.chain()
        source.write_text(source.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'hash'):
            audit.lineage(target, self.root)

    def test_repeating_prior_success_rejected(self):
        _, target = self.chain()
        data = json.loads(target.read_text())
        data['cases'] = [self.row(self.a, True)]
        target.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'selection'):
            audit.lineage(target, self.root)

    def test_running_batch_and_false_success_rejected(self):
        for status, rows in (('running', [self.row(self.a, True), self.row(self.b, False)]),
                             ('passed', [self.row(self.a, True), self.row(self.b, False)])):
            path = self.write('invalid.json', dict(status=status, cases=rows))
            with self.assertRaises(ValueError):
                audit.lineage(path, self.root)

    def test_external_evidence_rejected(self):
        with self.assertRaises(ValueError):
            audit.read(Path(__file__), self.root)


@unittest.skipUnless(os.environ.get('POSEIDON_AGENT_LINEAGE_REPORT'), 'requires saved completed paid-batch evidence')
class RealLineageEvidenceTests(unittest.TestCase):
    def test_saved_outputs_artifacts_and_cumulative_coverage(self):
        result = audit.audit(Path(os.environ['POSEIDON_AGENT_LINEAGE_REPORT']))
        self.assertEqual(result['status'], 'coverage_complete')
        self.assertEqual((result['planned'], result['passed']), (96, 96))
        self.assertEqual(len(result['family_passes']), 16)
        self.assertEqual(set(result['family_passes'].values()), {6})
        self.assertEqual(result['input_executions'], 384)
        self.assertEqual(result['private_key_directories_retained'], 0)
        self.assertFalse(result['poseidon_gpu_validated'])
        self.assertFalse(result['all_goal_requirements_complete'])
        self.assertEqual(result['new_api_calls'], 0)


if __name__ == '__main__':
    unittest.main()
