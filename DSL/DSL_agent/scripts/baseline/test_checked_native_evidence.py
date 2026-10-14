"""Opt-in audit of actual sandbox/SEAL native-function experiments; no API calls."""
import json
import os
from pathlib import Path
import unittest

from hecate_python_env import digest


@unittest.skipUnless(os.environ.get('POSEIDON_CHECKED_NATIVE_BATCH'), 'requires completed native sandbox experiment')
class CheckedNativeEvidenceTests(unittest.TestCase):
    def test_all_four_correct_and_real_wrong_order_counterexample(self):
        import numpy as np
        from seal_cpu_golden import compare
        root = Path(os.environ['POSEIDON_CHECKED_NATIVE_BATCH'])
        report = json.loads((root/'report.json').read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['agent_calls'], 0)
        self.assertEqual(len(report['cases']), 5)
        self.assertEqual([c['counterexample'] for c in report['cases']], [False]*4+[True])
        for row in report['cases']:
            run = Path(row['run'])
            result = json.loads((run/'report.json').read_text())
            self.assertEqual(result['agent_calls'], 0)
            self.assertFalse(result['llm_generation_validated'])
            self.assertFalse(result['poseidon_gpu_validated'])
            self.assertEqual(result['backend'], 'upstream_SEAL_HEVM_CPU')
            self.assertTrue(row['matched_expected'])
            for name, value in result['frozen_hashes'].items():
                self.assertEqual(digest(run/name), value)
            attempt = result['attempts'][0]
            out = run/'attempt-00/output'
            for name, value in attempt['artifact_hashes'].items():
                self.assertEqual(digest(out/name), value)
            self.assertTrue(attempt['compiled'])
            self.assertTrue(attempt['executed'])
            self.assertTrue(attempt['execution']['encrypted_execution'])
            self.assertFalse(attempt['execution']['bootstrap_executed'])
            self.assertEqual(attempt['trace']['construction'], 'validated_native_AST_to_Hecate_functions')
            self.assertFalse(attempt['trace']['candidate_python_executed'])
            actual = np.load(out/'decrypted.npy', allow_pickle=False)
            with np.load(run/'arrays.npz', allow_pickle=False) as arrays:
                reference = arrays['reference']
                inputs = arrays['inputs']
                independent = (inputs*.5+inputs+.375 if row['case'] == 'arithmetic-alias-chain'
                               else inputs[:, 0, :]-inputs[:, 1, :])
                np.testing.assert_allclose(reference, independent, atol=1e-15, rtol=0)
            comparison = compare(actual, reference, 1e-5, 1e-4)
            self.assertEqual(comparison, attempt['comparison'])
            self.assertEqual(comparison['passed'], not row['counterexample'])
            if row['counterexample']:
                self.assertEqual(attempt['failure_layer'], 'numerical_comparison')
            self.assertFalse((run/'private-keys').exists())
            self.assertTrue(json.loads((run/'key-cleanup-outcome.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
