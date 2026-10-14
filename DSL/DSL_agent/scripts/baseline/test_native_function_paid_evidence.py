"""Opt-in replay of live native-function evidence; never calls a paid API."""
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_FUNCTION_PAID_BATCH'),
                     'requires completed live native-function cohort')
class NativeFunctionPaidEvidenceTests(unittest.TestCase):
    def test_real_agent_native_calls_and_encrypted_differential_chain(self):
        from audit_native_function_batch import audit
        from native_function_exercises import EXERCISES
        root = Path(os.environ['POSEIDON_NATIVE_FUNCTION_PAID_BATCH'])
        result = audit(root)
        self.assertEqual(result['status'], 'covered')
        self.assertEqual(result['evidence_kind'], 'live_agent')
        self.assertEqual(result['passed'], 11)
        self.assertEqual(result['live_agent_cases'], 11)
        self.assertEqual(result['input_executions'], 44)
        self.assertEqual(result['compared_values'], 176)
        self.assertEqual([r['id'] for r in result['cases']], list(EXERCISES))
        self.assertTrue(all(r['cases'] for r in result['feature_matrix']))
        self.assertEqual(sum(r['evidence_kind'] == 'finite_output_influence'
                             for r in result['feature_matrix']), 10)
        self.assertEqual(sum(r['evidence_kind'] == 'trace_structural_only'
                             for r in result['feature_matrix']), 1)
        self.assertEqual(result['new_api_calls'], 0)
        self.assertEqual(result['new_fhe_executions'], 0)
        self.assertFalse(result['poseidon_gpu_validated'])
        self.assertFalse(result['all_upstream_semantics_proven'])

    def test_approved_configuration_and_request_limits(self):
        root = Path(os.environ['POSEIDON_NATIVE_FUNCTION_PAID_BATCH'])
        batch = json.loads((root/'report.json').read_text())
        expected = dict(service_provider='deepseek', model='deepseek-flash',
                        reasoning_effort='high', api_concurrency=10,
                        native_execution_concurrency=2, api_timeout=1200,
                        max_tokens=384000, max_repairs=3, provider_retries=3)
        self.assertEqual({k: batch[k] for k in expected}, expected)
        self.assertEqual(len(batch['cases']), 11)
        self.assertLessEqual(batch['summary']['api_calls'], 176)
        generations = 0
        for row in batch['cases']:
            run = Path(row['evidence'])
            report = json.loads((run/'report.json').read_text())
            metrics = report['provider_metrics']
            self.assertGreater(report['agent_calls'], 0)
            self.assertLessEqual(report['agent_calls'], 16)
            self.assertLessEqual(metrics['generation_attempts'], 4)
            self.assertEqual(len(metrics['calls']), report['agent_calls'])
            generations += metrics['generation_attempts']
            self.assertFalse((run/'private-keys').exists())
            cleanup = json.loads((run/'key-cleanup-outcome.json').read_text())
            self.assertTrue(cleanup['complete'])
        self.assertLessEqual(generations, 44)


if __name__ == '__main__':
    unittest.main()
