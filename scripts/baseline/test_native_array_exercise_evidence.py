"""Opt-in audit of completed array cohorts, no API or FHE execution."""
import os
from pathlib import Path
import unittest

from audit_native_array_batch import audit,audit_run


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_ARRAY_EXERCISE_GOLDENS'),'requires actual manual array cohort')
class NativeArrayManualEvidenceTests(unittest.TestCase):
    def test_all_ten_and_wrong_transpose_with_actual_frontend_observation(self):
        result=audit(Path(os.environ['POSEIDON_NATIVE_ARRAY_EXERCISE_GOLDENS']),manual=True)
        self.assertEqual((result['status'],result['passed'],result['live_agent_cases']),('covered',10,0))
        self.assertEqual((result['input_executions'],result['compared_values']),(40,160))
        self.assertTrue(result['counterexample']['wrong'])
        self.assertFalse(result['counterexample']['comparison']['passed'])
        self.assertTrue(all(x['cases'] for x in result['feature_matrix']))
        self.assertEqual(sum(x['evidence_kind']=='trace_structural_only' for x in result['feature_matrix']),1)
        self.assertEqual((result['new_api_calls'],result['new_fhe_executions']),(0,0))
        self.assertFalse(result['full_semantics_proven'])

    def test_manual_cannot_be_upgraded_to_agent(self):
        path=Path(os.environ['POSEIDON_NATIVE_ARRAY_EXERCISE_GOLDENS'])
        with self.assertRaises(ValueError): audit(path)
        result=audit(path,manual=True)
        with self.assertRaisesRegex(ValueError,'Generation kind mismatch'):
            audit_run(result['cases'][0]['evidence'],'na-reverse',live=True)


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_ARRAY_PAID_BATCH'),'requires actual paid array cohort')
class NativeArrayPaidEvidenceTests(unittest.TestCase):
    def test_live_array_generation_trace_and_encrypted_results(self):
        result=audit(Path(os.environ['POSEIDON_NATIVE_ARRAY_PAID_BATCH']))
        self.assertEqual((result['status'],result['passed'],result['live_agent_cases']),('covered',10,10))
        self.assertEqual(result['compared_values'],160)
        self.assertTrue(all(x['cases'] for x in result['feature_matrix']))
        self.assertIsNone(result['counterexample'])
        self.assertFalse(result['full_semantics_proven'])
        self.assertFalse(result['poseidon_gpu_validated'])


if __name__=='__main__': unittest.main()
