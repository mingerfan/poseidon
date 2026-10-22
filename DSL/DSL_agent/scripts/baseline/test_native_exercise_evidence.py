"""Audit real saved native exercise goldens; never call a model or rerun FHE."""
import os
from pathlib import Path
import unittest

from native_function_exercises import EXERCISES


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_EXERCISE_GOLDENS'),'requires actual manual native exercise evidence')
class NativeExerciseEvidenceTests(unittest.TestCase):
    def test_real_source_callsite_ciphertext_and_reference_chain(self):
        from audit_native_function_batch import audit
        root = Path(os.environ['POSEIDON_NATIVE_EXERCISE_GOLDENS'])
        result = audit(root,manual=True)
        self.assertEqual(result['status'],'covered')
        self.assertEqual(result['passed'],11)
        self.assertEqual(result['input_executions'],44)
        self.assertEqual(result['compared_values'],176)
        self.assertEqual(result['live_agent_cases'],0)
        self.assertEqual(result['new_api_calls'],0)
        self.assertEqual(result['new_fhe_executions'],0)
        self.assertFalse(result['poseidon_gpu_validated'])
        self.assertFalse(result['all_upstream_semantics_proven'])
        self.assertEqual([r['id'] for r in result['cases']],list(EXERCISES))
        self.assertEqual(len(result['feature_matrix']),11)
        self.assertTrue(all(r['cases'] for r in result['feature_matrix']))
        self.assertEqual(sum(r['evidence_kind'] == 'finite_output_influence' for r in result['feature_matrix']),10)
        self.assertEqual(sum(r['evidence_kind'] == 'trace_structural_only' for r in result['feature_matrix']),1)

    def test_manual_evidence_cannot_be_promoted_to_live_agent(self):
        from audit_native_function_batch import audit,audit_run
        root = Path(os.environ['POSEIDON_NATIVE_EXERCISE_GOLDENS'])
        with self.assertRaisesRegex(ValueError,'live native-function batch'):
            audit(root)
        result = audit(root,manual=True)
        with self.assertRaisesRegex(ValueError,'generation kind mismatch'):
            audit_run(result['cases'][0]['evidence'],'nf-scalar',live=True)


if __name__ == '__main__': unittest.main()
