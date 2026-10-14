"""Opt-in audit of the real v22 paid cohort; no generation or FHE rerun."""
import os
from pathlib import Path
import unittest

@unittest.skipUnless(os.environ.get('POSEIDON_V22_AGENT_BATCH'),'requires actual paid v22 evidence')
class ObjectUnaryPaidEvidenceTests(unittest.TestCase):
    def test_frozen_sources_types_artifacts_and_numeric_results(self):
        from audit_object_unary_batch import audit
        result=audit(Path(os.environ['POSEIDON_V22_AGENT_BATCH']))
        self.assertEqual(result['status'],'covered')
        self.assertEqual(result['passed'],8)
        self.assertEqual(result['compared_values'],128)
        self.assertEqual(len(result['feature_matrix']),12)
        self.assertTrue(all(row['cases'] for row in result['feature_matrix']))
        self.assertEqual(result['new_api_calls'],0)
        self.assertFalse(result['poseidon_gpu_validated'])
        self.assertFalse(result['all_upstream_semantics_proven'])

if __name__=='__main__':unittest.main()
