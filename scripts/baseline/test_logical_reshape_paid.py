"""Opt-in saved real Agent/CKKS evidence. Never makes new paid calls."""
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('POSEIDON_LOGICAL_RESHAPE_AGENT'), 'requires real paid reshape evidence')
class LogicalReshapePaidTests(unittest.TestCase):
    def test_three_real_candidates_and_independent_reference(self):
        from audit_logical_reshape_batch import audit
        report = audit(Path(os.environ['POSEIDON_LOGICAL_RESHAPE_AGENT']))
        self.assertEqual((report['status'], report['passed'], report['compared_values']), ('covered', 3, 48))
        self.assertEqual(report['summary']['api_calls'], 3)
        self.assertEqual(report['summary']['mean_repairs_all_completed'], 0)
        self.assertFalse(report['poseidon_gpu_validated'])
        self.assertFalse(report['all_semantics_verified'])
        self.assertEqual(report['new_api_calls'], 0)


@unittest.skipUnless(os.environ.get('POSEIDON_RECENT_CONSTRUCTION_AUDIT'), 'requires saved construction audit')
class RecentConstructionPaidTests(unittest.TestCase):
    def test_reaudit_all_four_frozen_feature_matrices(self):
        from audit_recent_construction import audit
        from audit_agent_lineage import metadata, RESULTS
        saved, _ = metadata(Path(os.environ['POSEIDON_RECENT_CONSTRUCTION_AUDIT']), RESULTS)
        actual = audit()
        import json
        self.assertEqual(json.loads(json.dumps(actual)), saved)
        self.assertEqual(actual['selected_cases'], 63)
        self.assertEqual(actual['compared_values'], 1008)
        self.assertFalse(actual['all_upstream_semantics_proven'])


if __name__ == '__main__':
    unittest.main()
