"""Opt-in real paid BatchNorm coverage and cumulative model-domain audit."""
import os
import json
from pathlib import Path
import unittest

@unittest.skipUnless(os.environ.get('POSEIDON_BATCH_NORM_AGENT'),'requires real paid BatchNorm batch')
class BatchNormPaidTests(unittest.TestCase):
    def test_six_real_generations_and_fixed_numeric_results(self):
        from audit_batch_norm_batch import audit
        r=audit(Path(os.environ['POSEIDON_BATCH_NORM_AGENT']))
        self.assertEqual((r['status'],r['passed'],r['compared_values']),('covered',6,88))
        self.assertEqual(r['summary']['api_calls'],6)
        self.assertEqual(r['summary']['mean_repairs_all_completed'],0)
        self.assertFalse(r['all_semantics_verified'])
        self.assertFalse(r['upstream_packing_helper_validated'])

@unittest.skipUnless(os.environ.get('POSEIDON_BATCH_NORM_CAPABILITIES'),'requires cumulative132-case audit')
class BatchNormCapabilityTests(unittest.TestCase):
    def test_132_cases_cover_current_14_operators_without_full_dsl_claim(self):
        from audit_agent_lineage import metadata,RESULTS
        from audit_model_capabilities import audit_models
        from historical_model_analysis import verify_sources,compare_historical
        p=Path(os.environ['POSEIDON_BATCH_NORM_CAPABILITIES'])
        report,_=metadata(p,RESULTS)
        self.assertEqual((report['total_agent_cases'],report['graph_cases'],report['legacy_catalog_cases']),(132,84,48))
        self.assertEqual(len(report['operator_matrix']),14)
        self.assertEqual(report['missing_graph_operators'],[])
        self.assertFalse(report['all_goal_requirements_complete'])
        verify_sources(report,Path(__file__).parent)
        actual=audit_models([Path(c['report']) for c in report['cohorts']])
        compare_historical(report,actual)
        self.assertEqual(actual['missing_graph_operators'],['concat','reshape'])

if __name__=='__main__':unittest.main()
