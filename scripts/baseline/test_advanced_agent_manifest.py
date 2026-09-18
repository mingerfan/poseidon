"""Exact next paid cohort; plan checks are offline, live evidence is opt-in."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

from custom_batch_manifest import load_manifest
from model_semantic_coverage import analyze_graph

BASE=Path(__file__).parent
MANIFEST=BASE/'cases/advanced-agent-12-manifest.json'


class AdvancedManifestTests(unittest.TestCase):
    def test_exact_existing_ten_plus_two_powers_and_no_hidden_model_substitution(self):
        rows,data,_=load_manifest(MANIFEST)
        original=load_manifest(BASE/'cases/advanced-shapes-manifest.json')[1]['cases']
        powers=[json.loads((BASE/'cases'/name).read_text()) for name in ('explicit-power-2.json','explicit-power-4.json')]
        self.assertEqual(data['cases'],original+powers)
        self.assertEqual(len(rows),12)
        features={f for c in data['cases'] for f in analyze_graph(c)['features']}
        self.assertTrue({'power.exponent.2','power.exponent.4','conv1d.groups.2','conv1d.dilation.3'} <= features)
        self.assertTrue({f'linear.output_width.{n}' for n in (5,6,7,8)} <= features)

    def test_twelve_case_plan_uses_flash_ten_workers_without_credentials(self):
        import run_agent_batch
        output=io.StringIO()
        with patch('sys.argv',['batch','--plan','--case-manifest',str(MANIFEST)]), \
             patch('agent_credentials.load_api_key',side_effect=AssertionError('not approved for live calls')), \
             redirect_stdout(output):
            self.assertEqual(run_agent_batch.main(),0)
        plan=json.loads(output.getvalue())
        self.assertEqual(len(plan['cases']),12)
        self.assertEqual((plan['service_provider'],plan['model'],plan['api_concurrency'],plan['agent_calls']),
                         ('deepseek','deepseek-flash',10,0))


@unittest.skipUnless(os.environ.get('POSEIDON_ADVANCED_12_AGENT_REPORT'),'requires approved real 12-case Agent run')
class AdvancedTwelveEvidenceTests(unittest.TestCase):
    def test_inventory_matches_actual_paid_batch_and_cleanup(self):
        from audit_agent_lineage import metadata,RESULTS,audit
        from dsl_semantic_inventory import inventory
        record=inventory()['advanced_agent_evidence']
        path=Path(os.environ['POSEIDON_ADVANCED_12_AGENT_REPORT'])
        report,digest=metadata(path,RESULTS)
        self.assertEqual(str(path),record['report'])
        self.assertEqual(digest,record['sha256'])
        self.assertEqual(report['custom_manifest']['sha256'],record['manifest_sha256'])
        self.assertEqual((report['model'],report['api_concurrency'],report['native_execution_concurrency']),
                         ('deepseek-flash',10,2))
        self.assertEqual(report['summary']['first_success'],dict(numerator=12,denominator=12,rate=1.0))
        self.assertEqual(report['summary']['api_calls'],12)
        self.assertEqual(report['summary']['transport_retries'],0)
        self.assertEqual(report['summary']['mean_repairs_all_completed'],0)
        self.assertEqual(audit(path)['private_key_directories_retained'],0)

    def test_same_twelve_graphs_real_ciphertexts_and_required_parameter_coverage(self):
        from audit_agent_lineage import audit,metadata,RESULTS
        from audit_model_capabilities import audit_models
        path=Path(os.environ['POSEIDON_ADVANCED_12_AGENT_REPORT'])
        actual=audit(path)
        expected={c['id']:c for c in load_manifest(MANIFEST)[1]['cases']}
        self.assertEqual((actual['planned'],actual['passed'],actual['input_executions']),(12,12,48))
        self.assertEqual(set(c['case'] for c in actual['cases']),set(expected))
        for row in actual['cases']:
            self.assertEqual((row['provider'],row['model']),('deepseek','deepseek-flash'))
            self.assertEqual(metadata(Path(row['evidence'])/'model.json',RESULTS)[0],expected[row['case']])
        coverage=audit_models([path])
        self.assertEqual((coverage['graph_cases'],coverage['legacy_catalog_cases']),(12,0))
        features={r['feature'] for r in coverage['feature_matrix']}
        self.assertTrue({'power.exponent.2','power.exponent.4','conv1d.groups.2','conv1d.dilation.3'} <= features)
        self.assertTrue({f'linear.output_width.{n}' for n in (5,6,7,8)} <= features)


@unittest.skipUnless(os.environ.get('POSEIDON_EXTENDED_CAPABILITY_REPORT'),'requires actual 126-case capability audit')
class CompletedCapabilityEvidenceTests(unittest.TestCase):
    def test_recomputed_126_cases_all_13_graph_operators_and_historical_split(self):
        from audit_agent_lineage import metadata,read,RESULTS
        from audit_model_capabilities import audit_models
        from dsl_semantic_inventory import inventory
        record=inventory()['model_capability_evidence']
        path=Path(os.environ['POSEIDON_EXTENDED_CAPABILITY_REPORT'])
        report,digest=metadata(path,RESULTS)
        self.assertEqual((str(path),digest),(record['report'],record['sha256']))
        self.assertEqual((report['total_agent_cases'],report['graph_cases'],report['legacy_catalog_cases']),(126,78,48))
        self.assertEqual(report['missing_graph_operators'],[])
        self.assertEqual(len(report['operator_matrix']),13)
        self.assertTrue(all(row['observed_cases'] for row in report['operator_matrix']))
        self.assertFalse(report['full_model_domain_verified'])
        self.assertFalse(report['tests_executed_by_summary'])
        from historical_model_analysis import verify_sources,compare_historical
        verify_sources(report,BASE)
        actual=audit_models([Path(c['report']) for c in report['cohorts']])
        compare_historical(report,actual)
        self.assertEqual(actual['missing_graph_operators'],['batch_norm','concat','reshape'])
        self.assertEqual(len(actual['operator_matrix']),16)


if __name__=='__main__':
    unittest.main()
