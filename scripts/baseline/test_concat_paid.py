"""Opt-in real concat evidence: rule, Agent and historical provenance, no new calls."""
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('POSEIDON_CONCAT_RULE'), 'requires saved real rule batch')
class ConcatRuleTests(unittest.TestCase):
    def test_rule_artifacts_and_independent_original_model_reference(self):
        self.check_rule(Path(os.environ['POSEIDON_CONCAT_RULE']),
                        Path(__file__).parent/'cases/concat-agent-6-manifest.json')

    def check_rule(self,root,manifest_path):
        import numpy as np
        from audit_agent_lineage import metadata, read, RESULTS
        from custom_batch_manifest import load_manifest
        from concat_evidence import reference_batch
        from seal_cpu_golden import compare
        report,_=metadata(root/'report.json',RESULTS)
        _,manifest,_=load_manifest(manifest_path)
        self.assertEqual(report['selected_descriptors'],manifest['cases'])
        self.assertEqual(len(report['cases']),len(manifest['cases']))
        self.assertEqual((report['status'],report['agent_calls']),('passed',0))
        self.assertEqual(report['parameters']['security_check'],'tc128')
        self.assertEqual(report['parameters']['modulus_bits'],[60]*14)
        for row,model in zip(report['cases'],manifest['cases']):
            folder=root/row['folder']
            self.assertEqual(row['status'],'passed')
            self.assertTrue(row['execution']['encrypted_execution'])
            self.assertFalse(row['execution']['bootstrap_executed'])
            for name,digest in row['frozen_hashes'].items():self.assertEqual(read(folder/name,folder)[1],digest)
            with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays['reference'],reference_batch(model,arrays['inputs']),atol=1e-12,rtol=1e-12)
                actual=compare(np.load(folder/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
            self.assertEqual(actual,row['comparison'])
        self.assertFalse((root/'private-keys').exists())
        self.assertTrue(metadata(root/'key-cleanup-outcome.json',RESULTS)[0]['complete'])


@unittest.skipUnless(os.environ.get('POSEIDON_CONCAT_AGENT'), 'requires saved real Agent batch')
class ConcatPaidTests(unittest.TestCase):
    def test_six_real_candidates_and_independent_reference(self):
        from audit_concat_batch import audit
        report=audit(Path(os.environ['POSEIDON_CONCAT_AGENT']))
        self.assertEqual((report['status'],report['passed'],report['compared_values']),('covered',6,72))
        self.assertEqual(report['new_api_calls'],0)
        self.assertFalse(report['poseidon_gpu_validated'])
        self.assertFalse(report['upstream_packing_helper_validated'])
        self.assertFalse(report['all_semantics_verified'])


@unittest.skipUnless(os.environ.get('POSEIDON_CONCAT_AXIS_RULE'), 'requires separate axis rule batch')
class ConcatAxisRuleTests(unittest.TestCase):
    def test_rule_axis_discriminator(self):
        ConcatRuleTests.check_rule(self,Path(os.environ['POSEIDON_CONCAT_AXIS_RULE']),
                                  Path(__file__).parent/'cases/concat-axis0-sensitive-1-manifest.json')


@unittest.skipUnless(os.environ.get('POSEIDON_PRE_CONCAT_CAPABILITIES'), 'requires original 135-case audit')
class PreConcatHistoryTests(unittest.TestCase):
    def test_135_case_history_does_not_gain_concat_evidence(self):
        from audit_agent_lineage import metadata,RESULTS
        from audit_model_capabilities import audit_models
        from historical_model_analysis import compare_historical,verify_sources
        report,_=metadata(Path(os.environ['POSEIDON_PRE_CONCAT_CAPABILITIES']),RESULTS)
        self.assertEqual((report['total_agent_cases'],len(report['operator_matrix'])),(135,15))
        verify_sources(report,Path(__file__).parent)
        actual=audit_models([Path(c['report']) for c in report['cohorts']])
        compare_historical(report,actual)
        self.assertEqual(actual['missing_graph_operators'],['concat'])


@unittest.skipUnless(os.environ.get('POSEIDON_CONCAT_AXIS_AGENT'), 'requires separate axis discriminator paid batch')
class ConcatAxisPaidTests(unittest.TestCase):
    def test_frozen_followup_not_relabelled_original_cohort(self):
        from audit_concat_batch import audit
        result=audit(Path(os.environ['POSEIDON_CONCAT_AXIS_AGENT']),
                     Path(__file__).parent/'cases/concat-axis0-sensitive-1-manifest.json')
        self.assertEqual((result['status'],result['planned'],result['passed'],result['compared_values']),('covered',1,1,8))
        self.assertFalse(result['all_semantics_verified'])


@unittest.skipUnless(os.environ.get('POSEIDON_CONCAT_STABILITY'), 'requires fresh-key offline replays')
class ConcatStabilityTests(unittest.TestCase):
    def test_exact_response_request_and_two_new_key_results(self):
        import numpy as np
        from audit_agent_lineage import metadata,read,RESULTS
        from concat_evidence import reference_batch
        from seal_cpu_golden import compare
        report,_=metadata(Path(os.environ['POSEIDON_CONCAT_STABILITY']),RESULTS)
        self.assertEqual((report['status'],report['new_api_calls'],len(report['cases'])),('passed',0,2))
        original=Path(report['original_run'])
        self.assertEqual(metadata(original/'report.json',RESULTS)[1],report['original_report_sha256'])
        model,_=metadata(original/'model.json',RESULTS)
        for row in report['cases']:
            run=Path(row['run']);saved,_=metadata(run/'report.json',RESULTS)
            self.assertEqual(saved['agent_calls'],0)
            self.assertFalse(saved['llm_generation_validated'])
            self.assertEqual(read(run/'request.json',run)[1],read(original/'request.json',original)[1])
            self.assertEqual(read(run/'attempt-00/response.txt',run)[1],report['response_sha256'])
            self.assertTrue(saved['attempts'][0]['execution']['encrypted_execution'])
            for name,digest in saved['frozen_hashes'].items():self.assertEqual(read(run/name,run)[1],digest)
            out=run/'attempt-00/output'
            for name,digest in saved['attempts'][0]['artifact_hashes'].items():self.assertEqual(read(out/name,out)[1],digest)
            with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays['reference'],reference_batch(model,arrays['inputs']),atol=1e-12,rtol=1e-12)
                actual=compare(np.load(out/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
            self.assertEqual(actual,row['comparison'])
            self.assertTrue(actual['passed'])
            self.assertTrue(metadata(run/'key-cleanup-outcome.json',RESULTS)[0]['complete'])


if __name__=='__main__':unittest.main()
