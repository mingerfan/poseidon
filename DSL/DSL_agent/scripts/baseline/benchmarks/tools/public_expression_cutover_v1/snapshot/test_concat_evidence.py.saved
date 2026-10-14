"""Opt-in actual Concat FHE evidence; never invokes a provider."""
import hashlib
import json
import os
from pathlib import Path
import unittest

@unittest.skipUnless(os.environ.get('POSEIDON_CONCAT_GOLDENS'),'requires saved real Concat goldens')
class ConcatEvidenceTests(unittest.TestCase):
    def test_exact_models_sources_security_ciphertexts_and_counterexamples(self):
        import numpy as np
        from audit_agent_lineage import metadata,read,RESULTS
        from run_concat_goldens import PLANS
        BASE=Path(__file__).parent
        def descriptor(name):return json.loads((BASE/'cases'/(name+'.json')).read_text())
        from model_graph import evaluate_reference
        from candidate_contract import validate_candidate
        from seal_cpu_golden import compare
        from zero_evidence import verify_zero_execution
        batch,_=metadata(Path(os.environ['POSEIDON_CONCAT_GOLDENS']),RESULTS)
        self.assertEqual((batch['status'],batch['agent_calls']),('passed',0))
        self.assertEqual(len(batch['cases']),len(PLANS))
        for row,(name,source_name,wrong) in zip(batch['cases'],PLANS):
            with self.subTest(name=source_name):
                self.assertEqual((row['case'],row['counterexample'],row['matched_expected']),(name,wrong,True))
                run=Path(row['run']);report,_=metadata(run/'report.json',RESULTS)
                self.assertEqual(report['agent_calls'],0)
                self.assertFalse(report['llm_generation_validated'])
                self.assertFalse(report['poseidon_gpu_validated'])
                self.assertEqual(report['backend'],'upstream_SEAL_HEVM_CPU')
                self.assertEqual(report['parameters']['security_check'],'tc128')
                self.assertEqual(report['parameters']['modulus_bits'],[60]*14)
                self.assertEqual(report['parameters']['polynomial_degree'],32768)
                for file,digest in report['frozen_hashes'].items():self.assertEqual(read(run/file,run)[1],digest)
                actual_model,_=metadata(run/'model.json',run)
                self.assertEqual(actual_model,descriptor(name))
                a=report['attempts'][0];out=run/'attempt-00/output'
                for file,digest in a['artifact_hashes'].items():self.assertEqual(read(out/file,out)[1],digest)
                payload,_=metadata(run/'attempt-00/trace-payload.json',run)
                self.assertEqual(payload['candidate']['hecate_source'],(BASE/'golden_cases/concat'/(source_name+'.py')).read_text())
                self.assertEqual(validate_candidate(payload['candidate'],payload['request']),a['static_check'])
                self.assertTrue(a['execution']['encrypted_execution'])
                self.assertEqual(a['execution']['input_batches'],4)
                self.assertFalse(a['execution']['bootstrap_executed'])
                self.assertFalse(a['trace']['candidate_python_executed'])
                verify_zero_execution(payload['request']['layout'],a['execution'])
                with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
                    from concat_evidence import reference_batch
                    expected=reference_batch(actual_model,arrays['inputs'])
                    np.testing.assert_allclose(arrays['reference'],expected,atol=1e-12,rtol=1e-12)
                    computed=compare(np.load(out/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
                self.assertEqual(a['comparison'],computed)
                self.assertEqual(computed['passed'],not wrong)
                if wrong:self.assertEqual(a['failure_layer'],'numerical_comparison')
                self.assertFalse((run/'private-keys').exists())

if __name__=='__main__':unittest.main()
