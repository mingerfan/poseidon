"""Opt-in actual scalar conversion evidence; missing records are explicit skips."""
import os
import unittest
import json
import hashlib
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
from function_construction import normalize
from scalar_conversion_exercises import EXERCISES

@unittest.skipUnless(os.environ.get('POSEIDON_SCALAR_CONVERSION_REPORT'),
                     'requires real scalar conversion CPU evidence')
class ScalarConversionEvidenceTests(unittest.TestCase):
    def test_exact_sources_artifacts_security_and_numerical_counterexamples(self):
        import numpy as np
        from candidate_contract import validate_candidate, canonical
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        root=Path(__file__).resolve().parents[2]
        batch=json.loads(Path(os.environ['POSEIDON_SCALAR_CONVERSION_REPORT']).read_text())
        self.assertEqual(batch['status'],'failed')
        self.assertEqual(batch['cases'][-1]['failure_layer'],'seal_runtime')
        self.assertFalse(batch['cases'][-1]['matched_expected'])
        failed_run=Path(batch['cases'][-1]['run'])
        self.assertIn('result ciphertext is transparent',(failed_run/'attempt-00/execute.log').read_text())
        follow=json.loads(Path(os.environ['POSEIDON_SCALAR_INDEX_COUNTEREXAMPLE']).read_text())
        self.assertEqual(follow['status'],'passed')
        self.assertEqual(follow['agent_calls'],0)
        self.assertEqual(len(follow['cases']),1)
        batch['cases']=batch['cases'][:-1]+follow['cases']
        self.assertEqual(batch['agent_calls'],0)
        names=(*EXERCISES,'wrong_int','wrong_item')
        self.assertEqual(len(batch['cases']),len(names))
        for name,row in zip(names,batch['cases']):
            with self.subTest(case=name):
                run=Path(row['run']).resolve()
                self.assertEqual(run.parent,Path(str(RESULTS)))
                data=json.loads((run/'report.json').read_text())
                self.assertEqual(data['agent_calls'],0)
                self.assertFalse(data['llm_generation_validated'])
                self.assertEqual(data['backend'],'upstream_SEAL_HEVM_CPU')
                self.assertFalse(data['poseidon_gpu_validated'])
                self.assertEqual(data['tolerance'],dict(atol=1e-5,rtol=1e-4))
                self.assertEqual(data['parameters']['security_check'],'tc128')
                self.assertEqual(data['parameters']['modulus_bits'],[60]*14)
                self.assertEqual(data['parameters']['polynomial_degree'],32768)
                for filename,digest in data['frozen_hashes'].items():
                    path=(run/filename).resolve()
                    self.assertTrue(path.is_relative_to(run))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
                item=data['attempts'][0]
                output=run/'attempt-00/output'
                for filename,digest in item['artifact_hashes'].items():
                    path=(output/filename).resolve()
                    self.assertTrue(path.is_relative_to(output))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
                req=json.loads((run/'request.json').read_text())
                self.assertEqual(req['task'],'hecate-function-synthesis-v21')
                self.assertEqual(req['request_id'],hashlib.sha256(canonical(
                    {k:v for k,v in req.items() if k!='request_id'})).hexdigest())
                source=(root/'scripts/baseline/golden_cases/scalar_conversion'/(name+'.py')).read_text()
                payload=json.loads((run/'attempt-00/trace-payload.json').read_text())
                self.assertEqual(payload['request'],req)
                self.assertEqual(payload['candidate']['hecate_source'],source)
                self.assertEqual(validate_candidate(payload['candidate'],req),item['static_check'])
                expanded=normalize(source,req['public_constants'],scalar_conversion=True)
                self.assertEqual(expanded['source'],(output/'normalized-source.py').read_text())
                self.assertEqual(expanded['construction'],json.loads((output/'construction.json').read_text()))
                self.assertEqual(expanded['derived_constants'],json.loads((output/'derived-constants.json').read_text()))
                self.assertEqual(item['artifact_gate'],inspect_artifacts(
                    (output/'lowered._hecate_golden.hevm').read_bytes(),
                    (output/'_hecate_golden.cst').read_bytes(),rotation_steps=(-3,-2,-1,1,2,3),expected_inputs=1))
                self.assertTrue(item['execution']['encrypted_execution'])
                self.assertFalse(item['execution']['bootstrap_executed'])
                self.assertEqual(item['execution']['input_batches'],4)
                self.assertFalse(item['trace']['candidate_python_executed'])
                with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
                    x=arrays['inputs']
                    expected=.5*(x*x if name=='sc-overlap' else x)+x+.375
                    np.testing.assert_allclose(arrays['reference'],expected,atol=1e-12,rtol=1e-12)
                    computed=compare(np.load(output/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
                self.assertEqual(computed,item['comparison'])
                self.assertEqual(computed['passed'],not name.startswith('wrong_'))
                self.assertTrue(row['matched_expected'])
                if name.startswith('wrong_'):
                    self.assertEqual(item['failure_layer'],'numerical_comparison')
                    self.assertGreater(computed['max_absolute_error'],.1)
                self.assertFalse((run/'private-keys').exists())



@unittest.skipUnless(os.environ.get('POSEIDON_V21_AGENT_BATCH'),'requires actual paid v21 evidence')
class ScalarPaidEvidenceTests(unittest.TestCase):
    def test_real_agent_case_and_feature_matrix(self):
        from audit_scalar_conversion_batch import audit
        result=audit(Path(os.environ['POSEIDON_V21_AGENT_BATCH']))
        self.assertEqual(result['status'],'covered')
        self.assertEqual(result['passed'],15)
        self.assertEqual(result['compared_values'],240)
        self.assertTrue(all(row['cases'] for row in result['feature_matrix']))
        self.assertEqual(result['new_api_calls'],0)
        self.assertFalse(result['poseidon_gpu_validated'])
        self.assertFalse(result['all_upstream_semantics_proven'])


if __name__=='__main__':unittest.main()
