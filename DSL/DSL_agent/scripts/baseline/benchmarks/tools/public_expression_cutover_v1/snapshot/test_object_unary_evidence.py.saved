"""Explicit real object-unary CPU evidence; never substitutes for paid Agent coverage."""
import os
import unittest
import json
import hashlib
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
from function_construction import normalize

@unittest.skipUnless(os.environ.get('POSEIDON_OBJECT_UNARY_REPORT'),
                     'requires real object unary CPU evidence')
class ObjectUnaryEvidenceTests(unittest.TestCase):
    def test_exact_sources_artifacts_security_and_numerical_counterexamples(self):
        import numpy as np
        from candidate_contract import validate_candidate, canonical
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        root=Path(__file__).resolve().parents[2]
        batch=json.loads(Path(os.environ['POSEIDON_OBJECT_UNARY_REPORT']).read_text())
        self.assertEqual(batch['status'],'passed')
        self.assertEqual(batch['agent_calls'],0)
        names=('neg','zero','view','positive','negative_call','positive_call','plain','mixed','wrong_neg','wrong_copy')
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
                self.assertEqual(req['task'],'hecate-function-synthesis-v22')
                self.assertEqual(req['request_id'],hashlib.sha256(canonical(
                    {k:v for k,v in req.items() if k!='request_id'})).hexdigest())
                source=(root/'scripts/baseline/golden_cases/object_unary'/(name+'.py')).read_text()
                payload=json.loads((run/'attempt-00/trace-payload.json').read_text())
                self.assertEqual(payload['request'],req)
                self.assertEqual(payload['candidate']['hecate_source'],source)
                self.assertEqual(validate_candidate(payload['candidate'],req),item['static_check'])
                expanded=normalize(source,req['public_constants'],object_unary=True)
                self.assertEqual(expanded['source'],(output/'normalized-source.py').read_text())
                self.assertEqual(expanded['construction'],json.loads((output/'construction.json').read_text()))
                self.assertEqual(expanded['derived_constants'],json.loads((output/'derived-constants.json').read_text()))
                self.assertEqual(item['artifact_gate'],inspect_artifacts(
                    (output/'lowered._hecate_golden.hevm').read_bytes(),
                    (output/'_hecate_golden.cst').read_bytes(),rotation_steps=(-3,-2,-1,1,2,3),expected_inputs=1))
                self.assertTrue(item['execution']['encrypted_execution'])
                if name in ('neg','zero','view','negative_call','mixed'):
                    self.assertGreater(item['artifact_gate']['opcode_counts'].get('2',0),0)
                elif not name.startswith('wrong_'):
                    self.assertEqual(item['artifact_gate']['opcode_counts'].get('2',0),0)
                self.assertFalse(item['execution']['bootstrap_executed'])
                self.assertEqual(item['execution']['input_batches'],4)
                self.assertFalse(item['trace']['candidate_python_executed'])
                with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
                    np.testing.assert_allclose(arrays['reference'],1.5*arrays['inputs']+.375,atol=1e-12,rtol=1e-12)
                    computed=compare(np.load(output/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
                self.assertEqual(computed,item['comparison'])
                self.assertEqual(computed['passed'],not name.startswith('wrong_'))
                self.assertTrue(row['matched_expected'])
                if name.startswith('wrong_'):
                    self.assertEqual(item['failure_layer'],'numerical_comparison')
                    self.assertGreater(computed['max_absolute_error'],.1)
                self.assertFalse((run/'private-keys').exists())



if __name__=='__main__':unittest.main()
