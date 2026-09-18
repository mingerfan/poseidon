"""Independent saved-artifact audit; never calls API or re-executes FHE."""
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np

from chunked_model_cases import cases
from chunked_input_abi import pack_inputs, validate_request_binding
from model_graph import evaluate_reference
from candidate_contract import validate_candidate, request_rotations, request_input_names, canonical
from compiler_configuration import verify_artifact_configuration
from seal_artifact_gate import inspect_artifacts
from seal_cpu_golden import compare


def read(path):return json.loads(path.read_text())


@unittest.skipUnless(os.environ.get('POSEIDON_CHUNKED_MODEL_GOLDENS'),'requires real chunked-input batch')
class ChunkedEvidenceTests(unittest.TestCase):
    def test_original_models_binding_encrypted_results_and_counterexamples(self):
        root=Path(os.environ['POSEIDON_CHUNKED_MODEL_GOLDENS'])
        batch=read(root/'report.json')
        self.assertEqual(batch['status'],'passed')
        self.assertEqual(batch['generator'],'deterministic_chunked_fx')
        self.assertEqual(batch['agent_calls'],0)
        expected=cases()
        self.assertEqual(len(batch['cases']),12)
        for i,row in enumerate(batch['cases']):
            run=Path(row['run']);report=read(run/'report.json')
            self.assertEqual(hashlib.sha256((run/'report.json').read_bytes()).hexdigest(),row['report_sha256'])
            self.assertEqual(row['status'],'passed')
            self.assertEqual(report['agent_calls'],0)
            self.assertFalse(report['llm_generation_validated'])
            self.assertEqual(report['backend'],'upstream_SEAL_HEVM_CPU')
            self.assertFalse(report['poseidon_gpu_validated'])
            for name,value in report['frozen_hashes'].items():
                self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(),value,name)
            descriptor=read(run/'model.json')
            self.assertEqual(descriptor,expected[i] if i<10 else expected[1 if i==10 else 5])
            req=read(run/'request.json');validate_request_binding(req)
            attempt=report['attempts'][0];directory=run/'attempt-00';out=directory/'output'
            response=read(directory/'response.txt');payload=read(directory/'trace-payload.json')
            self.assertEqual(payload['candidate'],response)
            self.assertEqual(payload['request'],req)
            self.assertEqual((directory/'candidate.py').read_text(),response['hecate_source'])
            self.assertEqual(canonical(validate_candidate(response,req)),canonical(attempt['static_check']))
            for name,value in attempt['artifact_hashes'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),value,name)
            gate=inspect_artifacts((out/'lowered._hecate_golden.hevm').read_bytes(),
                (out/'_hecate_golden.cst').read_bytes(),rotation_steps=request_rotations(req),
                expected_inputs=len(request_input_names(req)))
            verify_artifact_configuration(req,gate,req['compiler_profile_sha256'])
            execution=read(out/'execution.json')
            self.assertEqual(execution,attempt['execution'])
            self.assertTrue(execution['encrypted_execution'])
            self.assertFalse(execution['bootstrap_executed'])
            with np.load(run/'arrays.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['inputs'],pack_inputs(a['logical_inputs'],descriptor['input_shape']))
                flat=a['logical_inputs'].reshape(4,-1)
                independent=np.pad(flat,((0,0),(0,(-flat.shape[1])%4))).reshape(4,-1,4)
                np.testing.assert_array_equal(a['inputs'],independent)
                reference=np.array([evaluate_reference(descriptor,x.tolist()) for x in a['logical_inputs']])
                np.testing.assert_array_equal(a['reference'],reference)
            actual=np.load(out/'decrypted.npy',allow_pickle=False)
            comparison=compare(actual,reference,1e-5,1e-4)
            self.assertEqual(comparison,attempt['comparison'])
            self.assertEqual(comparison['passed'],not row['counterexample'])
            cleanup=read(run/'key-cleanup-outcome.json')
            self.assertTrue(cleanup['complete']);self.assertFalse((run/'private-keys').exists())


if __name__=='__main__':unittest.main()
