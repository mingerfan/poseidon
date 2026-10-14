"""Audit live larger-input Agent evidence, not a paid test launcher."""
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('POSEIDON_CHUNKED_PAID_BATCH'),'requires live schema-4 cohort')
class ChunkedPaidEvidenceTests(unittest.TestCase):
    def test_agent_response_original_reference_and_physical_binding(self):
        from audit_agent_lineage import audit, audit_case
        from candidate_contract import validate_candidate, request_input_names, request_rotations, canonical
        from chunked_input_abi import validate_request_binding, pack_inputs
        from chunked_model_cases import cases
        from model_graph import evaluate_reference
        from compiler_configuration import verify_artifact_configuration
        from seal_artifact_gate import inspect_artifacts
        from seal_cpu_golden import compare
        root=Path(os.environ['POSEIDON_CHUNKED_PAID_BATCH'])
        report=json.loads((root/'report.json').read_text())
        checked=audit(root/'report.json')
        self.assertEqual(checked['status'],'coverage_complete')
        self.assertEqual(checked['passed'],10)
        self.assertEqual([r['descriptor'] for r in report['cases']],cases())
        self.assertEqual(report['api_concurrency'],10)
        self.assertEqual(report['native_execution_concurrency'],2)
        self.assertEqual((report['model'],report['service_provider'],report['reasoning_effort']),
                         ('deepseek-flash','deepseek','high'))
        self.assertLessEqual(report['summary']['api_calls'],160)
        self.assertEqual(report['compiler_configuration']['name'],'seal-cpu-eva-w45-v1')
        for row in report['cases']:
            audit_case(row)
            run=Path(row['evidence']);child=json.loads((run/'report.json').read_text())
            self.assertLessEqual(child['provider_metrics']['generation_attempts'],4)
            request=json.loads((run/'request.json').read_text());validate_request_binding(request)
            attempt=next(a for a in reversed(child['attempts']) if a.get('numerically_correct'))
            directory=run/f"attempt-{attempt['index']:02d}";out=directory/'output'
            response=json.loads((directory/'response.txt').read_text())
            payload=json.loads((directory/'trace-payload.json').read_text())
            self.assertEqual(payload['candidate'],response);self.assertEqual(payload['request'],request)
            self.assertEqual((directory/'candidate.py').read_text(),response['hecate_source'])
            self.assertEqual(canonical(validate_candidate(response,request)),canonical(attempt['static_check']))
            self.assertEqual(json.loads((out/'execution.json').read_text()),attempt['execution'])
            gate=inspect_artifacts((out/'lowered._hecate_golden.hevm').read_bytes(),
                                  (out/'_hecate_golden.cst').read_bytes(),
                                  rotation_steps=request_rotations(request),expected_inputs=len(request_input_names(request)))
            verify_artifact_configuration(request,gate,request['compiler_profile_sha256'])
            with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
                descriptor=row['descriptor']
                np.testing.assert_array_equal(arrays['inputs'],pack_inputs(arrays['logical_inputs'],descriptor['input_shape']))
                flat=arrays['logical_inputs'].reshape(4,-1)
                independent=np.pad(flat,((0,0),(0,(-flat.shape[1])%4))).reshape(4,-1,4)
                np.testing.assert_array_equal(arrays['inputs'],independent)
                reference=np.array([evaluate_reference(descriptor,x.tolist()) for x in arrays['logical_inputs']])
                np.testing.assert_array_equal(reference,arrays['reference'])
            actual=np.load(out/'decrypted.npy',allow_pickle=False)
            self.assertEqual(compare(actual,reference,1e-5,1e-4),attempt['comparison'])
            cleanup=json.loads((run/'key-cleanup-outcome.json').read_text())
            self.assertTrue(cleanup['complete']);self.assertFalse((run/'private-keys').exists())
        self.assertEqual(checked['new_api_calls'],0)
        self.assertFalse(checked['all_goal_requirements_complete'])


@unittest.skipUnless(os.environ.get('POSEIDON_CHUNKED_TRACE_REPLAYS'),'requires original-candidate replay batch')
class ChunkedTraceReplayTests(unittest.TestCase):
    def test_original_valid_candidates_after_plain_binding_fix(self):
        from candidate_contract import validate_candidate, canonical
        from chunked_input_abi import pack_inputs, validate_request_binding
        from model_graph import evaluate_reference
        from seal_cpu_golden import compare
        root=Path(os.environ['POSEIDON_CHUNKED_TRACE_REPLAYS'])
        batch=json.loads((root/'report.json').read_text())
        self.assertEqual(batch['status'],'passed');self.assertEqual(batch['agent_calls'],0)
        self.assertEqual(batch['generator'],'saved_agent_candidate_replay')
        self.assertEqual([r['id'] for r in batch['cases']],['chunked-linear-2x4','chunked-residual-3x4'])
        for row in batch['cases']:
            old=Path(row['original_run']);run=Path(row['run'])
            report=json.loads((run/'report.json').read_text())
            original=json.loads((old/'report.json').read_text())
            self.assertEqual(original['attempts'][0]['failure_layer'],'dsl_trace')
            self.assertEqual(report['status'],'passed');self.assertEqual(report['agent_calls'],0)
            self.assertFalse(report['llm_generation_validated'])
            self.assertEqual(hashlib.sha256((run/'report.json').read_bytes()).hexdigest(),row['report_sha256'])
            self.assertEqual((old/'attempt-00/response.txt').read_bytes(),(run/'attempt-00/response.txt').read_bytes())
            self.assertEqual(hashlib.sha256((run/'attempt-00/response.txt').read_bytes()).hexdigest(),row['original_response_sha256'])
            request=json.loads((run/'request.json').read_text());validate_request_binding(request)
            self.assertEqual(request,json.loads((old/'request.json').read_text()))
            for name,value in report['frozen_hashes'].items():
                self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(),value)
            attempt=report['attempts'][0];out=run/'attempt-00/output'
            response=json.loads((run/'attempt-00/response.txt').read_text())
            payload=json.loads((run/'attempt-00/trace-payload.json').read_text())
            self.assertEqual(payload['candidate'],response)
            self.assertEqual(canonical(validate_candidate(response,request)),canonical(attempt['static_check']))
            for name,value in attempt['artifact_hashes'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),value)
            self.assertTrue(attempt['execution']['encrypted_execution'])
            self.assertFalse(attempt['execution']['bootstrap_executed'])
            descriptor=json.loads((run/'model.json').read_text())
            with np.load(run/'arrays.npz',allow_pickle=False) as a:
                reference=np.array([evaluate_reference(descriptor,x.tolist()) for x in a['logical_inputs']])
                np.testing.assert_array_equal(reference,a['reference'])
                np.testing.assert_array_equal(a['inputs'],pack_inputs(a['logical_inputs'],descriptor['input_shape']))
            actual=np.load(out/'decrypted.npy',allow_pickle=False)
            self.assertEqual(compare(actual,reference,1e-5,1e-4),attempt['comparison'])
            self.assertTrue(attempt['comparison']['passed'])
            self.assertFalse((run/'private-keys').exists())


if __name__=='__main__':unittest.main()
