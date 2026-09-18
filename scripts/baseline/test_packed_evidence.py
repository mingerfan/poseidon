"""Read saved encrypted evidence without new API calls or FHE executions."""
import json
import os
from pathlib import Path
import unittest

import numpy as np

from hecate_python_env import WORK,digest
from packed_input_abi import ABI,pack_inputs,validate_request,gate_options
from packed_model_cases import cases
from model_graph import evaluate_reference
from candidate_contract import request_rotations,request_input_names,validate_candidate,canonical
from seal_artifact_gate import inspect_artifacts
from seal_cpu_golden import compare


def read(p):return json.loads(p.read_text())


def check_run(test,run,expected=True,*,pre_prompt_fix_rule=False):
    report=read(run/'report.json');model=read(run/'model.json');request=read(run/'request.json')
    plan=validate_request(request)
    test.assertEqual(report['parameters']['modulus_bits'],[60]*14)
    test.assertEqual(report['parameters']['security_check'],'tc128')
    test.assertEqual(report['parameters']['polynomial_degree'],32768)
    test.assertEqual(report['waterline'],45)
    for name,sha in report['frozen_hashes'].items():test.assertEqual(digest(run/name),sha)
    attempt=report['attempts'][-1];output=run/f"attempt-{attempt['index']:02d}"/'output'
    directory=output.parent
    response=read(directory/'response.txt');payload=read(directory/'trace-payload.json')
    test.assertEqual(payload['candidate'],response);test.assertEqual(payload['request'],request)
    test.assertEqual((directory/'candidate.py').read_text(),response['hecate_source'])
    if pre_prompt_fix_rule:
        # Read-only historical evidence: this rule-only cohort predates the
        # guidance correction. Never weaken the live provider's current gate or
        # rewrite the old request/hash to make it look like a new experiment.
        from candidate_contract import SEMANTIC_GUIDANCE,PACKED_RULES
        from packed_input_abi import CONTRACT
        from hecate_contract import validate_function
        test.assertEqual(report['provider'],'scripted_replay');test.assertEqual(report['agent_calls'],0)
        test.assertEqual(request['semantic_guidance'],SEMANTIC_GUIDANCE)
        test.assertEqual(request['rules'],PACKED_RULES)
        test.assertEqual(set(response),{'schema','request_id','hecate_source'})
        test.assertEqual(response['schema'],1);test.assertEqual(response['request_id'],request['request_id'])
        checked=validate_function(response['hecate_source'],request['public_constants'],
            request['layout']['output_ciphertexts'],contract=CONTRACT,input_names=request_input_names(request),
            slot_period=plan['slot_period'])
        test.assertLessEqual(sum(checked['operator_counts'].values()),1024)
    else:checked=validate_candidate(response,request)
    test.assertEqual(canonical(checked),canonical(attempt['static_check']))
    for name,sha in attempt['artifact_hashes'].items():test.assertEqual(digest(output/name),sha)
    gate=inspect_artifacts((output/'lowered._hecate_golden.hevm').read_bytes(),
        (output/'_hecate_golden.cst').read_bytes(),rotation_steps=request_rotations(request),
        expected_inputs=len(request_input_names(request)),**gate_options(request['layout']))
    execution=attempt['execution']
    from compiler_configuration import verify_artifact_configuration
    verify_artifact_configuration(request,gate,request['compiler_profile_sha256'])
    test.assertEqual(read(output/'execution.json'),execution)
    test.assertTrue(execution['encrypted_execution']);test.assertFalse(execution['bootstrap_executed'])
    test.assertEqual(execution['execution_abi'],ABI);test.assertEqual(execution['input_slot_period'],plan['slot_period'])
    test.assertTrue(execution['rotation_key_check']['actual_key_file_verified'])
    test.assertEqual(execution['rotation_key_check']['required_steps'],gate['rotation_steps'])
    for batch in execution['ciphertext_metadata']:
        observed=batch.get('inputs',[batch.get('input')])
        test.assertEqual(len(observed),len(gate['arg_level']))
        for metadata,level,scale in zip(observed+batch['outputs'],gate['arg_level']+gate['res_level'],
                                       gate['arg_scale']+gate['res_scale']):
            test.assertEqual(metadata['data_modulus_count'],level)
            test.assertAlmostEqual(metadata['log2_scale'],scale,places=6)
            test.assertEqual(metadata['polynomials'],2)
    with np.load(run/'arrays.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['inputs'],pack_inputs(data['logical_inputs'],model['input_shape']))
        reference=np.asarray([evaluate_reference(model,x.tolist()) for x in data['logical_inputs']])
        np.testing.assert_array_equal(reference,data['reference'])
    actual=np.load(output/'decrypted.npy',allow_pickle=False)
    if len(request['layout']['output_shape'])>1:
        from packed_input_abi import decode_output
        logical=np.load(output/'decrypted-logical.npy',allow_pickle=False)
        test.assertEqual(digest(output/'decrypted-logical.npy'),attempt['logical_output_sha256'])
        test.assertEqual(attempt['logical_output_shape'],request['layout']['output_shape'])
        test.assertEqual(logical.shape,(4,*request['layout']['output_shape']))
        np.testing.assert_array_equal(logical,decode_output(actual,request['layout']))
        with np.load(run/'arrays.npz',allow_pickle=False) as data:
            np.testing.assert_array_equal(data['reference_logical'],reference.reshape(logical.shape))
    comparison=compare(actual,reference,1e-5,1e-4)
    test.assertEqual(comparison['passed'],expected)
    test.assertEqual(comparison,attempt['comparison'])
    test.assertTrue(read(run/'key-cleanup-outcome.json')['complete']);test.assertFalse((run/'private-keys').exists())
    return comparison


class PackedEvidenceTests(unittest.TestCase):
    def test_rule_golden_and_counterexamples(self):
        setting=os.environ.get('POSEIDON_PACKED_GOLDENS')
        if not setting:self.skipTest('Set POSEIDON_PACKED_GOLDENS to real FHE evidence')
        root=Path(setting);report=read(root/'report.json')
        self.assertEqual(digest(root/'report.json'),'ed7a8282f4c39a439ed9dabcf016465d02ca22bd0c7d759a6450b74eded003a1')
        self.assertEqual(report['status'],'passed');self.assertEqual(report['agent_calls'],0)
        self.assertEqual(report['generator'],'deterministic_packed_fx')
        self.assertEqual(len(report['cases']),14)
        self.assertEqual([r['id'] for r in report['cases'][:12]],[d['id'] for d in cases()])
        for row in report['cases']:
            run=Path(row['run']);self.assertEqual(digest(run/'report.json'),row['report_sha256'])
            check_run(self,run,not row['counterexample'],pre_prompt_fix_rule=True)

    def test_paid_real_agent_coverage(self):
        setting=os.environ.get('POSEIDON_PACKED_PAID_BATCH')
        if not setting:self.skipTest('Set POSEIDON_PACKED_PAID_BATCH to paid FHE evidence')
        from audit_agent_lineage import audit
        root=Path(setting);report=read(root/'report.json')
        self.assertEqual([r['descriptor'] for r in report['cases']],cases())
        self.assertEqual(report['status'],'passed')
        audited=audit(root/'report.json')
        self.assertEqual(audited['passed'],12);self.assertFalse(audited['all_goal_requirements_complete'])
        for row in report['cases']:
            child=read(Path(row['evidence'])/'report.json')
            self.assertGreater(child['agent_calls'],0);self.assertTrue(child['llm_generation_validated'])
            check_run(self,Path(row['evidence']))


if __name__=='__main__':unittest.main()
