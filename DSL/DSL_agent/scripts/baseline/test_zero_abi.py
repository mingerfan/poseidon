"""Explicit auxiliary encrypted zero is not x-x, plaintext replacement or bootstrap."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import unittest

from cipher_abi import ZERO_ARGUMENT, execution_options, has_zero_argument, physical_input_names
from hecate_contract import validate_function
from candidate_contract import make_request, request_input_names, validate_candidate, TASK_RULES
from seal_artifact_gate import inspect_artifacts
from dsl_grammar_coverage import analyze_source

BASE = Path(__file__).parent
CASES = ('zero-linear-pure', 'zero-linear-mixed', 'zero-mlp-hidden', 'zero-conv', 'zero-quad', 'zero-multiply')
GOLDENS = ('pure', 'mixed', 'hidden', 'conv', 'quad', 'multiply')


def hand_reference(case, batch):
    """Closed-form answers independent of FX, graph evaluator and compiler."""
    x = batch[0]
    if case == 'zero-linear-pure': return [0.0]
    if case == 'zero-linear-mixed':
        return [.375, math.fsum(w*v for w,v in zip([.5,-.25,.125,.75], x)) - .125]
    if case == 'zero-mlp-hidden': return [.125 - .5*x[0]*x[0]]
    if case == 'zero-conv': return [.25]*3
    if case == 'zero-quad':
        return [0.0, math.fsum(w*sum(values) for w,values in zip([.5,-.25,.125,.75], zip(*batch)))]
    if case == 'zero-multiply': return [0.0]*4
    if case == 'zero-scalar-mask': return [0.0, -.5*x[1]]
    raise ValueError('Unknown independent zero fixture')


def identity_artifact(nargs=2, dst=1):
    body = [40 + 8*(2*nargs+3), 0, nargs, 0, 13]
    body += [40]*nargs + [2]*nargs + [40, 2, dst]
    return struct.pack('<IIQQ', 0x4845564D, 24, nargs, 1) + struct.pack('<'+'Q'*len(body), *body), struct.pack('<q', 0)


class ZeroAbiTests(unittest.TestCase):
    def test_zero_evidence_binding_and_randomization_tampering(self):
        from zero_evidence import verify_zero_execution
        layout = dict(auxiliary_ciphertexts=[dict(ZERO_ARGUMENT)])
        execution = dict(encrypted_execution=True, bootstrap_executed=False, input_batches=4,
            encrypted_input_count=2, ciphertext_metadata=[dict(inputs=[dict(polynomials=2)]*2)]*4,
            auxiliary_encrypted_zero=dict(logical_input_count=1, index=1, fresh_per_batch=True,
                nontransparent=True, binding='trusted_client_public_key_encryption',
                ciphertext_fingerprints=[f'{i:016x}' for i in range(4)]))
        verify_zero_execution(layout, execution)
        for key, value in [('index', True), ('logical_input_count', 2), ('nontransparent', False),
                           ('fresh_per_batch', False), ('ciphertext_fingerprints', ['0'*16]*4)]:
            bad = copy.deepcopy(execution)
            bad['auxiliary_encrypted_zero'][key] = value
            with self.assertRaises(ValueError): verify_zero_execution(layout, bad)
        with self.assertRaises(ValueError): verify_zero_execution({}, execution)
        bad = copy.deepcopy(execution)
        bad['encrypted_input_count'] = 1
        with self.assertRaises(ValueError): verify_zero_execution(layout, bad)

    def test_exact_declaration_and_no_logical_input_redefinition(self):
        layout = dict(input_shape=[4], auxiliary_ciphertexts=[dict(ZERO_ARGUMENT)])
        self.assertEqual(physical_input_names(layout), ('x', 'zero_ct'))
        self.assertEqual(execution_options(layout), dict(expected_inputs=2, logical_inputs=1, encrypted_zero_input=True))
        for change in ({}, dict(ZERO_ARGUMENT, slot_period=True), dict(ZERO_ARGUMENT, value=[1]*4),
                       dict(ZERO_ARGUMENT, dsl_name='x'), dict(ZERO_ARGUMENT, source='candidate')):
            with self.assertRaises(ValueError):
                has_zero_argument(dict(layout, auxiliary_ciphertexts=[change]))
        for entries in ([], [dict(ZERO_ARGUMENT)]*2, dict(ZERO_ARGUMENT)):
            with self.assertRaises(ValueError):
                has_zero_argument(dict(layout, auxiliary_ciphertexts=entries))

    def test_one_to_four_logical_inputs_plus_exactly_one_zero(self):
        for count in range(1, 5):
            names = (*('x', 'y', 'z', 't')[:count], 'zero_ct')
            source = '@hc.func("' + ','.join(['c']*len(names)) + '")\ndef golden(' + ', '.join(names) + '):\n    return zero_ct\n'
            checked = validate_function(source, {}, contract='hecate-function-v4', input_names=names)
            self.assertEqual(checked['auxiliary_encrypted_zero'], 'zero_ct')
            observed = analyze_source(source, {}, contract='hecate-function-v4', input_names=names)
            self.assertIn(f'inputs.{count}', observed['output_dependency_counts'])
            self.assertTrue(observed['auxiliary_encrypted_zero'])
        for names in (('x',), ('x','z'), ('zero_ct','x'), ('x','y','z','t','fifth','zero_ct')):
            with self.assertRaises(ValueError):
                validate_function('invalid', {}, contract='hecate-function-v4', input_names=names)

    def test_legacy_contract_cannot_gain_zero_argument(self):
        payload = dict(static_check=dict(contract='hecate-function-v4'), fx_graph='constant zero graph',
            public_constants={}, constant_origins={}, layout=dict(input_shape=[4], output_ciphertexts=1,
                                                                 auxiliary_ciphertexts=[dict(ZERO_ARGUMENT)]))
        request = make_request(payload, dict(schema=2), 'a'*64)
        self.assertEqual(request['task'], 'hecate-function-synthesis-v5')
        self.assertEqual(request_input_names(request), ('x','zero_ct'))
        for mode in ('old-task', 'missing-zero'):
            bad = copy.deepcopy(request)
            if mode == 'old-task':
                bad['task'] = 'hecate-function-synthesis-v1'
                bad['rules'] = TASK_RULES[bad['task']][1]
            else:
                bad['layout'].pop('auxiliary_ciphertexts')
            with self.assertRaises(ValueError):
                request_input_names(bad)

    def test_zero_operation_identity_artifact_requires_initialized_result(self):
        for nargs in (2, 5):
            hevm, cst = identity_artifact(nargs, nargs-1)
            gate = inspect_artifacts(hevm, cst, expected_inputs=nargs)
            self.assertEqual(gate['opcode_counts'], {})
            self.assertEqual(gate['res_dst'], [nargs-1])
        hevm, cst = identity_artifact(2, 2)
        with self.assertRaises(ValueError):
            inspect_artifacts(hevm, cst, expected_inputs=2)
        with self.assertRaises(ValueError):
            inspect_artifacts(*identity_artifact(5, 4), expected_inputs=6)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires pinned CPU Torch')
class ZeroTranslationTests(unittest.TestCase):
    def test_six_models_rule_manual_torch_and_independent_reference(self):
        import numpy as np
        import torch
        from candidate_trace import evaluate_tree
        from fx_to_hecate import translate
        from model_catalog import build_model, test_inputs
        from model_graph import evaluate_reference
        from multi_input_fixtures import fixture_inputs

        class Cipher:
            def __init__(self, value): self.value = np.asarray(value, dtype=float)
            def __add__(self, other): return Cipher(self.value + (other.value if isinstance(other, Cipher) else other))
            def __sub__(self, other): return Cipher(self.value - (other.value if isinstance(other, Cipher) else other))
            def __mul__(self, other): return Cipher(self.value * (other.value if isinstance(other, Cipher) else other))
            def __neg__(self): return Cipher(-self.value)
            def rotate(self, step): return Cipher(np.roll(self.value, -step))

        for case, golden in zip((*CASES, 'zero-scalar-mask'), (*GOLDENS, 'mask')):
            data = json.loads((BASE/'cases'/f'{case}.json').read_text())
            model, shape = build_model(data)
            translated = translate(model, shape)
            request = make_request(translated, data, 'a'*64)
            self.assertEqual(translated['static_check']['contract'], 'hecate-function-v4')
            names = request_input_names(request)
            multiple = data['schema'] == 3
            originals = np.asarray(fixture_inputs(4)) if multiple else test_inputs(shape).reshape(4,1,4)
            for source in (translated['hecate_source'], (BASE/'golden_cases/encrypted_zero'/f'{golden}.py').read_text()):
                validate_candidate(dict(schema=1, request_id=request['request_id'], hecate_source=source), request)
                for batch in originals:
                    tensors = ([torch.from_numpy(x.copy()) for x in batch] if multiple else
                               [torch.from_numpy(batch[0].reshape(shape).copy())])
                    supplied = {s['name']:x.tolist() for s,x in zip(data['inputs'], batch)} if multiple else batch[0].reshape(shape).tolist()
                    expected = evaluate_reference(data, supplied)
                    np.testing.assert_allclose(expected, hand_reference(case, batch), atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(model(*tensors).detach().numpy(), expected, atol=1e-12, rtol=1e-12)
                    bound = dict(zip(names[:-1], map(Cipher, batch)))
                    bound['zero_ct'] = Cipher([0]*4)
                    constants = {k:np.asarray(v) for k,v in translated['public_constants'].items()}
                    result = evaluate_tree(source, constants, encrypted_inputs=bound)
                    result = result if isinstance(result, list) else [result]
                    actual = [result[i].value[j] for i,j in translated['layout']['output_selectors']]
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            # These direct hand values ensure zero cases are not only graph/self comparisons.
            values = np.asarray([.5,-1,.25,-.75])
            if case == 'zero-linear-mixed':
                np.testing.assert_allclose(evaluate_reference(data, values.tolist()), [.375, -.15625])
            if case == 'zero-mlp-hidden':
                self.assertEqual(evaluate_reference(data, values.tolist()), [0.0])


@unittest.skipUnless(os.environ.get('POSEIDON_ZERO_GOLDEN_REPORT'), 'requires real encrypted-zero golden batch')
class ZeroEvidenceTests(unittest.TestCase):
    def test_real_ciphertexts_references_artifacts_counterexamples_and_cleanup(self):
        import numpy as np
        from audit_agent_lineage import metadata, read
        from hecate_python_env import WORK, ROOT
        from seal_cpu_golden import compare, KEY_BUILD
        from model_catalog import test_inputs
        from multi_input_fixtures import fixture_inputs
        from run_zero_goldens import PLANS
        from zero_evidence import verify_zero_execution

        results = WORK/'results'
        batch, _ = metadata(Path(os.environ['POSEIDON_ZERO_GOLDEN_REPORT']), results)
        self.assertEqual((batch['status'], batch['agent_calls']), ('passed', 0))
        self.assertEqual(len(batch['cases']), len(PLANS))
        for saved, (case, golden, wrong) in zip(batch['cases'], PLANS):
            self.assertEqual((saved['case'], saved['counterexample'], saved['matched_expected']), (case, wrong, True))
            run = Path(saved['run'])
            report, _ = metadata(run/'report.json', results)
            self.assertEqual(report['agent_calls'], 0)
            self.assertFalse(report['llm_generation_validated'])
            self.assertEqual(report['backend'], 'upstream_SEAL_HEVM_CPU')
            self.assertFalse(report['poseidon_gpu_validated'])
            self.assertEqual(report['tolerance'], dict(atol=1e-5, rtol=1e-4))
            params = report['parameters']
            self.assertEqual((params['seal_version'], params['polynomial_degree'], params['security_check'],
                              params['modulus_bits'], params['parameters_set']), ('4.0.0', 32768, 'tc128', [60]*14, True))
            self.assertEqual(report['metadata_observer_sha256'], read(KEY_BUILD/'libseal_golden_metadata.so', WORK)[1])
            self.assertEqual(report['metadata_observer_source_sha256'], read(ROOT/'scripts/baseline/seal_keys/metadata.cpp', ROOT)[1])
            for path, digest in report['frozen_hashes'].items():
                self.assertEqual(read(run/path, run)[1], digest)
            model, _ = metadata(run/'model.json', run)
            self.assertEqual(model, json.loads((BASE/'cases'/f'{case}.json').read_text()))
            self.assertEqual(len(report['attempts']), 1)
            attempt = report['attempts'][0]
            output = run/'attempt-00/output'
            self.assertTrue({'lowered._hecate_golden.hevm', '_hecate_golden.cst', 'lowered.ckks.mlir'} <= set(attempt['artifact_hashes']))
            for path, digest in attempt['artifact_hashes'].items():
                self.assertEqual(read(output/path, output)[1], digest)
            payload, _ = metadata(run/'attempt-00/trace-payload.json', run)
            request, _ = metadata(run/'request.json', run)
            self.assertEqual(payload['request'], request)
            self.assertEqual(payload['candidate']['hecate_source'], (BASE/'golden_cases/encrypted_zero'/f'{golden}.py').read_text())
            self.assertEqual(validate_candidate(payload['candidate'], request), attempt['static_check'])
            self.assertEqual(request['task'], 'hecate-function-synthesis-v5')
            self.assertEqual(attempt['trace']['frontend'], 'real_Hecate')
            self.assertFalse(attempt['trace']['candidate_python_executed'])
            execution = attempt['execution']
            verify_zero_execution(request['layout'], execution)
            self.assertTrue(execution['rotation_key_check']['actual_key_file_verified'])
            gate = inspect_artifacts((output/'lowered._hecate_golden.hevm').read_bytes(),
                (output/'_hecate_golden.cst').read_bytes(), rotation_steps=(-3,-2,-1,1,2,3),
                expected_inputs=len(request_input_names(request)))
            self.assertEqual(gate, attempt['artifact_gate'])
            with np.load(run/'arrays.npz', allow_pickle=False) as arrays:
                expected_inputs = (np.asarray(fixture_inputs(4)) if model['schema'] == 3 else
                                   test_inputs(model['input_shape']).reshape(4,4))
                np.testing.assert_array_equal(arrays['inputs'], expected_inputs)
                logical_batches = expected_inputs if model['schema'] == 3 else expected_inputs[:,None,:]
                expected = np.asarray([hand_reference(case, inputs) for inputs in logical_batches])
                np.testing.assert_allclose(arrays['reference'], expected, rtol=1e-12, atol=1e-12)
                comparison = compare(np.load(output/'decrypted.npy', allow_pickle=False), arrays['reference'], 1e-5, 1e-4)
            self.assertEqual(comparison, attempt['comparison'])
            self.assertEqual(comparison['passed'], not wrong)
            if wrong: self.assertEqual(attempt['failure_layer'], 'numerical_comparison')
            self.assertFalse((run/'private-keys').exists())
            # Tampered bookkeeping must not turn a public zero into an encrypted-input claim.
            for key, value in [('index', 0), ('nontransparent', False), ('fresh_per_batch', False),
                               ('ciphertext_fingerprints', ['0'*16]*4), ('binding', 'candidate')]:
                bad = copy.deepcopy(execution)
                bad['auxiliary_encrypted_zero'][key] = value
                with self.assertRaises(ValueError): verify_zero_execution(request['layout'], bad)


@unittest.skipUnless(os.environ.get('POSEIDON_ZERO_RULE_REPORT'), 'requires actual seven-case rule batch')
class ZeroRuleEvidenceTests(unittest.TestCase):
    def test_rule_baseline_real_ciphertexts_independent_reference_and_key_cleanup(self):
        import numpy as np
        from audit_agent_lineage import metadata, read
        from hecate_python_env import WORK
        from seal_cpu_golden import compare
        from zero_evidence import verify_zero_execution
        path = Path(os.environ['POSEIDON_ZERO_RULE_REPORT'])
        report, _ = metadata(path, WORK/'results')
        self.assertEqual((report['status'], report['agent_calls']), ('passed', 0))
        self.assertEqual([c['case'] for c in report['cases']], [*CASES, 'zero-scalar-mask'])
        self.assertEqual(report['backend'], 'upstream_SEAL_HEVM_CPU')
        self.assertFalse(report['poseidon_gpu_validated'])
        self.assertEqual(report['parameters']['security_check'], 'tc128')
        self.assertEqual(report['parameters']['modulus_bits'], [60]*14)
        for case in report['cases']:
            folder = path.parent/case['folder']
            self.assertEqual(case['status'], 'passed')
            for name, digest in case['frozen_hashes'].items():
                self.assertEqual(read(folder/name, folder)[1], digest)
            verify_zero_execution(case['layout'], case['execution'])
            self.assertTrue(case['execution']['rotation_key_check']['actual_key_file_verified'])
            with np.load(folder/'arrays.npz', allow_pickle=False) as data:
                inputs = data['inputs'] if case['descriptor']['schema'] == 3 else data['inputs'][:,None,:]
                expected = [hand_reference(case['case'], batch) for batch in inputs]
                np.testing.assert_allclose(data['reference'], expected, atol=1e-12, rtol=1e-12)
                self.assertEqual(compare(np.load(folder/'decrypted.npy', allow_pickle=False), data['reference'], 1e-5, 1e-4), case['comparison'])
        self.assertFalse((path.parent/'private-keys').exists())
        cleanup, _ = metadata(path.parent/'key-cleanup-outcome.json', path.parent)
        self.assertTrue(cleanup['complete'])


@unittest.skipUnless(os.environ.get('POSEIDON_ZERO_AGENT_REPORT'), 'requires actual seven-case Agent cohort')
class ZeroAgentEvidenceTests(unittest.TestCase):
    def test_seven_live_generations_reaudit_with_explicit_zero_input(self):
        from audit_agent_lineage import audit, metadata
        from hecate_python_env import WORK
        from zero_evidence import verify_zero_execution
        report = audit(Path(os.environ['POSEIDON_ZERO_AGENT_REPORT']))
        self.assertEqual((report['status'], report['planned'], report['passed']), ('coverage_complete', 7, 7))
        self.assertEqual(set(c['case'] for c in report['cases']), {*CASES, 'zero-scalar-mask'})
        self.assertEqual(report['private_key_directories_retained'], 0)
        self.assertEqual(report['input_executions'], 28)
        self.assertFalse(report['all_goal_requirements_complete'])
        for row in report['cases']:
            self.assertEqual((row['provider'], row['model']), ('deepseek', 'deepseek-flash'))
            run = Path(row['evidence'])
            candidate, _ = metadata(run/'report.json', WORK/'results')
            request, _ = metadata(run/'request.json', run)
            self.assertEqual(request['task'], 'hecate-function-synthesis-v5')
            attempt = next(a for a in reversed(candidate['attempts']) if a.get('numerically_correct'))
            verify_zero_execution(request['layout'], attempt['execution'])


if __name__ == '__main__':
    unittest.main()
