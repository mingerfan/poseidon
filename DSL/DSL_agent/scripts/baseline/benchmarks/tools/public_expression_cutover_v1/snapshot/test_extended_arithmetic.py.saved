"""Versioned arithmetic grammar and real upstream Expr dispatch (C-ABI double).

These unit tests do not claim real encryption; run_extended_arithmetic_goldens.py
is the separate, sandboxed Dacapo/SEAL execution gate.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
import unittest

from candidate_contract import (make_request, validate_candidate, request_input_names,
                                SEMANTIC_GUIDANCE, canonical)
from candidate_trace import evaluate_tree
from cipher_abi import ZERO_ARGUMENT
from hecate_contract import validate_function
from test_frontend_augmented_ops import frontend_double


def fragment(body, names=('x',)):
    return '@hc.func("' + ','.join('c' for _ in names) + '")\ndef golden(' + ', '.join(names) + '):\n    ' + body.replace('\n', '\n    ') + '\n'


def request(zero=False, count=1, extended=True):
    layout = dict(input_shape=[4], output_shape=[4], output_ciphertexts=1,
                  output_selectors=[[0, i] for i in range(4)])
    if count > 1:
        layout.pop('input_shape')
        layout['inputs'] = [dict(name='input'+str(i), dsl_name=n, shape=[4])
                            for i, n in enumerate(('x', 'y', 'z', 't')[:count])]
    if zero:
        layout['auxiliary_ciphertexts'] = [copy.deepcopy(ZERO_ARGUMENT)]
    return make_request(dict(fx_graph='test arithmetic', public_constants={'w': [.5], 'b': .375},
        constant_origins={'w': 'weight', 'b': 'bias'}, layout=layout),
        dict(schema=2, id='arithmetic-unit'), 'a'*64, extended_arithmetic=extended)


class ExtendedArithmeticTests(unittest.TestCase):
    def check(self, body, constants=None, names=('x',), outputs=1):
        source = fragment(body, names)
        return validate_function(source, constants or {'w': .5, 'b': .375}, outputs,
                                 contract='hecate-function-v5', input_names=names)

    def test_reverse_arithmetic_all_public_forms(self):
        for value in (.25, [.25], [.25, -.5, .75, 1.]):
            for op in ('+', '-', '*'):
                check = self.check('return w '+op+' x', {'w': value})
                self.assertEqual(sum(check['operator_counts'].values()), 1)
                self.assertFalse(check['encrypted_correctness_checked'])

    def test_augmented_chain_rebinding_and_counts(self):
        result = self.check('a = x\nx *= w\nx += a\nx = -x\nx -= b\nreturn -x')
        self.assertEqual(result['operator_counts'], dict(add=1, multiply=1, rotate=0, subtract=1, negate=2))

    def test_real_expr_dispatch_preserves_alias_and_subtract_order(self):
        expr = frontend_double()
        for op, opcode in (('+', 6), ('-', 7), ('*', 8)):
            source = fragment('alias = x\nx '+op+'= w\nreturn [alias, x]')
            self.check('alias = x\nx '+op+'= w\nreturn [alias, x]', outputs=2)
            x = expr('input')
            values = evaluate_tree(source, {'w': .25}, x)
            self.assertIs(values[0], x)
            self.assertEqual(values[1].obj, ('binary', opcode, 'input', ('plain', .25)))
            self.assertEqual(x.obj, 'input')

    def test_rhs_is_evaluated_before_rebinding(self):
        expr = frontend_double()
        source = fragment('alias = x\nx -= x * w\nx = x + alias\nreturn x')
        self.check('alias = x\nx -= x * w\nx = x + alias\nreturn x')
        result = evaluate_tree(source, {'w': .5}, expr('x'))
        self.assertEqual(result.obj, ('binary', 6, ('binary', 7, 'x',
            ('binary', 8, 'x', ('plain', .5))), 'x'))

    def test_explicit_plain_left_dispatch(self):
        expr = frontend_double()
        for op, opcode in (('+', 6), ('-', 7), ('*', 8)):
            result = evaluate_tree(fragment('return w '+op+' x'), {'w': expr('plain-vector')}, expr('input'))
            self.assertEqual(result.obj, ('binary', opcode, 'plain-vector', 'input'))

    def test_no_public_mutation_undefined_target_or_new_effects(self):
        for body in ('w += x\nreturn x', 'w = x\nreturn x', 'a -= x\nreturn a',
                     'x /= w\nreturn x', 'x[0] += w\nreturn x', 'x.obj = w\nreturn x',
                     'return w + b', 'x = w\nreturn x', 'return open(w)',
                     'for a in x:\n    x += a\nreturn x', 'return x.bootstrap()',
                     'return x + 1', 'return x.__class__', 'return x.rotate(4)'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.check(body)

    def test_all_physical_signatures_and_zero_readonly(self):
        for count in range(1, 5):
            for zero in (False, True):
                req = request(zero, count)
                names = request_input_names(req)
                source = fragment('x += w\nreturn x', names)
                validated = validate_candidate(dict(schema=1, request_id=req['request_id'], hecate_source=source), req)
                self.assertEqual(len(validated['inputs']), count+int(zero))
                if zero:
                    for stmt in ('zero_ct += x', 'zero_ct = x'):
                        with self.assertRaises(ValueError):
                            self.check(stmt+'\nreturn x', names=names)

    def test_legacy_rules_and_guidance_immutable(self):
        old = request(extended=False)
        old_raw = canonical(old)
        new = request()
        self.assertNotEqual(new['request_id'], old['request_id'])
        self.assertEqual(old['semantic_guidance'], SEMANTIC_GUIDANCE)
        self.assertEqual(canonical(old), old_raw)
        for version in range(5):
            names = ('x', 'y') if version == 3 else ('x', 'zero_ct') if version == 4 else ('x',)
            for body in ('x += w\nreturn x', 'return w - x', 'x = x + w\nreturn x'):
                with self.subTest(version=version, body=body), self.assertRaises(ValueError):
                    validate_function(fragment(body, names), {'w': .5}, contract='hecate-function-v'+str(version), input_names=names)

    def test_augmented_operations_cannot_bypass_budget(self):
        req = request()
        with self.assertRaisesRegex(ValueError, 'operation budget'):
            validate_candidate(dict(schema=1, request_id=req['request_id'],
                hecate_source=fragment('x += w\n'*257+'return x')), req)

    def test_cli_version_is_forwarded(self):
        from run_candidate import parse_args, forward_options
        args = parse_args(['--case', 'fixture.json', '--prepare', '--extended-arithmetic'])
        self.assertIn('--extended-arithmetic', forward_options(args))
        self.assertNotIn('--extended-arithmetic', forward_options(parse_args(['--case', 'fixture.json', '--prepare'])))

    def test_coverage_tracks_value_versions_not_last_variable_spelling(self):
        from dsl_grammar_coverage import analyze_source
        source = fragment('a = x\nx *= w\nx += b\nreturn a')
        result = analyze_source(source, {'w': [.5], 'b': .375}, contract='hecate-function-v5')
        self.assertEqual(result['dead_assignments'], ['x@1', 'x@2'])
        self.assertNotIn('multiply.length1', result['output_dependency_counts'])
        self.assertEqual(result['output_dependency_counts']['statement.alias'], 1)
        source = fragment('x = w - x\na = x\nx *= b\nreturn a + x')
        result = analyze_source(source, {'w': [.5], 'b': .375}, contract='hecate-function-v5')
        self.assertEqual(result['dead_assignments'], [])
        self.assertEqual(result['output_dependency_counts']['subtract.reverse.length1'], 1)
        self.assertEqual(result['output_dependency_counts']['statement.augmented.multiply'], 1)


@unittest.skipUnless(os.environ.get('POSEIDON_EXTENDED_ARITHMETIC_REPORT'),
                     'requires actual sandboxed Dacapo/SEAL evidence')
class ExtendedArithmeticEvidenceTests(unittest.TestCase):
    def test_artifacts_independent_reference_and_wrong_program_rejection(self):
        import numpy as np
        from run_extended_arithmetic_goldens import PLANS
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        from candidate_contract import request_rotations
        root = Path(__file__).resolve().parents[2]
        results = Path(str(RESULTS)).resolve()
        batch = json.loads(Path(os.environ['POSEIDON_EXTENDED_ARITHMETIC_REPORT']).read_text())
        self.assertEqual(batch['status'], 'passed')
        self.assertEqual(batch['agent_calls'], 0)
        self.assertEqual(len(batch['cases']), len(PLANS))
        for row, (case, golden, wrong) in zip(batch['cases'], PLANS):
            self.assertEqual(row['case'], case)
            self.assertEqual(row['counterexample'], wrong)
            self.assertTrue(row['matched_expected'])
            run = Path(row['run']).resolve()
            self.assertTrue(run.is_relative_to(results))
            report = json.loads((run/'report.json').read_text())
            self.assertEqual(report['agent_calls'], 0)
            self.assertFalse(report['llm_generation_validated'])
            self.assertFalse(report['poseidon_gpu_validated'])
            self.assertEqual(report['backend'], 'upstream_SEAL_HEVM_CPU')
            self.assertEqual(report['tolerance'], dict(atol=1e-5, rtol=1e-4))
            self.assertEqual(report['parameters']['security_check'], 'tc128')
            self.assertEqual(report['parameters']['modulus_bits'], [60]*14)
            self.assertEqual(report['parameters']['polynomial_degree'], 32768)
            self.assertEqual(report['frontend_python_sha256'], hashlib.sha256(
                (root/'third_party/dacapo/python/hecate/hecate/expr.py').read_bytes()).hexdigest())
            for name, digest in report['frozen_hashes'].items():
                self.assertTrue((run/name).resolve().is_relative_to(run))
                self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(), digest)
            self.assertEqual(len(report['attempts']), 1)
            item = report['attempts'][0]
            self.assertTrue(item['executed'])
            self.assertTrue(item['execution']['encrypted_execution'])
            self.assertFalse(item['execution']['bootstrap_executed'])
            self.assertEqual(item['execution']['input_batches'], 4)
            self.assertFalse(item['trace']['candidate_python_executed'])
            output = run/'attempt-00/output'
            self.assertTrue({'candidate_trace.mlir', 'lowered.ckks.mlir',
                'lowered._hecate_golden.hevm', '_hecate_golden.cst'} <= set(item['artifact_hashes']))
            for name, digest in item['artifact_hashes'].items():
                self.assertTrue((output/name).resolve().is_relative_to(output))
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), digest)
            payload = json.loads((run/'attempt-00/trace-payload.json').read_text())
            req, candidate = payload['request'], payload['candidate']
            self.assertEqual(req['task'], 'hecate-function-synthesis-v6')
            body = {k: v for k, v in req.items() if k != 'request_id'}
            self.assertEqual(hashlib.sha256(canonical(body)).hexdigest(), req['request_id'])
            self.assertEqual(candidate['hecate_source'],
                (root/'scripts/baseline/golden_cases/extended_arithmetic'/(golden+'.py')).read_text())
            self.assertEqual(validate_candidate(candidate, req), item['static_check'])
            inspected = inspect_artifacts((output/'lowered._hecate_golden.hevm').read_bytes(),
                (output/'_hecate_golden.cst').read_bytes(), rotation_steps=request_rotations(req), expected_inputs=1)
            self.assertEqual(inspected, item['artifact_gate'])
            with np.load(run/'arrays.npz', allow_pickle=False) as arrays:
                x = arrays['inputs']
                # Independent formulas, not evaluation of generated DSL or decrypted intermediates.
                if case == 'broadcast-add-one':
                    expected = x + .375
                elif case == 'broadcast-mul-one':
                    expected = x * -.5
                elif case == 'broadcast-sub-one':
                    expected = x - .375
                elif case == 'broadcast-sub-four':
                    expected = x - np.array([.125, -.25, .5, .75])
                elif case == 'arithmetic-alias-chain':
                    expected = 1.5*x + .375
                else:
                    self.fail('Unspecified reference')
                np.testing.assert_allclose(arrays['reference'], expected, atol=1e-12, rtol=1e-12)
                actual = np.load(output/'decrypted.npy', allow_pickle=False)
                comparison = compare(actual, arrays['reference'], 1e-5, 1e-4)
            self.assertEqual(comparison, item['comparison'])
            self.assertEqual(comparison['passed'], not wrong)
            if wrong:
                self.assertEqual(item['failure_layer'], 'numerical_comparison')
                self.assertGreater(comparison['max_absolute_error'], .1)
            self.assertFalse((run/'private-keys').exists())


if __name__ == '__main__':
    unittest.main()
