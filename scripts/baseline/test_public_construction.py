"""Public construction normalization: semantics, resource boundaries and isolation.

Symbolic doubles below are unit evidence, never real FHE execution.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
import unittest

from candidate_contract import make_request, validate_candidate, request_input_names
from candidate_trace import evaluate_tree
from hecate_contract import validate_function
from public_construction import normalize
from test_extended_arithmetic import fragment
from test_frontend_augmented_ops import frontend_double


class PublicConstructionTests(unittest.TestCase):
    def expand(self, body, constants=None, names=('x',), outputs=1):
        return normalize(fragment(body, names), {} if constants is None else constants,
                         outputs, input_names=names)

    def dispatch(self, result, constants=None):
        expr = frontend_double()
        return evaluate_tree(result['source'], {} if constants is None else constants, expr('x'))

    def test_reduction_loop_expands_in_order(self):
        result = self.expand('total = x\nfor step in range(1, 3):\n    total += total.rotate(step)\nreturn total')
        actual = self.dispatch(result).obj
        pair = ('binary', 6, 'x', ('rotate', 'x', 1))
        self.assertEqual(actual, ('binary', 6, pair, ('rotate', pair, 2)))
        self.assertEqual(result['check']['rotation_steps'], [1, 2])
        self.assertEqual(result['construction']['loop_iterations'], 2)
        self.assertFalse(result['construction']['candidate_python_executed'])

    def test_list_alias_mutation_is_distinct_from_cipher_rebinding(self):
        result = self.expand('items = [x]\nother = items\nold = items[0]\nitems[0] = -x\nreturn [old, other[0]]', outputs=2)
        actual = self.dispatch(result)
        self.assertEqual([x.obj for x in actual], ['x', ('unary', 13, 'x')])
        self.assertEqual(result['construction']['container_writes'], 1)
        result = self.expand('items = [x]\nother = items\nitems = [-x]\nreturn other[0]')
        self.assertEqual(self.dispatch(result).obj, 'x')

    def test_append_alias_and_negative_index(self):
        result = self.expand('items = []\nother = items\nitems.append(-x)\nreturn other[-1]')
        self.assertEqual(self.dispatch(result).obj, ('unary', 13, 'x'))

    def test_nested_containers_and_unpack_snapshot(self):
        result = self.expand('a = x\nb = -x\na, b = (b, a)\nitems = [[a], (b,)]\nreturn (items[0][0], items[1][0])', outputs=2)
        self.assertEqual([x.obj for x in self.dispatch(result)], [('unary', 13, 'x'), 'x'])

    def test_public_if_and_conditional_expression(self):
        result = self.expand('values = []\nfor i in range(2):\n    if i == 0:\n        values.append(x)\n    else:\n        values.append(-x)\nreturn values[1] if len(values) == 2 else x')
        self.assertEqual(self.dispatch(result).obj, ('unary', 13, 'x'))
        self.assertEqual(result['construction']['public_branches'], 3)

    def test_integer_arithmetic_and_negative_range_steps(self):
        result = self.expand('value = x\nfor i in range(3, 0, -1):\n    step = (i * 2 - 1) // 2 + 1\n    value = value.rotate(-step)\nreturn value')
        self.assertEqual(result['check']['rotation_steps'], [-3, -2, -1])
        self.assertEqual(self.dispatch(result).obj, ('rotate', ('rotate', ('rotate', 'x', -3), -2), -1))

    def test_empty_loop_preserves_binding_and_does_not_define_target(self):
        result = self.expand('value = x\nfor i in range(0):\n    value = -x\nreturn value')
        self.assertEqual(self.dispatch(result).obj, 'x')
        with self.assertRaisesRegex(ValueError, 'Undefined'):
            self.expand('for i in range(0):\n    value = x\nreturn value')

    def test_public_list_iteration_observes_append_like_python(self):
        result = self.expand('values = [0]\nout = x\nfor i in values:\n    if i < 2:\n        values.append(i + 1)\n    out = -out\nreturn out')
        self.assertEqual(result['construction']['loop_iterations'], 3)
        self.assertEqual(result['check']['operator_counts']['negate'], 3)

    def test_loop_unpacking_and_multiple_inputs(self):
        result = self.expand('pairs = [(x, y)]\nfor a, b in pairs:\n    total = a - b\nreturn total', names=('x', 'y'))
        expr = frontend_double()
        actual = evaluate_tree(result['source'], {}, encrypted_inputs={'x': expr('x'), 'y': expr('y')})
        self.assertEqual(actual.obj, ('binary', 7, 'x', 'y'))

    def test_named_public_values_are_not_modified(self):
        constants = {'w': [.5, -.25, .75, 1.]}
        before = copy.deepcopy(constants)
        result = self.expand('weights = (w,)\nout = x\nfor weight in weights:\n    out *= weight\nreturn out', constants)
        self.assertEqual(constants, before)
        self.assertFalse(result['construction']['constants_changed'])

    def test_cipher_dependent_controls_are_rejected(self):
        for body in ('if x:\n    x = -x\nreturn x', 'return -x if x == 0 else x',
                     'for i in x:\n    x = -x\nreturn x', 'return [x][x]',
                     'return x[0]', 'return x.rotate(x)', 'return x.rotate(len(x))'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.expand(body)

    def test_no_hidden_effects_even_in_empty_public_branch(self):
        for statement in ('open("secret")', 'import os', 'x.__class__()', 'eval("x")',
                          'while True:\n        x = -x', 'x = [i for i in range(2)]'):
            with self.subTest(statement=statement), self.assertRaises(ValueError):
                self.expand('for i in range(0):\n    '+statement+'\nreturn x')

    def test_list_tuple_and_public_operand_boundaries(self):
        for body in ('a = (x,)\na[0] = x\nreturn x', 'a = [x]\nreturn a[2]',
                     'a = [x]\nreturn a[:]', 'a, b = (x,)\nreturn a',
                     'w = x\nreturn x', 'return w[0] * x', 'return x + 1',
                     'a = [x]\na.append(a)\nreturn x', 'a = [x]\na[0] = a\nreturn x',
                     'a = [x]\nb = [a]\na.append(b)\nreturn x',
                     'a = [x]\na[0] += x\nreturn x'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.expand(body, {'w': [.5]})

    def test_expansion_resource_limits(self):
        for body in ('for i in range(129):\n    x = -x\nreturn x',
                     'for i in range(128):\n    for j in range(128):\n        a = i+j\nreturn x',
                     'for i in range(128):\n    for j in range(3):\n        x = -x\nreturn x',
                     'values = [0]\nfor i in values:\n    values.append(i+1)\nreturn x',
                     'i = 1048576*1048576\nreturn x',
                     'for i in range(1, 2, 0):\n    x = -x\nreturn x',
                     'i = 1//0\nreturn x'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.expand(body)

    def test_normalization_is_deterministic_and_avoids_name_capture(self):
        body = 'constructed0 = x\nfor i in range(2):\n    constructed0 = -constructed0\nreturn constructed0'
        a = self.expand(body)
        self.assertEqual(a, self.expand(body))
        self.assertIn('constructed1', a['source'])
        self.assertFalse(any(type(n) in (ast.For, ast.If, ast.Subscript) for n in ast.walk(ast.parse(a['source']))))

    def test_versions_cli_and_request_do_not_change_old_semantics(self):
        payload = dict(fx_graph='x', public_constants={}, constant_origins={}, layout=dict(
            input_shape=[4], output_shape=[4], output_ciphertexts=1, output_selectors=[[0, i] for i in range(4)]))
        req = make_request(payload, dict(schema=2, id='unit'), 'a'*64, public_construction=True)
        self.assertEqual(req['task'], 'hecate-function-synthesis-v7')
        self.assertEqual(request_input_names(req), ('x',))
        source = fragment('values = [x]\nreturn values[0]')
        candidate = dict(schema=1, request_id=req['request_id'], hecate_source=source)
        self.assertEqual(validate_candidate(candidate, req)['contract'], 'hecate-function-v6')
        for version in range(6):
            names = ('x', 'y') if version == 3 else ('x', 'zero_ct') if version == 4 else ('x',)
            with self.assertRaises(ValueError):
                validate_function(fragment('values = [x]\nreturn values[0]', names), {},
                                  contract='hecate-function-v'+str(version), input_names=names)
        from run_candidate import parse_args, forward_options
        self.assertIn('--public-construction', forward_options(parse_args(['--case', 'case.json', '--prepare', '--public-construction'])))

    def test_output_coverage_uses_expanded_dependencies_not_loop_counts(self):
        from dsl_grammar_coverage import analyze_source
        source = fragment('dead = x\nfor i in range(2):\n    dead = -dead\nreturn x')
        result = analyze_source(source, {}, contract='hecate-function-v6')
        self.assertEqual(result['public_construction']['loop_iterations'], 2)
        self.assertNotIn('negate.cipher', result['output_dependency_counts'])
        self.assertEqual(len(result['dead_assignments']), 2)


@unittest.skipUnless(os.environ.get('POSEIDON_PUBLIC_CONSTRUCTION_REPORT'), 'requires real Dacapo/SEAL evidence')
class PublicConstructionEvidenceTests(unittest.TestCase):
    def test_saved_expansion_independent_reference_and_actual_encryption(self):
        import numpy as np
        from run_public_construction_goldens import PLANS
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        from candidate_contract import canonical, request_rotations
        root = Path(__file__).resolve().parents[2]
        batch = json.loads(Path(os.environ['POSEIDON_PUBLIC_CONSTRUCTION_REPORT']).read_text())
        self.assertEqual(batch['status'], 'passed')
        self.assertEqual(batch['agent_calls'], 0)
        self.assertEqual(len(batch['cases']), len(PLANS))
        for row, (case, golden, wrong) in zip(batch['cases'], PLANS):
            self.assertEqual((row['case'], row['counterexample']), (case, wrong))
            self.assertTrue(row['matched_expected'])
            run = Path(row['run']).resolve()
            self.assertTrue(run.is_relative_to(str(RESULTS)))
            report = json.loads((run/'report.json').read_text())
            self.assertEqual(report['agent_calls'], 0)
            self.assertFalse(report['llm_generation_validated'])
            self.assertFalse(report['poseidon_gpu_validated'])
            self.assertEqual(report['tolerance'], dict(atol=1e-5, rtol=1e-4))
            self.assertEqual(report['parameters']['security_check'], 'tc128')
            self.assertEqual(report['parameters']['modulus_bits'], [60]*14)
            self.assertEqual(report['parameters']['polynomial_degree'], 32768)
            self.assertEqual(report['parameters']['seal_version'], '4.0.0')
            self.assertEqual(report['source_hashes']['public_construction.py'], hashlib.sha256(
                (root/'scripts/baseline/public_construction.py').read_bytes()).hexdigest())
            for name, digest in report['frozen_hashes'].items():
                self.assertTrue((run/name).resolve().is_relative_to(run))
                self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(), digest)
            item = report['attempts'][0]
            self.assertTrue(item['execution']['encrypted_execution'])
            self.assertFalse(item['execution']['bootstrap_executed'])
            self.assertEqual(item['execution']['input_batches'], 4)
            self.assertFalse(item['trace']['candidate_python_executed'])
            output = run/'attempt-00/output'
            self.assertTrue({'normalized-source.py', 'construction.json', 'lowered.ckks.mlir',
                'lowered._hecate_golden.hevm', '_hecate_golden.cst'} <= set(item['artifact_hashes']))
            for name, digest in item['artifact_hashes'].items():
                self.assertTrue((output/name).resolve().is_relative_to(output))
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), digest)
            payload = json.loads((run/'attempt-00/trace-payload.json').read_text())
            req, candidate = payload['request'], payload['candidate']
            self.assertEqual(req['task'], 'hecate-function-synthesis-v7')
            self.assertEqual(req['request_id'], hashlib.sha256(canonical({k: v for k, v in req.items()
                if k != 'request_id'})).hexdigest())
            source = (root/'scripts/baseline/golden_cases/public_construction'/(golden+'.py')).read_text()
            self.assertEqual(candidate['hecate_source'], source)
            self.assertEqual(validate_candidate(candidate, req), item['static_check'])
            expanded = normalize(source, req['public_constants'], req['layout']['output_ciphertexts'],
                                 input_names=request_input_names(req))
            self.assertEqual(expanded['source'], (output/'normalized-source.py').read_text())
            self.assertEqual(expanded['construction'], json.loads((output/'construction.json').read_text()))
            self.assertEqual(item['artifact_gate'], inspect_artifacts(
                (output/'lowered._hecate_golden.hevm').read_bytes(), (output/'_hecate_golden.cst').read_bytes(),
                rotation_steps=request_rotations(req), expected_inputs=len(request_input_names(req))))
            with np.load(run/'arrays.npz', allow_pickle=False) as arrays:
                x = arrays['inputs']
                if case == 'construction-linear':
                    expected = x @ np.array([[.5, -.25, .125, .75], [-.375, .25, .5, -.125]]).T
                elif case == 'arithmetic-alias-chain':
                    expected = 1.5*x + .375
                elif case == 'custom-dual-subtract':
                    expected = x[:, 0, :] - x[:, 1, :]
                else:
                    self.fail('Missing independent formula')
                np.testing.assert_allclose(arrays['reference'], expected, atol=1e-12, rtol=1e-12)
                comparison = compare(np.load(output/'decrypted.npy', allow_pickle=False), arrays['reference'], 1e-5, 1e-4)
            self.assertEqual(comparison, item['comparison'])
            self.assertEqual(comparison['passed'], not wrong)
            if wrong:
                self.assertEqual(item['failure_layer'], 'numerical_comparison')
                self.assertGreater(comparison['max_absolute_error'], .1)
            self.assertFalse((run/'private-keys').exists())


if __name__ == '__main__':
    unittest.main()
