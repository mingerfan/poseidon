"""Scoped helper semantics; symbolic unit evidence is separate from real FHE."""
import ast
import hashlib
import json
import os
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
import unittest

from function_construction import normalize
from candidate_trace import evaluate_tree
from test_extended_arithmetic import fragment
from test_frontend_augmented_ops import frontend_double


def program(helpers, body, names=('x',)):
    return helpers + '\n' + fragment(body, names)


class FunctionConstructionTests(unittest.TestCase):
    def run_source(self, helpers, body, constants=None, outputs=1):
        constants = {} if constants is None else constants
        result = normalize(program(helpers, body), constants, outputs)
        expr = frontend_double()
        value = evaluate_tree(result['source'], constants, expr('x'))
        return value, result

    def test_composed_calls_and_local_names_are_isolated(self):
        helpers = ('def negate(a):\n    value = -a\n    return value\n'
                   'def twice(a):\n    value = negate(a)\n    return negate(value)\n')
        value, result = self.run_source(helpers, 'value = x\nout = twice(x)\nreturn [value, out]', outputs=2)
        self.assertEqual([v.obj for v in value], ['x', ('unary', 13, ('unary', 13, 'x'))])
        self.assertEqual(result['construction']['helper_calls'], 3)
        self.assertEqual(result['construction']['max_call_depth'], 2)

    def test_parameter_rebinding_does_not_change_caller(self):
        value, _ = self.run_source('def change(a):\n    a = -a\n    return a\n',
                                   'out = change(x)\nreturn [x, out]', outputs=2)
        self.assertEqual([v.obj for v in value], ['x', ('unary', 13, 'x')])

    def test_list_parameter_and_return_keep_shared_reference(self):
        helpers = 'def change(items):\n    items[0] = -items[0]\n    return items\n'
        value, _ = self.run_source(helpers, 'items = [x]\nother = change(items)\nother.append(x)\nreturn [items[0], items[-1]]', outputs=2)
        self.assertEqual([v.obj for v in value], [('unary', 13, 'x'), 'x'])

    def test_arguments_evaluate_once_left_to_right(self):
        helpers = ('def change(items):\n    items[0] = -items[0]\n    return items[0]\n'
                   'def subtract(a, b):\n    return a - b\n')
        value, result = self.run_source(helpers, 'items = [x]\nreturn subtract(items[0], change(items))')
        self.assertEqual(value.obj, ('binary', 7, 'x', ('unary', 13, 'x')))
        self.assertEqual(result['construction']['container_writes'], 1)

    def test_helpers_see_lexical_globals_not_caller_locals(self):
        helpers = ('def inner(a):\n    return a * weight\n'
                   'def outer(a):\n    weight = -a\n    return inner(a)\n')
        value, _ = self.run_source(helpers, 'return outer(x)', {'weight': .5})
        self.assertEqual(value.obj, ('binary', 8, 'x', ('plain', .5)))
        with self.assertRaisesRegex(ValueError, 'Undefined'):
            self.run_source('def inner(a):\n    return a + hidden\n', 'hidden = x\nreturn inner(x)')

    def test_parameters_can_shadow_public_names_without_mutating_manifest(self):
        values = {'weight': .5}
        value, _ = self.run_source('def change(weight):\n    weight = -weight\n    return weight\n', 'return change(x)', values)
        self.assertEqual(value.obj, ('unary', 13, 'x'))
        self.assertEqual(values, {'weight': .5})

    def test_local_before_assignment_does_not_fall_back_to_global(self):
        with self.assertRaisesRegex(ValueError, 'unbound local'):
            self.run_source('def bad(a):\n    out = a * weight\n    weight = a\n    return out\n',
                            'return bad(x)', {'weight': .5})

    def test_each_loop_invocation_gets_a_fresh_frame(self):
        helpers = 'def maybe(a, i):\n    if i == 0:\n        value = a\n    return value\n'
        with self.assertRaisesRegex(ValueError, 'unbound local'):
            self.run_source(helpers, 'for i in range(2):\n    out = maybe(x, i)\nreturn out')

    def test_early_return_unwinds_only_current_call_including_loop(self):
        helpers = 'def find(a):\n    for i in range(3):\n        if i == 1:\n            return -a\n    return a\n'
        value, result = self.run_source(helpers, 'out = find(x)\nreturn out + x')
        self.assertEqual(value.obj, ('binary', 6, ('unary', 13, 'x'), 'x'))
        self.assertEqual(result['construction']['loop_iterations'], 2)
        value, _ = self.run_source('', 'if 1 < 2:\n    return -x\nreturn x')
        self.assertEqual(value.obj, ('unary', 13, 'x'))

    def test_void_helper_statements_and_none_not_ciphertext(self):
        value, _ = self.run_source('def append(items, a):\n    items.append(-a)\n',
                                   'items = []\nappend(items, x)\nreturn items[0]')
        self.assertEqual(value.obj, ('unary', 13, 'x'))
        for body in ('return nothing(x)', 'nothing(x)'):
            with self.assertRaises(ValueError):
                self.run_source('def nothing(a):\n    return\n', body)

    def test_higher_order_arguments_and_returned_function_values(self):
        helpers = ('def negate(a):\n    return -a\n'
                   'def select():\n    return negate\n'
                   'def apply(function, a):\n    return function(a)\n')
        value, result = self.run_source(helpers, 'function = select()\nreturn apply(function, x)')
        self.assertEqual(value.obj, ('unary', 13, 'x'))
        self.assertEqual(result['construction']['helper_calls'], 3)

    def test_bounded_recursive_construction_and_failure_limit(self):
        helpers = 'def repeat(a, n):\n    if n == 0:\n        return a\n    return repeat(-a, n-1)\n'
        value, result = self.run_source(helpers, 'return repeat(x, 3)')
        self.assertEqual(value.obj, ('unary', 13, ('unary', 13, ('unary', 13, 'x'))))
        self.assertEqual(result['construction']['helper_calls'], 4)
        with self.assertRaisesRegex(ValueError, 'resource limit'):
            self.run_source('def loop(a):\n    return loop(a)\n', 'return loop(x)')

    def test_function_arity_and_declaration_boundaries(self):
        for helpers, body in [
            ('def f(a):\n    return a\n', 'return f(x, x)'),
            ('def f(a):\n    return a\n', 'return f()'),
            ('def f(a=x):\n    return a\n', 'return f()'),
            ('def f(*args):\n    return args[0]\n', 'return f(x)'),
            ('def f(a):\n    return a\n', 'return f(a=x)'),
            ('@hc.func("c")\ndef f(a):\n    return a\n', 'return f(x)'),
            ('def f(a):\n    def inner():\n        return a\n    return inner()\n', 'return f(x)'),
            ('def f(a):\n    return a\ndef f(a):\n    return -a\n', 'return f(x)')]:
            with self.subTest(helpers=helpers, body=body), self.assertRaises(ValueError):
                self.run_source(helpers, body)

    def test_external_calls_and_encrypted_branch_fail_closed(self):
        for helpers, body in [
            ('def unused(a):\n    return open("secret")\n', 'return x'),
            ('def f(a):\n    return a.__class__\n', 'return f(x)'),
            ('def f(a):\n    import os\n    return a\n', 'return f(x)'),
            ('def f(a):\n    if a:\n        return -a\n    return a\n', 'return f(x)'),
            ('def apply(f, a):\n    return f(a)\n', 'return apply(x, x)')]:
            with self.subTest(helpers=helpers), self.assertRaises(ValueError):
                self.run_source(helpers, body)

    def test_syntax_and_coverage_are_not_execution_claims(self):
        from dsl_grammar_coverage import analyze_source
        source = program('def negate(a):\n    return -a\n', 'dead = negate(x)\nreturn x')
        result = analyze_source(source, {}, contract='hecate-function-v7')
        self.assertEqual(result['public_construction']['helper_calls'], 1)
        self.assertNotIn('negate.cipher', result['output_dependency_counts'])
        self.assertFalse(result['semantic_equivalence_proven'])
        normalized = normalize(source, {})['source']
        self.assertEqual(sum(type(n) is ast.FunctionDef for n in ast.walk(ast.parse(normalized))), 1)

    def test_versioned_request_cli_and_legacy_rejection(self):
        from candidate_contract import make_request, validate_candidate
        from hecate_contract import validate_function
        from run_candidate import parse_args, forward_options
        payload = dict(fx_graph='x', public_constants={}, constant_origins={}, layout=dict(input_shape=[4],
            output_shape=[4], output_ciphertexts=1, output_selectors=[[0, i] for i in range(4)]))
        req = make_request(payload, dict(schema=2, id='unit'), 'a'*64, function_composition=True)
        source = program('def f(a):\n    return a\n', 'return f(x)')
        checked = validate_candidate(dict(schema=1, request_id=req['request_id'], hecate_source=source), req)
        self.assertEqual(checked['contract'], 'hecate-function-v7')
        self.assertEqual(req['task'], 'hecate-function-synthesis-v8')
        for contract in ('hecate-function-v5', 'hecate-function-v6'):
            with self.assertRaises(ValueError):
                validate_function(source, {}, contract=contract)
        self.assertIn('--function-composition', forward_options(parse_args(['--case', 'case.json', '--prepare', '--function-composition'])))


class FunctionGoldenPlainTests(unittest.TestCase):
    def test_manual_programs_match_independent_formulas_before_native_run(self):
        from run_function_composition_goldens import PLANS
        root = Path(__file__).resolve().parent
        class Packed:
            def __init__(self, values):
                values = values if type(values) in (list, tuple) else [values]
                self.values = tuple(values * 4 if len(values) == 1 else values)
            def __add__(self, other):
                return Packed([a+b for a, b in zip(self.values, other.values)])
            def __sub__(self, other):
                return Packed([a-b for a, b in zip(self.values, other.values)])
            def __mul__(self, other):
                return Packed([a*b for a, b in zip(self.values, other.values)])
            def __neg__(self):
                return Packed([-a for a in self.values])
            def rotate(self, step):
                step %= 4
                return Packed(self.values[step:]+self.values[:step])
        weights = [[.5, -.25, .125, .75], [-.375, .25, .5, -.125]]
        for case, golden, wrong in PLANS:
            constants = dict(c0=weights[0], c1=weights[1]) if case == 'construction-linear' else dict(
                c0=[.5], c1=[.125 if case == 'explicit-power-4' else .375])
            source = (root/'golden_cases/function_composition'/(golden+'.py')).read_text()
            outputs = 2 if case == 'construction-linear' else 1
            expanded = normalize(source, constants, outputs)
            detected = False
            for x in ([0.]*4, [.5, -1., .25, -.75], [-1., 1., -1., 1.]):
                if case == 'construction-linear':
                    expected = [sum(a*b for a, b in zip(x, row)) for row in weights]
                elif case == 'explicit-power-4':
                    expected = [.5*v**4+.125 for v in x]
                else:
                    expected = [1.5*v+.375 for v in x]
                value = evaluate_tree(expanded['source'], {k: Packed(v) for k, v in constants.items()}, Packed(x))
                actual = [v.values[0] for v in value] if type(value) is list else value.values
                equal = all(abs(a-b) < 1e-12 for a, b in zip(actual, expected))
                detected |= not equal
                if not wrong:
                    self.assertTrue(equal, (case, actual, expected))
            if wrong:
                self.assertTrue(detected, case)


@unittest.skipUnless(os.environ.get('POSEIDON_FUNCTION_REPORTS'), 'requires real helper-composition evidence lineage')
class FunctionEvidenceTests(unittest.TestCase):
    def test_actual_function_expansion_and_encrypted_differential_results(self):
        import numpy as np
        from run_function_composition_goldens import PLANS
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        from candidate_contract import validate_candidate, request_input_names, request_rotations, canonical
        root = Path(__file__).resolve().parents[2]
        selected = {}
        # Ordered report list: an explicit rerun replaces only the same case/role.
        for filename in json.loads(os.environ['POSEIDON_FUNCTION_REPORTS']):
            batch = json.loads(Path(filename).read_text())
            self.assertEqual(batch['agent_calls'], 0)
            for row in batch['cases']:
                selected[(row['case'], row['counterexample'])] = row
        self.assertEqual(set(selected), {(case, wrong) for case, _, wrong in PLANS})
        for case, golden, wrong in PLANS:
            row = selected[(case, wrong)]
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
            # Historical request-v8 evidence retains its actual producer hash.
            # It is NOT evidence of execution by today's closure-capable engine.
            # Below we separately reproduce and compare its old-mode expansion.
            self.assertIn(report['source_hashes']['function_construction.py'], {
                '8eea9b31450d19c9ff5f3bccaeca6d3f8e881bdac8a92e7b73244f168e8e1155',
                hashlib.sha256((root/'scripts/baseline/function_construction.py').read_bytes()).hexdigest()})
            for name, digest in report['frozen_hashes'].items():
                self.assertTrue((run/name).resolve().is_relative_to(run))
                self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(), digest)
            self.assertEqual(len(report['attempts']), 1)
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
            self.assertEqual(req['task'], 'hecate-function-synthesis-v8')
            self.assertEqual(req['request_id'], hashlib.sha256(canonical({k: v for k, v in req.items()
                if k != 'request_id'})).hexdigest())
            source = (root/'scripts/baseline/golden_cases/function_composition'/(golden+'.py')).read_text()
            self.assertEqual(candidate['hecate_source'], source)
            self.assertEqual(validate_candidate(candidate, req), item['static_check'])
            expanded = normalize(source, req['public_constants'], req['layout']['output_ciphertexts'],
                                 input_names=request_input_names(req))
            self.assertEqual(expanded['source'], (output/'normalized-source.py').read_text())
            self.assertEqual(expanded['construction'], json.loads((output/'construction.json').read_text()))
            self.assertGreater(expanded['construction']['helper_calls'], 0)
            self.assertEqual(item['artifact_gate'], inspect_artifacts(
                (output/'lowered._hecate_golden.hevm').read_bytes(), (output/'_hecate_golden.cst').read_bytes(),
                rotation_steps=request_rotations(req), expected_inputs=1))
            with np.load(run/'arrays.npz', allow_pickle=False) as arrays:
                x = arrays['inputs']
                if case == 'construction-linear':
                    expected = x @ np.array([[.5, -.25, .125, .75], [-.375, .25, .5, -.125]]).T
                elif case == 'explicit-power-4':
                    expected = .5*x**4+.125
                else:
                    expected = 1.5*x+.375
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
