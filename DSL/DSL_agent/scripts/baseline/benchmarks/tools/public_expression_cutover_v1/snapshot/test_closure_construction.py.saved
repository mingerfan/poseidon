"""Closure binding tests. Symbolic/preflight checks are not FHE evidence."""
import ast
import hashlib
import json
import os
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
import unittest

from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_frontend_augmented_ops import frontend_double


class ClosureTests(unittest.TestCase):
    def run_source(self, helpers, body, constants=None, outputs=1):
        constants = {} if constants is None else constants
        result = normalize(program(helpers, body), constants, outputs, closures=True)
        expr = frontend_double()
        value = evaluate_tree(result['source'], constants, expr('x'))
        return value, result

    def test_capture_observes_later_rebinding_not_snapshot(self):
        value, result = self.run_source('', 'value = x\ndef read():\n    return value\nvalue = -x\nreturn read()')
        self.assertEqual(value.obj, ('unary', 13, 'x'))
        self.assertEqual(result['construction']['closure_instances'], 1)

    def test_escaped_frames_are_separate_between_factory_calls(self):
        helpers = 'def make(a):\n    def read():\n        return a\n    return read\n'
        value, _ = self.run_source(helpers, 'first = make(x)\nsecond = make(-x)\nreturn [first(), second()]', outputs=2)
        self.assertEqual([v.obj for v in value], ['x', ('unary', 13, 'x')])

    def test_siblings_share_nonlocal_cell_after_return(self):
        helpers = ('def make(a):\n    def update():\n        nonlocal a\n        a = -a\n'
                   '    def read():\n        return a\n    return [update, read]\n')
        value, result = self.run_source(helpers, 'update, read = make(x)\nold = read()\nupdate()\nreturn [old, read()]', outputs=2)
        self.assertEqual([v.obj for v in value], ['x', ('unary', 13, 'x')])
        self.assertEqual(result['construction']['nonlocal_writes'], 1)

    def test_nonlocal_finds_nearest_binding_across_intermediate_frame(self):
        helpers = ('def make(a):\n    def middle():\n        def write():\n            nonlocal a\n'
                   '            a = -a\n        return write\n    update = middle()\n    update()\n    return a\n')
        value, _ = self.run_source(helpers, 'return make(x)')
        self.assertEqual(value.obj, ('unary', 13, 'x'))

    def test_nonlocal_shadow_does_not_change_farther_frame(self):
        helpers = ('def make(a):\n    def middle(a):\n        def write():\n            nonlocal a\n'
                   '            a = -a\n        write()\n        return a\n    b = middle(a)\n    return [a, b]\n')
        value, _ = self.run_source(helpers, 'return make(x)', outputs=2)
        self.assertEqual([v.obj for v in value], ['x', ('unary', 13, 'x')])

    def test_loop_closures_share_final_loop_binding(self):
        value, _ = self.run_source('', 'items = []\nfor i in range(3):\n    def read():\n        return x if i == 2 else -x\n    items.append(read)\nfirst, second, third = items\nreturn [first(), second(), third()]', outputs=3)
        self.assertEqual([v.obj for v in value], ['x']*3)

    def test_static_locals_do_not_leak_from_child_scope(self):
        value, _ = self.run_source('', 'def child():\n    weight = x\n    return weight\nreturn x * weight', {'weight': .5})
        self.assertEqual(value.obj, ('binary', 8, 'x', ('plain', .5)))

    def test_read_before_local_or_cell_initialization_fails(self):
        for body in (
            'def read():\n    return value\nreturn read()\nvalue = x',
            'def read():\n    result = x * weight\n    weight = x\n    return result\nreturn read()',
            'result = x * weight\nweight = x\nreturn result',
        ):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, 'unbound local'):
                self.run_source('', body, {'weight': .5})

    def test_lexical_environment_never_dynamic_caller(self):
        helpers = ('def apply(f, hidden):\n    return f()\n'
                   'def make(a):\n    def read():\n        return a\n    return read\n')
        value, _ = self.run_source(helpers, 'f = make(x)\nreturn apply(f, -x)')
        self.assertEqual(value.obj, 'x')
        with self.assertRaisesRegex(ValueError, 'Undefined'):
            self.run_source('def read():\n    return hidden\n', 'hidden = x\nreturn read()')

    def test_captured_list_alias_mutation_without_nonlocal(self):
        value, result = self.run_source('', 'items = [x]\ndef change():\n    items[0] = -items[0]\nchange()\nreturn items[0]')
        self.assertEqual(value.obj, ('unary', 13, 'x'))
        self.assertEqual(result['construction']['nonlocal_writes'], 0)

    def test_nonlocal_can_initialize_cell_and_shadow_constant_in_helper(self):
        helpers = ('def make(a):\n    def change():\n        nonlocal weight\n        weight = -a\n'
                   '    if 0:\n        weight = a\n    change()\n    return weight\n')
        value, _ = self.run_source(helpers, 'return make(x)', {'weight': .5})
        self.assertEqual(value.obj, ('unary', 13, 'x'))

    def test_nonlocal_legality_checked_even_in_unused_helper(self):
        for helpers in (
            'def unused():\n    nonlocal missing\n    return missing\n',
            'def unused(a):\n    nonlocal a\n    return a\n',
            'def outer(a):\n    def bad():\n        result = a\n        nonlocal a\n        return result\n    return a\n',
            'def bad():\n    nonlocal weight\n    return weight\n',
        ):
            with self.subTest(helpers=helpers), self.assertRaisesRegex(ValueError, 'Invalid lexical scope'):
                self.run_source(helpers, 'return x', {'weight': .5})

    def test_nested_recursion_and_resource_limits(self):
        value, _ = self.run_source('', 'def f(a, n):\n    if n == 0:\n        return a\n    return f(-a, n-1)\nreturn f(x, 2)')
        self.assertEqual(value.obj, ('unary', 13, ('unary', 13, 'x')))
        with self.assertRaisesRegex(ValueError, 'resource limit'):
            self.run_source('', 'def f():\n    return f()\nreturn f()')
        with self.assertRaisesRegex(ValueError, 'Closure instance resource limit'):
            self.run_source('', 'for i in range(2):\n    for j in range(65):\n        def f():\n            return x\nreturn x')

    def test_harness_bindings_remain_read_only_and_no_external_capability(self):
        for body in ('def f():\n    global x\n    x = -x\nf()\nreturn x',
                     'def f():\n    return open(x)\nreturn x',
                     'def f():\n    return x.__class__\nreturn x',
                     'def f(a=x):\n    return a\nreturn f()',
                     'def f():\n    nonlocal weight\n    weight = x\nif 0:\n    weight = x\nf()\nreturn x'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.run_source('', body, {'weight': .5})

    def test_version_cli_coverage_and_old_rejection(self):
        from candidate_contract import make_request, validate_candidate, valid_semantic_guidance
        from hecate_contract import validate_function
        from dsl_grammar_coverage import analyze_source
        from run_candidate import parse_args, forward_options
        payload = dict(fx_graph='x', public_constants={}, constant_origins={}, layout=dict(input_shape=[4],
            output_shape=[4], output_ciphertexts=1, output_selectors=[[0, i] for i in range(4)]))
        req = make_request(payload, dict(schema=2, id='unit'), 'a'*64, closures=True)
        source = program('', 'def read():\n    return -x\nreturn read()')
        checked = validate_candidate(dict(schema=1, request_id=req['request_id'], hecate_source=source), req)
        self.assertEqual(checked['contract'], 'hecate-function-v8')
        self.assertEqual(req['task'], 'hecate-function-synthesis-v9')
        self.assertTrue(valid_semantic_guidance(req))
        self.assertEqual(analyze_source(source, {}, contract='hecate-function-v8')['public_construction']['closure_instances'], 1)
        self.assertIn('--closures', forward_options(parse_args(['--case', 'case.json', '--prepare', '--closures'])))
        for version in range(8):
            with self.subTest(version=version), self.assertRaises(ValueError):
                validate_function(source, {}, contract='hecate-function-v'+str(version))


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


class ClosureGoldenPlainTests(unittest.TestCase):
    def test_goldens_against_independent_formula(self):
        from run_closure_goldens import PLANS
        root = Path(__file__).resolve().parent
        weights = [[.5, -.25, .125, .75], [-.375, .25, .5, -.125]]
        for case, golden, wrong in PLANS:
            constants = dict(c0=weights[0], c1=weights[1]) if case == 'construction-linear' else dict(c0=[.5], c1=[.375])
            source = (root/'golden_cases/closures'/(golden+'.py')).read_text()
            expanded = normalize(source, constants, 2 if case == 'construction-linear' else 1, closures=True)
            detected = False
            for x in ([0.]*4, [.5, -1., .25, -.75], [-1., 1., -1., 1.]):
                expected = ([sum(a*b for a, b in zip(x, row)) for row in weights]
                            if case == 'construction-linear' else [1.5*v+.375 for v in x])
                value = evaluate_tree(expanded['source'], {k: Packed(v) for k, v in constants.items()}, Packed(x))
                actual = [v.values[0] for v in value] if type(value) is list else value.values
                equal = len(actual) == len(expected) and all(abs(a-b) < 1e-12 for a, b in zip(actual, expected))
                detected |= not equal
                if not wrong:
                    self.assertTrue(equal, (golden, actual, expected))
            if wrong:
                self.assertTrue(detected, golden)


@unittest.skipUnless(os.environ.get('POSEIDON_CLOSURE_REPORT'), 'requires actual closure CPU evidence')
class ClosureEvidenceTests(unittest.TestCase):
    def test_actual_artifacts_immutable_reference_and_numerical_results(self):
        from run_closure_goldens import PLANS
        historical = {'function_construction.py': 'e8ed5b9db77274d4360d4b1031d427c088ff48c986d857a627ba01734bc05e91',
                      'lexical_scope.py': 'd73967bd2bb47afb0edb7395cf51ccf94f670844a68a8ecafe6fd557fc21fd01'}
        self.verify_evidence(os.environ['POSEIDON_CLOSURE_REPORT'], PLANS, 'closures',
            'hecate-function-synthesis-v9', dict(closures=True), historical)

    def verify_evidence(self, filename, plans, golden_folder, task, options, historical):
        import numpy as np
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        from candidate_contract import validate_candidate, request_input_names, request_rotations, canonical
        root = Path(__file__).resolve().parents[2]
        if type(filename) is list:
            selected = {}
            for path in filename:
                original = json.loads(Path(path).read_text())
                self.assertEqual(original['agent_calls'], 0)
                for row in original['cases']:
                    command = row['command']
                    golden_name = Path(command[command.index('--golden-file')+1]).stem
                    self.assertIn(golden_name, {p[1] for p in plans})
                    selected[golden_name] = row
            self.assertEqual(set(selected), {p[1] for p in plans})
            batch = dict(cases=[selected[p[1]] for p in plans])
        else:
            batch = json.loads(Path(filename).read_text())
            self.assertEqual(batch['status'], 'passed')
            self.assertEqual(batch['agent_calls'], 0)
        self.assertEqual(len(batch['cases']), len(plans))
        # Several distinct closure programs share one reference model. Match by
        # ordered plan AND exact golden source, never collapse by case/role.
        for row, (case, golden, wrong) in zip(batch['cases'], plans):
            with self.subTest(golden=golden):
                self.assertEqual((row['case'], row['counterexample']), (case, wrong))
                self.assertTrue(row['matched_expected'])
                run = Path(row['run']).resolve()
                self.assertTrue(run.is_relative_to(str(RESULTS)))
                report = json.loads((run/'report.json').read_text())
                self.assertEqual(report['agent_calls'], 0)
                self.assertFalse(report['llm_generation_validated'])
                self.assertFalse(report['poseidon_gpu_validated'])
                self.assertEqual(report['backend'], 'upstream_SEAL_HEVM_CPU')
                self.assertEqual(report['tolerance'], dict(atol=1e-5, rtol=1e-4))
                self.assertEqual(report['parameters']['security_check'], 'tc128')
                self.assertEqual(report['parameters']['modulus_bits'], [60]*14)
                self.assertEqual(report['parameters']['polynomial_degree'], 32768)
                for name in historical:
                    # Preserve actual request-v9 producer, not a new-execution claim.
                    # Exact old-mode normalization is independently reproduced below.
                    recorded = historical[name]
                    accepted = set(recorded) if type(recorded) in (list,tuple) else {recorded}
                    self.assertIn(report['source_hashes'][name], {*accepted, hashlib.sha256(
                        (root/'scripts/baseline'/name).read_bytes()).hexdigest()})
                for name, digest in report['frozen_hashes'].items():
                    self.assertTrue((run/name).resolve().is_relative_to(run))
                    self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(), digest)
                self.assertEqual(len(report['attempts']), 1)
                item = report['attempts'][0]
                self.assertTrue(item['execution']['encrypted_execution'])
                self.assertFalse(item['execution']['bootstrap_executed'])
                self.assertEqual(item['execution']['input_batches'], 4)
                self.assertEqual(item['execution']['encrypted_input_count'], 1)
                self.assertFalse(item['trace']['candidate_python_executed'])
                output = run/'attempt-00/output'
                self.assertTrue({'normalized-source.py', 'construction.json', 'lowered.ckks.mlir',
                    'lowered._hecate_golden.hevm', '_hecate_golden.cst'} <= set(item['artifact_hashes']))
                for name, digest in item['artifact_hashes'].items():
                    self.assertTrue((output/name).resolve().is_relative_to(output))
                    self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), digest)
                payload = json.loads((run/'attempt-00/trace-payload.json').read_text())
                req, candidate = payload['request'], payload['candidate']
                self.assertEqual(req['task'], task)
                self.assertEqual(req['request_id'], hashlib.sha256(canonical({k: v for k, v in req.items()
                    if k != 'request_id'})).hexdigest())
                source = (root/'scripts/baseline/golden_cases'/golden_folder/(golden+'.py')).read_text()
                self.assertEqual(candidate['hecate_source'], source)
                self.assertEqual(validate_candidate(candidate, req), item['static_check'])
                expanded = normalize(source, req['public_constants'], req['layout']['output_ciphertexts'],
                                     input_names=request_input_names(req), **options)
                self.assertEqual(expanded['source'], (output/'normalized-source.py').read_text())
                self.assertEqual(expanded['construction'], json.loads((output/'construction.json').read_text()))
                if task in ('hecate-function-synthesis-v14','hecate-function-synthesis-v15','hecate-function-synthesis-v16','hecate-function-synthesis-v17','hecate-function-synthesis-v18'):
                    self.assertIn('derived-constants.json',item['artifact_hashes'])
                    self.assertEqual(expanded['derived_constants'],json.loads((output/'derived-constants.json').read_text()))
                    if task in ('hecate-function-synthesis-v14','hecate-function-synthesis-v16','hecate-function-synthesis-v17'):
                        self.assertGreater(expanded['construction']['derived_constant_count'],0)
                    self.assertFalse(expanded['construction']['input_constants_changed'])
                    self.assertEqual({k:expanded['constants'][k] for k in req['public_constants']},req['public_constants'])
                if task == 'hecate-function-synthesis-v17':
                    self.assertGreater(expanded['construction']['public_polynomial_operations'],0)
                    self.assertGreater(expanded['construction']['public_numpy_functions'],0)
                if task == 'hecate-function-synthesis-v18':
                    self.assertGreater(expanded['construction']['object_array_operations'],0)
                    self.assertEqual(expanded['construction']['schema'],12)
                if task == 'hecate-function-synthesis-v16':
                    self.assertGreater(expanded['construction']['public_string_calls'],0)
                if task == 'hecate-function-synthesis-v11':
                    self.assertGreater(expanded['construction']['comprehensions'], 0)
                else:
                    self.assertGreater(expanded['construction']['helper_calls'], 0)
                if task == 'hecate-function-synthesis-v9':
                    self.assertGreater(expanded['construction']['closure_instances'], 0)
                self.assertEqual(item['artifact_gate'], inspect_artifacts(
                    (output/'lowered._hecate_golden.hevm').read_bytes(), (output/'_hecate_golden.cst').read_bytes(),
                    rotation_steps=request_rotations(req), expected_inputs=1))
                with np.load(run/'arrays.npz', allow_pickle=False) as arrays:
                    x = arrays['inputs']
                    expected = (x @ np.array([[.5, -.25, .125, .75], [-.375, .25, .5, -.125]]).T
                                if case == 'construction-linear' else
                                4*x**3-3*x if case == 'construction-chebyshev3' else
                                2*x**3-1.25*x if case == 'construction-genpoly3' else
                                2*x**5-.5*x**3-.625*x if case == 'construction-genpoly5' else
                                4*x**7-5*x**5+2*x**3-.4375*x if case == 'construction-genpoly7' else
                                2*x**3-1.25*x+.5*x*x-.25 if case == 'construction-genpoly-even' else 1.5*x+.375)
                    if case in ('construction-sumslots3','construction-sumslots4'):
                        expected = sum(np.roll(x,-j,axis=-1) for j in range(int(case[-1])))
                    np.testing.assert_allclose(arrays['reference'], expected, atol=1e-12, rtol=1e-12)
                    comparison = compare(np.load(output/'decrypted.npy', allow_pickle=False), arrays['reference'], 1e-5, 1e-4)
                self.assertEqual(comparison, item['comparison'])
                self.assertEqual(comparison['passed'], not wrong)
                if wrong:
                    self.assertEqual(item['failure_layer'], 'numerical_comparison')
                    self.assertGreater(comparison['max_absolute_error'], .1)
                self.assertFalse((run/'private-keys').exists())
