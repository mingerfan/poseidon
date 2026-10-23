"""R8f2 flat activation tail invariant and unchanged frozen packages."""

from collections import Counter
from functools import lru_cache
import gc
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent import cost
from pt_agent.backends import EmitBackend, MockBackend, RefreshBackend
from pt_agent.backends.base import _Plain
from pt_agent.dsl.parser import parse
from pt_agent.export import export_package
from pt_agent.lower import decode_outputs, lower, pack_inputs
from pt_agent.paths import REPO, pt_result
from pt_agent.planners.agent_tools import _exact_check
from pt_agent.tensor_sim import run_tensor
from pt_agent.tests._artifacts import DATA, WEIGHTS, require
from pt_agent.tests.test_p2_export import cases, package_modules


FROZEN = pt_result('r8e7-reconfirm-20261022T122558')
R6B = pt_result('r6b-refresh-20261021T025529/REPORT.json')
MODELS = pt_result('models-v2')
FAILED_MLP = pt_result('r8f-20261023T092453/acceptance/mismatch/'
                       'e1__mlp__random__051/program.ptdsl')
OLD_LOWER = 'fb6d7f3c:DSL/DSL_agent/scripts/pt_agent/lower.py'


@lru_cache(maxsize=1)
def old_lower():
    """Load the spec-approved pre-fix trusted implementation without a checkout."""
    text = subprocess.run(['git', 'show', OLD_LOWER], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout
    name = 'pt_agent._r8f2_before_lower'
    module = ModuleType(name)
    module.__package__ = 'pt_agent'
    sys.modules[name] = module
    exec(compile(text, OLD_LOWER, 'exec'), module.__dict__)
    return module.lower


def activation_source(acts):
    lines = ['@fhe.program(slots=16384)', 'def flat_tail(v: Cipher[1, 4]):',
             '    x = pack(v, layout="flat")']
    for attrs in acts:
        lines.append(f'    x = act(x, {attrs})')
    lines.append('    return x')
    return '\n'.join(lines)+'\n'


def triggering_mlp():
    text = ('@fhe.program(slots=16384)\ndef flat_mlp(v: Cipher[1, 4]):\n'
            '    x = pack(v, layout="flat")\n'
            '    x = linear(x, w=W["first.weight"], b=W["first.bias"])\n'
            '    x = act(x, fn="relu", method="cheb", deg=7, domain=(-2.,3.))\n'
            '    y = linear(x, w=W["last.weight"], b=W["last.bias"])\n'
            '    return y\n')
    weights = {'first.weight': np.eye(4, dtype=np.float64),
               'first.bias': np.array([.1, -.1, .05, -.05]),
               'last.weight': np.array([[.3, -.1, .2, .4], [-.2, .5, -.3, .1]]),
               'last.bias': np.array([.2, -.2])}
    return text, parse(text), weights


def mock_values(program, weights, images):
    backend = MockBackend(program.header.slots, batch=len(images))
    backend.bind(pack_inputs(program, [images]))
    values, layouts = lower(program, weights, backend)
    return values, layouts


def emitted(program, weights, lowering=lower):
    backend = EmitBackend(program.header.slots)
    active = backend
    if program.header.refresh == 'agent':
        config = cost.PROFILES['qwen48']
        active = RefreshBackend(backend, **{key: config[key] for key in
                                           ('input_level', 'boot_in_level', 'boot_out_level')})
    lowering(program, weights, active)
    return backend.render()


def assert_render_equal(test, left, right):
    left_text, left_constants = left
    right_text, right_constants = right
    test.assertEqual(left_text, right_text)
    test.assertEqual(list(left_constants), list(right_constants))
    for name in left_constants:
        test.assertTrue(np.array_equal(left_constants[name], right_constants[name]), name)


class ClassifiedEmit(EmitBackend):
    """Distinguish plaintext addition while keeping the real emitted operations."""
    def __init__(self, slots):
        super().__init__(slots)
        self.kinds = Counter()

    def _apply(self, kind, left, right=None, *, step=None):
        label = 'add_cp' if kind == 'add' and isinstance(right, _Plain) else kind
        self.kinds[label] += 1
        return super()._apply(kind, left, right, step=step)


class FlatTailTests(unittest.TestCase):
    def assert_tail_zero(self, acts):
        images = np.array([[-.4, .1, .2, .3], [.1, -.2, .3, -.1]], dtype=np.float64)
        # Each prefix exposes the slots immediately after its final activation.
        for end in range(1, len(acts)+1):
            program = parse(activation_source(acts[:end]))
            values, layouts = mock_values(program, {}, images)
            tail = values[0].value[:, layouts[0].size:]
            self.assertTrue(np.all(tail == 0.0),
                            f'activation {end}: max tail {float(np.max(np.abs(tail)))}')

    def test_cheb_symmetric_and_asymmetric_and_square(self):
        for attrs in (
                'fn="relu", method="cheb", deg=7, domain=(-2.,2.)',
                'fn="relu", method="cheb", deg=15, domain=(-2.,3.)',
                'fn="square", method="exact"'):
            with self.subTest(activation=attrs):
                self.assert_tail_zero([attrs])

    def test_each_consecutive_activation_restores_tail(self):
        self.assert_tail_zero([
            'fn="relu", method="cheb", deg=7, domain=(-2.,3.)',
            'fn="silu", method="cheb", deg=15, domain=(-2.,2.)',
            'fn="square", method="exact"',
        ])

    def test_exact_sigmoid_and_exp_tail_strictly_zero(self):
        images = np.array([[-.4, .1, .2, .3], [.1, -.2, .3, -.1]], dtype=np.float64)
        for fn in ('sigmoid', 'exp'):
            with self.subTest(fn=fn):
                program = parse(activation_source([
                    f'fn="{fn}", method="cheb", deg=7, domain=(-2.,3.)']))
                backend = MockBackend(program.header.slots, batch=len(images))
                backend.bind(pack_inputs(program, [images]))
                values, layouts = lower(program, {}, backend, act_mode='exact')
                tail = values[0].value[:, layouts[0].size:]
                self.assertTrue(np.all(tail == 0.0),
                                f'{fn}: max tail {float(np.max(np.abs(tail)))}')

    def test_linear_sigmoid_linear_passes_exact_check(self):
        text, _, weights = triggering_mlp()
        program = parse(text.replace('fn="relu"', 'fn="sigmoid"'))
        images = np.array([[-.4, .1, .2, .3], [.1, -.2, .3, -.1],
                           [.25, -.35, -.15, .2], [-.3, .4, .1, -.2]], dtype=np.float64)
        first = images @ weights['first.weight'].T + weights['first.bias']
        activated = 1.0 / (1.0 + np.exp(-first))
        reference = activated @ weights['last.weight'].T + weights['last.bias']
        ctx = SimpleNamespace(weights=weights, probe_images=images,
                              probe_act_inputs=[first], val_exact_logits=reference,
                              act_tensors=['first_linear'])
        result = _exact_check(program, ctx)
        self.assertTrue(result['ok'], result)
        self.assertIsNone(result['first_mismatch'])
        self.assertEqual(result['act_count'], [1, 1])
        self.assertLessEqual(result['logit_max_diff'], 1e-9)

    def test_new_package_matches_fixed_lower_text_and_constants(self):
        text, program, weights = triggering_mlp()
        with tempfile.TemporaryDirectory() as directory:
            package = export_package(text, program, weights, directory)
            reference = emitted(program, weights)
            backend = EmitBackend(program.header.slots)
            with package_modules(package) as runtime:
                runtime.invoke(backend, weights)
            assert_render_equal(self, reference, backend.render())

    def test_new_package_matches_fixed_lower_outputs(self):
        text, program, weights = triggering_mlp()
        images = np.array([[-.4, .1, .2, .3], [.1, -.2, .3, -.1]], dtype=np.float64)
        values, layouts = mock_values(program, weights, images)
        expected = decode_outputs([value.value for value in values], layouts)[0]
        with tempfile.TemporaryDirectory() as directory:
            package = export_package(text, program, weights, directory)
            with package_modules(package) as runtime:
                actual = runtime.run_numpy(weights, [images])[0]
        self.assertTrue(np.array_equal(actual, expected))


@require(DATA[-1])
class MinimaxFlatTailTests(unittest.TestCase):
    assert_tail_zero = FlatTailTests.assert_tail_zero

    def test_minimax_tail_strictly_zero(self):
        self.assert_tail_zero(['fn="relu", method="minimax_comp", '
                               'degrees=(15,15,27), domain=(-2.,2.)'])


@require(REPO/'.git')
class FlatTailCostTests(unittest.TestCase):
    def test_only_one_plain_add_and_no_new_boots(self):
        _, program, weights = triggering_mlp()
        old = old_lower()
        previous, current = ClassifiedEmit(program.header.slots), ClassifiedEmit(program.header.slots)
        old(program, weights, previous)
        lower(program, weights, current)
        self.assertEqual(current.kinds-previous.kinds, Counter(add_cp=1))
        self.assertEqual(previous.kinds-current.kinds, Counter())
        with patch.object(cost, 'lower', old):
            before = cost.estimate(program, weights)
        after = cost.estimate(program, weights)
        self.assertEqual(after['boots_pred'], before['boots_pred'])
        self.assertEqual(after['max_depth'], before['max_depth'])
        self.assertEqual(after['ops_total'], before['ops_total']+1)
        for kind in before['counts']:
            self.assertEqual(after['counts'][kind], before['counts'][kind]+(kind == 'add'), kind)


@require(DATA, WEIGHTS, MODELS, FAILED_MLP)
class FailedMlpTests(unittest.TestCase):
    def test_random_051_heldout_100_matches_tensor(self):
        from pt_agent.data.cifar import heldout_set
        from pt_agent.models.zoo import get

        model = get('mlp', MODELS)
        weights = model.import_graph().weights
        images = model.adapt(heldout_set(100, normalization=model.normalization)[0])
        program = parse(FAILED_MLP.read_text())
        maximum = 0.
        for start in range(0, 100, 20):
            batch = images[start:start+20]
            values, layouts = mock_values(program, weights, batch)
            expected = decode_outputs([value.value for value in values], layouts)[0]
            actual = run_tensor(program, weights, batch, cache=False)
            delta = np.abs(actual-expected)
            maximum = max(maximum, float(delta.max()))
            self.assertTrue(np.all(delta <= 1e-9*(1+np.abs(expected))),
                            f'batch {start}: max difference {float(delta.max())}')
        print(f'R8f2 random/051: 100 heldout, max_abs_diff={maximum}', flush=True)


@require(REPO/'.git', FROZEN, R6B, DATA, WEIGHTS, MODELS)
class FrozenEmitTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_seven_frozen_programs_unchanged(self):
        from pt_agent.models.zoo import get

        programs = cases()
        self.assertEqual(len(programs), 7)
        for label, model, path in programs:
            with self.subTest(program=label):
                weights = get(model, MODELS).import_graph().weights
                program = parse(path.read_text())
                before = emitted(program, weights, old_lower())
                after = emitted(program, weights)
                assert_render_equal(self, before, after)
                print(f'R8f2 frozen {label}: render and {len(before[1])} constants unchanged', flush=True)
                del before, after, weights
                gc.collect()


if __name__ == '__main__':
    unittest.main()
