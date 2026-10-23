"""R8f tensor arithmetic, unchanged decisions and process-local cache checks."""

import copy
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pt_agent import tensor_sim
from pt_agent.backends import BackendError, MockBackend
from pt_agent.cost import PROFILES
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.lower import LoweringError, decode_outputs, lower, pack_inputs
from pt_agent.paths import pt_result
from pt_agent.planners import agent_tools, confirm, tuner
from pt_agent.tests._artifacts import DATA, WEIGHTS, require


PREFIX_LIMIT = 3 * 2**29
FROZEN = pt_result('r8e7-reconfirm-20261022T122558/models')
MODELS = pt_result('models-v2')
FIXED_SOURCE = ('@fhe.program(slots=16384)\ndef fixed(v: Cipher[1, 2]):\n'
                '    x = pack(v, layout="flat")\n'
                '    x = act(x, fn="relu", method="cheb", deg=7, domain=(-2.0, 2.0))\n'
                '    return x\n')
# Captured on fb6d7f3c before the R8f implementation changed agent_tools.py.
FIXED_ROW = dict(deg=7, degrees=None, domain=(-2.0, 2.0), graph_tensor='pre_relu',
                 max_u=.5, method='cheb', obs_max=.5, obs_min=-1., program_line=4)
FIXED_VALIDATION = dict(acc_drop_pp=25., agree=3, boot_input_bound=None,
                        boot_input_max_abs=0., diverged=False, diverged_indices=[],
                        exact_top1=100., feasible=False, images_done=4, max_u=.5,
                        n_diverged=0, rows=[FIXED_ROW], top1=75., total=4,
                        worst_acts=[FIXED_ROW])


def fixed_context():
    images = np.array([[-1., .5], [.25, -.5], [.1, .2], [-.2, -.1]], dtype=np.float64)
    return SimpleNamespace(weights={}, profile='qwen48', val_images=images,
                           val_exact_logits=np.maximum(images, 0),
                           val_labels=np.array([1, 0, 1, 0]), act_tensors=['pre_relu'],
                           val_split='test[9000:10000]')


def mock_logits(program, weights, images, observer=None):
    backend = MockBackend(program.header.slots, batch=len(images))
    backend.bind(pack_inputs(program, [images]))
    values, layouts = lower(program, weights, backend, observer=observer)
    return decode_outputs([value.value for value in values], layouts)[0]


def flat_source(body, features=4):
    return (f'@fhe.program(slots=16384)\ndef small(v: Cipher[1, {features}]):\n'
            '    x = pack(v, layout="flat")\n' + body + '    return y\n')


def prefix_program(degree=15):
    # The final activation depends on both sides of a residual DAG.
    return parse(flat_source(
        '    a = act(x, fn="relu", method="cheb", deg=15, domain=(-4., 4.))\n'
        '    b = add(a, x)\n'
        '    c = act(b, fn="silu", method="cheb", deg=15, domain=(-4., 4.))\n'
        '    d = add(c, a)\n'
        f'    y = act(d, fn="relu", method="cheb", deg={degree}, domain=(-4., 4.))\n'))


def renamed(program):
    names = [param.name for param in program.params]
    names += [output for op in program.ops for output in op.outputs]
    mapping = {name: f'renamed_{index}' for index, name in enumerate(names)}
    return replace(program, name='renamed_program',
                   params=tuple(replace(p, name=mapping[p.name], lineno=p.lineno+10)
                                for p in program.params),
                   ops=tuple(replace(op, inputs=tuple(mapping[name] for name in op.inputs),
                                     outputs=tuple(mapping[name] for name in op.outputs),
                                     lineno=op.lineno+10) for op in program.ops),
                   outputs=tuple(mapping[name] for name in program.outputs))


class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        tensor_sim.set_prefix_cache_limit(PREFIX_LIMIT)
        tensor_sim.clear_caches()
        self.rng = np.random.default_rng(20261023)

    def tearDown(self):
        tensor_sim.clear_caches()
        tensor_sim.set_prefix_cache_limit(PREFIX_LIMIT)

    def assert_tensor_matches_mock(self, source, weights, images):
        program = parse(source)
        expected = mock_logits(program, weights, images)
        actual = tensor_sim.run_tensor(program, weights, images, cache=False)
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.all(np.abs(actual-expected) <= 1e-9*(1+np.abs(expected))),
                        f'max logit difference {float(np.max(np.abs(actual-expected)))}')

    def test_01_conv_with_and_without_bn_stride_1_and_2(self):
        images = self.rng.normal(0, .2, (3, 2, 4, 4))
        weights = dict(w=self.rng.normal(0, .1, (4, 2, 3, 3)),
                       b=self.rng.normal(0, .1, 4))
        weights.update({'bn.weight': self.rng.uniform(.5, 1.5, 4),
                        'bn.bias': self.rng.normal(0, .1, 4),
                        'bn.running_mean': self.rng.normal(0, .1, 4),
                        'bn.running_var': self.rng.uniform(.5, 1.5, 4)})
        for stride in (1, 2):
            for bn in (False, True):
                with self.subTest(stride=stride, bn=bn):
                    source = ('@fhe.program(slots=16384)\ndef conv(v: Cipher[1,2,4,4]):\n'
                              '    x = pack(v, layout="mux", gap=1)\n'
                              f'    y = conv2d(x, w=W["w"], b=W["b"], stride={stride}, pad=1'
                              + (', bn="bn"' if bn else '') + ')\n    return y\n')
                    self.assert_tensor_matches_mock(source, weights, images)

    def test_02_cheb_square_add_and_refresh(self):
        images = self.rng.uniform(-.5, .5, (3, 4))
        cases = [
            '    y = act(x, fn="relu", method="cheb", deg=7, domain=(-2.,2.))\n',
            '    y = act(x, fn="silu", method="cheb", deg=15, domain=(-2.,2.))\n',
            '    y = act(x, fn="square", method="exact")\n',
            '    a = act(x, fn="square", method="exact")\n    y = add(a,x)\n',
            '    a = refresh(x)\n    y = act(a, fn="square", method="exact")\n',
        ]
        for body in cases:
            with self.subTest(body=body):
                self.assert_tensor_matches_mock(flat_source(body), {}, images)

    def test_03_downsample_mean_and_relayout(self):
        images = self.rng.uniform(-1., 1., (3, 2, 4, 4))
        header = ('@fhe.program(slots=16384)\ndef pool(v: Cipher[1,2,4,4]):\n'
                  '    x = pack(v, layout="mux", gap=1)\n')
        bodies = [
            ('    a = slice(x, axis=2, start=0, end=None, step=2)\n'
             '    b = slice(a, axis=3, start=0, end=4, step=2)\n'
             '    y = pad(b, pads=(0,0,0,0,1,1))\n'),
            '    y = mean(x, axis=(2,3))\n',
            '    a = mean(x, axis=(2,3))\n    y = relayout(a, layout="flat")\n',
        ]
        for body in bodies:
            with self.subTest(body=body):
                self.assert_tensor_matches_mock(header+body+'    return y\n', {}, images)

    def test_04_linear_bsgs_and_diag(self):
        images = self.rng.uniform(-1., 1., (3, 4))
        weights = dict(w=self.rng.normal(0, .1, (3, 4)), b=self.rng.normal(0, .1, 3))
        for algo in ('bsgs', 'diag'):
            for bias in (False, True):
                with self.subTest(algo=algo, bias=bias):
                    body = (f'    y = linear(x, w=W["w"], algo="{algo}"'
                            + (', b=W["b"]' if bias else '') + ')\n')
                    self.assert_tensor_matches_mock(flat_source(body), weights, images)

    def test_05_bad_conv_has_same_lowering_error(self):
        images = np.ones((1, 2, 4, 4), dtype=np.float64)
        weights = dict(w=np.ones((4, 2, 3, 3)))
        source = ('@fhe.program(slots=16384)\ndef conv(v: Cipher[1,2,4,4]):\n'
                  '    x = pack(v, layout="mux")\n'
                  '    y = conv2d(x, w=W["w"], stride=1, pad=0)\n    return y\n')
        program = parse(source)
        errors = []
        for runner in (mock_logits, tensor_sim.run_tensor):
            with self.assertRaises(LoweringError) as caught:
                runner(program, weights, images)
            errors.append((caught.exception.op_id, caught.exception.lineno, caught.exception.msg))
        self.assertEqual(errors[0], errors[1])

    def test_06_diverged_indices_match(self):
        source = flat_source('    y = act(x, fn="relu", method="cheb", deg=255, '
                             'domain=(-0.001,0.001))\n', features=2)
        ctx = fixed_context()
        ctx.val_images = np.array([[0.,0.], [1.,-1.], [.0001,-.0001], [.5,.4]])
        ctx.val_exact_logits = np.maximum(ctx.val_images, 0)
        reports = [agent_tools.validate(parse(source), ctx, engine=engine, cache=False,
                                       boot_nodes=[], stop_on_divergence=False)
                   for engine in ('mock', 'tensor')]
        self.assertEqual(reports[0]['diverged_indices'], [1, 3])
        self.assertEqual(reports[0]['diverged_indices'], reports[1]['diverged_indices'])
        self.assertEqual(reports[0]['n_diverged'], reports[1]['n_diverged'])

    def test_07_tensor_never_estimates(self):
        with patch.object(agent_tools, 'estimate', side_effect=AssertionError('estimate called')):
            result = agent_tools.validate(parse(FIXED_SOURCE), fixed_context(), engine='tensor')
        self.assertIsNone(result['boot_input_max_abs'])

    def test_08_boot_constraints_require_mock(self):
        for field in ('boot_input_bound', 'boot_abs_error'):
            with self.subTest(field=field), patch.dict(PROFILES['qwen48'], {field: .1}):
                with self.assertRaises(ValueError):
                    agent_tools.validate(parse(FIXED_SOURCE), fixed_context(), engine='tensor')

    def test_09_mock_matches_prechange_dictionary(self):
        actual = agent_tools.validate(parse(FIXED_SOURCE), fixed_context(),
                                      engine='mock', boot_nodes=[])
        self.assertEqual(actual, FIXED_VALIDATION)

    def test_10_confirm_always_uses_mock(self):
        self.assertNotIn('engine', inspect.signature(confirm.confirm_select).parameters)
        ctx = fixed_context()
        candidate = confirm.Candidate(FIXED_SOURCE, 'llm', 1, 1., 0)
        with patch.object(confirm, 'build_confirm_context', return_value=ctx), \
                patch.object(agent_tools, 'estimate', return_value={'boot_nodes': []}), \
                patch.object(agent_tools, 'MockBackend', wraps=MockBackend) as mock, \
                patch.object(tensor_sim, 'run_tensor', side_effect=AssertionError('tensor used')):
            result = confirm.confirm_select([candidate], ctx)
        self.assertEqual(len(result['evals']), 1)
        self.assertGreater(mock.call_count, 0)
        with self.assertRaises(TypeError):
            confirm.confirm_select([], ctx, engine='tensor')

    def test_11_prefix_hits_survive_last_degree_change_and_renaming(self):
        images = self.rng.uniform(-.2, .2, (4, 4))
        weights = {}
        first, second = prefix_program(), prefix_program(7)
        tensor_sim.run_tensor(first, weights, images, cache=True)
        before = tensor_sim.cache_info()['prefix'].copy()
        cached = tensor_sim.run_tensor(second, weights, images, cache=True)
        after = tensor_sim.cache_info()['prefix'].copy()
        self.assertEqual(after['misses'], before['misses'])
        self.assertGreaterEqual(after['hits']-before['hits'], 3)
        uncached = tensor_sim.run_tensor(second, weights, images, cache=False)
        self.assertTrue(np.array_equal(cached, uncached))
        actual = tensor_sim.run_tensor(renamed(second), weights, images, cache=True)
        renamed_stats = tensor_sim.cache_info()['prefix']
        self.assertEqual(renamed_stats['misses'], after['misses'])
        self.assertGreaterEqual(renamed_stats['hits']-after['hits'], 3)
        self.assertTrue(np.array_equal(actual, uncached))

    def test_12_lru_eviction_and_oversize(self):
        images = self.rng.uniform(-.2, .2, (4, 4))
        weights = {}
        program = parse(flat_source(
            '    a = act(x, fn="relu", method="cheb", deg=15, domain=(-4.,4.))\n'
            '    b = act(a, fn="silu", method="cheb", deg=15, domain=(-4.,4.))\n'
            '    y = act(b, fn="relu", method="cheb", deg=7, domain=(-4.,4.))\n'))
        expected = tensor_sim.run_tensor(program, weights, images, cache=False)
        tensor_sim.set_prefix_cache_limit(images.nbytes)
        actual = tensor_sim.run_tensor(program, weights, images, cache=True)
        stats = tensor_sim.cache_info()['prefix']
        self.assertTrue(np.array_equal(actual, expected))
        self.assertGreater(stats['evictions'], 0)
        self.assertLessEqual(stats['bytes'], images.nbytes)
        again = tensor_sim.run_tensor(program, weights, images, cache=True)
        self.assertTrue(np.array_equal(again, expected))
        tensor_sim.clear_caches()
        tensor_sim.set_prefix_cache_limit(images.nbytes-1)
        actual = tensor_sim.run_tensor(program, weights, images, cache=True)
        stats = tensor_sim.cache_info()['prefix']
        self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(stats['entries'], 0)
        self.assertEqual(stats['bytes'], 0)
        self.assertGreater(stats['oversize'], 0)

    def test_13_result_cache_isolation(self):
        ctx, program = fixed_context(), parse(FIXED_SOURCE)
        first = agent_tools.validate(program, ctx, engine='tensor', cache=True)
        expected = copy.deepcopy(first)
        first['rows'][0]['obs_min'] = -999.
        first['worst_acts'].clear()
        first['diverged_indices'].append(123)
        second = agent_tools.validate(program, ctx, engine='tensor', cache=True)
        self.assertEqual(second, expected)
        second['rows'][0]['domain'] = ('changed',)
        third = agent_tools.validate(program, ctx, engine='tensor', cache=True)
        self.assertEqual(third, expected)
        self.assertGreaterEqual(tensor_sim.cache_info()['result']['hits'], 2)

    def test_14_cost_cache_keeps_source_locations_and_copies(self):
        original = parse(FIXED_SOURCE)
        moved = replace(original, ops=tuple(replace(op, lineno=op.lineno+10,
                                                    call_lines=(30,)) for op in original.ops))
        self.assertEqual(print_program(original), print_program(moved))
        calls, weights = [], {}

        def estimator(program, weights, *, profile):
            calls.append(program)
            return dict(hotspots=[dict(line=op.lineno, call_lines=list(op.call_lines))
                                  for op in program.ops], boot_nodes=[1])

        first = tensor_sim.cached_estimate(original, weights, 'qwen48', estimator)
        first['boot_nodes'].append(99)
        second = tensor_sim.cached_estimate(original, weights, 'qwen48', estimator)
        self.assertEqual(second['boot_nodes'], [1])
        second['hotspots'][0]['line'] = -100
        self.assertEqual(tensor_sim.cached_estimate(original, weights, 'qwen48', estimator)
                         ['hotspots'][0]['line'], original.ops[0].lineno)
        shifted = tensor_sim.cached_estimate(moved, weights, 'qwen48', estimator)
        self.assertEqual(shifted['hotspots'][0], dict(line=moved.ops[0].lineno, call_lines=[30]))
        self.assertEqual(len(calls), 2)


@require(DATA[-1])
class MinimaxTensorTests(unittest.TestCase):
    # Keep the artifact-dependent minimax check separate from the pure tests.
    setUp = TensorTests.setUp
    tearDown = TensorTests.tearDown
    assert_tensor_matches_mock = TensorTests.assert_tensor_matches_mock

    def test_minimax(self):
        images = self.rng.uniform(-.5, .5, (3, 4))
        source = flat_source('    y = act(x, fn="relu", method="minimax_comp", '
                             'degrees=(15,15,27), domain=(-2.,2.))\n')
        self.assert_tensor_matches_mock(source, {}, images)


@require(DATA, WEIGHTS, MODELS, FROZEN/'resnet20/program.ptdsl',
         FROZEN/'resnet20_silu/program.ptdsl')
class FrozenTuneCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        tensor_sim.set_prefix_cache_limit(PREFIX_LIMIT)
        tensor_sim.clear_caches()

    def tearDown(self):
        tensor_sim.clear_caches()

    def compare_frozen_tune(self, name):
        from pt_agent.models.zoo import get

        model = get(name, MODELS)
        graph = model.import_graph()
        calibration = json.loads((FROZEN/name/'calibration.json').read_text())
        ctx = agent_tools.build_context(graph, calibration, n_val=1000,
                                        normalization=model.normalization)
        program = parse((FROZEN/name/'program.ptdsl').read_text())
        validate = agent_tools.validate
        traces, outcomes = [], []
        for use_cache in (True, False):
            tensor_sim.clear_caches()
            requests = []

            def tracked(candidate, context, n=None, **kwargs):
                result = validate(candidate, context, n, **kwargs)
                images = context.val_images if n is None else context.val_images[:n]
                logits = []
                for start in range(0, len(images), agent_tools.VAL_BATCH):
                    batch = images[start:start+agent_tools.VAL_BATCH]
                    try:
                        value = tensor_sim.run_tensor(candidate, context.weights, batch,
                                                      cache=use_cache, image_id=id(context.val_images),
                                                      batch_start=start)
                        logits.append((start, value.copy()))
                    except BackendError:
                        for offset in range(len(batch)):
                            try:
                                value = tensor_sim.run_tensor(candidate, context.weights,
                                                              batch[offset:offset+1], cache=use_cache,
                                                              image_id=id(context.val_images),
                                                              batch_start=start+offset)
                                logits.append((start+offset, value.copy()))
                            except BackendError:
                                logits.append((start+offset, None))
                                if kwargs.get('stop_on_divergence', True):
                                    break
                        if logits[-1][1] is None and kwargs.get('stop_on_divergence', True):
                            break
                requests.append((print_program(candidate), n, copy.deepcopy(result), logits))
                return result

            with patch.object(tuner, 'validate', side_effect=tracked):
                outcome = tuner.tune(program, ctx, engine='tensor', cache=use_cache,
                                     budget_s=2400, max_sweeps=1)
            traces.append(requests)
            outcomes.append(outcome)
        self.assertEqual(outcomes[0].history, outcomes[1].history)
        self.assertEqual(outcomes[0].source, outcomes[1].source)
        self.assertEqual(outcomes[0].changes, outcomes[1].changes)
        self.assertEqual(outcomes[0].full_evals, outcomes[1].full_evals)
        self.assertEqual(outcomes[0].coarse_evals, outcomes[1].coarse_evals)
        self.assertEqual(len(traces[0]), len(traces[1]))
        comparisons = 0
        for index, (cached, uncached) in enumerate(zip(*traces)):
            with self.subTest(model=name, validation=index):
                self.assertEqual(cached[:3], uncached[:3])
                self.assertEqual(len(cached[3]), len(uncached[3]))
                for (left_start, left), (right_start, right) in zip(cached[3], uncached[3]):
                    self.assertEqual(left_start, right_start)
                    if left is None or right is None:
                        self.assertIs(left, right)
                    else:
                        self.assertTrue(np.array_equal(left, right),
                                        f'{name} validation {index} batch {left_start}')
                    comparisons += 1
        print(f'R8f cache exact model={name} validations={len(traces[0])} '
              f'logit_batches={comparisons} history={len(outcomes[0].history)}', flush=True)

    def test_resnet20_full_tune(self):
        self.compare_frozen_tune('resnet20')

    def test_silu_full_tune(self):
        self.compare_frozen_tune('resnet20_silu')


if __name__ == '__main__':
    unittest.main()
