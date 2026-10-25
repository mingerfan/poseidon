"""U3b2 consumed-weight diagnostics, faithful artifacts and unchanged v4 feedback."""

import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check, consumed_weights, referenced_weights
from pt_agent.intake.graph import Graph, Node, TensorInfo
from pt_agent.models.info import ModelInfo
from pt_agent.models.zoo import get
from pt_agent.paths import REPO, pt_result
from pt_agent.planners import agent_tools, prompts
from pt_agent.tensor_sim import clear_caches
from pt_agent import tensor_sim
from pt_agent.tests._artifacts import DATA, WEIGHTS, require
from pt_agent.tests.test_p2_export import cases, FROZEN, MODELS, R6B
from pt_agent.tests.test_r8f2_feedback import SOURCE, context as feedback_context


U3A = pt_result('u3a-complete-20261024T161354727454/REPORT.json')
U3B = pt_result('u3b-agent-20261024T170907')
RERUN_ORDER = ('mobilenet_relu', 'vgg16_relu_maxpool', 'alexnet_relu_maxpool')
ARTIFACT_RESULTS = []


def linear_fixture():
    weights = {'kernel': np.eye(2), 'offset': np.zeros(2),
               'orphan.bias': np.ones(2), 'num_batches_tracked': np.zeros(())}
    graph = Graph('dense', [Node('dense_node', 'linear', ['v'], ['out'],
                  dict(weight='kernel', bias='offset'), 'test')],
                  {'v': TensorInfo('v', (1, 2), 'input')}, weights, ['v'], ['out'])
    source = ('@fhe.program(slots=16384)\ndef dense(v: Cipher[1, 2]):\n'
              '    x = pack(v, layout="flat")\n'
              '    out = linear(x, w=W["kernel"])\n    return out\n')
    return graph, source


def bn_fixture():
    weights = dict(kernel=np.ones((2, 1, 1, 1)), offset=np.zeros(2))
    weights.update({'norm.'+k: np.ones(2) for k in ('weight', 'bias', 'running_mean', 'running_var')})
    weights['norm.num_batches_tracked'] = np.zeros(())
    graph = Graph('normed', [
        Node('convolution', 'conv2d', ['v'], ['conv'], dict(weight='kernel', bias='offset'), 'test'),
        Node('normalization', 'batchnorm', ['conv'], ['out'], dict(weight='norm.weight',
            bias='norm.bias', mean='norm.running_mean', var='norm.running_var'), 'test')],
        {'v': TensorInfo('v', (1, 1, 2, 2), 'input')}, weights, ['v'], ['out'])
    source = ('@fhe.program(slots=16384)\ndef normed(v: Cipher[1, 1, 2, 2]):\n'
              '    x = pack(v, layout="mux", gap=1)\n'
              '    out = conv2d(x, w=W["kernel"], b=W["offset"], pad=0)\n'
              '    return out\n')
    return graph, source


def checked(source, graph):
    return check(parse(source), {k: v.shape for k, v in graph.weights.items()}, graph=graph)


def unused(result):
    return [d for d in result.diags if d.code == 'E_WEIGHT_UNUSED']


def names(result):
    return [json.loads(d.msg.split(' 未被使用：', 1)[0]) for d in unused(result)]


class WeightUsageTests(unittest.TestCase):
    def tearDown(self):
        clear_caches()

    def test_linear_bias_role_location_and_consumption_not_suffix(self):
        graph, source = linear_fixture()
        result = checked(source, graph)
        self.assertEqual(names(result), ['offset'])
        diag = unused(result)[0]
        for token in ('linear', 'dense_node', 'bias', 'b=W["offset"]'):
            self.assertIn(token, diag.msg)
        self.assertEqual(diag.lineno, 4)
        self.assertTrue(check(parse(source), {k: v.shape for k, v in graph.weights.items()}).ok)
        self.assertTrue(checked(source.replace('w=W["kernel"]', 'w=W["kernel"], b=W["offset"]'), graph).ok)

    def test_missing_bn_reports_four_roles_and_fused_conv_line(self):
        graph, source = bn_fixture()
        result = checked(source, graph)
        self.assertEqual(set(names(result)), {'norm.'+k for k in ('weight', 'bias', 'running_mean', 'running_var')})
        for diag in unused(result):
            self.assertIn('batchnorm normalization', diag.msg)
            self.assertEqual(diag.lineno, 4)
        fixed = source.replace('pad=0)', 'pad=0, bn="norm")')
        self.assertTrue(checked(fixed, graph).ok)
        self.assertNotIn('norm.num_batches_tracked', consumed_weights(graph))

    def test_explicit_separate_bias_and_static_macros_are_references(self):
        graph, source = linear_fixture()
        explicit = source.replace('    return out', '    y = bias(out, b=W["offset"])\n    return y')
        self.assertTrue(checked(explicit, graph).ok)
        macro = ('@fhe.program(slots=16384)\ndef dense(v: Cipher[1, 2]):\n'
                 '    x = pack(v, layout="flat")\n    out = block(x, "ker", "off")\n    return out\n'
                 'def block(x, k, b):\n'
                 '    y = linear(x, w=W[k + "nel"], b=W[b + "set"])\n    return y\n')
        self.assertTrue(checked(macro, graph).ok)
        bad = macro.replace(', b=W[b + "set"]', '')
        diag = unused(checked(bad, graph))[0]
        self.assertEqual((diag.lineno, diag.call_lines), (7, (4,)))

    def test_weight_valued_graph_inputs_and_nonstandard_consumed_buffer(self):
        graph, source = linear_fixture()
        graph.nodes.append(Node('consume_buffer', 'add', ['out', 'num_batches_tracked'], ['sum'], {}, 'test'))
        graph.nodes.append(Node('consume_other', 'mul', ['sum'], ['final'], {'const': 'orphan.bias'}, 'test'))
        result = checked(source, graph)
        self.assertEqual(set(names(result)), {'offset', 'num_batches_tracked', 'orphan.bias'})
        self.assertIn('input[1]', next(d.msg for d in unused(result) if '"num_batches_tracked"' in d.msg))
        self.assertIn('const', next(d.msg for d in unused(result) if '"orphan.bias"' in d.msg))

    def test_shared_missing_weight_lists_each_consumer(self):
        graph, source = linear_fixture()
        graph.nodes.append(Node('dense_again', 'linear', ['out'], ['last'],
                           dict(weight='kernel', bias='offset'), 'test'))
        result = checked(source, graph)
        self.assertEqual(names(result), ['offset'])
        for token in ('dense_node', 'dense_again'):
            self.assertIn(token, unused(result)[0].msg)

    def test_profile_or_prompt_enables_error_before_exact(self):
        graph, source = linear_fixture()
        ctx = SimpleNamespace(graph=graph, weight_shapes={k:v.shape for k,v in graph.weights.items()},
                              prompt_version='r8-v4')
        for prompt, profile in (('r8-v4', 'tensor'), ('r8-v5', 'mock')):
            with self.subTest(prompt=prompt, profile=profile), \
                    patch.object(agent_tools, '_exact_check', side_effect=AssertionError('exact ran')):
                ctx.prompt_version = prompt
                result = agent_tools.evaluate(source, ctx, engine_profile=profile)
                self.assertEqual(result['stage'], 'typecheck')
                self.assertIsNone(result['exact'])
                self.assertEqual(result['errors'][0]['code'], 'E_WEIGHT_UNUSED')
                self.assertEqual(result['errors'][0]['line_text'], source.splitlines()[3].strip())

    def test_all_missing_weights_reach_feedback_beyond_twenty(self):
        graph, source = linear_fixture()
        for i in range(24):
            name = f'missing{i}'
            graph.weights[name] = np.ones(2)
            graph.nodes.append(Node(f'consumer{i}', 'add', ['out'], [f'out{i}'], {'const':name}, 'test'))
        ctx = SimpleNamespace(graph=graph, weight_shapes={k:v.shape for k,v in graph.weights.items()},
                              prompt_version='r8-v5')
        result = agent_tools.evaluate(source, ctx)
        self.assertEqual(len(result['errors']), 25)
        self.assertTrue(all(e['code']=='E_WEIGHT_UNUSED' for e in result['errors']))

    def test_next_iteration_repair_requires_a_parsable_weight_reference(self):
        from pt_agent.experiments.u3b2_weight_usage_rerun import weight_events
        _, source = linear_fixture()
        error = dict(code='E_WEIGHT_UNUSED', message='"kernel" 未被使用：linear weight')
        first = dict(iteration=1, fb=dict(stage='typecheck', errors=[error]), messages=[])
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for text, expected in (('invalid python!', None), (source, True),
                                   (source.replace('linear(x, w=W["kernel"])', 'scale(x, c=1.)'), False)):
                following = dict(iteration=2, fb=dict(stage='parse' if expected is None else 'complete'),
                    messages=[dict(role='assistant', content=json.dumps(dict(program=text)))])
                (directory/'agent_trace.jsonl').write_text('\n'.join(map(json.dumps, (first, following)))+'\n')
                event = weight_events(directory)[0]
                self.assertIs(event['fixed_next'], expected)
                self.assertEqual(event['next_iteration'], 2)


@require(REPO/'.git')
class FrozenFeedbackTests(unittest.TestCase):
    def test_r8f2_feedback_fixture_matches_pre_change_v4_text(self):
        # Execute only trusted repository Python, never generated DSL.
        filename = 'DSL/DSL_agent/scripts/pt_agent/planners/agent_tools.py'
        source = subprocess.run(['git', 'show', 'HEAD:'+filename], cwd=REPO,
                                check=True, capture_output=True, text=True).stdout
        before = ModuleType('pt_agent.planners._u3b2_before')
        before.__package__ = 'pt_agent.planners'
        exec(compile(source, filename, 'exec'), before.__dict__)
        ctx = feedback_context()
        ctx.prompt_version = 'r8-v4'
        ctx.weights['unwritten'] = np.zeros(2)
        ctx.weight_shapes['unwritten'] = (2,)
        ctx.graph = Graph('feedback', [Node('first', 'linear', ['v'], ['h'],
                         dict(weight='identity', bias='unwritten'), 'test')], {}, ctx.weights, ['v'], ['h'])
        clear_caches()
        expected = before.evaluate(SOURCE, ctx, cache=False)
        clear_caches()
        actual = agent_tools.evaluate(SOURCE, ctx, cache=False)
        self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))
        self.assertEqual(prompts.feedback_message(actual, 1, None),
                         prompts.feedback_message(expected, 1, None))
        ARTIFACT_RESULTS.append(dict(kind='v4-feedback', fixture='R8f2', baseline='HEAD', equal=True))


@require(U3A, U3B, FROZEN, R6B, MODELS, DATA, WEIGHTS)
class RetainedWeightUsageTests(unittest.TestCase):
    def test_01_all_nine_u3a_rule_programs_have_no_unused_weights(self):
        for entry in json.loads(U3A.read_text())['models']:
            path = Path(entry['acceptance']).parent
            graph = ModelInfo.load(path).graph
            result = checked((path/'program.ptdsl').read_text(), graph)
            self.assertTrue(result.ok, [str(d) for d in result.diags])
            ARTIFACT_RESULTS.append(dict(kind='rule', model=entry['model'], unused=[]))
            del graph, result
            gc.collect()

    def test_02_four_frozen_and_seven_p2_programs_have_no_unused_weights(self):
        all_cases = [(f'development-{name}', name, FROZEN/'models'/name/'program.ptdsl')
                     for name in ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp')]+cases()
        for model in ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp'):
            graph = get(model, MODELS).import_graph()
            for label, name, path in all_cases:
                if name != model:
                    continue
                result = checked(path.read_text(), graph)
                self.assertTrue(result.ok, [str(d) for d in result.diags])
                ARTIFACT_RESULTS.append(dict(kind='frozen-or-p2', label=label, model=model, unused=[]))
            del graph
            gc.collect()

    def test_03_every_retained_relu_exact_failure_reports_conv_bias(self):
        cases_by_name = {e['model']:Path(e['acceptance']).parent for e in json.loads(U3A.read_text())['models']}
        seen = 0
        for model in RERUN_ORDER:
            graph = ModelInfo.load(cases_by_name[model]).graph
            shapes = {k:v.shape for k,v in graph.weights.items()}
            first_bias = next(n.attrs['bias'] for n in graph.nodes if n.op=='conv2d')
            for line in (U3B/'models'/model/'agent_trace.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row['fb']['stage'] != 'exact':
                    continue
                source = json.loads(next(m['content'] for m in row['messages'] if m['role']=='assistant'))['program']
                ctx = SimpleNamespace(graph=graph, weight_shapes=shapes, prompt_version='r8-v5')
                with patch.object(agent_tools, '_exact_check', side_effect=AssertionError('exact ran')):
                    fb = agent_tools.evaluate(source, ctx, engine_profile='tensor')
                self.assertEqual(fb['stage'], 'typecheck')
                missing = names(checked(source, graph))
                self.assertIn(first_bias, missing)
                self.assertEqual(len(fb['errors']), len(missing))
                consumed = consumed_weights(graph)
                errors = []
                for error in fb['errors']:
                    name = json.loads(error['message'].split(' 未被使用：', 1)[0])
                    errors.append(dict(weight=name, consumers=[dict(node=n.id, op=n.op, role=role)
                        for n,role in consumed[name]], **error))
                ARTIFACT_RESULTS.append(dict(kind='failure-replay', model=model,
                    iteration=row['iteration'], weights=missing, errors=errors))
                print(f'U3b2 replay {model}/{row["iteration"]}: '+json.dumps(missing), flush=True)
                seen += 1
            del graph
            gc.collect()
        self.assertEqual(seen, 10)


def previous_tensor_module():
    """Load trusted pre-§6 repository code for arithmetic comparisons only."""
    filename = 'DSL/DSL_agent/scripts/pt_agent/tensor_sim.py'
    source = subprocess.run(['git', 'show', 'HEAD:'+filename], cwd=REPO,
                            check=True, capture_output=True, text=True).stdout
    before = ModuleType('pt_agent._u3b2_tensor_before')
    before.__package__ = 'pt_agent'
    sys.modules[before.__name__] = before
    exec(compile(source, filename, 'exec'), before.__dict__)
    return before


@require(REPO/'.git', DATA[-1])
class TensorMemoryTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        clear_caches()
        self.limit = tensor_sim._prefix_limit
        tensor_sim.set_prefix_cache_limit(tensor_sim.DEFAULT_PREFIX_BYTES)

    def tearDown(self):
        clear_caches()
        tensor_sim.set_prefix_cache_limit(self.limit)

    def test_resource_adaptive_prefix_capacity(self):
        for address, expected in ((4*1024**3, int(.2*4*1024**3)),
                                  (16*1024**3, tensor_sim.DEFAULT_PREFIX_BYTES),
                                  (resource.RLIM_INFINITY, tensor_sim.DEFAULT_PREFIX_BYTES)):
            with self.subTest(address=address), patch.object(tensor_sim.resource, 'getrlimit',
                    return_value=(address, resource.RLIM_INFINITY)):
                self.assertEqual(tensor_sim.cache_info()['prefix']['max_bytes'], expected)
        with patch.object(tensor_sim.resource, 'getrlimit', return_value=(4*1024**3, 6*1024**3)):
            tensor_sim.set_prefix_cache_limit(17)
            self.assertEqual(tensor_sim.cache_info()['prefix']['max_bytes'], 17)

    def test_large_cheb_activation_array_equal_and_workspace_bound(self):
        from pt_agent.kernels.poly import cheb_eval, cheb_fit
        values = np.linspace(-3., 3., 20*64*32*32).reshape(20, 64, 32, 32)
        coeffs = cheb_fit('silu', 127, -4., 4.)
        evaluate = lambda part: cheb_eval(None, part*.25, coeffs)
        expected = evaluate(values)
        calls = []
        def tracked(part):
            arrays = tensor_sim._polynomial_workspace_arrays(127)
            workspace = part.nbytes*arrays+128*8*(arrays+1)
            self.assertLessEqual(workspace, 256*1024**2)
            calls.append(part.size)
            return evaluate(part)
        actual = tensor_sim._polynomial_chunks(values, tracked, 127)
        self.assertGreater(len(calls), 1)
        self.assertTrue(np.array_equal(actual, expected))
        ARTIFACT_RESULTS.append(dict(kind='memory-chunk-cheb', shape=list(values.shape),
            degree=127, chunks=len(calls), array_equal=True, workspace_limit=256*1024**2))

    def test_large_minimax_array_equal(self):
        from pt_agent.kernels.poly import cheb_eval, minimax_coeffs
        values = np.linspace(-1., 1., 128*8192).reshape(128, 8192)[:, ::-1]
        coefficients = minimax_coeffs((15, 15, 27))
        def evaluate(part):
            sign = part*.5
            for coeffs in coefficients:
                sign = cheb_eval(None, sign, coeffs)
            return part*(sign+.5)
        expected = evaluate(values)
        actual = tensor_sim._polynomial_chunks(values,
            lambda part: tensor_sim._minimax_relu(part, coefficients, 2.), 27)
        self.assertTrue(np.array_equal(actual, expected))
        ARTIFACT_RESULTS.append(dict(kind='memory-chunk-minimax', shape=list(values.shape),
            noncontiguous=True, array_equal=True))

    def test_act_and_maxpool_use_chunks_without_arithmetic_changes(self):
        before = previous_tensor_module()
        values = np.linspace(-.25, .25, 2*3*4*4).reshape(2, 3, 4, 4)
        for kind in ('act', 'maxpool'):
            for method in ('cheb', 'minimax_comp'):
                attrs = ('fn="relu", ' if kind == 'act' else 'k=(3,3), stride=(2,2), pad=(1,1), ')
                attrs += ('method="cheb", deg=127' if method == 'cheb' else
                          'method="minimax_comp", degrees=(15,15,27)')
                source = ('@fhe.program(slots=16384)\ndef memory(x: Cipher[1,3,4,4]):\n'
                          '    a = pack(x, layout="mux", gap=1)\n'
                          f'    b = {kind}(a, {attrs}, domain=(-2.,2.))\n    return b\n')
                program = parse(source)
                with self.subTest(kind=kind, method=method):
                    expected = before.run_tensor(program, {}, values, cache=False)
                    with patch.object(tensor_sim, 'POLYNOMIAL_WORKSPACE_BYTES', 65536), \
                            patch.object(tensor_sim, '_polynomial_chunks', wraps=tensor_sim._polynomial_chunks) as chunks:
                        actual = tensor_sim.run_tensor(program, {}, values, cache=False)
                    self.assertGreater(chunks.call_count, 0)
                    self.assertTrue(np.array_equal(actual, expected))


@require(REPO/'.git', FROZEN/'models', MODELS, DATA, WEIGHTS)
class FrozenMemoryValidationTests(unittest.TestCase):
    def test_four_frozen_tensor_validations_equal_pre_change(self):
        import torch
        torch.set_num_threads(1)
        before = previous_tensor_module()
        for name in ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp'):
            model = get(name, MODELS)
            graph = model.import_graph()
            calibration = json.loads((FROZEN/'models'/name/'calibration.json').read_text())
            ctx = agent_tools.build_context(graph, calibration, n_val=1000,
                                            normalization=model.normalization)
            program = parse((FROZEN/'models'/name/'program.ptdsl').read_text())
            logits = []
            with self.subTest(model=name):
                with patch.object(agent_tools, 'run_tensor', before.run_tensor):
                    expected = agent_tools.validate(program, ctx, engine='tensor', cache=False,
                        stop_on_divergence=False, output_callback=lambda value: logits.append(value))
                actual = agent_tools.validate(program, ctx, engine='tensor', cache=False,
                    stop_on_divergence=False, output_callback=lambda value: logits.append(value))
                self.assertEqual(actual, expected)
                self.assertTrue(np.array_equal(logits[0], logits[1], equal_nan=True))
                ARTIFACT_RESULTS.append(dict(kind='frozen-memory-validation', model=name,
                    images=1000, fields_equal=True, logits_array_equal=True, metrics=actual))
                print('U3b2 §6 frozen '+name+': 1000 images, all fields/logits equal', flush=True)
            del graph, ctx, logits, program
            before.clear_caches()
            clear_caches()
            gc.collect()


if __name__ == '__main__':
    unittest.main()
