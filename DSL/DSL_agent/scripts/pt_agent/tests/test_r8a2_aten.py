from collections import Counter
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pt_agent.analyzer import calibrate, reference_forward
from pt_agent.data.cifar import calibration_set
from pt_agent.dsl.ir import canonical
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.intake import IntakeError, import_module
from pt_agent.intake.aten_frontend import CONTROL_FLOW_REASON
from pt_agent.intake.coverage import report
from pt_agent.models.zoo import get
from pt_agent.planners.rule import plan
from pt_agent.tests.test_r3_ops import CASES
from pt_agent.tests.test_r8a_models import StaticPool
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import DATA, require


BASE = pt_result('run-resnet20-rule-20261020T165817')
RESNET_COUNTS = dict(conv2d=19, batchnorm=19, act=19, add=9, slice=4, pad=2, mean=1, linear=1)


def parameter_independent_canonical(program):
    result = canonical(program)
    names = {'p:'+param[0]: 'p:'+str(index) for index, param in enumerate(result['params'])}
    for index, param in enumerate(result['params']):
        param[0] = str(index)
    for op in result['ops']:
        op[1] = [names.get(value, value) for value in op[1]]
    result['outputs'] = [names.get(value, value) for value in result['outputs']]
    return result


@require(DATA, BASE)
class AtenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261022)
        torch.set_num_threads(2)

    def compare(self, module, graph, images):
        model = copy.deepcopy(module).double().eval()
        with torch.no_grad():
            expected = np.concatenate([model(torch.from_numpy(images[i:i+1])).numpy()
                                       for i in range(len(images))])
        actual = reference_forward(graph, {graph.inputs[0]: images})[graph.outputs[0]]
        self.assertEqual(actual.shape, expected.shape)
        self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)

    def check_r3(self, index):
        module, shape, _ = CASES[index]
        module = copy.deepcopy(module).double().eval()
        inputs = torch.randn(shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
        graph = import_module(module, (inputs,), 'op'+str(index+1), frontend='aten')
        self.compare(module, graph, inputs.numpy())
        if index == 23:
            self.assertEqual(report(graph)['needs_composition_ops'], ['adaptive_avgpool'])
            self.assertEqual(report(graph)['needs_composition'], 1)
        elif index == 22:
            self.assertEqual(report(graph)['needs_composition'], 0)

    def resnet(self, name):
        zm = get(name, Path('/unused'))
        graph = zm.import_graph()
        self.assertEqual(len(graph.nodes), 74)
        self.assertEqual(dict(Counter(node.op for node in graph.nodes)), RESNET_COUNTS)
        self.assertIn('layer1.0.conv1.weight', graph.weights)
        state = zm.module.state_dict()
        for key, value in graph.weights.items():
            self.assertIn(key, state)
            self.assertEqual(value.shape, tuple(state[key].shape))
            self.assertFalse(value.flags.writeable)
        images, _ = calibration_set(normalization=zm.normalization)
        self.compare(zm.module, graph, images[:2])
        return zm, graph, images[:2]

    def test_resnet20_canonical(self):
        zm, graph, _ = self.resnet('resnet20')
        fx = import_module(zm.module, (zm.example_input,), zm.name)
        frozen = json.loads((BASE/'calibration.json').read_text())
        translated = {}
        self.assertEqual(len(fx.nodes), len(graph.nodes))
        for old, new in zip(fx.nodes, graph.nodes):
            self.assertEqual(old.op, new.op)
            for before, after in zip(old.outputs, new.outputs):
                translated[after] = frozen[before]
        actual = parse(plan(graph, translated, deg=127, margin=1.5))
        expected = parse((BASE/'program.ptdsl').read_text())
        self.assertEqual(parameter_independent_canonical(actual), parameter_independent_canonical(expected))

    def test_resnet20_silu(self):
        _, graph, images = self.resnet('resnet20_silu')
        self.assertTrue(all(node.attrs['fn'] == 'silu' for node in graph.nodes if node.op == 'act'))
        calib = calibrate(graph, {graph.inputs[0]: images})
        program = parse(plan(graph, calib, deg=127, margin=1.5))
        self.assertTrue(check(program, {key: value.shape for key, value in graph.weights.items()}).ok)
        self.assertEqual(len([op for op in program.ops if op.kind == 'act']), 19)
        self.assertTrue(all(op.attrs['fn'] == 'silu' for op in program.ops if op.kind == 'act'))

    def test_size_shape_writings(self):
        images = np.random.default_rng(7).normal(size=(2, 4, 8, 8))
        for style in range(4):
            with self.subTest(style=style):
                model = StaticPool(style).eval()
                graph = import_module(model, (torch.zeros(1, 4, 8, 8),), 'shape', frontend='aten')
                self.compare(model, graph, images)
                self.assertEqual([node.op for node in graph.nodes], ['mean'])

    def test_shape_control_flow_and_channel_shuffle(self):
        class Shuffle(nn.Module):
            def forward(self, x):
                n, c, h, w = x.shape
                if c == 4:
                    x = x.view(n, 2, c//2, h, w).transpose(1, 2).reshape(n, c, h, w)
                return F.hardswish(x)

        model = Shuffle()
        graph = import_module(model, (torch.zeros(1, 4, 4, 4),), 'shuffle', frontend='aten')
        self.compare(model, graph, np.random.default_rng(7).normal(size=(2, 4, 4, 4)))

    def test_parameter_names_and_sequential(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                              nn.Flatten(), nn.Linear(4*4*4, 5)).eval()
        graph = import_module(model, (torch.zeros(1, 3, 4, 4),), 'sequential', frontend='aten')
        self.assertTrue({'0.weight', '1.running_mean', '4.weight', '4.bias'} <= graph.weights.keys())
        self.compare(model, graph, np.random.default_rng(7).normal(size=(2, 3, 4, 4)))

    def test_collect_maxpool_and_transposed_convolution(self):
        model = nn.Sequential(nn.MaxPool2d(2), nn.ConvTranspose2d(1, 1, 2)).eval()
        with self.assertRaises(IntakeError) as caught:
            import_module(model, (torch.zeros(1, 1, 4, 4),), 'unsupported', frontend='aten')
        reasons = [entry['reason'] for entry in caught.exception.entries]
        self.assertNotIn('max pooling has no polynomial lowering in v1', reasons)
        self.assertIn('transposed convolution', reasons)
        from pt_agent.intake.graph_exec import forward
        from pt_agent.intake.coverage import check as check_graph
        sample = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
        model = nn.Sequential(nn.MaxPool2d(2)).double().eval()
        graph = import_module(model, (sample,), 'maxpool', frontend='aten')
        self.assertIn('maxpool', [node.op for node in graph.nodes])
        actual = forward(graph, {graph.inputs[0]: sample.numpy()})[graph.outputs[0]]
        self.assertLessEqual(float(np.max(np.abs(actual-model(sample).numpy()))), 1e-12)
        checked = check_graph(graph)
        self.assertTrue(checked['ok'])
        from pt_agent.analyzer import calibrate
        from pt_agent.dsl.parser import parse
        from pt_agent.planners.rule import plan
        from pt_agent.tests.test_u3_kernels import mock
        arrays = {graph.inputs[0]: sample.numpy()}
        program = parse(plan(graph, calibrate(graph, arrays)))
        np.testing.assert_allclose(mock(program, graph.weights, sample.numpy(), exact=True), actual,
                                   atol=1e-12, rtol=0)

    def test_data_dependent_control_flow(self):
        class Dynamic(nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x
                return -x

        with self.assertRaises(IntakeError) as caught:
            import_module(Dynamic(), (torch.zeros(1, 8),), 'dynamic', frontend='aten')
        self.assertEqual(caught.exception.entries[0]['op'], 'trace')
        self.assertEqual(caught.exception.entries[0]['reason'], CONTROL_FLOW_REASON)


def case_test(index):
    def run(self):
        self.check_r3(index)
    return run


for _index in range(len(CASES)):
    setattr(AtenTests, f'test_r3_op_{_index+1:02d}', case_test(_index))
