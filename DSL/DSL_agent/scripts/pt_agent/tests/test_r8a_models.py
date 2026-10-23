from collections import Counter
import copy
import gc
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pt_agent.analyzer import calibrate, reference_forward
from pt_agent.backends import MockBackend
from pt_agent.data.cifar import calibration_set, normalize
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.intake import IntakeError, import_module
from pt_agent.lower import decode_outputs, lower, pack_inputs
from pt_agent.models.zoo import (CIFAR_NORMALIZATION, CnnPlain, IMAGENET_NORMALIZATION,
                                 Mlp, ZooModel, get)
from pt_agent.planners.rule import plan
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import DATA, WEIGHTS, require


BASE = pt_result('run-resnet20-rule-20261020T165817')


class StaticPool(nn.Module):
    def __init__(self, style):
        super().__init__()
        self.style = style

    def forward(self, x):
        if self.style == 0:
            return F.avg_pool2d(x, x.size()[3]).view(x.size(0), -1)
        if self.style == 1:
            return torch.reshape(F.avg_pool2d(x, x.shape[3]), (x.shape[0], -1))
        if self.style == 2:
            return torch.flatten(F.avg_pool2d(x, x.size(dim=3)), start_dim=x.size(dim=0))
        return F.avg_pool2d(x, x.size(3)).flatten(x.size(0))


@require(DATA, WEIGHTS, BASE)
class ModelsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261022)
        torch.set_num_threads(2)

    def compare_reference(self, model, graph, images):
        with torch.no_grad():
            reference = copy.deepcopy(model).double().eval()
            expected = np.concatenate([reference(torch.from_numpy(images[i:i+1])).numpy()
                                       for i in range(len(images))])
        actual = reference_forward(graph, {graph.inputs[0]: images})[graph.outputs[0]]
        self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)

    def test_size_shape_and_flatten(self):
        images = np.random.default_rng(1).normal(size=(2, 4, 8, 8))
        for style in range(4):
            with self.subTest(style=style):
                model = StaticPool(style).eval()
                graph = import_module(model, (torch.zeros(1, 4, 8, 8),), 'pool', frontend="aten")
                self.assertEqual([node.op for node in graph.nodes], ['mean'])
                self.assertEqual(graph.nodes[0].attrs, dict(axes=(2, 3), keepdim=False))
                self.assertEqual(graph.nodes[0].outputs, graph.outputs)
                self.compare_reference(model, graph, images)

    def test_global_pool_without_flatten_is_retained(self):
        model = nn.AvgPool2d(8).eval()
        graph = import_module(model, (torch.zeros(1, 4, 8, 8),), 'pool')
        self.assertEqual([node.op for node in graph.nodes], ['avgpool'])

    def test_functionals_are_restored_after_success_and_trace_error(self):
        names = ('avg_pool2d', 'max_pool2d', 'adaptive_avg_pool2d')
        originals = {name: id(getattr(F, name)) for name in names}
        import_module(StaticPool(0), (torch.zeros(1, 4, 8, 8),), 'pool')
        self.assertEqual({name: id(getattr(F, name)) for name in names}, originals)

        class Bad(nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x
                return -x

        with self.assertRaises(IntakeError):
            import_module(Bad(), (torch.zeros(1, 4, 8, 8),), 'bad')
        self.assertEqual({name: id(getattr(F, name)) for name in names}, originals)

    def test_size_value_as_tensor_is_rejected(self):
        class Bad(nn.Module):
            def forward(self, x):
                return x+x.size(0)

        with self.assertRaises(IntakeError) as caught:
            import_module(Bad(), (torch.zeros(1, 4, 8, 8),), 'bad')
        self.assertEqual(caught.exception.entries[0]['reason'], 'size value used as tensor')

    def test_silu_import(self):
        zm = get('resnet20_silu', Path('/unused'))
        graph = import_module(zm.module, (zm.example_input,), zm.name)
        original = get('resnet20', Path('/unused'))
        original_graph = import_module(original.module, (original.example_input,), original.name)
        self.assertEqual(len(graph.nodes), 74)
        self.assertEqual(Counter(node.op for node in graph.nodes), Counter(node.op for node in original_graph.nodes))
        self.assertTrue(all(node.attrs['fn'] == 'silu' for node in graph.nodes if node.op == 'act'))
        images, _ = calibration_set(normalization=zm.normalization)
        self.compare_reference(zm.module, graph, images[:2])

    def test_small_model_imports(self):
        for name, model, shape, counts in [
                ('cnn_plain', CnnPlain(), (1, 3, 32, 32), dict(conv2d=3, batchnorm=3, act=3, mean=1, linear=1)),
                ('mlp', Mlp(), (1, 3072), dict(linear=2, act=1))]:
            with self.subTest(model=name):
                model.eval()
                graph = import_module(model, (torch.zeros(shape),), name)
                self.assertEqual(dict(Counter(node.op for node in graph.nodes)), counts)
                images = np.random.default_rng(2).normal(size=(2,)+shape[1:])
                self.compare_reference(model, graph, images)

    def rule_exact(self, model, shape, name):
        model.eval()
        graph = import_module(model, (torch.zeros(shape),), name)
        images = np.random.default_rng(3).normal(size=(2,)+shape[1:])
        calibration = calibrate(graph, {graph.inputs[0]: images})
        source = plan(graph, calibration, deg=127, margin=1.5)
        program = parse(source)
        self.assertTrue(check(program, {key: value.shape for key, value in graph.weights.items()}).ok)
        backend = MockBackend(16384, batch=2)
        backend.bind(pack_inputs(program, [images]))
        values, layouts = lower(program, graph.weights, backend, act_mode='exact')
        actual = decode_outputs([value.value for value in values], layouts)[0]
        expected = reference_forward(graph, {graph.inputs[0]: images})[graph.outputs[0]]
        self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)
        del backend, values
        gc.collect()
        return source

    def test_mlp_rule_flat_path(self):
        source = self.rule_exact(Mlp(), (1, 3072), 'mlp')
        program = parse(source)
        self.assertEqual(program.params[0].shape, (1, 3072))
        self.assertEqual(program.ops[0].kind, 'pack')
        self.assertEqual(program.ops[0].attrs['layout'], 'flat')
        self.assertNotIn('relayout', source)

    def test_cnn_rule_exact(self):
        self.rule_exact(CnnPlain(), (1, 3, 32, 32), 'cnn_plain')

    def test_resnet20_rule_text_is_unchanged(self):
        zm = get('resnet20', Path('/unused'))
        graph = import_module(zm.module, (zm.example_input,), zm.name)
        calibration = json.loads((BASE/'calibration.json').read_text())
        self.assertEqual(plan(graph, calibration, deg=127, margin=1.5), (BASE/'program.ptdsl').read_text())

    def test_normalization_and_model_input_adaptation(self):
        pixels = np.arange(3*32*32, dtype=np.uint8).reshape(1, 3, 32, 32)
        mean, std = CIFAR_NORMALIZATION
        expected = (pixels.astype(np.float64)/255-np.asarray(mean).reshape(1, 3, 1, 1))/np.asarray(std).reshape(1, 3, 1, 1)
        self.assertEqual(float(np.max(np.abs(normalize(pixels, CIFAR_NORMALIZATION)-expected))), 0)
        self.assertEqual(float(np.max(np.abs(normalize(pixels)-normalize(pixels, IMAGENET_NORMALIZATION)))), 0)
        images = normalize(pixels)
        zm = ZooModel('mlp', Mlp().eval(), torch.zeros(1, 3072), IMAGENET_NORMALIZATION)
        self.assertEqual(zm.adapt(images).shape, (1, 3072))
        zm.name = 'cnn_plain'
        self.assertIs(zm.adapt(images), images)
