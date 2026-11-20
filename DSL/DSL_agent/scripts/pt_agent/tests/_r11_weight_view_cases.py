"""R11 storage semantics using finite tensor fixtures in the intake environment."""

import copy
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


INNER_CASE_COUNT = 12
_GLOBAL_WEIGHT = None


class WeightViewCases(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        self.np, self.torch = np, torch

    def private_module(self):
        torch = self.torch
        module = torch.nn.Module()
        module.register_parameter('weight', torch.nn.Parameter(
            torch.tensor([[2., -1.], [1., 3.]], dtype=torch.float64)))
        module.register_parameter('tied', module.weight)
        module.register_buffer('offset', torch.tensor([4., 5.], dtype=torch.float64))
        return copy.deepcopy(module)

    def mapper(self, private):
        from pt_agent.intake.aten_frontend import _AtenMapper
        from pt_agent.intake.graph import Graph
        names = {value.data_ptr(): name for name, value in
                 list(private.named_parameters())+list(private.named_buffers())}
        graph = Graph('weights', [], {}, {}, [], [])
        return _AtenMapper(graph, SimpleNamespace(graph=SimpleNamespace(nodes=[])), names)

    def import_model(self, model):
        from pt_agent.intake import import_module
        torch = self.torch
        example = (torch.tensor([[1., 2.]], dtype=torch.float64),)
        probes = tuple((torch.tensor([[float(i+2), -1.]], dtype=torch.float64),)
                       for i in range(3))
        with patch.object(torch, 'randn', side_effect=AssertionError('Random probe called')):
            return import_module(model, example, 'weight_views', frontend='aten',
                                 validation_inputs=probes)

    def linear_module(self, *, tied=False):
        torch = self.torch
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[2., -1.], [1., 3.]], dtype=torch.float64))
                if tied:
                    self.alias = self.weight

            def forward(self, x):
                first = torch.nn.functional.linear(x, self.weight)
                return first+torch.nn.functional.linear(x, self.alias) if tied else first
        return Linear().eval()

    def check_graph_result(self, graph, expected):
        from pt_agent.intake.graph_exec import forward
        actual = forward(graph, {graph.inputs[0]: self.np.array([[1., 2.]], dtype=self.np.float64)})
        self.np.testing.assert_allclose(actual[graph.outputs[0]], expected, atol=1e-12, rtol=0)

    def test_private_parameter_and_buffer_share_readonly_storage(self):
        private = self.private_module()
        mapper = self.mapper(private)
        for name, value in (('weight', private.weight), ('offset', private.offset)):
            with self.subTest(name=name):
                self.assertEqual(mapper.weight(name, value), name)
                array = mapper.graph.weights[name]
                self.assertEqual(array.__array_interface__['data'][0], value.data_ptr())
                self.assertTrue(self.np.shares_memory(array, value.detach().numpy()))
                self.assertEqual(array.dtype, self.np.float64)
                self.assertFalse(array.flags.writeable)
                self.assertEqual(mapper.graph.tensors[name].shape, tuple(value.shape))
                self.assertEqual(mapper.graph.tensors[name].kind, 'weight')
                with self.assertRaises(ValueError):
                    array.flat[0] = -123.

    def test_original_user_model_is_isolated_after_import(self):
        model = self.linear_module()
        graph = self.import_model(model)
        snapshot = graph.weights['weight'].copy()
        self.assertNotEqual(graph.weights['weight'].__array_interface__['data'][0], model.weight.data_ptr())
        with self.torch.no_grad():
            model.weight.fill_(99.)
        self.np.testing.assert_array_equal(graph.weights['weight'], snapshot)
        self.check_graph_result(graph, self.np.array([[0., 7.]]))

    def test_tied_parameters_and_transpose_keep_private_storage(self):
        private = self.private_module()
        self.assertIs(private.weight, private.tied)
        mapper = self.mapper(private)
        mapper.weight('first', private.weight)
        mapper.weight('tied', private.tied)
        mapper.weight('transposed', private.weight.t())
        first, tied, transposed = (mapper.graph.weights[key] for key in ('first', 'tied', 'transposed'))
        self.assertTrue(self.np.shares_memory(first, tied))
        self.assertTrue(self.np.shares_memory(first, transposed))
        self.assertEqual(transposed.__array_interface__['data'][0], private.weight.data_ptr())
        self.assertEqual(transposed.strides, private.weight.detach().numpy().T.strides)
        self.np.testing.assert_array_equal(transposed, first.T)
        self.assertTrue(all(not array.flags.writeable for array in (first, tied, transposed)))

    def test_tied_module_import_preserves_values(self):
        graph = self.import_model(self.linear_module(tied=True))
        self.assertEqual(list(graph.weights), ['weight'])
        self.assertFalse(graph.weights['weight'].flags.writeable)
        self.check_graph_result(graph, self.np.array([[0., 14.]]))

    def test_numpy_view_survives_private_module_collection(self):
        private = self.private_module()
        mapper = self.mapper(private)
        mapper.weight('weight', private.weight)
        array = mapper.graph.weights['weight']
        expected, pointer = array.copy(), private.weight.data_ptr()
        module_reference = weakref.ref(private)
        del private, mapper
        gc.collect()
        self.assertIsNone(module_reference())
        self.assertEqual(array.__array_interface__['data'][0], pointer)
        self.np.testing.assert_array_equal(array, expected)
        self.assertFalse(array.flags.writeable)

    def test_graph_save_load_preserves_values_and_metadata(self):
        from pt_agent.intake.graph import Graph
        graph = self.import_model(self.linear_module())
        with TemporaryDirectory() as directory:
            graph.save(Path(directory))
            recovered = Graph.load(Path(directory))
        self.assertEqual(recovered.name, graph.name)
        self.assertEqual(recovered.inputs, graph.inputs)
        self.assertEqual(recovered.outputs, graph.outputs)
        self.assertEqual(recovered.nodes, graph.nodes)
        self.assertEqual(recovered.tensors, graph.tensors)
        self.assertEqual(set(recovered.weights), set(graph.weights))
        for name in graph.weights:
            self.np.testing.assert_array_equal(recovered.weights[name], graph.weights[name])
            self.assertFalse(recovered.weights[name].flags.writeable)
        self.check_graph_result(recovered, self.np.array([[0., 7.]]))

    def test_registered_float32_uses_snapshot_copy(self):
        private = self.torch.nn.Module()
        private.register_buffer('weight', self.torch.tensor([2., 3.], dtype=self.torch.float32))
        mapper = self.mapper(private)
        mapper.weight('weight', private.weight)
        array = mapper.graph.weights['weight']
        self.assertEqual(array.dtype, self.np.float64)
        self.assertNotEqual(array.__array_interface__['data'][0], private.weight.data_ptr())
        private.weight.fill_(81.)
        self.np.testing.assert_array_equal(array, [2., 3.])
        self.assertFalse(array.flags.writeable)

    def test_non_tensor_uses_snapshot_copy(self):
        mapper = self.mapper(self.private_module())
        original = self.np.array([2., 3.], dtype=self.np.float64)
        mapper.weight('array', original)
        array = mapper.graph.weights['array']
        self.assertFalse(self.np.shares_memory(array, original))
        original[:] = 81.
        self.np.testing.assert_array_equal(array, [2., 3.])
        self.assertFalse(array.flags.writeable)

    def test_original_mapper_keeps_default_copy(self):
        from pt_agent.intake.graph import Graph
        from pt_agent.intake.torch_mapping import Mapper
        source = self.torch.tensor([2., 3.], dtype=self.torch.float64)
        mapper = Mapper(Graph('original', [], {}, {}, [], []), None)
        mapper.weight('weight', source)
        array = mapper.graph.weights['weight']
        self.assertNotEqual(array.__array_interface__['data'][0], source.data_ptr())
        source.fill_(81.)
        self.np.testing.assert_array_equal(array, [2., 3.])
        self.assertFalse(array.flags.writeable)

    def test_unregistered_derived_tensor_uses_snapshot_copy(self):
        private = self.private_module()
        mapper = self.mapper(private)
        derived = private.offset*2
        mapper.weight('derived', derived)
        array = mapper.graph.weights['derived']
        self.assertNotEqual(array.__array_interface__['data'][0], derived.data_ptr())
        derived.fill_(81.)
        self.np.testing.assert_array_equal(array, [8., 10.])
        self.assertFalse(array.flags.writeable)

    def test_global_captured_tensor_import_is_snapshot(self):
        global _GLOBAL_WEIGHT
        torch = self.torch
        _GLOBAL_WEIGHT = torch.tensor([[2., -1.], [1., 3.]], dtype=torch.float64)
        class GlobalWeight(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.linear(x, _GLOBAL_WEIGHT)
        graph = self.import_model(GlobalWeight())
        original = {key: value.copy() for key, value in graph.weights.items()}
        self.assertTrue(original)
        self.assertTrue(all(array.__array_interface__['data'][0] != _GLOBAL_WEIGHT.data_ptr()
                            for array in graph.weights.values()))
        _GLOBAL_WEIGHT.fill_(81.)
        for key, value in graph.weights.items():
            self.np.testing.assert_array_equal(value, original[key])
        self.check_graph_result(graph, self.np.array([[0., 7.]]))
        _GLOBAL_WEIGHT = None

    def test_closure_captured_tensor_import_is_snapshot(self):
        torch = self.torch
        captured = torch.tensor([[2., -1.], [1., 3.]], dtype=torch.float64)
        class ClosureWeight(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.linear(x, captured)
        graph = self.import_model(ClosureWeight())
        original = {key: value.copy() for key, value in graph.weights.items()}
        self.assertTrue(original)
        self.assertTrue(all(array.__array_interface__['data'][0] != captured.data_ptr()
                            for array in graph.weights.values()))
        captured.fill_(81.)
        for key, value in graph.weights.items():
            self.np.testing.assert_array_equal(value, original[key])
        self.check_graph_result(graph, self.np.array([[0., 7.]]))


if __name__ == '__main__':
    unittest.main()
