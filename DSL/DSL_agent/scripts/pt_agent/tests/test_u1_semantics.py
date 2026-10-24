"""Semantic imports remain separate from lowering and from the frozen FX path."""

import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from pt_agent.intake import import_module, IntakeError
from pt_agent.intake.coverage import check, report
from pt_agent.intake.graph import Graph
from pt_agent.intake.graph_exec import forward


class SemanticTests(unittest.TestCase):
    def assert_model(self, model, sample):
        model = model.double().eval()
        graph = import_module(model, (sample,), 'semantics', frontend='aten')
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = forward(graph, {graph.inputs[0]: sample.numpy()})[graph.outputs[0]]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            graph.save(directory)
            loaded = Graph.load(directory)
            np.testing.assert_array_equal(forward(loaded, {graph.inputs[0]: sample.numpy()})[graph.outputs[0]], actual)
        return graph

    def test_pool_variants_and_kernel_boundary(self):
        sample = torch.arange(-20, 29, dtype=torch.float64).reshape(1, 1, 7, 7)
        for kwargs in (dict(kernel_size=2), dict(kernel_size=3, padding=1, stride=2, ceil_mode=True),
                       dict(kernel_size=2, dilation=2, stride=1)):
            with self.subTest(kwargs=kwargs):
                graph = self.assert_model(nn.MaxPool2d(**kwargs), sample)
                self.assertEqual(report(graph)['needs_composition_ops'], ['maxpool'])
                diag = check(graph)['diags'][0]
                self.assertEqual(diag['op_id'], graph.nodes[0].id)
                self.assertEqual(diag['lineno'], 1)
                self.assertIn('maxpool', diag['reason'])

    def test_unary_and_broadcast_semantics(self):
        class Functions(nn.Module):
            def forward(self, x):
                y = torch.erf(torch.sqrt(x)) + torch.rsqrt(x) + torch.reciprocal(x) + torch.pow(x, 2)
                return y.unsqueeze(1).expand(1, 3, 4)
        graph = self.assert_model(Functions(), torch.arange(1, 5, dtype=torch.float64).reshape(1, 4))
        self.assertIn('broadcast', [node.op for node in graph.nodes])
        self.assertTrue(all(diag['op_id'] and diag['lineno'] for diag in check(graph)['diags']))

    def test_constant_position_subgraph_folds_but_mask_stays_encrypted(self):
        class Masked(nn.Module):
            def forward(self, x, mask):
                positions = torch.arange(4, dtype=torch.float64).cos().reshape(1, 4)
                return (x+positions)*(1-mask)
        sample = torch.arange(4, dtype=torch.float64).reshape(1, 4)
        mask = torch.tensor([[0., 1., 0., 1.]], dtype=torch.float64)
        graph = import_module(Masked(), (sample, mask), 'masked', frontend='aten')
        self.assertEqual(len(graph.inputs), 2)
        actual = forward(graph, dict(zip(graph.inputs, (sample.numpy(), mask.numpy()))))[graph.outputs[0]]
        np.testing.assert_allclose(actual, Masked()(sample, mask).numpy(), atol=1e-12, rtol=0)
        self.assertNotIn('embedding', [node.op for node in graph.nodes])
        self.assertIn(graph.inputs[1], [name for node in graph.nodes for name in node.inputs])

    def test_fx_maxpool_contract_is_preserved(self):
        with self.assertRaises(IntakeError):
            import_module(nn.MaxPool2d(2), (torch.zeros(1, 1, 4, 4),), 'old_fx')


if __name__ == '__main__':
    unittest.main()
