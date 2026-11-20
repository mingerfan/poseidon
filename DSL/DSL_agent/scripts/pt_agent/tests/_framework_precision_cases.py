"""Tiny reference-only checks of explicit framework intermediate rounding."""

import unittest


class FrameworkPrecisionCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cls.np, cls.torch = np, torch

    def test_expanded_rmsnorm_matches_original_mixed_precision(self):
        np, torch = self.np, self.torch
        from pt_agent.intake.graph import Graph, Node, TensorInfo
        from pt_agent.intake.framework_precision import expand_reference_precision
        from pt_agent.intake.graph_exec import forward
        weight = np.array([.7, -.3, 1.25, 2.], dtype=np.float64)
        graph = Graph('rms', [Node('rms', 'rmsnorm', ['x'], ['y'], {'weight': 'w', 'eps': 1e-6}, 'original rms')],
                      {'x': TensorInfo('x', (1, 2, 4), 'act'), 'y': TensorInfo('y', (1, 2, 4), 'act'),
                       'w': TensorInfo('w', (4,), 'weight')}, {'w': weight}, ['x'], ['y'])
        expanded = expand_reference_precision(graph, {'rms': 'torch.float32'})
        value = np.array([[[.123456789, -.55667788, 2.25, -.00012345], [11.75, .0023, -.18, .7]]], dtype=np.float64)
        x = torch.from_numpy(value).float()
        expected = (torch.from_numpy(weight)*(x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+1e-6)).double()).numpy()
        observed = forward(expanded, {'x': value})['y']
        self.assertTrue(np.array_equal(observed, expected))
        self.assertIs(expanded.weights['w'], weight)
        self.assertEqual(graph.nodes[0].op, 'rmsnorm')
        self.assertEqual(len(graph.nodes), 1)

    def test_expanded_masked_softmax_matches_original_dtype(self):
        np, torch = self.np, self.torch
        from pt_agent.intake.graph import Graph, Node, TensorInfo
        from pt_agent.intake.framework_precision import expand_reference_precision
        from pt_agent.intake.graph_exec import forward
        shape = (1, 1, 3, 3)
        graph = Graph('softmax', [Node('softmax', 'softmax', ['x', 'mask'], ['y'],
            {'keep': 'causal', 'mask': 'mask', 'axis': -1}, 'original attention')],
            {name: TensorInfo(name, shp, 'act') for name, shp in [('x', shape), ('y', shape), ('mask', (1, 3))]},
            {}, ['x', 'mask'], ['y'])
        expanded = expand_reference_precision(graph, {'softmax': 'torch.float32'})
        value = np.arange(9, dtype=np.float64).reshape(shape)*.123456789-.25
        mask = np.array([[1., 1., 0.]], dtype=np.float64)
        keep = torch.arange(3)[None, :] <= torch.arange(3)[:, None]
        keep = keep & torch.from_numpy(mask).bool()[:, None, None, :]
        expected = torch.softmax(torch.from_numpy(value).masked_fill(~keep, -float('inf')),
                                 dim=-1, dtype=torch.float32).double().numpy()
        observed = forward(expanded, {'x': value, 'mask': mask})['y']
        self.assertTrue(np.array_equal(observed, expected))

    def test_unknown_precision_and_unproven_node_are_rejected(self):
        from pt_agent.intake.graph import Graph, Node, TensorInfo
        from pt_agent.intake.framework_precision import expand_reference_precision
        graph = Graph('plain', [Node('add', 'add', ['x'], ['y'], {'scalar': .5}, '')],
                      {name: TensorInfo(name, (1, 2), 'act') for name in ('x', 'y')}, {}, ['x'], ['y'])
        with self.assertRaises(ValueError):
            expand_reference_precision(graph, {'add': 'torch.float32'})
        with self.assertRaises(ValueError):
            expand_reference_precision(graph, {'missing': 'torch.float32'})
        with self.assertRaises(ValueError):
            expand_reference_precision(graph, {'add': 'torch.float16'})
