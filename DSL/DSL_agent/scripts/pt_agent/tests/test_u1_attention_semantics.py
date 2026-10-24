"""U1 §12: individual attention operations, with no model-specific mapping."""

import unittest

import numpy as np
import torch
from torch import nn

from pt_agent.intake import import_module
from pt_agent.intake.graph_exec import forward


def values(shape, scale=1000):
    count = int(np.prod(shape))
    return (torch.arange(count, dtype=torch.float64).reshape(shape) % 137-68)/scale


class AttentionSemanticTests(unittest.TestCase):
    def compare(self, fn, *inputs):
        class Function(nn.Module):
            def forward(self, *args):
                return fn(*args)
        model = Function().double().eval()
        graph = import_module(model, tuple(inputs), 'operation', frontend='aten')
        expected = model(*inputs)
        actual = forward(graph, dict(zip(graph.inputs, [x.numpy() for x in inputs])))
        outputs = expected if isinstance(expected, tuple) else (expected,)
        self.assertEqual(len(outputs), len(graph.outputs))
        for name, target in zip(graph.outputs, outputs):
            np.testing.assert_allclose(actual[name], target.numpy(), atol=1e-12, rtol=0, equal_nan=True)
        return graph

    def test_broadcast_gqa(self):
        self.compare(lambda x: x.unsqueeze(2).expand(1, 2, 2, 80, 8), values((1, 2, 80, 8)))

    def test_expand_then_reshape_gqa(self):
        self.compare(lambda x: x.unsqueeze(2).expand(1, 2, 2, 80, 8).reshape(1, 4, 80, 8),
                     values((1, 2, 80, 8)))

    def test_rotate_slices_concat_and_negation(self):
        for heads in (2, 4):
            self.compare(lambda x: torch.cat((-x[..., 4:8], x[..., :4]), dim=-1),
                         values((1, heads, 80, 8)))

    def test_nonunit_slice(self):
        self.compare(lambda x: x[:, :, 1:80:3, ::2], values((1, 4, 80, 8)))

    def test_compare_and_where(self):
        mask = torch.tensor([[1.]*49+[0.]*31], dtype=torch.float64)
        base = torch.full((1, 1, 80, 80), torch.finfo(torch.float64).min, dtype=torch.float64).triu(1)
        self.compare(lambda x: torch.where((base+x[:, None, None, :]) == 0,
                                           torch.finfo(torch.float64).min, base), mask)

    def test_explicit_casts_and_rms_arithmetic(self):
        self.compare(lambda x: (x.float()*torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)+1e-6)).double(),
                     values((1, 80, 32)))

    def test_gqa_batched_matmul(self):
        self.compare(lambda q, k, v: torch.bmm(torch.bmm(q, k), v),
                     values((4, 80, 8)), values((4, 8, 80)), values((4, 80, 8)))

    def test_masked_softmax_minimum_and_infinity(self):
        x = values((1, 4, 80, 80))
        for minimum in (torch.finfo(torch.float64).min, float('-inf')):
            mask = torch.full((1, 1, 80, 80), minimum, dtype=torch.float64).triu(1)
            self.compare(lambda a, m: torch.softmax((a+m).float(), dim=-1).double(), x, mask)

    def test_effect_only_write_into_static_view_keeps_input_dependency(self):
        def fn(x):
            base = torch.zeros((1, 1, 80, 80), dtype=torch.float64)
            view = base[:, :, :, :80]
            view.copy_(x[:, None, None, :].expand(1, 1, 80, 80))
            return base+1
        self.compare(fn, values((1, 80)))


if __name__ == '__main__':
    unittest.main()
