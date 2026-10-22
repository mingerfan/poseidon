import copy
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pt_agent.analyzer import reference_forward
from pt_agent.intake import import_module
from pt_agent.intake.coverage import report


class Layer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class Function(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class Constant(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(8, generator=torch.Generator().manual_seed(7)))

    def forward(self, x):
        return x+self.b


class DropIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.5)
        self.identity = nn.Identity()

    def forward(self, x):
        return F.relu(self.identity(self.drop(x)))


def batchnorm():
    result = nn.BatchNorm2d(4).double().eval()
    rng = torch.Generator().manual_seed(7)
    with torch.no_grad():
        result.running_mean.copy_(torch.randn(4, generator=rng, dtype=torch.float64))
        result.running_var.copy_(torch.rand(4, generator=rng, dtype=torch.float64)+0.5)
        result.weight.copy_(torch.randn(4, generator=rng, dtype=torch.float64))
        result.bias.copy_(torch.randn(4, generator=rng, dtype=torch.float64))
    return result


def split_add(x):
    a, b = torch.split(x, 3, dim=1)
    return a+b


torch.manual_seed(7)
CASES = [
    (Layer(nn.Conv2d(3, 8, 3, padding=1)), (1, 3, 8, 8), ["conv2d"]),
    (Layer(nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8, bias=False)), (1, 8, 8, 8), ["conv2d"]),
    (Layer(nn.Conv2d(4, 6, 3, dilation=2)), (1, 4, 9, 9), ["conv2d"]),
    (Layer(nn.Linear(16, 5)), (1, 16), ["linear"]),
    (Function(lambda x: torch.matmul(x, x.transpose(1, 2))), (1, 4, 6), ["permute", "matmul"]),
    (Function(lambda x: x+x*2.0-1.0), (1, 8), ["mul", "add", "sub"]),
    (Function(lambda x: 3.0-x), (1, 8), ["sub"]),
    (Function(lambda x: x/4.0), (1, 8), ["mul"]),
    (Function(lambda x: -x), (1, 8), ["mul"]),
    (Constant(), (1, 8), ["add"]),
    (Layer(batchnorm()), (1, 4, 5, 5), ["batchnorm"]),
    (Layer(nn.LayerNorm(8)), (1, 3, 8), ["layernorm"]),
    (Function(F.relu), (1, 16), ["act"]),
    (Layer(nn.GELU()), (1, 16), ["act"]),
    (Layer(nn.GELU(approximate="tanh")), (1, 16), ["act"]),
    (Layer(nn.SiLU()), (1, 16), ["act"]),
    (Function(torch.sigmoid), (1, 16), ["act"]),
    (Function(torch.tanh), (1, 16), ["act"]),
    (Layer(nn.Hardswish()), (1, 16), ["act"]),
    (Function(torch.exp), (1, 16), ["act"]),
    (Function(lambda x: F.softmax(x, dim=-1)), (1, 3, 8), ["softmax"]),
    (Layer(nn.AvgPool2d(2)), (1, 4, 8, 8), ["avgpool"]),
    (Layer(nn.AdaptiveAvgPool2d(1)), (1, 4, 8, 8), ["avgpool"]),
    (Layer(nn.AdaptiveAvgPool2d(3)), (1, 4, 8, 8), ["adaptive_avgpool"]),
    (Function(lambda x: torch.flatten(x, 1)), (1, 2, 3, 4), ["flatten"]),
    (Function(lambda x: x.view(1, -1)), (1, 2, 3, 4), ["reshape"]),
    (Function(lambda x: x.permute(0, 2, 1)), (1, 3, 5), ["permute"]),
    (Function(lambda x: torch.cat([x, x*2.0], dim=1)), (1, 4), ["mul", "concat"]),
    (Function(lambda x: x[:, :, ::2, 1:]), (1, 2, 6, 6), ["slice", "slice"]),
    (Function(split_add), (1, 6), ["split", "add"]),
    (Function(lambda x: F.pad(x, (0, 0, 0, 0, 2, 2))), (1, 4, 3, 3), ["pad"]),
    (Function(lambda x: x.mean(dim=(2, 3))), (1, 4, 5, 5), ["mean"]),
    (Function(lambda x: torch.sum(x, dim=1, keepdim=True)), (1, 4, 5), ["sum"]),
    (DropIdentity(), (1, 8), ["act"]),
]


class OpTests(unittest.TestCase):
    def check_case(self, number):
        original, shape, expected = CASES[number-1]
        module = copy.deepcopy(original).double().eval()
        x = torch.randn(shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
        graph = import_module(module, (x,), f"op{number}")
        self.assertEqual([node.op for node in graph.nodes], expected)
        actual = reference_forward(graph, {"x": x.numpy()})[graph.outputs[0]]
        with torch.no_grad():
            expected_value = module(x).numpy()
        self.assertEqual(actual.shape, expected_value.shape)
        self.assertLessEqual(float(np.max(np.abs(actual-expected_value))), 1e-9)
        self.assertEqual([node.id for node in graph.nodes], [f"n{i}" for i in range(len(graph.nodes))])
        if number == 6:
            self.assertEqual(graph.nodes[0].attrs["scalar"], 2.0)
            self.assertEqual(graph.nodes[2].attrs["scalar"], 1.0)
        elif number == 7:
            self.assertTrue(graph.nodes[0].attrs["reverse"])
        elif number in (8, 9):
            self.assertEqual(graph.nodes[0].attrs["scalar"], 0.25 if number == 8 else -1.0)
        elif number == 10:
            self.assertEqual(graph.nodes[0].attrs["const"], "b")
        elif 13 <= number <= 20:
            self.assertEqual(graph.nodes[0].attrs["fn"], ["relu", "gelu", "gelu_tanh", "silu", "sigmoid", "tanh", "hardswish", "exp"][number-13])
        elif number == 21:
            self.assertEqual(graph.nodes[0].attrs["axis"], 2)
        elif number == 23:
            self.assertEqual(report(graph)["needs_composition"], 0)
        elif number == 24:
            self.assertEqual(report(graph)["needs_composition"], 1)
            self.assertEqual(report(graph)["needs_composition_ops"], ["adaptive_avgpool"])
        elif number == 26:
            self.assertEqual(graph.nodes[0].attrs["shape"], (1, 24))
        elif number == 34:
            self.assertNotIn("drop", graph.tensors)
            self.assertNotIn("identity", graph.tensors)
            self.assertEqual(set(graph.tensors), {"x", graph.outputs[0]})

    def check_batch(self, number):
        original, shape, _ = CASES[number-1]
        module = copy.deepcopy(original).double().eval()
        graph = import_module(module, (torch.zeros(shape, dtype=torch.float64),), "batch")
        x = torch.randn((3,)+shape[1:], generator=torch.Generator().manual_seed(7), dtype=torch.float64)
        actual = reference_forward(graph, {"x": x.numpy()})[graph.outputs[0]]
        with torch.no_grad():
            expected = torch.cat([module(sample[None]) for sample in x]).numpy()
        self.assertEqual(actual.shape, expected.shape)
        self.assertLessEqual(float(np.max(np.abs(actual-expected))), 1e-9)

    def test_batch_conv(self):
        self.check_batch(1)

    def test_batch_batchnorm(self):
        self.check_batch(11)


def case_test(number):
    def run(self):
        self.check_case(number)
    return run


for _number in range(1, len(CASES)+1):
    setattr(OpTests, f"test_op_{_number:02d}", case_test(_number))
