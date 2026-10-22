import unittest

import torch
from torch import nn
from torch.nn import functional as F

from pt_agent.intake import IntakeError, import_module


class UnsupportedOps(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return F.relu(torch.sort(self.pool(x))[0])


class Dynamic(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x*2
        return x


class Divide(nn.Module):
    def forward(self, x):
        return x/x


class IntegerIndex(nn.Module):
    def forward(self, x):
        return x[:, 0]


class IntakeErrorTests(unittest.TestCase):
    def test_collect_all_unsupported(self):
        with self.assertRaises(IntakeError) as caught:
            import_module(UnsupportedOps(), (torch.randn(1, 2, 4, 4),), "unsupported")
        entries = caught.exception.entries
        self.assertTrue(any("MaxPool2d" in entry["reason"] for entry in entries))
        self.assertTrue(any("sort" in entry["target"] for entry in entries))
        for entry in entries:
            self.assertEqual(set(entry), {"fx_node", "op", "target", "reason"})

    def test_dynamic_control(self):
        with self.assertRaises(IntakeError) as caught:
            import_module(Dynamic(), (torch.randn(1, 8),), "dynamic")
        self.assertEqual(caught.exception.entries[0]["op"], "trace")

    def test_activation_divisor(self):
        with self.assertRaisesRegex(IntakeError, "activation tensor as divisor"):
            import_module(Divide(), (torch.ones(1, 8),), "divide")

    def test_integer_index(self):
        with self.assertRaisesRegex(IntakeError, "integer indexing"):
            import_module(IntegerIndex(), (torch.randn(1, 3, 4),), "integer")
