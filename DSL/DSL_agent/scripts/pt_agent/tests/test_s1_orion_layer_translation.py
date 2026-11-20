"""Tiny deterministic translation checks; no fit, compiler, or scheme init."""
import copy
import unittest

import torch
import torch.nn.functional as F
import orion.nn as on
from orion.core.tracer import OrionTracer

from pt_agent.experiments.s1_orion_layer_translation import build_from_torch


def fixed(module):
    with torch.no_grad():
        for name, value in module.state_dict().items():
            if name.endswith('running_var'):
                value.fill_(1.25)
            elif value.is_floating_point():
                value.fill_(0.125)
            else:
                value.fill_(7)
    module.eval()
    return module


class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1)
        self.bn = torch.nn.BatchNorm2d(2, eps=2e-5)
        self.act = torch.nn.SiLU()
        self.fc = torch.nn.Linear(2, 3)

    def forward(self, x):
        y = self.act(self.bn(self.conv(x)))
        y = self.act(self.bn(y))
        y = F.avg_pool2d(y, y.size()[3])
        return self.fc(y.view(y.size(0), -1))


class TranslationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.arange(32, dtype=torch.float64).reshape(1, 2, 4, 4) / 100

    def test_shared_state_and_activation_calls_are_independent_and_exact(self):
        original = fixed(Shared().double())
        snapshot = {k: v.clone() for k, v in original.state_dict().items()}
        translated, report = build_from_torch(original, self.x)
        with torch.no_grad():
            torch.testing.assert_close(translated(self.x), original(self.x), rtol=1e-12, atol=1e-12)
        acts = [m for m in translated.modules() if isinstance(m, on.SiLU)]
        bns = [m for m in translated.modules() if isinstance(m, on.BatchNorm2d)]
        self.assertEqual([m.degree for m in acts], [63, 63])
        self.assertEqual(len(bns), 2)
        self.assertNotEqual(bns[0].weight.data_ptr(), bns[1].weight.data_ptr())
        self.assertEqual([m.eps for m in bns], [2e-5, 2e-5])
        self.assertTrue(report['copied_state_values_equal'])
        self.assertEqual(report['fit_calls'], 0)
        self.assertEqual(report['compile_calls'], 0)
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(snapshot[name], value))

    def test_generic_unequal_channel_padding_and_positive_spatial_strides(self):
        class Selector(torch.nn.Module):
            def forward(self, x):
                return F.pad(x[:, :, ::3, ::2], (0, 0, 0, 0, 1, 2), 'constant', 0)
        x = torch.arange(126, dtype=torch.float64).reshape(1, 2, 7, 9)
        original = Selector()
        translated, report = build_from_torch(original, x)
        self.assertTrue(torch.equal(translated(x), original(x)))
        self.assertEqual(report['selectors'][0]['channel_pad'], [1, 2])
        self.assertEqual(report['selectors'][0]['stride'], [3, 2])
        conv = next(m for m in translated.modules() if isinstance(m, on.Conv2d))
        self.assertEqual(set(conv.weight.detach().flatten().tolist()), {0.0, 1.0})
        self.assertFalse(conv.weight.requires_grad)

    def test_evaluation_dropout_module_pool_and_flatten(self):
        original = fixed(torch.nn.Sequential(torch.nn.AvgPool2d(2), torch.nn.Dropout(),
            torch.nn.Flatten(), torch.nn.Linear(8, 2)).double())
        translated, report = build_from_torch(original, self.x)
        torch.testing.assert_close(translated(self.x), original(self.x), rtol=1e-12, atol=1e-12)
        pools = [m for m in translated.modules() if isinstance(m, on.AvgPool2d)]
        self.assertEqual(len(pools), 1)
        self.assertTrue(torch.equal(pools[0].weight, torch.full_like(pools[0].weight, 0.25)))
        self.assertTrue(any(r.get('role') == 'Dropout' for r in report['original_to_translated_calls']))

    def test_official_tracer_inlines_wrapper_to_Orion_leaf_modules(self):
        translated, _ = build_from_torch(fixed(Shared().double()), self.x)
        traced = OrionTracer().trace_model(translated)
        calls = [n for n in traced.graph.nodes if n.op == 'call_module']
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(isinstance(traced.get_submodule(str(n.target)), on.Module) for n in calls))
        self.assertTrue(callable(translated.set_scheme))
        self.assertTrue(callable(translated.set_margin))

    def test_source_float32_state_copied_losslessly_without_mutation(self):
        original = fixed(torch.nn.Sequential(torch.nn.Linear(4, 2)))
        before = copy.deepcopy(original.state_dict())
        translated, report = build_from_torch(original, torch.ones(1, 4, dtype=torch.float64))
        self.assertTrue(report['copied_state_values_equal'])
        self.assertTrue(all(v.dtype == torch.float64 for v in translated.parameters()))
        for key, value in original.state_dict().items():
            self.assertEqual(value.dtype, before[key].dtype)
            self.assertTrue(torch.equal(value, before[key]))

    def test_unknown_ops_and_nonzero_padding_reject(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported Torch layer'):
            build_from_torch(torch.nn.Sequential(torch.nn.ReLU()).double(), self.x)
        class BadPad(torch.nn.Module):
            def forward(self, x):
                return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 1, 1), value=2)
        with self.assertRaisesRegex(ValueError, 'Tensor slicing'):
            build_from_torch(BadPad(), self.x)


if __name__ == '__main__':
    unittest.main()
