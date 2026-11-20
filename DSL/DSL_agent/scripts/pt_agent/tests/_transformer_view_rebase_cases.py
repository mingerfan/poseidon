"""Tiny deterministic original Torch graphs for consumed token views."""

import unittest


class MaterializedViewCases(unittest.TestCase):
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

    def _check(self, mode):
        np, torch = self.np, self.torch
        from pt_agent.intake import import_module
        from pt_agent.intake.transformer import normalize
        from pt_agent.intake.graph_exec import forward

        class TokenViews(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('cos', torch.tensor([[[[1., .5, 1., .5], [.25, .75, .25, .75],
                                                            [-.5, 1., -.5, 1.], [.5, -.25, .5, -.25]]]], dtype=torch.float64))
                self.register_buffer('sin', torch.tensor([[[[0., .25, 0., .25], [.5, -.25, .5, -.25],
                                                            [.75, .5, .75, .5], [-.25, .75, -.25, .75]]]], dtype=torch.float64))

            def forward(self, x):
                q = x.reshape(1, 4, 2, 4).transpose(1, 2)
                rotated = torch.cat((-q[..., 2:], q[..., :2]), dim=-1)
                if mode == 'rotary':
                    value = q*self.cos + rotated*self.sin
                elif mode == 'act':
                    value = torch.nn.functional.silu(rotated)
                else:
                    value = rotated + q
                return value.transpose(1, 2).reshape(1, 4, 8)

        module = TokenViews().eval()
        x = torch.arange(32, dtype=torch.float64).reshape(1, 4, 8)/16 - .75
        probes = tuple((x + shift,) for shift in (.125, -.25, .375))
        raw = import_module(module, (x,), 'token_views', frontend='aten', validation_inputs=probes)
        normalized = normalize(raw)
        for (probe,) in ((x,), *probes):
            with torch.no_grad():
                expected = module(probe).numpy()
            arrays = {raw.inputs[0]: probe.numpy()}
            observed = forward(normalized, arrays)[normalized.outputs[0]]
            raw_result = forward(raw, arrays)[raw.outputs[0]]
            self.assertTrue(np.all(np.abs(raw_result - expected) <= 1e-12 + 1e-12*np.abs(expected)))
            self.assertTrue(np.all(np.abs(observed - expected) <= 1e-12 + 1e-12*np.abs(expected)),
                            (mode, float(np.max(np.abs(observed - expected)))))
        return normalized

    def test_rotary_consumes_half_permutation_once(self):
        graph = self._check('rotary')
        self.assertEqual(sum(node.op == 'shuffle' for node in graph.nodes), 1)

    def test_activation_consumes_permuted_input_once(self):
        self._check('act')

    def test_two_input_pointwise_consumes_permutation_once(self):
        self._check('add')
