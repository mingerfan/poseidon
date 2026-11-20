"""Deterministic small arithmetic regressions, without model calibration."""

import unittest


class StableSoftmaxReferenceCases(unittest.TestCase):
    def test_large_absolute_scores_and_disparate_rows(self):
        import numpy as np
        from pt_agent.kernels.stable_softmax_reference import fixed_shift_squaring
        x = np.array([[1010., 1009., 1008., 1007.], [-12., -11., -10., -9.]], dtype=np.float64)
        mask = np.array([[True, True, True, False], [True, True, False, False]])
        p, metadata = fixed_shift_squaring(x, mask, shift=1010., steps=8)
        y = np.where(mask, x, -np.inf)
        e = np.exp(y-y.max(-1, keepdims=True))
        expected = e/e.sum(-1, keepdims=True)
        self.assertTrue(np.all(np.abs(p-expected) <= 1e-12+1e-12*np.abs(expected)))
        self.assertFalse(metadata['dynamic_max_used'])
        self.assertEqual(len(metadata['stages']), 9)
        self.assertTrue(np.all(p[~mask] == 0))

    def test_zero_steps_and_causal_broadcast(self):
        import numpy as np
        from pt_agent.kernels.stable_softmax_reference import fixed_shift_squaring
        x = np.arange(18, dtype=np.float64).reshape(2, 3, 3)*.07
        mask = np.arange(3)[None, :] <= np.arange(3)[:, None]
        p, _ = fixed_shift_squaring(x, mask, shift=1.2, steps=0)
        y = np.where(mask, x, -np.inf)
        e = np.exp(y-y.max(-1, keepdims=True))
        self.assertTrue(np.all(np.abs(p-e/e.sum(-1, keepdims=True)) <= 1e-12))

    def test_invalid_configuration_keeps_failures_visible(self):
        import numpy as np
        from pt_agent.kernels.stable_softmax_reference import fixed_shift_squaring
        x = np.array([[1010., 1009.]], dtype=np.float64)
        with self.assertRaises((FloatingPointError, ValueError)):
            fixed_shift_squaring(x, np.ones_like(x, dtype=bool), shift=0., steps=0)
        with self.assertRaises(ValueError):
            fixed_shift_squaring(x, np.zeros_like(x, dtype=bool), shift=1010., steps=8)


if __name__ == '__main__':
    unittest.main()
