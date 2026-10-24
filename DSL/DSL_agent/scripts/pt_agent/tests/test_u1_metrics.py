"""Thresholds, divergent cases, and unchanged labelled classification fields."""

import unittest

import numpy as np

from pt_agent.metrics import compare, passes


class MetricTests(unittest.TestCase):
    def test_labelled_float_order_and_fields(self):
        reference = np.array([[1., 0.], [0., 1.], [1., 0.]])
        approx = np.array([[1., 0.], [1., 0.], [np.nan, np.nan]])
        labels = np.array([0, 1, 1])
        actual = compare(reference, approx, labels)
        top1 = 100.0*1/3
        exact_top1 = float(100*np.mean(reference.argmax(axis=1) == labels))
        self.assertEqual(actual, dict(agree=1, total=3, top1=top1, exact_top1=exact_top1,
                                      acc_drop_pp=exact_top1-top1))

    def test_unlabelled_boundary_and_divergence(self):
        reference = np.tile([1., 0.], (200, 1))
        approx = reference.copy(); approx[0] = [0, 1]
        result = compare(reference, approx, None)
        self.assertEqual(result, dict(agree=199, total=200, agreement_pct=99.5))
        self.assertTrue(passes(result, {'agreement_pct': 99.5}))
        approx[1] = np.nan
        self.assertFalse(passes(compare(reference, approx, None), {'agreement_pct': 99.5}))

    def test_lm_uses_last_real_position_and_two_conditions(self):
        reference = np.array([[[1., 0.], [0., 2.], [9., 0.]], [[1., 0.], [5., 0.], [0., 7.]]])
        approx = reference.copy(); approx[:, -1] = [0, 50]
        result = compare(dict(values=reference, task='causal_lm', lengths=[2, 1]), approx, None)
        self.assertEqual(result, dict(agree=2, total=2, agreement_pct=100., logit_rel_err=0.))
        self.assertTrue(passes(result, {'agreement_pct': 99., 'logit_rel_err': .01}))
        approx[0, 1, 1] = 2.1
        result = compare(dict(values=reference, task='causal_lm', lengths=[2, 1]), approx, [1, 0])
        self.assertEqual(result['true_top1'], 100.)
        self.assertFalse(passes(result, {'agreement_pct': 99., 'logit_rel_err': .01}))

    def test_regression_zero_and_shape_errors(self):
        self.assertEqual(compare(np.zeros((2, 1)), np.zeros((2, 1)), None)['relative_error'], 0.)
        self.assertTrue(passes(compare(np.ones((2, 1)), np.full((2, 1), 1.01), None), {'relative_error': .010000000000001}))
        with self.assertRaises(ValueError):
            compare(np.zeros((2, 1)), np.zeros((3, 1)), None)


if __name__ == '__main__':
    unittest.main()
