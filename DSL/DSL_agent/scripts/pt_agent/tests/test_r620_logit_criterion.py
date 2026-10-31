"""A1 §6.20: unlabelled classification also bounds the logit relative error."""

import unittest

import numpy as np

from pt_agent.metrics import compare, passes
from pt_agent.models.info import metric_for
from pt_agent.transformer_validation import add_required_logit_error


class MetricVersionTests(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(metric_for('classification', has_labels=False)[1], {'agreement_pct': 99.5})
        self.assertEqual(metric_for('classification', has_labels=False, version='a1-6.20')[1],
                         {'agreement_pct': 99.5, 'logit_rel_err': 1e-2})
        self.assertEqual(metric_for('classification', has_labels=True, version='a1-6.20')[1], {'acc_drop_pp': 0.5})
        self.assertEqual(metric_for('causal_lm', has_labels=False, version='a1-6.20')[1],
                         {'agreement_pct': 99.0, 'logit_rel_err': 1e-2})
        with self.assertRaises(ValueError):
            metric_for('classification', has_labels=False, version='other')


class RequiredErrorTests(unittest.TestCase):
    exact = np.array([[2.0, -1.0], [0.5, 4.0]])

    def measure(self, approx, threshold):
        return add_required_logit_error(compare(self.exact, approx, None), threshold, self.exact, approx)

    def test_old_threshold_keeps_fields(self):
        approx = self.exact+.01
        self.assertEqual(self.measure(approx, {'agreement_pct': 99.5}), dict(agree=2, total=2, agreement_pct=100.0))

    def test_new_threshold_rejects_agreeing_but_inaccurate_outputs(self):
        threshold = metric_for('classification', has_labels=False, version='a1-6.20')[1]
        close = self.measure(self.exact+.02, threshold)
        self.assertAlmostEqual(close['logit_rel_err'], .02/4.0)
        self.assertTrue(passes(close, threshold))
        far = self.measure(self.exact+np.array([[.3, 0.], [0., .3]]), threshold)
        self.assertEqual(far['agree'], 2)
        self.assertAlmostEqual(far['logit_rel_err'], .3/4.0)
        self.assertFalse(passes(far, threshold))

    def test_nonfinite_output_fails(self):
        threshold = metric_for('classification', has_labels=False, version='a1-6.20')[1]
        approx = self.exact.copy()
        approx[1, 0] = np.nan
        measured = self.measure(approx, threshold)
        self.assertIsNone(measured['logit_rel_err'])
        self.assertFalse(passes(measured, threshold))


if __name__ == '__main__':
    unittest.main()
