"""A1 §6.19: tuner alignment margin (u <= 0.80) and half logit-error limit (tune_rule v8-6.19)."""

from types import SimpleNamespace
import unittest

from pt_agent.domain_criteria import maximum_u
from pt_agent.margin import minimum_domain
from pt_agent.planners import transformer_tuner as tuner


def ctx(threshold):
    return SimpleNamespace(model_info=SimpleNamespace(threshold=threshold))


class AlignmentTests(unittest.TestCase):
    def test_log_point_reaches_the_target_position(self):
        lo, hi = 888.343842523836, 181205.05006163998
        domain = tuner.margin_domain(lo, hi, log_scale=True)
        for got, expected in zip(domain, (456.9644829633131, 352264.55546963104)):
            self.assertLess(abs(got-expected), 1e-9*expected)
        u = maximum_u([229225.78856814431], domain, fn='rsqrt', version='r8-v10')
        self.assertAlmostEqual(u, 0.871, places=3)
        self.assertAlmostEqual(maximum_u([lo, hi], domain, fn='rsqrt', version='r8-v10'), 0.80, places=12)

    def test_asymmetric_linear_point_widens_the_short_side(self):
        domain = tuner.margin_domain(-1.0, 10.0, log_scale=False)
        self.assertEqual(domain, (-2.375, 15.0))
        self.assertAlmostEqual(maximum_u([-1.0], domain), 0.842, places=3)
        self.assertAlmostEqual(maximum_u([10.0], domain), 0.424, places=3)

    def test_symmetric_point_keeps_m(self):
        self.assertEqual(tuner.margin_domain(-2.0, 2.0, log_scale=False), minimum_domain(-2.0, 2.0))

    def test_degenerate_range_keeps_m(self):
        self.assertEqual(tuner.margin_domain(3.0, 3.0, log_scale=True), minimum_domain(3.0, 3.0))
        self.assertEqual(tuner.margin_domain(3.0, 3.0, log_scale=False), minimum_domain(3.0, 3.0))

    def test_domain_contains_m(self):
        for lo, hi, log_scale in ((0.5, 12.0, True), (-13.0, 8.0, False), (2.0, 3.0, False), (1e-3, 4.0, True)):
            with self.subTest(lo=lo, hi=hi):
                m, d = minimum_domain(lo, hi), tuner.margin_domain(lo, hi, log_scale=log_scale)
                self.assertLessEqual(d[0], m[0])
                self.assertGreaterEqual(d[1], m[1])


class AccuracyTests(unittest.TestCase):
    def test_half_limit(self):
        causal = ctx(dict(agreement_pct=99.0, logit_rel_err=1e-2))
        classification = ctx(dict(agreement_pct=99.5))
        self.assertEqual(tuner.move_logit_limit(causal, 'v8-6.19'), 5e-3)
        self.assertEqual(tuner.move_logit_limit(classification, 'v8-6.19'), 5e-3)
        self.assertEqual(tuner.move_logit_limit(causal, 'v8-6.17'), 1e-2)
        self.assertTrue(tuner.move_accuracy_ok(dict(logit_rel_err=.0049), causal, 'v8-6.19'))
        self.assertFalse(tuner.move_accuracy_ok(dict(logit_rel_err=.0051), causal, 'v8-6.19'))
        self.assertTrue(tuner.move_accuracy_ok(dict(logit_rel_err=.0099), causal, 'v8-6.17'))

    def test_rule_selection(self):
        self.assertEqual(tuner.validate_tune_rule('r8-v10', None), 'v8-6.17')
        self.assertEqual(tuner.validate_tune_rule('r8-v10', 'v8-6.19'), 'v8-6.19')
        self.assertTrue(tuner.convergent_rule('v8-6.19'))
        with self.assertRaises(ValueError):
            tuner.validate_tune_rule('r8-v4', 'v8-6.19')


if __name__ == '__main__':
    unittest.main()
