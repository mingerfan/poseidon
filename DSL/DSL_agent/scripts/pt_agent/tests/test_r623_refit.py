"""A1 §6.23: widened points are refit to their original accuracy."""

import unittest

from pt_agent.planners import margin_refit as refit
from pt_agent.planners import transformer_tuner as tuner


class RefitTests(unittest.TestCase):
    def check(self, attrs, domain):
        setting, record = refit.refit(attrs, domain)
        new = dict(attrs, domain=domain, **setting)
        self.assertEqual(record['status'], 'within')
        self.assertLessEqual(refit.max_error(new), record['target']*(1+refit.SLACK))
        self.assertLessEqual(refit.max_error(dict(attrs, domain=domain)), float('inf'))
        return setting, record

    def test_remez_rsqrt_raises_degree(self):
        attrs = dict(fn='rsqrt', method='remez', deg=15, domain=(1.0, 10.0), fit='relative')
        setting, record = self.check(attrs, (0.5, 20.0))
        self.assertGreater(setting['deg'], 15)
        self.assertGreater(refit.max_error(dict(attrs, domain=(0.5, 20.0))), record['target'])

    def test_newton_inv_keeps_error(self):
        attrs = dict(fn='inv', method='newton', deg0=7, iters=2, domain=(0.1, 2.0))
        setting, _ = self.check(attrs, (0.05, 4.0))
        self.assertEqual(set(setting), {'deg0', 'iters'})

    def test_cheb_activation(self):
        attrs = dict(fn='silu', method='cheb', deg=15, domain=(-8.0, 8.0))
        setting, _ = self.check(attrs, (-16.0, 16.0))
        self.assertGreaterEqual(setting['deg'], 15)

    def test_degree_limits(self):
        self.assertEqual(refit.degrees(127), [7, 15, 31, 63, 127])
        self.assertEqual(refit.degrees(15), [7, 15, 31, 63])
        self.assertEqual(refit.degrees(15, 'cheb'), [7, 15, 31, 63, 127, 255])
        # Qwen line-29 silu: cheb 127 on the widened domain misses the original error; 255 meets it.
        attrs = dict(fn='silu', method='cheb', deg=127, domain=(-36.0, 34.5))
        setting, _ = self.check(attrs, (-47.6312, 45.1083))
        self.assertEqual(setting['deg'], 255)

    def test_same_domain_keeps_cheapest_within(self):
        attrs = dict(fn='rsqrt', method='remez', deg=31, domain=(1.0, 10.0))
        setting, record = self.check(attrs, (1.0, 10.0))
        self.assertLessEqual(setting['deg'], 31)

    def test_composite_is_kept(self):
        setting, record = refit.refit(dict(fn='relu', method='composite', degrees=(7, 7), eps=0.05,
                                           domain=(-1.0, 1.0)), (-2.0, 2.0))
        self.assertEqual(setting, {})
        self.assertEqual(record['status'], 'kept')

    def test_rule(self):
        self.assertIn('v8-6.23', tuner.REFIT_RULES)
        self.assertEqual(tuner.MARGIN_DOMAIN_FACTORS['v8-6.23'], 2.0)
        self.assertEqual(tuner.validate_tune_rule('r8-v10', 'v8-6.23'), 'v8-6.23')
        self.assertTrue(tuner.margin_rule('v8-6.23') and tuner.accuracy_rule('v8-6.23'))


if __name__ == '__main__':
    unittest.main()
