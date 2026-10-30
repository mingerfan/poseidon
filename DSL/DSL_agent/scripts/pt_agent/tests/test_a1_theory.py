"""A1 §1.8 inverse theory remains finite over the specified ratio range."""

import math
import unittest

from pt_agent.planners.transformer_rule import inverse_theory


class InverseTheoryTests(unittest.TestCase):
    def test_moderate_ratios_match_original_formulas_and_records(self):
        for ratio in (342.2334589845186,144.2694173659632):
            target=1e-3;q=(ratio-1)/(ratio+1)
            old_iters=0 if q<=target else math.ceil(math.log(math.log(target)/math.log(q),2))
            old_degree=math.ceil(math.acosh(1/target)/math.acosh((ratio+1)/(ratio-1)))-1
            result=inverse_theory((1.,ratio),target)
            self.assertEqual(result['optimal_constant_newton_iters'],old_iters)
            self.assertEqual(result['relative_minimax_polynomial_degree'],old_degree)
            self.assertIsInstance(result['optimal_constant_newton_iters'],int)
            self.assertIsInstance(result['relative_minimax_polynomial_degree'],int)
            self.assertFalse(result['approximate'])
            if ratio>300:self.assertEqual((old_iters,old_degree),(11,70))

    def test_extreme_finite_ratios_keep_finite_floating_diagnostics(self):
        for ratio in (9.4e264,1.7e308):
            result=inverse_theory((1.,ratio))
            self.assertTrue(result['approximate'])
            self.assertTrue(math.isfinite(result['optimal_constant_newton_iters']))
            self.assertTrue(math.isfinite(result['relative_minimax_polynomial_degree']))
            self.assertIsInstance(result['optimal_constant_newton_iters'],int)
            self.assertIsInstance(result['relative_minimax_polynomial_degree'],float)
            self.assertGreater(result['relative_minimax_polynomial_degree'],2**53)

    def test_null_cases_preserve_domain_and_reason(self):
        for domain,reason in (((0.,1.),'nonpositive lower bound'),
                              ((-1.,1.),'nonpositive lower bound'),
                              ((1.,float('inf')),'ratio not finite'),
                              ((1e-308,1e308),'ratio not finite'),
                              ((1.,1.),'ratio <= 1'),((2.,1.),'ratio <= 1')):
            with self.subTest(domain=domain):
                result=inverse_theory(domain)
                self.assertIsNone(result['optimal_constant_newton_iters'])
                self.assertIsNone(result['relative_minimax_polynomial_degree'])
                self.assertEqual(result['reason'],reason);self.assertEqual(result['domain'],domain)
                self.assertIn('log1p',result['newton_formula']);self.assertIn('log1p',result['polynomial_formula'])
                self.assertIn('与原式数学等价',result['note'])

    def test_zero_iteration_case(self):
        result=inverse_theory((1.,1.001))
        self.assertEqual(result['optimal_constant_newton_iters'],0)


if __name__=='__main__':unittest.main()
