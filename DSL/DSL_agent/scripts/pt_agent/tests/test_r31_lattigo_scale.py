"""Source-level scale rule examples; no model or crypto execution."""
from fractions import Fraction
import unittest

from pt_agent.experiments.r31_lattigo_scale_diagnostic import addition_factors,bootstrap_normalization_factor


class ScaleRuleChecks(unittest.TestCase):
    def test_integer_truncation_and_max_scale_weight_both_operand_orders(self):
        left,right,out=addition_factors(Fraction(101),Fraction(100))
        self.assertEqual((left,right,out),(Fraction(1),Fraction(100,101),Fraction(101)))
        self.assertEqual(addition_factors(Fraction(100),Fraction(101)),(right,left,out))
        self.assertEqual(addition_factors(Fraction(100),Fraction(100)),(Fraction(1),Fraction(1),Fraction(100)))
        self.assertEqual(addition_factors(Fraction(250),Fraction(100)),(Fraction(1),Fraction(4,5),Fraction(250)))

    def test_modup_metadata_ratio_and_coefficient_integer_have_distinct_effects(self):
        q0=36028797019488257;default=Fraction(2**43)
        self.assertEqual(bootstrap_normalization_factor(default,q0),(Fraction(1),16,32))
        changed=default*Fraction(101,100)
        self.assertEqual(bootstrap_normalization_factor(changed,q0),(Fraction(101,100),16,32))


if __name__=='__main__':unittest.main()
