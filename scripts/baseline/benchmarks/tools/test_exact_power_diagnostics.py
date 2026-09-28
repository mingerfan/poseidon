import unittest
from fractions import Fraction as F
from exact_power_diagnostics import power_coefficients,estrin
class ExactPowerTests(unittest.TestCase):
    def test_all_basis_terms(self):
        for degree in range(129):
            c=[0.]*(degree+1);c[-1]=1.
            p=power_coefficients(c)
            for x in (F(-1),F(-2,3),F(0),F(1,5),F(1)):
                a,b=F(1),x
                for _ in range(2,degree+1):a,b=b,2*x*b-a
                expected=a if degree==0 else b
                self.assertEqual(sum(v*x**i for i,v in enumerate(p)),expected)
    def test_nonzero_tiny_terms_retained(self):
        self.assertEqual(power_coefficients([1e-38,0.,1e-22])[0],F(1e-38)-F(1e-22))
    def test_normal_and_weighted_evaluation(self):
        coeff=[.25,-.1,.125,1e-22,-.0625]
        for x in (-1.,-.5,0.,.25,1.):
            exact=sum(v*F(x)**i for i,v in enumerate(power_coefficients(coeff)))
            self.assertAlmostEqual(estrin(x,coeff,0.),float(exact),places=14)
            self.assertAlmostEqual(estrin(x,coeff,0.,weight=.75),float(exact)*.75,places=14)
    def test_size_bound(self):
        with self.assertRaises(ValueError):power_coefficients([1.]*130)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):power_coefficients([])
if __name__=="__main__":unittest.main()
