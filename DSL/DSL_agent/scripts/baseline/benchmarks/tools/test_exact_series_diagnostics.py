import unittest
import numpy as np
from fractions import Fraction
from exact_series_diagnostics import multiply_argument,parity_series,finite_literals
class ExactSeriesTests(unittest.TestCase):
    def test_argument_fusion_all_degrees(self):
        x=np.linspace(-1,1,41)
        for degree in range(128):
            c=[0.]*degree+[.125]
            fused=finite_literals(multiply_argument(c,.25))
            np.testing.assert_allclose(np.polynomial.chebyshev.chebval(x,fused),
                x*(np.polynomial.chebyshev.chebval(x,c)+.25),atol=1e-13,rtol=1e-12)
    def test_parity_with_mixed_sign_and_tiny_terms(self):
        x=np.linspace(-1,1,41)
        rng=np.random.default_rng(42)
        for degree in (1,2,3,7,15,27,95,128):
            c=rng.uniform(-.1,.1,degree+1).tolist();c[-1]=1e-38
            even,odd=parity_series(c);y=2*x*x-1
            result=np.polynomial.chebyshev.chebval(y,finite_literals(even))
            if odd:result+=x*np.polynomial.chebyshev.chebval(y,finite_literals(odd))
            np.testing.assert_allclose(result,np.polynomial.chebyshev.chebval(x,c),atol=2e-12,rtol=2e-12)
            self.assertTrue((even if degree%2==0 else odd)[-1])
    def test_zero_constant_and_t0_special_case(self):
        self.assertEqual(multiply_argument([2.],3.),[0,5])
        self.assertEqual(multiply_argument([0.,2.]),[1,0,1])
        self.assertEqual(parity_series([1.,0.,2.,0.,3.]),([1,2,3],[0,0]))
    def test_no_silent_underflow_or_limit_relaxation(self):
        for values in ([float("nan")],[],[1.]*130):
            with self.assertRaises(ValueError):multiply_argument(values)
        with self.assertRaises(ValueError):finite_literals([Fraction(1,10**400)])
        with self.assertRaises(ValueError):finite_literals([Fraction(1025)])
if __name__=="__main__":unittest.main()
