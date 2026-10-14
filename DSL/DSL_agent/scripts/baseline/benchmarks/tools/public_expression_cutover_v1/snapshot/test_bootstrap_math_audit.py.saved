import cmath
import math
import unittest
from bootstrap_math_audit import fft,encode_math,decode_math,clenshaw,chebyshev

class BootstrapMathAuditTests(unittest.TestCase):
    def test_fft_against_dft_definition(self):
        values=[complex(i/8,(-1)**i/10) for i in range(8)]
        expected=[sum(v*cmath.exp(-2j*math.pi*k*j/8) for j,v in enumerate(values)) for k in range(8)]
        self.assertLess(max(abs(x-y) for x,y in zip(fft(values),expected)),1e-13)

    def test_embedding_against_polynomial_at_odd_roots(self):
        values=[.125,-.25,.375,-.5,.1,.2,-.3,.4]
        coefficients=encode_math(values);n=len(coefficients);pos=1
        for expected in values:
            z=cmath.exp(1j*math.pi*pos/n)
            self.assertLess(abs(sum(a*z**j for j,a in enumerate(coefficients))-expected),1e-13)
            pos=5*pos%(2*n)

    def test_roundtrip_complex_and_constant(self):
        values=[complex(i/17,(-1)**i/5) for i in range(8)]
        self.assertLess(max(abs(x-y) for x,y in zip(values,decode_math(encode_math(values)))),1e-13)
        c=encode_math([.5]*8)
        self.assertLess(abs(c[0]-.5),1e-13)
        self.assertLess(max(abs(x) for x in c[1:]),1e-13)

    def test_chebyshev_against_trigonometric_definition(self):
        coeff=[.1,-.5,.03,.02,0.,.001]
        for x in (-1.,-.75,-.1,0.,.1,.75,1.):
            expected=sum(c*math.cos(k*math.acos(x)) for k,c in enumerate(coeff))
            self.assertAlmostEqual(clenshaw(coeff,x),expected,places=14)
            self.assertAlmostEqual(chebyshev(32,x),math.cos(32*math.acos(x)),places=13)

    def test_reject_invalid_fft_length(self):
        for values in ([],[1,2,3]):
            with self.assertRaises(ValueError):fft(values)
