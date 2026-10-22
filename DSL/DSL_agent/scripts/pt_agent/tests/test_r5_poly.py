import math
import unittest

import numpy as np

from pt_agent.backends import CountBackend, MockBackend
from pt_agent.kernels import KernelError, act_cheb, cheb_eval


class PolyTests(unittest.TestCase):
    def test_random_coefficients(self):
        rng = np.random.default_rng(20261029)
        for d in (1, 2, 3, 7, 8, 15, 16, 31, 63, 127):
            with self.subTest(degree=d):
                coeffs = rng.normal(size=d+1)
                inputs = rng.uniform(-1, 1, 64)
                mock = MockBackend(64)
                mock.bind([inputs])
                value = cheb_eval(mock, mock.input(0), coeffs)
                mock.output([value])
                expected = np.polynomial.chebyshev.chebval(inputs, coeffs)
                self.assertLessEqual(float(np.max(np.abs(mock.outputs[0]-expected))), 1e-12)
                counter = CountBackend(64)
                result = cheb_eval(counter, counter.input(0), coeffs)
                counter.output([result])
                self.assertLessEqual(counter.stats()["max_depth"], math.ceil(math.log2(d+1))+1)

    def test_exact_depths(self):
        rng = np.random.default_rng(20261029)
        for d, depth in ((15, 5), (31, 6), (63, 7), (127, 8)):
            with self.subTest(degree=d):
                counter = CountBackend(64)
                value = cheb_eval(counter, counter.input(0), rng.normal(size=d+1))
                counter.output([value])
                self.assertEqual(counter.stats()["max_depth"], depth)

    def test_relu_counts_and_approximation(self):
        for degree, expected, depth in ((15, dict(mul_cp=13, add=20, mul_cc=7, sub=4), 6),
                                        (127, dict(mul_cp=121, add=145, mul_cc=24, sub=17), 9)):
            with self.subTest(degree=degree):
                counter = CountBackend(4096)
                value = act_cheb(counter, counter.input(0), "relu", degree, (-13.7, 27.2))
                counter.output([value])
                for op, count in expected.items():
                    self.assertEqual(counter.stats()["counts"][op], count)
                self.assertEqual(counter.stats()["max_depth"], depth)
                if degree == 127:
                    x = np.linspace(-13.7, 27.2, 4096)
                    mock = MockBackend(4096)
                    mock.bind([x])
                    value = act_cheb(mock, mock.input(0), "relu", degree, (-13.7, 27.2))
                    self.assertLessEqual(float(np.max(np.abs(value.value-np.maximum(x, 0)))), 0.05)

    def test_constant_polynomial_errors(self):
        for coeffs in ([2.0], [2.0, 0.0, 0.0]):
            with self.subTest(coeffs=coeffs), self.assertRaises(KernelError):
                counter = CountBackend(64)
                cheb_eval(counter, counter.input(0), coeffs)
