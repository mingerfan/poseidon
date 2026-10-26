"""Pure U5 exchange and the §6 shared Newton arithmetic contract."""

import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.backends import CountBackend
from pt_agent.kernels.approx import (ApproximationError, act_remez, newton,
                                     remez, remez_report, tolerance, composite_coeffs,
                                     composite_report, composite, search_composite)
from pt_agent.kernels.poly import cheb_eval, cheb_fit
from pt_agent.kernels import approx


class RemezTests(unittest.TestCase):
    def test_smooth_fit_and_alternation(self):
        c = remez('exp', 7, (-1.0, 1.0))
        x = np.linspace(-1.0, 1.0, 65536)
        error = np.max(np.abs(cheb_eval(None, x, c)-np.exp(x)))
        old = np.max(np.abs(cheb_eval(None, x, cheb_fit('exp', 7, -1, 1))-np.exp(x)))
        report = remez_report('exp', 7, (-1.0, 1.0))
        self.assertLessEqual(error, old+report['tau'])
        errors = np.asarray(report['errors'])
        self.assertGreaterEqual(len(errors), 9)
        self.assertTrue(np.all(errors[:-1]*errors[1:] < 0))
        allowed = max(1e-6*abs(report['E']),report['tau'])
        self.assertTrue(np.all(np.abs(np.abs(errors)-abs(report['E'])) <= allowed))

    def test_float64_floor_for_silu63(self):
        report = remez_report('silu', 63, (-8.0, 8.0))
        self.assertEqual(report['criterion'],'float64-floor')
        self.assertLessEqual(report['gap'],report['tau'])
        self.assertGreater(report['tau'],1e-6*abs(report['E']))
        self.assertLessEqual(report['iterations'],50)

    def test_exchange_with_degenerate_initial_ripple(self):
        for fn in ('sigmoid','tanh'):
            c = remez(fn,7,(-8.0,8.0))
            report = remez_report(fn,7,(-8.0,8.0))
            self.assertTrue(any(row.get('exchange') == 'single-extremum' for row in report['history']))
            errors = np.asarray(report['errors'])
            self.assertGreaterEqual(len(errors),9)
            allowed = max(1e-6*abs(report['E']),report['tau'])
            self.assertTrue(np.all(np.abs(np.abs(errors)-abs(report['E'])) <= allowed))

    def test_composite_stage_domains_and_odd_coefficients(self):
        coefficients = composite_coeffs((3,3),0.2)
        reports = composite_report((3,3),0.2)
        for c in coefficients:
            np.testing.assert_array_equal(c[::2],0.0)
        error = reports[0]['max_error']
        self.assertEqual(reports[1]['domain'],(1-error,1+error))
        self.assertLessEqual(reports[1]['max_error'],error)
        x = np.linspace(-1,1,255)
        y = composite(None,x,(3,3),(-1,1),0.2)
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(composite(None,np.zeros(1),(3,3),(-1,1),0.2)[0],0.0)

    def test_composite_search_respects_depth_and_error(self):
        result = search_composite(8,0.2,0.05)
        self.assertLessEqual(result['depth'],8)
        self.assertLessEqual(result['error'],0.05)
        self.assertTrue(1 <= len(result['degrees']) <= 4)
        self.assertTrue(all(1 <= d <= 63 and d % 2 for d in result['degrees']))

    def test_odd_chebyshev_columns_recurrence_and_cosine(self):
        grid = np.linspace(-1,1,1025)
        for degree in (3,15,29,63):
            odd = np.arange(1,degree+1,2)
            recurrence = np.polynomial.chebyshev.chebvander(grid,degree)[:,1::2]
            cosine = np.cos(np.arccos(grid)[:,None]*odd[None,:])
            np.testing.assert_allclose(recurrence,cosine,atol=1e-12,rtol=1e-12)

    def test_odd29_converges_without_power_basis_conversion(self):
        with patch('numpy.polynomial.chebyshev.poly2cheb',side_effect=AssertionError('power conversion')):
            coefficients = composite_coeffs((29,),0.2)
        report = composite_report((29,),0.2)[0]
        self.assertLessEqual(report['iterations'],50)
        self.assertLessEqual(report['gap'],max(1e-6*abs(report['E']),report['tau']))
        np.testing.assert_array_equal(coefficients[0][::2],0.0)

    def test_search_excludes_failed_candidate_and_keeps_searching(self):
        approx._composite.cache_clear()
        search_composite.cache_clear()
        original = composite_coeffs
        def fitting(degrees,eps):
            if degrees == (3,):
                with patch('pt_agent.kernels.approx.MAX_ITERATIONS',1):
                    return original(degrees,eps)
            return original(degrees,eps)
        with patch('pt_agent.kernels.approx.composite_coeffs',side_effect=fitting):
            found = search_composite(8,0.2,0.05)
        self.assertLessEqual(found['error'],0.05)
        self.assertGreaterEqual(found['excluded'],1)
        failed = next(row for row in found['exclusions'] if row['degrees'] == (3,))
        self.assertEqual(failed['stage'],1)
        self.assertEqual(failed['iterations'],1)
        self.assertEqual(failed['gap'],failed['max_error']-failed['E'])
        self.assertNotEqual(found['degrees'],(3,))
        search_composite.cache_clear()

    def test_explicit_composite_nonconvergence_still_raises(self):
        approx._composite.cache_clear()
        with patch('pt_agent.kernels.approx.MAX_ITERATIONS',1):
            with self.assertRaises(ApproximationError):
                composite_coeffs((3,),0.2)

    def test_nonconvergence_is_explicit(self):
        with patch('pt_agent.kernels.approx.MAX_ITERATIONS', 1):
            with self.assertRaises(ApproximationError) as caught:
                remez('exp', 3, (-8.0, 8.0))
        self.assertEqual(caught.exception.iterations, 1)
        self.assertIn('max|err|=', str(caught.exception))
        self.assertIn('E=', str(caught.exception))

    def test_newton_depth_formula(self):
        for fn, increment in (('inv', 2), ('rsqrt', 3)):
            domain = (0.5, 2.0)
            for degree in (3, 7):
                coefficients = remez(fn, degree, domain)
                backend = CountBackend(8)
                initial = act_remez(backend, backend.input(0), fn, degree, domain,
                                    coefficients=coefficients)
                for iters in range(4):
                    b = CountBackend(8)
                    value = newton(b, b.input(0), fn, degree, iters, domain,
                                   coefficients=coefficients)
                    self.assertEqual(value.depth, initial.depth+increment*iters)
                    self.assertEqual(b.stats()['max_depth'], value.depth)

    def test_newton_numpy_operation_order(self):
        x = np.linspace(0.5, 2.0, 51)
        for fn in ('inv', 'rsqrt'):
            coefficients = remez(fn, 3, (0.5, 2.0))
            y = act_remez(None, x, fn, 3, (0.5, 2.0), coefficients=coefficients)
            if fn == 'rsqrt':
                x_half = x*0.5
            for iters in range(4):
                actual = newton(None, x, fn, 3, iters, (0.5, 2.0),
                                coefficients=coefficients)
                np.testing.assert_array_equal(actual, y)
                if fn == 'inv':
                    t = x*y
                    t = 2-t
                    y = y*t
                else:
                    t = y*y
                    t = x_half*t
                    t = 1.5-t
                    y = y*t


if __name__ == '__main__':
    unittest.main()
