"""U4 §11 diagnostic hooks and the fixed-order Newton error model."""

import unittest
import numpy as np

from pt_agent.dsl.parser import parse
from pt_agent.transformer_tensor import run,polynomial
from pt_agent.experiments.u4_point_diagnosis import distribution,errors,newton_error_model,newton_profile
from pt_agent.tests.test_u4_pipeline import PROGRAM


class PointDiagnosisTests(unittest.TestCase):
    def fixture(self):
        weights=dict(w=np.eye(8)*.3,b=np.arange(8)*.001,gamma=np.ones(8),beta=np.zeros(8),
                     head=np.cos(np.arange(8*41)).reshape(41,8)*.01)
        arrays=dict(x=np.sin(np.arange(2*3*8).reshape(2,3,8))*.2,
                    m=np.array([[1,1,0],[1,1,1]],dtype=float))
        return parse(PROGRAM),weights,arrays

    def test_hooks_preserve_program_and_exact_outputs(self):
        program,weights,arrays=self.fixture()
        for exact in (False,True):
            points=[];steps=[]
            def evaluate(op,role,value,attrs,mask):
                points.append((op.id,role))
                return polynomial(value,attrs,exact)
            expected=run(program,weights,arrays,act_mode='exact' if exact else 'program')
            actual=run(program,weights,arrays,point_evaluator=evaluate,
                       step_observer=lambda op,value:steps.append(op.id))
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual([role for _,role in points],['rsqrt','exp','inv'])
            self.assertEqual(steps,[op.id for op in program.ops])

    def test_inverse_residual_squaring_and_failure_region(self):
        result=newton_error_model('inv',np.array([.8,1.2]))
        self.assertEqual(result['required_iters'],3)
        np.testing.assert_allclose(result['history'],[.2,.04,.0016,.00000256],atol=2e-16,rtol=0)
        failed=newton_error_model('inv',np.array([-1.,2.1]))
        self.assertFalse(failed['converges'])
        self.assertIsNone(failed['required_iters'])

    def test_rsqrt_relative_model_and_sign(self):
        z=np.array([.8,1.2])
        result=newton_error_model('rsqrt',z)
        measured=[]
        for _ in result['history']:
            measured.append(float(np.max(np.abs(1-z))))
            z=z*(1.5-.5*z*z)
        np.testing.assert_array_equal(result['history'],measured)
        self.assertFalse(newton_error_model('rsqrt',np.array([-1.]))['converges'])

    def test_actual_error_and_nonfinite_distribution(self):
        measured=distribution(np.array([0.,2.,4.,np.nan]),(1.,3.))
        self.assertEqual(measured['nonfinite'],1)
        self.assertEqual(measured['outside'],3)
        self.assertEqual(measured['quantiles']['p50'],2.)
        local=errors(np.array([1.,0.]),np.array([0.,0.]))
        self.assertEqual(local['zero_reference_nonzero'],1)
        self.assertTrue(np.isinf(local['max_rel']))

    def test_domain_profile_matches_u5_initialization(self):
        from pt_agent.kernels import approx
        from pt_agent.kernels.poly import cheb_eval
        profile=newton_profile('inv',7,(.5,1.5))
        self.assertTrue(profile['converges'])
        for endpoint in profile['endpoints']:
            x=endpoint['x']
            c=approx.remez('inv',7,(.5,1.5))
            y=cheb_eval(None,np.array([x*2-2]),c)[0]
            self.assertEqual(endpoint['y0'],y)
        self.assertEqual(profile['required_depth'],profile['initial_depth']+2*profile['required_iters'])


if __name__=='__main__':unittest.main()
