"""U4 §11.8 weighted exchange and end-to-end approximation fit fields."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.backends import EmitBackend
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.export import export_package
from pt_agent.kernels import approx
from pt_agent.kernels.poly import cheb_eval
from pt_agent.lower import lower
from pt_agent.planners import transformer_rule as rule
from pt_agent.tensor_sim import run_tensor
from pt_agent.tests._artifacts import RESULTS_DIR,require
from pt_agent.tests.test_p2_export import emit_pair,package_modules
from pt_agent.tests.test_u4_kernels import mock
from pt_agent.tests.test_u4_pipeline import PROGRAM


def error(fn,degree,domain,fit):
    x=np.unique(np.r_[np.linspace(*domain,65537),np.geomspace(*domain,65537)])
    c=approx.remez(fn,degree,domain,fit=fit)
    actual=cheb_eval(None,x*(2/(domain[1]-domain[0]))-sum(domain)/(domain[1]-domain[0]),c)
    return float(np.max(np.abs(actual/approx.function(fn,x)-1)))


class RelativeFitTests(unittest.TestCase):
    def test_inverse_ratio342_matches_relative_minimax_bound(self):
        domain=(1/342.,1.)
        for degree in (15,31,63):
            with self.subTest(degree=degree):
                measured=error('inv',degree,domain,'relative')
                theory=1/np.cosh((degree+1)*np.arccosh(343/341))
                self.assertLessEqual(abs(measured-theory),1e-6*theory+1e-12)
                self.assertLessEqual(measured,error('inv',degree,domain,'absolute'))

    def test_rsqrt_and_exp_relative_fits(self):
        for fn,domain,degree in (('rsqrt',(1/342.,1.),31),('exp',(.1,2.5),7)):
            with self.subTest(fn=fn):
                self.assertLessEqual(error(fn,degree,domain,'relative'),error(fn,degree,domain,'absolute')+1e-14)
        with self.assertRaises(Exception):approx.remez('silu',7,(-1,1),fit='relative')

    def test_rsqrt_linear_relative_minimax_closed_form(self):
        # For x in [1/4,1], equality at the endpoints gives b=-7a/4.
        # The sole interior extremum of sqrt(x)*(a*x+b) is x=7/12.
        # Equal ripples then give E=(M-m)/(M+m).
        m=3/4;maximum=(7/6)*np.sqrt(7/12)
        theory=(maximum-m)/(maximum+m)
        measured=error('rsqrt',1,(.25,1.),'relative')
        self.assertLessEqual(abs(measured-theory),1e-6*theory+1e-12)
        self.assertLessEqual(measured,error('rsqrt',1,(.25,1.),'absolute'))

    def test_all_candidates_include_fit_and_use_complete_selection_key(self):
        for fn,domain in (('inv',(1/342.,1.)),('rsqrt',(.1,2.)),('exp',(-2.,.3))):
            with self.subTest(fn=fn):
                report=rule.select_approximation(fn,domain,exp_ladder=True)
                candidates=report['candidates']
                self.assertEqual({c['fit'] for c in candidates},{'absolute','relative'})
                legal=[c for c in candidates if c['feasible']]
                chosen=min(legal,key=lambda c:(c['depth'],c['relative_error'],c['method'],
                    c.get('deg',c.get('deg0',0)),c.get('iters',0),c['fit']))
                self.assertEqual(report['selected'],{k:chosen[k] for k in ('method','deg','deg0','iters','fit') if k in chosen})
                self.assertEqual(report['selected_depth'],chosen['depth'])
                if fn=='exp':self.assertEqual({c['deg'] for c in candidates},{7,15,31,63})
                self.assertTrue(all(c.get('iters',0)<=5 and c.get('deg',c.get('deg0',0))<=63 for c in candidates))

    def test_role_fields_parse_roundtrip_and_invalid_fit(self):
        text=PROGRAM.replace('rsqrt_method=', 'rsqrt_fit="relative",rsqrt_method=')
        text=text.replace('exp_method=','exp_fit="relative",exp_method=').replace('inv_method=','inv_fit="relative",inv_method=')
        program=parse(text)
        restored=parse(print_program(program))
        for op in restored.ops:
            if op.kind=='layernorm':self.assertEqual(op.attrs['rsqrt_fit'],'relative')
            if op.kind=='softmax':self.assertEqual(op.attrs['exp_fit'],'relative');self.assertEqual(op.attrs['inv_fit'],'relative')
        with self.assertRaises(Exception):parse(text.replace('fit="relative"','fit="unknown"'))
        step=text.replace('mask=am,','mask=am,steps=1,step_inv_fit="relative",step_inv_method="remez",step_inv_domain=(0.1,1.5),step_inv_deg=31,')
        self.assertEqual(next(o for o in parse(step).ops if o.kind=='softmax').attrs['step_inv_fit'],'relative')

    def test_default_absolute_emit_and_arrays_equal(self):
        weights=dict(w=np.eye(8),b=np.zeros(8),gamma=np.ones(8),beta=np.zeros(8),head=np.ones((41,8)))
        text=PROGRAM.replace('rsqrt_method=','rsqrt_fit="absolute",rsqrt_method=').replace('exp_method=','exp_fit="absolute",exp_method=').replace('inv_method=','inv_fit="absolute",inv_method=')
        pairs=[]
        for source in (PROGRAM,text):
            backend=EmitBackend(1024);lower(parse(source),weights,backend);pairs.append(backend.render())
        self.assertEqual(pairs[0][0],pairs[1][0]);self.assertEqual(pairs[0][1].keys(),pairs[1][1].keys())
        for name in pairs[0][1]:np.testing.assert_array_equal(pairs[0][1][name],pairs[1][1][name])

    def test_relative_coefficients_shared_by_mock_tensor_and_package(self):
        text=PROGRAM.replace('rsqrt_method=','rsqrt_fit="relative",rsqrt_method=').replace('exp_method=','exp_fit="relative",exp_method=').replace('inv_method=','inv_fit="relative",inv_method=')
        text=text.replace('inv_domain=(0.01,100),inv_deg0=7','inv_domain=(0.1,10),inv_deg0=15')
        text=text.replace('mask=am,','mask=am,steps=1,step_inv_fit="relative",step_inv_method="remez",step_inv_domain=(0.1,1.5),step_inv_deg=31,')
        weights=dict(w=np.eye(8)*.3,b=np.arange(8)*.001,gamma=np.ones(8),beta=np.zeros(8),head=np.cos(np.arange(328)).reshape(41,8)*.01)
        arrays=[np.sin(np.arange(48).reshape(2,3,8))*.2,np.array([[1.,1.,0.],[1.,1.,1.]])]
        actual,_,_=mock(text,weights,arrays)
        tensor=run_tensor(parse(text),weights,dict(zip(('x','m'),arrays)))
        np.testing.assert_allclose(actual,tensor,atol=1e-9,rtol=1e-9)
        with tempfile.TemporaryDirectory() as directory:
            package=export_package(text,parse(text),weights,Path(directory))
            left,right=emit_pair(text,weights,package)
            self.assertEqual(left[0],right[0]);self.assertEqual(left[1].keys(),right[1].keys())
            for name in left[1]:np.testing.assert_array_equal(left[1][name],right[1][name])
            with package_modules(package) as runtime,patch('numpy.linalg.solve',side_effect=AssertionError('refit')):
                result=runtime.run_numpy(weights,arrays)[0]
            np.testing.assert_array_equal(result,actual)


SNAPSHOT=RESULTS_DIR/'u4a-remaining-20261027T031028306606/baseline-tree'


@require(SNAPSHOT/'DSL/DSL_agent/scripts/pt_agent/kernels/approx.py')
class AbsoluteCompatibilityTests(unittest.TestCase):
    def test_default_coefficients_equal_previous_code(self):
        from pt_agent.experiments.u2_multi_layout import baseline_package
        previous=baseline_package(SNAPSHOT)('kernels.approx')
        for fn,domain in (('exp',(-2.,1.)),('inv',(1/342.,1.)),('rsqrt',(.1,1.))):
            for degree in (7,15,31):
                with self.subTest(fn=fn,degree=degree):
                    before=previous.remez(fn,degree,domain)
                    np.testing.assert_array_equal(approx.remez(fn,degree,domain),before)
                    np.testing.assert_array_equal(approx.remez(fn,degree,domain,fit='absolute'),before)


if __name__=='__main__':unittest.main()
