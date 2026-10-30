"""U4 §11 public transforms and finite, depth-ordered parameter choices."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from pt_agent.backends import CountBackend,EmitBackend
from pt_agent.dsl.parser import parse
from pt_agent.export import export_package
from pt_agent.kernels import approx
from pt_agent.lower import lower
from pt_agent.planners.transformer_rule import select_approximation,softmax_adjustments
from pt_agent.tensor_sim import run_tensor
from pt_agent.transformer_tensor import run
from pt_agent.transformer_validation import exact_check
from pt_agent.tests.test_p2_export import emit_pair,package_modules
from pt_agent.tests.test_u4_kernels import mock
from pt_agent.tests.test_u4_pipeline import PROGRAM


class AdjustmentTests(unittest.TestCase):
    def fixture(self):
        weights=dict(w=np.eye(8)*.3,b=np.arange(8)*.001,gamma=np.ones(8),beta=np.zeros(8),
                     head=np.cos(np.arange(8*41)).reshape(41,8)*.01)
        arrays=dict(x=np.sin(np.arange(2*3*8).reshape(2,3,8))*.2,
                    m=np.array([[1,1,0],[1,1,1]],dtype=float))
        text=PROGRAM.replace('mask=am,','mask=am,shift=1.0,inv_scale=2.0,')
        text=text.replace('exp_deg=15','exp_deg=31').replace('inv_domain=(0.01,100)','inv_domain=(0.01,3)')
        text=text.replace('inv_deg0=7','inv_deg0=31')
        return text,weights,arrays

    def test_neutral_attributes_keep_emit_and_constants(self):
        signatures=[]
        for text in (PROGRAM,PROGRAM.replace('mask=am,','mask=am,shift=0.0,inv_scale=1.0,')):
            backend=EmitBackend(1024)
            _,weights,_=self.fixture()
            lower(parse(text),weights,backend)
            signatures.append(backend.render())
        self.assertEqual(signatures[0][0],signatures[1][0])
        self.assertEqual(signatures[0][1].keys(),signatures[1][1].keys())
        for key in signatures[0][1]:np.testing.assert_array_equal(signatures[0][1][key],signatures[1][1][key])

    def test_shift_scale_tensor_mock_and_exact_semantics(self):
        text,weights,arrays=self.fixture();program=parse(text)
        for exact in (False,True):
            actual=run_tensor(program,weights,arrays,act_mode='exact' if exact else 'program')
            target,_,_=mock(text,weights,list(arrays.values()),exact=exact)
            np.testing.assert_allclose(actual,target,atol=1e-9,rtol=1e-9)
        previous=run(parse(PROGRAM),weights,arrays,act_mode='exact')
        current=run(program,weights,arrays,act_mode='exact')
        np.testing.assert_allclose(current,previous,atol=1e-12,rtol=1e-12)

    def test_shift_scale_package_arithmetic(self):
        text,weights,arrays=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            package=export_package(text,parse(text),weights,Path(directory))
            left,right=emit_pair(text,weights,package)
            self.assertEqual(left[0],right[0])
            self.assertEqual(left[1].keys(),right[1].keys())
            for key in left[1]:np.testing.assert_array_equal(left[1][key],right[1][key])
            target,_,_=mock(text,weights,list(arrays.values()))
            with package_modules(package) as runtime:actual=runtime.run_numpy(weights,list(arrays.values()))[0]
            np.testing.assert_array_equal(actual,target)

    def test_single_and_multi_token_shift_scale_masks(self):
        for width in (7,601):
            x=np.sin(np.arange(2*3*width).reshape(2,3,width))*.1
            mask=np.ones((2,width));mask[:,-2:]=0
            text=f'''@fhe.program(slots=1024)
def sm(x: Cipher[1,3,{width}],m: Mask[1,{width}]):
 a=pack(x,layout="token")
 am=pack(m,layout="flat")
 y=softmax(a,mask=am,shift=0.1,inv_scale={float(width)},exp_method="remez",exp_domain=(-0.3,0.1),exp_deg=15,inv_method="newton",inv_domain=(0.3,2),inv_deg0=15,inv_iters=2)
 return y
'''
            for exact in (False,True):
                target,_,_=mock(text,{},[x,mask],exact=exact)
                actual=run(parse(text),{},dict(x=x,m=mask),act_mode='exact' if exact else 'program')
                np.testing.assert_allclose(actual,target,atol=1e-9,rtol=1e-9)
                if exact:
                    exp=np.exp(x)*mask[:,None,:]
                    np.testing.assert_allclose(actual,exp/exp.sum(-1,keepdims=True),atol=1e-12,rtol=1e-12)

    def test_exact_check_compares_transformed_points(self):
        text,weights,arrays=self.fixture();observations=[]
        expected=run(parse(PROGRAM),weights,arrays,act_mode='exact',
                     observer=lambda name,value,cfg:observations.append(value))
        ctx=SimpleNamespace(val_inputs=arrays,weights=weights,val_exact_logits=expected,
                            probe_act_inputs=observations,act_tensors=['norm','exp','inv'])
        self.assertTrue(exact_check(parse(text),ctx,engine='tensor')['ok'])

    def test_parameter_choice_minimizes_count_depth(self):
        for fn,domain in (('exp',(-2.,1.)),('inv',(.5,1.5)),('rsqrt',(.5,1.5))):
            with self.subTest(fn=fn):
                report=select_approximation(fn,domain)
                self.assertIsNotNone(report['selected'])
                feasible=[c for c in report['candidates'] if c['feasible']]
                self.assertEqual(report['selected_depth'],min(c['depth'] for c in feasible))
                settings=dict(report['selected'],fn=fn,domain=domain)
                backend=CountBackend(8)
                result=approx.activation(backend,backend.input(0),settings)
                self.assertEqual(result.depth,report['selected_depth'])
                if fn!='exp':
                    newton=[c for c in report['candidates'] if c['method']=='newton']
                    self.assertEqual({(c['deg0'],c['iters']) for c in newton},
                                     {(d,i) for d in (7,15,31) for i in range(1,6)})

    def test_public_transform_uses_calibration_only(self):
        transformed=softmax_adjustments(dict(min=-2,max=3),dict(min=.2,max=20))
        self.assertEqual(transformed['shift'],3)
        self.assertEqual(transformed['inv_scale'],20*np.exp(-3))
        self.assertEqual(transformed['exp_domain'],(-6.,1.5))
        self.assertEqual(transformed['inv_domain'],(.2/1.5/20,1.5))
        before=(20*1.5)/(.2/1.5)
        self.assertAlmostEqual(transformed['inv_domain'][1]/transformed['inv_domain'][0],before)

    def test_invalid_public_attributes_and_prompt_scope(self):
        from pt_agent.capabilities import render_capabilities
        for attrs in ('shift=1e309','inv_scale=0','inv_scale=-1','inv_scale=1e309'):
            with self.subTest(attrs=attrs),self.assertRaises(Exception):parse(PROGRAM.replace('mask=am,','mask=am,'+attrs+','))
        self.assertIn('inv_scale',render_capabilities('r8-v7'))
        for version in ('r8-v4','r8-v5','r8-v6'):self.assertNotIn('inv_scale',render_capabilities(version))


if __name__=='__main__':unittest.main()
