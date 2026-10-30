"""U4 §11.4–§11.6: shared inverse, exact identity and unchanged zero steps."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from pt_agent.backends import EmitBackend
from pt_agent.dsl.parser import parse
from pt_agent.export import export_package
from pt_agent.kernels import approx
from pt_agent.lower import lower
from pt_agent.tensor_sim import run_tensor
from pt_agent.transformer_validation import points
from pt_agent.planners import transformer_rule as rule
from pt_agent.tests.test_p2_export import package_modules,emit_pair
from pt_agent.tests.test_u4_kernels import mock
from pt_agent.tests.test_u4_pipeline import PROGRAM


def fixture(steps, width=7, causal=False):
    extra=(f',steps={steps},step_inv_method="newton",step_inv_domain=(0.01,1.5),'
           'step_inv_deg0=15,step_inv_iters=2') if steps else ''
    return f'''@fhe.program(slots=1024)
def sm(x: Cipher[1,3,{width}],m: Mask[1,{width}]):
 a=pack(x,layout="token")
 am=pack(m,layout="flat")
 y=softmax(a,keep="{'causal' if causal else 'all'}",mask=am,shift=0.7,inv_scale=3.0,exp_method="remez",exp_domain=(-2,2),exp_deg=15,inv_method="newton",inv_domain=(0.01,100),inv_deg0=7,inv_iters=2{extra})
 return y
'''


class SquaredSoftmaxTests(unittest.TestCase):
    def test_identity_random_scores_mask_causal_k0_to_6(self):
        rng=np.random.default_rng(20261026)
        for causal in (False,True):
            x=rng.normal(size=(2,3,7))*3
            mask=np.array([[1,0,1,1,0,1,0],[1,1,0,1,1,0,1]],dtype=float)
            expected=run_tensor(parse(fixture(0,causal=causal)),{},dict(x=x,m=mask),act_mode='exact')
            for steps in range(7):
                with self.subTest(causal=causal,steps=steps):
                    program=parse(fixture(steps,causal=causal))
                    tensor=run_tensor(program,{},dict(x=x,m=mask),act_mode='exact')
                    mock_values,_,_=mock(fixture(steps,causal=causal),{},[x,mask],exact=True)
                    np.testing.assert_allclose(tensor,expected,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(mock_values,expected,atol=1e-12,rtol=1e-12)

    def test_multiblock_identity_and_zero_tail(self):
        x=np.sin(np.arange(2*3*601)).reshape(2,3,601)*2
        mask=np.ones((2,601));mask[:,1::3]=0
        for steps in (1,5):
            tensor=run_tensor(parse(fixture(steps,601)),{},dict(x=x,m=mask),act_mode='exact')
            actual,values,layouts=mock(fixture(steps,601),{},[x,mask],exact=True)
            np.testing.assert_allclose(actual,tensor,atol=1e-12,rtol=1e-12)
            for value,(_,_,block) in zip(values,layouts[0].blocks):self.assertFalse(value.value[...,block.size:].any())

    def test_zero_steps_emit_text_constants(self):
        weights=dict(w=np.eye(8),b=np.zeros(8),gamma=np.ones(8),beta=np.zeros(8),head=np.ones((41,8)))
        emitted=[]
        for text in (PROGRAM,PROGRAM.replace('mask=am,','mask=am,steps=0,')):
            backend=EmitBackend(1024);lower(parse(text),weights,backend);emitted.append(backend.render())
        self.assertEqual(emitted[0][0],emitted[1][0])
        self.assertEqual(emitted[0][1].keys(),emitted[1][1].keys())
        for name in emitted[0][1]:np.testing.assert_array_equal(emitted[0][1][name],emitted[1][1][name])

    def test_analytical_bounds_cover_calibration_and_stages(self):
        x=np.sin(np.arange(2*3*7)).reshape(2,3,7)*3
        cfg=rule.squared_softmax_adjustments(dict(min=float(x.min()),max=float(x.max())),7)
        self.assertEqual(cfg['steps'],4)
        a,b=cfg['exp_domain'];self.assertLessEqual(b-a,1)
        t=(x-cfg['shift'])*(2.0**-cfg['steps'])
        self.assertGreaterEqual(t.min(),a);self.assertLessEqual(t.max(),b)
        e=np.exp(t);s=e.sum(-1,keepdims=True)/cfg['inv_scale']
        self.assertGreaterEqual(s.min(),cfg['inv_domain'][0]);self.assertLessEqual(s.max(),cfg['inv_domain'][1])
        p=e/e.sum(-1,keepdims=True)
        for _ in range(cfg['steps']):
            q=p*p;total=q.sum(-1,keepdims=True)
            self.assertGreaterEqual(total.min(),cfg['step_inv_domain'][0])
            self.assertLessEqual(total.max(),cfg['step_inv_domain'][1]);p=q/total

    def test_nonconvergent_candidates_record_context_without_retries(self):
        failure=approx.ApproximationError('test exchange',iterations=50,max_error=3,error_amplitude=2,history=[])
        rule._select_approximation.cache_clear()
        with patch.object(rule,'_relative_profile',side_effect=failure) as fit:
            report=rule.select_approximation('inv',(.02,1.5))
        self.assertEqual(fit.call_count,8)
        self.assertEqual(report['status'],'infeasible')
        self.assertEqual(len(report['candidates']),38)
        for candidate in report['candidates']:
            self.assertFalse(candidate['feasible']);self.assertEqual(candidate['reason'],'nonconvergent')
            self.assertIn(candidate.get('deg',candidate.get('deg0')),(7,15,31,63))
            self.assertEqual(candidate['iterations'],50)
            self.assertEqual(candidate['max_error'],3);self.assertEqual(candidate['abs_E'],2)
        rule._select_approximation.cache_clear()

    def test_shared_step_point_updates_and_package(self):
        text=PROGRAM.replace('mask=am,','mask=am,steps=2,step_inv_method="newton",step_inv_domain=(0.1,1.5),step_inv_deg0=15,step_inv_iters=2,')
        text=text.replace('inv_domain=(0.01,100),inv_deg0=7','inv_domain=(0.1,10),inv_deg0=15')
        weights=dict(w=np.eye(8)*.3,b=np.arange(8)*.001,gamma=np.ones(8),beta=np.zeros(8),
                     head=np.cos(np.arange(8*41)).reshape(41,8)*.01)
        arrays=[np.sin(np.arange(2*3*8).reshape(2,3,8))*.2,np.array([[1,1,0],[1,1,1]],dtype=float)]
        program=parse(text);descriptors=points(program)
        self.assertEqual([p.role for p in descriptors],['rsqrt','exp','inv','step_inv[0]','step_inv[1]'])
        descriptors[-1].update(dict(iters=3))
        self.assertEqual(points(program)[-2].attrs['iters'],3)
        program=parse(text)
        with tempfile.TemporaryDirectory() as temporary:
            package=export_package(text,program,weights,Path(temporary))
            left,right=emit_pair(text,weights,package)
            self.assertEqual(left[0],right[0])
            self.assertEqual(left[1].keys(),right[1].keys())
            for name in left[1]:np.testing.assert_array_equal(left[1][name],right[1][name])
            target,_,_=mock(text,weights,arrays)
            with package_modules(package) as runtime:actual=runtime.run_numpy(weights,arrays)[0]
            np.testing.assert_array_equal(actual,target)
            tensor=run_tensor(program,weights,dict(zip(('x','m'),arrays)))
            np.testing.assert_allclose(actual,tensor,atol=1e-9,rtol=1e-9)


if __name__=='__main__':unittest.main()
