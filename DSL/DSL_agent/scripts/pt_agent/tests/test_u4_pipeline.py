"""U4 DSL contracts, independent tensor execution and standalone packages."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

from pt_agent.backends import EmitBackend
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.dsl.ir import canonical
from pt_agent.dsl.typecheck import check
from pt_agent.export import export_package
from pt_agent.lower import lower
from pt_agent.tensor_sim import run_tensor
from pt_agent.tests.test_p2_export import package_modules,emit_pair
from pt_agent.tests.test_u4_kernels import mock


PROGRAM='''@fhe.program(slots=1024)
def token_fixture(x: Cipher[1,3,8],m: Mask[1,3]):
 a=pack(x,layout="token")
 am=pack(m,layout="flat")
 q=linear(a,w=W["w"],b=W["b"],algo="bsgs")
 n=layernorm(q,w=W["gamma"],b=W["beta"],eps=0.01,rsqrt_method="newton",rsqrt_domain=(0.01,0.2),rsqrt_deg0=7,rsqrt_iters=2)
 s=matmul(n,n,transpose_b=True,heads=2,kv_heads=2,scale=0.5)
 p=softmax(s,axis=-1,keep="causal",mask=am,exp_method="remez",exp_domain=(-5,5),exp_deg=15,inv_method="newton",inv_domain=(0.01,100),inv_deg0=7,inv_iters=2)
 h=matmul(p,n,transpose_b=False,heads=2,kv_heads=2,scale=None)
 last=select(h,am)
 o=linear(last,w=W["head"],algo="bsgs")
 return o
'''


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.weights=dict(w=np.eye(8)*.3,b=np.arange(8)*.001,gamma=np.ones(8),beta=np.zeros(8),
                          head=np.cos(np.arange(8*41)).reshape(41,8)*.01)
        self.arrays=[np.sin(np.arange(2*3*8).reshape(2,3,8))*.2,np.array([[1,1,0],[1,1,1]],dtype=float)]

    def test_tensor_mock_and_exact(self):
        p=parse(PROGRAM)
        for exact in (False,True):
            target,_,_=mock(PROGRAM,self.weights,self.arrays,exact=exact)
            actual=run_tensor(p,self.weights,dict(zip([v.name for v in p.params],self.arrays)),
                              act_mode='exact' if exact else 'program')
            np.testing.assert_allclose(actual,target,atol=1e-9,rtol=1e-9)

    def test_package_emit_constants_and_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=parse(PROGRAM)
            package=export_package(PROGRAM,p,self.weights,Path(tmp))
            left,right=emit_pair(PROGRAM,self.weights,package)
            self.assertEqual(left[0],right[0])
            self.assertEqual(set(left[1]),set(right[1]))
            for key in left[1]:np.testing.assert_array_equal(left[1][key],right[1][key])
            target,_,_=mock(PROGRAM,self.weights,self.arrays)
            with package_modules(package) as runtime:
                actual=runtime.run_numpy(self.weights,self.arrays)[0]
            np.testing.assert_array_equal(actual,target)

    def test_parser_roundtrip_and_invalid_contracts(self):
        p=parse(PROGRAM)
        self.assertEqual(canonical(p),canonical(parse(print_program(p))))
        for before,after in [('rsqrt_method="newton",',''),('exp_method="remez",',''),
                             ('rsqrt_domain=(0.01,0.2)','rsqrt_domain=(-1,1)'),
                             ('heads=2,kv_heads=2','heads=3,kv_heads=2')]:
            with self.subTest(change=before),self.assertRaises(Exception):parse(PROGRAM.replace(before,after))
        bad=PROGRAM.replace('select(h,am)','select(h,a)')
        self.assertFalse(check(parse(bad),{k:v.shape for k,v in self.weights.items()}).ok)


if __name__=='__main__':unittest.main()
