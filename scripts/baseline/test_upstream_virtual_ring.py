"""Full virtual-slot representation checks. Public vectors are NOT FHE evidence."""
import unittest,sys
import numpy as np
from upstream_adapters.virtual_ring import Context,VirtualExpr,Block,NT,proxy_type
from test_upstream_spatial import Vector as BaseVector

class Vector(BaseVector):
    def __neg__(self):return Vector(-self.data)

def expand(x):
    a=np.zeros(NT);p=x.context.period
    for index,block in x.values.items():
        v=np.asarray(block.value if block.public else block.value.data[:p])
        if not block.public:
            np.testing.assert_array_equal(block.value.data,np.tile(v,len(block.value.data)//p))
        a[index*p:(index+1)*p]=v
    return a

def input_expr(p,n=None,cls=VirtualExpr):
    n=n or p;c=Context(p)
    v=np.arange(1,p+1,dtype=float)/p
    return cls.input(c,Vector(np.tile(v,16384//p)),n),np.pad(v[:n],(0,NT-n))

class VirtualRingTests(unittest.TestCase):
    def test_full_ring_rotations_and_combinations(self):
        for p in (4,8,16,32,64,128,256):
            for step in (0,1,-1,p,-p,p+1,-p-1,32767,32768,65535,65536,-65537):
                x,a=input_expr(p,p-1);y=x.rotate(step)
                np.testing.assert_array_equal(expand(y),np.roll(a,-step))
                np.testing.assert_array_equal(expand(y.rotate(-step)),a)
                np.testing.assert_array_equal(expand(y+x),np.roll(a,-step)+a)
    def test_nonperiodic_far_tail_retained(self):
        x,a=input_expr(8);b=np.zeros(NT);b[[0,23,32768,65535]]=[.1,.2,.3,.4]
        y=x+b
        np.testing.assert_array_equal(expand(y),a+b)
        np.testing.assert_array_equal(expand(y.rotate(-1)*b),np.roll(a+b,1)*b)
        self.assertIn(8191,x.context.constants[0]["nonzero_blocks"])
        self.assertEqual(x.context.constants[0]["entries"],NT)
        for bad in (b[:-1],np.r_[b[:-1],np.nan]):
            with self.assertRaises(ValueError):x+bad
    def test_public_and_cipher_arithmetic(self):
        x,a=input_expr(8);b=np.zeros(NT);b[:8]=np.arange(8)/8
        for y,expected in ((x-b,a-b),(b-x,b-a),(x*b,a*b),(-x,-a)):
            np.testing.assert_array_equal(expand(y),expected)
        z=x*np.zeros(NT);self.assertEqual(z.values,{})
        y=z+b;np.testing.assert_array_equal(expand(y),b)
        result=y.export_prefix(5,Vector(np.zeros(16384)))
        np.testing.assert_array_equal(result.data,np.tile(np.r_[b[:5],np.zeros(3)],2048))
        self.assertTrue(x.context.events[-1]["encrypted_zero_argument_used"])
    def test_projection_is_explicit_and_late(self):
        x,a=input_expr(8);y=x.rotate(3)
        self.assertIn(8191,y.values)
        r=y.export_prefix(4,Vector(np.zeros(16384)))
        np.testing.assert_array_equal(r.data[:8],np.r_[np.roll(a,-3)[:4],np.zeros(4)])
        self.assertIn(8191,x.context.events[-1]["discarded_nonzero_blocks"])
        np.testing.assert_array_equal(expand(y.rotate(-3)),a)
    def test_bounds_and_context_rejected(self):
        with self.assertRaises(ValueError):Context(65536)
        x,_=input_expr(4);z,_=input_expr(4)
        with self.assertRaises(ValueError):x+z
        with self.assertRaises(ValueError):x.rotate(1.0)
        with self.assertRaises(ValueError):x+1
        with self.assertRaises(ValueError):x+np.ones(NT)
        c=Context(4,limit=1);v=VirtualExpr.input(c,Vector(np.ones(16384)),3)
        with self.assertRaisesRegex(ValueError,"operation budget"):v.rotate(1)
        with self.assertRaises(ValueError):VirtualExpr(c,{c.blocks:Block((1.,0.,0.,0.),1,True)})

class ActualVirtualHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
        cls.proxy=proxy_type(cls.helpers.hc.Expr)
    def test_original_linear_dense_full_ring(self):
        import torch
        from types import SimpleNamespace
        from upstream_adapters.periodic_ring import proxy_type as dense_type
        dense=dense_type(self.helpers.hc.Expr)
        for n in (1,2,3,5,8,9,15):
            for m in (1,2,3,4,7):
                p=1<<max(2,(max(n,m)-1).bit_length());x,a=input_expr(p,n,self.proxy)
                weight=torch.arange(m*n,dtype=torch.float64).reshape(m,n)/128-.25
                bias=torch.arange(m,dtype=torch.float64)/16
                linear=SimpleNamespace(weight=weight,bias=bias)
                y=self.helpers.HE_Linear(None,np.array([x],dtype=object),linear)[0]
                d=dense(Vector(a),NT,{"operations":[],"rotations":[]})
                expected=self.helpers.HE_Linear(None,np.array([d],dtype=object),linear)[0]
                with self.subTest(n=n,m=m):
                    np.testing.assert_allclose(expand(y),expected.value.data,atol=1e-12,rtol=1e-12)
    def test_zero_padding_linear_math(self):
        import torch
        from types import SimpleNamespace
        for n in (1,2,3,5,8,9,15):
            for m in (1,2,3,4,7):
                size=max(n,m);p=1<<max(2,(size-1).bit_length())
                x,a=input_expr(p,n,self.proxy)
                w=np.arange(m*n,dtype=float).reshape(m,n)/128-.25;b=np.arange(m)/16
                linear=SimpleNamespace(weight=torch.tensor(np.pad(w,((0,0),(0,size-n))),dtype=torch.float64),
                                       bias=torch.tensor(b,dtype=torch.float64))
                y=self.helpers.HE_Linear(None,np.array([x],dtype=object),linear)[0]
                with self.subTest(n=n,m=m):
                    np.testing.assert_allclose(expand(y)[:m],w@a[:n]+b,atol=1e-12,rtol=1e-12)

if __name__=="__main__":unittest.main()
