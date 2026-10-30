"""U4 primitive checks, with static numeric inputs and independent float64 targets."""

import unittest
import numpy as np
import torch

from pt_agent.backends import MockBackend, CountBackend
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.dsl.ir import canonical
from pt_agent.dsl.typecheck import check
from pt_agent.lower import lower, pack_inputs, decode_outputs
from pt_agent.kernels.layout import (TokenLayout, ScoreLayout, FlatLayout,
                                     layout_record, layout_from_record, pack_sequence, unpack_sequence)
from pt_agent.kernels import transformer as tr


def mock(text, weights, arrays, *, exact=False):
    program = parse(text)
    result = check(program, {k:v.shape for k,v in weights.items()})
    if not result.ok:
        raise AssertionError(result.diags)
    backend = MockBackend(program.header.slots, batch=len(arrays[0]))
    backend.bind(pack_inputs(program, arrays))
    values, layouts = lower(program, weights, backend, act_mode='exact' if exact else 'program')
    return decode_outputs([v.value for v in values], layouts)[0], values, layouts


class LayoutTests(unittest.TestCase):
    def test_roundtrip_single_multi_and_scores(self):
        for layout in (TokenLayout(1024,3,13), TokenLayout(1024,3,601), ScoreLayout(1024,3,2,5)):
            with self.subTest(layout=layout):
                shape = (layout.L, layout.D) if isinstance(layout, TokenLayout) else (layout.H,layout.L,layout.Lk)
                x = np.arange(np.prod(shape), dtype=float).reshape(shape)/100
                packed = pack_sequence(x, layout)
                np.testing.assert_array_equal(unpack_sequence(packed,layout), x)
                self.assertEqual(layout_from_record(layout_record(layout)), layout)
                for v,(_,_,block) in zip(packed,layout.blocks):
                    self.assertFalse(v[block.size:].any())


class PrimitiveTests(unittest.TestCase):
    def close(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)

    def test_tall_route_and_rotations(self):
        for O in (3*256,4*256,5*256,4*256+3):
            I = 256
            x = (np.arange(I)%19-9)/100
            w = np.sin(np.arange(O*I).reshape(O,I))*.01
            bias = np.cos(np.arange(O))*.01
            b=MockBackend(1024); b.bind([np.pad(x,(0,1024-I))])
            y,layout=tr.linear(b,[b.input(0)],FlatLayout(1024,I),w,bias)
            ys=tr.values(y,layout)
            actual=np.concatenate([v.value[:block.size] for v,(_,_,block) in zip(ys,layout.blocks)]) if hasattr(layout,'blocks') else y.value[:O]
            self.close(actual,w@x+bias)
            for v,(_,_,block) in zip(ys, tr.layout_blocks(layout)):
                self.assertFalse(v.value[block.size:].any())
        rotations=[]
        for O in (512,1025,2049):
            b=CountBackend(1024)
            tr.tall_linear(b,b.input(0),FlatLayout(1024,16),np.ones((O,16)))
            rotations.append(b.stats()['counts']['rot'])
        self.assertEqual(rotations,[21]*3)

    def test_layernorm_single_and_multi(self):
        for D in (13,601):
            x=np.sin(np.arange(2*3*D).reshape(2,3,D))*.3
            w=np.cos(np.arange(D))*.1+1; bias=np.sin(np.arange(D))*.1
            text=f'''@fhe.program(slots=1024)
def layer(x: Cipher[1,3,{D}]):
 a=pack(x,layout="token")
 y=layernorm(a,w=W["w"],b=W["b"],eps=0.001,rsqrt_method="newton",rsqrt_domain=(0.01,0.2),rsqrt_deg0=7,rsqrt_iters=2)
 return y
'''
            exact,_,_=mock(text,dict(w=w,b=bias),[x],exact=True)
            target=torch.nn.functional.layer_norm(torch.tensor(x),(D,),torch.tensor(w),torch.tensor(bias),.001).numpy()
            self.close(exact,target)
            approximate,values,layouts=mock(text,dict(w=w,b=bias),[x])
            for v,(_,_,block) in zip(values,layouts[0].blocks):
                self.assertFalse(v.value[...,block.size:].any())
            self.assertTrue(np.isfinite(approximate).all())

    def test_qk_pv_mha_gqa_multi(self):
        for H,Hkv,dh in ((2,2,3),(4,2,3),(2,1,300)):
            L=3;D=H*dh;K=Hkv*dh
            q=np.sin(np.arange(2*L*D).reshape(2,L,D))*.1
            k=np.cos(np.arange(2*L*K).reshape(2,L,K))*.1
            v=np.sin(np.arange(2*L*K).reshape(2,L,K))*.2
            text=f'''@fhe.program(slots=1024)
def attention(q: Cipher[1,{L},{D}],k: Cipher[1,{L},{K}],v: Cipher[1,{L},{K}]):
 a=pack(q,layout="token")
 b=pack(k,layout="token")
 c=pack(v,layout="token")
 p=matmul(a,b,transpose_b=True,heads={H},kv_heads={Hkv},scale=0.5)
 y=matmul(p,c,transpose_b=False,heads={H},kv_heads={Hkv},scale=None)
 return y
'''
            got,_,_=mock(text,{},[q,k,v])
            qt=q.reshape(2,L,H,dh).transpose(0,2,1,3)
            kt=k.reshape(2,L,Hkv,dh).transpose(0,2,1,3).repeat(H//Hkv,axis=1)
            vt=v.reshape(2,L,Hkv,dh).transpose(0,2,1,3).repeat(H//Hkv,axis=1)
            target=((qt@kt.swapaxes(-1,-2))*.5)@vt
            self.close(got,target.transpose(0,2,1,3).reshape(2,L,D))

    def test_softmax_masks_and_multi(self):
        for D in (7,601):
            x=np.sin(np.arange(2*3*D).reshape(2,3,D))*.1
            mask=np.ones((2,D));mask[:,-2:]=0
            text=f'''@fhe.program(slots=1024)
def sm(x: Cipher[1,3,{D}],m: Mask[1,{D}]):
 a=pack(x,layout="token")
 am=pack(m,layout="flat")
 y=softmax(a,axis=-1,keep="all",mask=am,exp_method="remez",exp_domain=(-0.5,0.5),exp_deg=15,inv_method="newton",inv_domain=({D/3},{2*D}),inv_deg0=7,inv_iters=2)
 return y
'''
            got,_,_=mock(text,{},[x,mask],exact=True)
            exp=np.exp(x)*mask[:,None,:]
            self.close(got,exp/exp.sum(-1,keepdims=True))
            p=parse(text);self.assertEqual(canonical(parse(print_program(p))),canonical(p))

    def test_shuffle_and_encrypted_selection(self):
        x=np.sin(np.arange(2*3*6).reshape(2,3,6))
        mask=np.array([[0,0,1],[0,1,0]],dtype=float)
        text='''@fhe.program(slots=1024)
def sh(x: Cipher[1,3,6],m: Mask[1,3]):
 a=pack(x,layout="token")
 am=pack(m,layout="flat")
 h=shuffle(a,index=(3,4,5,0,1,2),sign=(-1,-1,-1,1,1,1))
 y=select(h,am)
 return y
'''
        got,_,_=mock(text,{},[x,mask])
        target=np.concatenate((-x[...,3:],x[...,:3]),axis=-1)
        self.close(got,np.sum(target*mask[...,None],axis=1,keepdims=True))


if __name__=='__main__':
    unittest.main()
