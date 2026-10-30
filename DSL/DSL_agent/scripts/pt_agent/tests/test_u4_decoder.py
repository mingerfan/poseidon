"""U4b decoder primitives: independent float64 targets in single/multiple blocks."""

import unittest
import numpy as np
import torch

from pt_agent.dsl.parser import parse
from pt_agent.tensor_sim import run_tensor
from pt_agent.tests.test_u4_kernels import mock


class DecoderPrimitiveTests(unittest.TestCase):
    def compare(self,text,weights,arrays,target):
        program=parse(text)
        for exact in (False,True):
            actual,values,layouts=mock(text,weights,list(arrays.values()),exact=exact)
            tensor=run_tensor(program,weights,arrays,act_mode='exact' if exact else 'program')
            np.testing.assert_allclose(actual,tensor,atol=1e-9,rtol=1e-9)
            if exact:np.testing.assert_allclose(actual,target,atol=1e-9,rtol=1e-9)
            for value,(_,_,block) in zip(values,layouts[0].blocks):
                self.assertFalse(value.value[...,block.size:].any())

    def test_rmsnorm_single_and_multiple_blocks(self):
        for width in (13,601):
            with self.subTest(width=width):
                x=np.sin(np.arange(6*width).reshape(2,3,width))*.3
                gamma=1+np.cos(np.arange(width))*.1
                text=f'''@fhe.program(slots=1024)
def rms(x: Cipher[1,3,{width}]):
 a=pack(x,layout="token")
 y=rmsnorm(a,w=W["gamma"],eps=0.001,rsqrt_method="newton",rsqrt_domain=(0.01,0.2),rsqrt_deg0=7,rsqrt_iters=2)
 return y
'''
                tx=torch.tensor(x);target=(tx*torch.rsqrt((tx*tx).mean(-1,keepdim=True)+.001)*torch.tensor(gamma)).numpy()
                self.compare(text,dict(gamma=gamma),dict(x=x),target)

    def test_rope_per_head_static_shuffle_and_position_constants(self):
        for width in (12,600):
            with self.subTest(width=width):
                head=width//2;half=head//2
                index=tuple(v for start in (0,head) for v in (*range(start+half,start+head),*range(start,start+half)))
                sign=tuple(([-1]*half+[1]*half)*2)
                x=np.cos(np.arange(6*width).reshape(2,3,width))*.2
                angle=np.arange(3)[:,None]*np.linspace(.01,.5,width)[None,:]
                c,s=np.cos(angle),np.sin(angle)
                text=f'''@fhe.program(slots=1024)
def rope(x: Cipher[1,3,{width}]):
 a=pack(x,layout="token")
 r=shuffle(a,index={index!r},sign={sign!r})
 u=mul(a,W["cos"])
 v=mul(r,W["sin"])
 y=add(u,v)
 return y
'''
                tx=torch.tensor(x);heads=tx.reshape(2,3,2,head)
                rotated=torch.cat((-heads[...,half:],heads[...,:half]),-1).reshape_as(tx)
                target=(tx*torch.tensor(c)+rotated*torch.tensor(s)).numpy()
                self.compare(text,dict(cos=c,sin=s),dict(x=x),target)

    def test_swiglu_single_and_multiple_blocks(self):
        for width in (13,601):
            with self.subTest(width=width):
                x=np.sin(np.arange(6*width).reshape(2,3,width))*.2
                y=np.cos(np.arange(6*width).reshape(2,3,width))*.3
                text=f'''@fhe.program(slots=1024)
def swiglu(x: Cipher[1,3,{width}],y: Cipher[1,3,{width}]):
 a=pack(x,layout="token")
 b=pack(y,layout="token")
 s=act(a,fn="silu",method="cheb",deg=127,domain=(-0.4,0.4))
 z=mul(s,b)
 return z
'''
                self.compare(text,{},dict(x=x,y=y),(torch.nn.functional.silu(torch.tensor(x))*torch.tensor(y)).numpy())

    def test_gqa_causal_and_padding_single_and_multiple_blocks(self):
        for heads,kv_heads,dh in ((4,2,3),(2,1,300)):
            with self.subTest(heads=heads,dh=dh):
                width=heads*dh;kv_width=kv_heads*dh
                q=np.sin(np.arange(6*width).reshape(2,3,width))*.1
                k=np.cos(np.arange(6*kv_width).reshape(2,3,kv_width))*.1
                v=np.sin(np.arange(6*kv_width).reshape(2,3,kv_width))*.2
                mask=np.array([[1.,1.,0.],[1.,1.,1.]])
                text=f'''@fhe.program(slots=1024)
def gqa(q: Cipher[1,3,{width}],k: Cipher[1,3,{kv_width}],v: Cipher[1,3,{kv_width}],m: Mask[1,3]):
 a=pack(q,layout="token")
 b=pack(k,layout="token")
 c=pack(v,layout="token")
 am=pack(m,layout="flat")
 s=matmul(a,b,transpose_b=True,heads={heads},kv_heads={kv_heads},scale=0.5)
 p=softmax(s,keep="causal",mask=am,exp_method="remez",exp_domain=(-2,2),exp_deg=15,inv_method="newton",inv_domain=(0.3,4),inv_deg0=7,inv_iters=2)
 y=matmul(p,c,transpose_b=False,heads={heads},kv_heads={kv_heads},scale=None)
 return y
'''
                tq=torch.tensor(q).reshape(2,3,heads,dh).transpose(1,2)
                tk=torch.tensor(k).reshape(2,3,kv_heads,dh).transpose(1,2).repeat_interleave(heads//kv_heads,dim=1)
                tv=torch.tensor(v).reshape(2,3,kv_heads,dh).transpose(1,2).repeat_interleave(heads//kv_heads,dim=1)
                scores=(tq@tk.transpose(-1,-2))*.5
                keep=torch.ones((3,3),dtype=torch.bool).tril()[None,None]&torch.tensor(mask).bool()[:,None,None,:]
                target=(torch.softmax(scores.masked_fill(~keep,-torch.inf),-1)@tv).transpose(1,2).reshape(2,3,width).numpy()
                self.compare(text,{},dict(q=q,k=k,v=v,m=mask),target)

    def test_encrypted_last_selection_then_single_tall_head(self):
        width=16;outputs=1201
        x=np.sin(np.arange(96).reshape(2,3,width))*.2
        sel=np.array([[0.,0.,1.],[0.,1.,0.]])
        weight=np.cos(np.arange(outputs*width).reshape(outputs,width))*.01
        bias=np.sin(np.arange(outputs))*.01
        text=f'''@fhe.program(slots=1024)
def lm(x: Cipher[1,3,{width}],s: Mask[1,3]):
 a=pack(x,layout="token")
 am=pack(s,layout="flat")
 last=select(a,am)
 y=linear(last,w=W["w"],b=W["b"],algo="bsgs")
 return y
'''
        all_positions=torch.nn.functional.linear(torch.tensor(x),torch.tensor(weight),torch.tensor(bias))
        expected=torch.stack((all_positions[0,2],all_positions[1,1]))[:,None,:].numpy()
        self.compare(text,dict(w=weight,b=bias),dict(x=x,s=sel),expected)


if __name__=='__main__':unittest.main()
