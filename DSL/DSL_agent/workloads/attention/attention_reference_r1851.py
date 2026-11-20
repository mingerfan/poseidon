"""Independent NumPy complex-RoPE and PyTorch matrix-RoPE Attention references."""
import math
import numpy as np
import torch
def numpy_forward(model,inputs,*,ideal=False,trace=False):
    w=model['constants'];x=np.asarray(inputs['x'],dtype=np.float64);ck=np.asarray(inputs['cached_k'],dtype=np.float64);cv=np.asarray(inputs['cached_v'],dtype=np.float64)
    assert x.shape==(2,4)and ck.shape==cv.shape==(2,1,2)
    assert all(np.isfinite(v).all()and np.max(np.abs(v))<=1 for v in [x,ck,cv])
    q,k,v=[(x@np.asarray(w[name],dtype=np.float64).T).reshape(2,2,2).transpose(1,0,2)for name in ['wq','wk','wv']]
    phase=np.exp(1j*np.asarray([1.,2.],dtype=np.float64))
    def rotary(value):
        z=(value[:,:,0]+1j*value[:,:,1])*phase
        return np.stack([z.real,z.imag],axis=-1)
    q,k=rotary(q),rotary(k);keys=np.concatenate([ck,k],axis=1);values=np.concatenate([cv,v],axis=1);scores=np.matmul(q,keys.transpose(0,2,1))*w['scale'];assert np.isfinite(scores).all()and np.min(scores)>=-1 and np.max(scores)<=1
    exponent=np.exp(scores)if ideal else np.polynomial.polynomial.polyval(scores,[1.,1.,.5,1/6,1/24]);exponent=exponent*np.asarray(w['causal_allow']);denominator=exponent.sum(axis=-1,keepdims=True);assert denominator.min()>=1 and denominator.max()<=4
    inverse=1/denominator if ideal else(1-(1-denominator/4)**8)/denominator
    probability=exponent*inverse;attended=np.matmul(probability,values);output=attended.transpose(1,0,2).reshape(2,4)@np.asarray(w['wo']).T
    result=dict(output=output,keys=keys,values=values)
    if trace:result['_trace']=dict(scores=scores,denominator=denominator,probabilities=probability,head_outputs=attended)
    return result
def torch_forward(model,inputs,*,ideal=False):
    w=model['constants'];x=torch.tensor(inputs['x'],dtype=torch.float64);ck=torch.tensor(inputs['cached_k'],dtype=torch.float64);cv=torch.tensor(inputs['cached_v'],dtype=torch.float64)
    with torch.inference_mode():
        q,k,v=[torch.nn.functional.linear(x,torch.tensor(w[name],dtype=torch.float64)).reshape(2,2,2).permute(1,0,2)for name in ['wq','wk','wv']]
        def rotary(value):
            rows=[]
            for t in range(2):
                c,s=math.cos(t+1),math.sin(t+1);rotation=torch.tensor([[c,s],[-s,c]],dtype=torch.float64);rows.append(value[:,t,:]@rotation)
            return torch.stack(rows,dim=1)
        q,k=rotary(q),rotary(k);keys=torch.cat([ck,k],dim=1);values=torch.cat([cv,v],dim=1);scores=torch.matmul(q,keys.transpose(-2,-1))*w['scale'];allow=torch.tensor(w['causal_allow'],dtype=torch.bool)
        if ideal:probability=torch.softmax(scores.masked_fill(~allow,-torch.inf),dim=-1)
        else:
            term=torch.ones_like(scores);exponent=torch.ones_like(scores)
            for degree in range(1,5):term=term*scores/degree;exponent=exponent+term
            masked=exponent*allow;denominator=masked.sum(dim=-1,keepdim=True);residual=1-denominator/4;inverse=(1+residual)/4
            for _ in range(2):residual=residual.square();inverse=inverse*(1+residual)
            probability=masked*inverse
        attended=torch.matmul(probability,values);merged=attended.permute(1,0,2).reshape(2,4);output=torch.nn.functional.linear(merged,torch.tensor(w['wo'],dtype=torch.float64))
        return{name:value.numpy()for name,value in dict(output=output,keys=keys,values=values).items()}
