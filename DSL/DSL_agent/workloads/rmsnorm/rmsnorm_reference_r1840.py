"""Independent original and declared RMSNorm references; no FHE execution."""
import math
import numpy as np
import torch
def numpy_forward(model,x,*,ideal=False):
    node=next(n for n in model['nodes']if n['op']=='rms_norm');p=node['attrs']['approximation'];x=np.asarray(x,dtype=np.float64)
    variance=np.sum(np.square(x),axis=-1,keepdims=True)/x.shape[-1]+node['attrs']['eps']
    assert np.all(variance>=p['domain'][0])and np.all(variance<=p['domain'][1])
    inverse=1/np.sqrt(variance)if ideal else np.full_like(variance,1/math.sqrt(p['domain'][1]))
    if not ideal:
        for _ in range(p['iterations']):inverse=.5*inverse*(3-variance*np.square(inverse))
    return x*inverse*np.asarray(model['constants'][node['inputs'][1]],dtype=np.float64)
def torch_forward(model,x,*,ideal=False):
    node=next(n for n in model['nodes']if n['op']=='rms_norm');p=node['attrs']['approximation'];x=torch.as_tensor(x,dtype=torch.float64)
    with torch.inference_mode():
        variance=x.square().mean(dim=-1,keepdim=True)+node['attrs']['eps']
        assert torch.all(variance>=p['domain'][0])and torch.all(variance<=p['domain'][1])
        inverse=variance.rsqrt()if ideal else torch.ones_like(variance)/math.sqrt(p['domain'][1])
        if not ideal:
            for _ in range(p['iterations']):inverse=inverse+inverse*(1-variance*inverse*inverse)*.5
        return(x*inverse*torch.tensor(model['constants'][node['inputs'][1]],dtype=torch.float64)).numpy()
