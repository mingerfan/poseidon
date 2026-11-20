"""B-owned whole square-MLP references; original public learned values preserved."""
import numpy as np
import torch
from torch.nn import functional as F

def numpy_forward(model,x):
    assert [n['op']for n in model['nodes']]==['linear','square','linear']
    first,last=model['nodes'][0],model['nodes'][2];values=model['constants']
    w0,b0=(np.asarray(values[name],dtype=np.float64)for name in first['inputs'][1:])
    w1,b1=(np.asarray(values[name],dtype=np.float64)for name in last['inputs'][1:])
    hidden=np.asarray(x,dtype=np.float64)@w0.T+b0
    return(hidden*hidden)@w1.T+b1

def torch_forward(model,x):
    first,last=model['nodes'][0],model['nodes'][2];values=model['constants']
    weights=[torch.tensor(values[name],dtype=torch.float64)for node in(first,last)for name in node['inputs'][1:]]
    with torch.inference_mode():
        hidden=F.linear(torch.as_tensor(x,dtype=torch.float64),weights[0],weights[1])
        return F.linear(hidden*hidden,weights[2],weights[3]).numpy()
