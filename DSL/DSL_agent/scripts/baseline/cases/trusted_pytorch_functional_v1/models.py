"""Small functional normalization and population-variance source examples."""
import torch
from torch.nn import functional as F

class Layer(torch.nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        if variant==1:
            self.register_buffer("gain",torch.tensor([.5,.75],dtype=torch.float64))
            self.register_buffer("bias",torch.tensor([.125,-.0625],dtype=torch.float64))
    def forward(self,x):
        if self.variant==0:return F.layer_norm(x,(4,),eps=1.)
        if self.variant==1:
            y=F.layer_norm(input=x,normalized_shape=[2],weight=self.gain,bias=self.bias,eps=1.)
            return y+x*.125
        y=F.layer_norm(x,(2,2),None,None,1.)
        return torch.stack([y.mean(2),y.sum(2)],dim=-1)

class Group(torch.nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        if variant==1:
            self.register_buffer("gain",torch.tensor([.5,.75],dtype=torch.float64))
            self.register_buffer("bias",torch.tensor([.125,-.0625],dtype=torch.float64))
    def forward(self,x):
        if self.variant==0:return F.group_norm(x,1,eps=1.)
        if self.variant==1:return F.group_norm(input=x,num_groups=2,weight=self.gain,bias=self.bias,eps=1.)
        y=F.group_norm(x,2,None,None,1.)
        return y+x.mean(-1,True)*.125

class Variance(torch.nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
    def forward(self,x):
        if self.variant==0:return torch.var(input=x,dim=-1,correction=0,keepdim=True)
        if self.variant==1:return x.var((0,2),False,True)+x.mean((0,2),True)
        v=x.var(dim=(),unbiased=False).reshape(1)
        return torch.stack([v,x.mean().reshape(1)],dim=0)

def make_layer(variant=0):return Layer(variant)
def make_group(variant=0):return Group(variant)
def make_variance(variant=0):return Variance(variant)
