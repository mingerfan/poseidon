"""Small trusted sources for approximation selection acceptance."""
import torch
from torch.nn import functional as F

class ModuleProfiles(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch=torch.nn.Sequential(torch.nn.ReLU(),torch.nn.LeakyReLU(.2))
    def forward(self,x):return self.branch(x*.125)

class NodeProfiles(torch.nn.Module):
    def __init__(self):
        super().__init__();self.act=torch.nn.ReLU()
    def forward(self,x):
        a=self.act(x*.125);b=self.act(x*.25);c=torch.relu(x*.5)
        return a+b+c

class Leaf(torch.nn.Module):
    def forward(self,x):
        y=F.relu(x)
        for _ in range(x.shape[-1]):y=y+y*.125
        return y

class NestedScopes(torch.nn.Module):
    def __init__(self):
        super().__init__();self.left=Leaf();self.right=Leaf()
    def forward(self,x):return self.left(x*.125)+self.right(x*.25)

class Functional(torch.nn.Module):
    def forward(self,x):
        y=F.gelu(x*.125);z=F.silu(x*.125)
        return {"activation":y+z,"gate":torch.sigmoid(x*.125)*torch.tanh(x*.125)}

class NormProfiles(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a=torch.nn.LayerNorm(2,eps=.5,elementwise_affine=False)
        self.b=torch.nn.LayerNorm(2,eps=.25,elementwise_affine=False)
    def forward(self,x):return self.a(x*.125)+self.b(x*.25)

class Compound(torch.nn.Module):
    def __init__(self):
        super().__init__();self.sm=torch.nn.Softmax(dim=-1)
    def forward(self,x):return F.softmax(x*.125,dim=-1)+self.sm(x*.25)

def make(kind):
    return {"modules":ModuleProfiles,"nodes":NodeProfiles,"nested":NestedScopes,
            "functions":Functional,"norms":NormProfiles,"compound":Compound}[kind]()
