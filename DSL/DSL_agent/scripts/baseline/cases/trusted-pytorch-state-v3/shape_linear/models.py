"""Deterministic small input fixtures for trusted source capture."""
import torch
from torch import nn
from torch.nn import functional as F

def fixed_parameters(module,seed):
    generator=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_((torch.rand(parameter.shape,generator=generator)-0.5)*0.25)
    return module

class MLP(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(4,width),nn.Linear(width,2)])
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.layers[0](input=x)
        x=self.layers[1](x*x)
        return x.unsqueeze(0).squeeze(0).contiguous()

def make_mlp(width,*,seed):
    return fixed_parameters(MLP(width),seed)

class ConvPool(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,channels,1),nn.AvgPool2d((1,2)))
    def forward(self,x):
        x=self.layers(x)
        return x.reshape(x.shape).contiguous()
def make_conv(*,channels,seed):
    return fixed_parameters(ConvPool(channels),seed)

class Residual(nn.Module):
    def __init__(self,width):
        super().__init__();self.project=nn.Linear(width,width)
    def forward(self,x):
        result=self.project(x.reshape(x.shape))+x
        return result.unsqueeze(-1).squeeze(-1).contiguous()
def make_residual(width,*,seed):
    return fixed_parameters(Residual(width),seed)

class Normalization(nn.Module):
    def __init__(self,channels):
        super().__init__();self.norm=nn.BatchNorm1d(channels)
        self.dropout=nn.Dropout(0.2)
        with torch.no_grad():
            self.norm.running_mean.copy_(torch.tensor([0.03125,-0.0625]))
            self.norm.running_var.copy_(torch.tensor([0.75,1.25]))
    def forward(self,x):
        normalized=self.dropout(self.norm(x))
        return (normalized,x-normalized)
def make_normalization(channels,*,seed):
    return fixed_parameters(Normalization(channels),seed)

class MatrixFusion(nn.Module):
    def forward(self,x,y):
        product=torch.matmul(x,y)
        return {"product":product,"branches":[torch.add(input=x,other=y),
                                              torch.sub(input=x,other=y)]}
def make_matrix(*,scale):
    # Factory scalar is a public configuration field, not a dispatch key.
    if scale!=1: raise ValueError("The fixture uses an unscaled matrix product")
    return MatrixFusion()

class ShapeLinear(nn.Module):
    def __init__(self,width,seed):
        super().__init__()
        generator=torch.Generator().manual_seed(seed)
        self.weight=nn.Parameter((torch.rand((width,4),generator=generator,dtype=torch.float64)-.5)*.25)
        self.bias=nn.Parameter((torch.rand((width,),generator=generator,dtype=torch.float64)-.5)*.125)
    def forward(self,x):
        x=x.reshape(x.shape[0],-1)
        value=F.linear(input=x,weight=self.weight,bias=self.bias)
        value=value.unsqueeze(0).squeeze(0).contiguous()
        return [value,torch.mul(input=value,other=.125)]
def make_shape(width,*,seed):
    return ShapeLinear(width,seed)

class WeightedMatrixFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(.125,dtype=torch.float64))
    def forward(self,x,y):
        product=torch.matmul(x,y)*self.scale
        return {"product":product,"branches":[x+y,x-y]}
def make_weighted_matrix(*,scale):
    if scale!=1: raise ValueError("The fixture uses a unit configured scale")
    return WeightedMatrixFusion()
