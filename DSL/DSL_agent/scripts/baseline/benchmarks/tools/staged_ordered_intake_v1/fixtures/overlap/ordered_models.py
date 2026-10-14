import torch
from torch.nn import functional as F

class LinearAliases(torch.nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear=torch.nn.Linear(in_features,out_features)
    def forward(self,x):
        y=self.linear(x)
        before=y.square()
        view=y[:,0:1]
        view.add_(.125)
        return {"current":y,"previous":before,"part":view}

class ConvAliases(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.Conv1d(1,1,2)
    def forward(self,x):
        y=F.avg_pool1d(self.conv(x),2)
        before=y.square()
        y.mul_(.5)
        return y,before

class ResidualAliases(torch.nn.Module):
    def forward(self,x,b):
        y=x+b
        before=y.square()
        x.copy_(y)
        return [x,before,x+b]

class NormAliases(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=torch.nn.LayerNorm(2,eps=.1)
    def forward(self,x):
        y=self.norm(x)
        before=y.square()
        y.add_(.125)
        return y,before

class MatrixAliases(torch.nn.Module):
    def forward(self,x,b):
        y=x@b
        before=y.square()
        y.transpose(0,1).mul_(.5)
        return {"current":y,"previous":before}

class WeightView(torch.nn.Module):
    def __init__(self,split=False):
        super().__init__()
        self.split=split
        self.register_buffer("weight",torch.zeros(2,4) if split else torch.zeros(3,2))
    def forward(self,x):
        weight=self.weight.split(2,dim=1)[1] if self.split else self.weight.T
        y=x@weight
        before=y.square()
        y.add_(.125)
        return y,before

class SplitAliases(torch.nn.Module):
    def forward(self,x):
        parts=x.split(2,dim=1)
        before=x.square()
        parts[1].add_(.125)
        return x,before,parts[1]

class BroadcastCopy(torch.nn.Module):
    def forward(self,x,b):
        before=x.square()
        x.copy_(b)
        return x,before

class PublicMutation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight",torch.zeros(2))
    def forward(self,x):
        self.weight.add_(.25)
        return x+self.weight

class Overlap(torch.nn.Module):
    def forward(self,x):
        x[:,1:3].copy_(x[:,:2])
        return x

class Dynamic(torch.nn.Module):
    def forward(self,x):
        if x.sum()>0:
            return x
        return x.square()

def build(kind="linear",in_features=3,out_features=2,float32=False):
    if kind=="linear":model=LinearAliases(in_features,out_features)
    elif kind=="conv":model=ConvAliases()
    elif kind=="residual":model=ResidualAliases()
    elif kind=="norm":model=NormAliases()
    elif kind=="matrix":model=MatrixAliases()
    elif kind=="transpose_weight":model=WeightView(False)
    elif kind=="split_weight":model=WeightView(True)
    elif kind=="split":model=SplitAliases()
    elif kind=="broadcast_copy":model=BroadcastCopy()
    elif kind=="public_mutation":model=PublicMutation()
    elif kind=="overlap":model=Overlap()
    elif kind=="dynamic":model=Dynamic()
    else:raise ValueError("Unknown fixture")
    return model.float().eval() if float32 else model.double().eval()
