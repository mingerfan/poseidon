import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.17098708779405736, 0.10326203500834485], [0.06151488069834726, 0.25223130439079067]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
    def forward(self,x,*,z):
        snapshot=x.clone()
        c=torch.ones_like(self.w,dtype=torch.int64).masked_fill(self.mask.reshape(2,1),-2).double()
        return {'y':x*(c+.125)+z+self.w,'saved':[snapshot,x,None]}
def make():
    return Model().eval()
