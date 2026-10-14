import torch
from torch.nn import functional as F
from helpers import masked_selection
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.30217202409601857, -0.17743589422009676], [-0.36970988995692206, 0.3988392493103625]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
    def forward(self,x,*,z):
        snapshot=x.clone()
        c=masked_selection(x,z,self.mask,self.w)
        return {'y':c+z+self.w,'saved':[snapshot,x,None]}
def make():
    return Model().eval()
