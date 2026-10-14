import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([0.32473438050385395, -0.21091092126679145, -0.29992334637703816, 0.2096139072742499],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False, False, True],dtype=torch.bool))
    def forward(self,x):
        snapshot=x.clone()
        c=torch.where(self.mask,x,self.w)
        return c*self.w+snapshot
def make():
    return Model().eval()
