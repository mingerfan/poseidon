import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([-0.38447172732223267, -0.2823702507669599, -0.29887332341907175, -0.36720457382922983],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False, False, True],dtype=torch.bool))
    def forward(self,x):
        snapshot=x.clone()
        base=x.clone()
        c=base.masked_fill_(self.mask,.125)
        return c*self.w+snapshot
def make():
    return Model().eval()
