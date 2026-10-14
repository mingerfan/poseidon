import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([-0.10496437454689395, 0.17441800939231034, -0.3101348856882334, -0.14105847784519848],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False, False, True],dtype=torch.bool))
    def forward(self,x):
        snapshot=x.clone()
        c=x.where(self.mask,self.w)
        return c*self.w+snapshot
def make():
    return Model().eval()
