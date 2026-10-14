import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([0.3286344316340474, 0.1530197681277592, 0.21905598593455355, 0.256624362263117],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False, False, True],dtype=torch.bool))
    def forward(self,x):
        snapshot=x.clone()
        c=self.w.masked_fill(self.mask,.125)
        return x*(c+.125)+self.w
def make():
    return Model().eval()
