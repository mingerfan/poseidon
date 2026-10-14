import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([0.3477283504412787, -0.2027404559646521, 0.3662690037345109, 0.010660433635286626],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False, False, True],dtype=torch.bool))
    def forward(self,x):
        snapshot=x.clone()
        c=torch.where(self.mask,self.w,.25)
        return x*(c+.125)+self.w
def make():
    return Model().eval()
