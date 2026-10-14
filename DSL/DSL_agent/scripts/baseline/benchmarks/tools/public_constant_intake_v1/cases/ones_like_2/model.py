import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.40839418981674647, 0.2618458041860246], [0.034578620419564055, 0.2713800166444082]],dtype=torch.float64))
    def forward(self,x,*,z):
        c=torch.ones_like(self.w.float(),dtype=torch.float32).double()
        return {'y':x*(c+.125)+z+self.w,'saved':[x,None]}
def make():
    return Model().eval()
