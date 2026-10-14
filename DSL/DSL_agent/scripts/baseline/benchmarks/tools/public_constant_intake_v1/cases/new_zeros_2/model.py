import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.03109866274721318, 0.1605421031618141], [-0.4249257359225613, -0.3386697493561462]],dtype=torch.float64))
    def forward(self,x,*,z):
        c=x.new_zeros(())
        return {'y':x*(c+.125)+z+self.w,'saved':[x,None]}
def make():
    return Model().eval()
