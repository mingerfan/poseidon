import torch
from torch.nn import functional as F
from helpers import constant_like
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.05477534585961169, 0.10390238366479104], [0.06122338090794305, 0.29353321029952234]],dtype=torch.float64))
    def forward(self,x,*,z):
        c=constant_like(x,self.w)
        return {'y':x*(c+.125)+z+self.w,'saved':[x,None]}
def make():
    return Model().eval()
