import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.zeros((2,2),dtype=torch.float64))
        self.register_buffer("mask",torch.zeros((2,2),dtype=torch.bool))
        self.register_buffer("i",torch.zeros((2,2),dtype=torch.int64))
        self.register_buffer("scalar",torch.tensor(0.,dtype=torch.float64))
        self.register_buffer("proj",torch.zeros((2,4),dtype=torch.float64))
        self.register_buffer("bias",torch.zeros(2,dtype=torch.float64))
    def forward(self,x,*,z):
        c=torch.stack((x,self.w.flip(0)),dim=-1).flatten(1)
        return F.linear(c,self.proj,self.bias)+z
def make():
    return Model().eval()
