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
    def forward(self,x):
        c=torch.stack((self.w.float(),self.i),dim=-1)
        return x.unsqueeze(-1)*c
def make():
    return Model().eval()
