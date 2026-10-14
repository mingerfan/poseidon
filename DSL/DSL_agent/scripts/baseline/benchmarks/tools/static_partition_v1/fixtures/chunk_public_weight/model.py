import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight",torch.zeros(4,2,dtype=torch.float64))
    def forward(self,x):
        a,b=self.weight.chunk(2,dim=0)
        return F.linear(x,a)+F.linear(x,b)
def make(): return Model().double().eval()
