import torch
from torch import nn
class Model(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.linear=nn.Linear(width,width,dtype=torch.float64)
    def forward(self,x): return self.linear(x)
def build(width=2): return Model(width)
