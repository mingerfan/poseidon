"""Owned static source; explicitly trusted capture only."""
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.zeros(2,dtype=torch.float64))
    def forward(self,x):
        y=torch.mul(input=x, other=self.public)
        return y.reshape(x.numel())
def make_model():
    return Model().double().eval()
