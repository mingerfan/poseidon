"""Owned static source; explicitly trusted capture only."""
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.zeros(2,dtype=torch.float64))
    def forward(self,x):
        y=x - self.public
        return y.reshape(y.shape[0]*y.shape[1])
def make_model():
    return Model().double().eval()
