"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.ReflectionPad2d((1, 0, 0, 1))
    def forward(self, x):
        y=F.pad(x, (1, 0, 0, 1), mode='reflect', value=0.0)
        return y.reshape(y.size(0), -1).flatten()
def make_model():
    return Model().double().eval()
