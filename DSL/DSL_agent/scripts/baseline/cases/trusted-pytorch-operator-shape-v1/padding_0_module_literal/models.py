"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.ConstantPad1d((1, 2), .125)
    def forward(self, x):
        y=self.operation(x)
        return y.reshape(6)
def make_model():
    return Model().double().eval()
