"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.ConstantPad1d((1, 2), .125)
    def forward(self, x):
        y=F.pad(x, (1, 2), mode='constant', value=0.125)
        return y.reshape(y.numel())
def make_model():
    return Model().double().eval()
