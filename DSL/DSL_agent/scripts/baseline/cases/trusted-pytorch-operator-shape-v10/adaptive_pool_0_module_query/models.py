"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.AdaptiveAvgPool1d((2,))
    def forward(self, x):
        y=self.operation(x)
        return y.reshape(y.numel())
def make_model():
    return Model().double().eval()
