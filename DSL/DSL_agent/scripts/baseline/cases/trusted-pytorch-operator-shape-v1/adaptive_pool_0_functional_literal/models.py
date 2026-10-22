"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.AdaptiveAvgPool1d((2,))
    def forward(self, x):
        y=F.adaptive_avg_pool1d(x, output_size=(2,))
        return y.reshape(2)
def make_model():
    return Model().double().eval()
