"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.AdaptiveAvgPool2d((2, 2))
    def forward(self, x):
        y=F.adaptive_avg_pool2d(x, output_size=(2, 2))
        return y.reshape(8)
def make_model():
    return Model().double().eval()
