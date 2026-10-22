"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.AdaptiveAvgPool2d((1, 3))
    def forward(self, x):
        y=F.adaptive_avg_pool2d(x, output_size=(1, 3))
        return y.reshape(y.shape[0], -1).flatten()
def make_model():
    return Model().double().eval()
