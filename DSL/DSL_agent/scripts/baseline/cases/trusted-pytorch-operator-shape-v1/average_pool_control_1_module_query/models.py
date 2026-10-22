"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.AvgPool2d((2, 2), stride=(1, 1), padding=(1, 1), count_include_pad=False)
    def forward(self, x):
        y=self.operation(x)
        return y.reshape(y.size(0), -1).flatten()
def make_model():
    return Model().double().eval()
