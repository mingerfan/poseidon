"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.AvgPool1d((2,), stride=(1,), padding=(0,), count_include_pad=True)
    def forward(self, x):
        y=F.avg_pool1d(x, kernel_size=(2,), stride=(1,), padding=(0,), count_include_pad=True)
        return y.reshape(y.numel())
def make_model():
    return Model().double().eval()
