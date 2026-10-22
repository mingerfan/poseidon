"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.AvgPool2d((2, 3), stride=(2, 1), padding=(0, 0), count_include_pad=True)
    def forward(self, x):
        y=F.avg_pool2d(x, kernel_size=(2, 3), stride=(2, 1), padding=(0, 0), count_include_pad=True)
        return y.reshape(y.shape[0], -1).flatten()
def make_model():
    return Model().double().eval()
