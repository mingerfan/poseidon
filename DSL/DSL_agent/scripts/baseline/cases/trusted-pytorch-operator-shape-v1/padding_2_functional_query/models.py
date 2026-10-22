"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.ReplicationPad2d((0, 1, 1, 0))
    def forward(self, x):
        y=F.pad(x, (0, 1, 1, 0), mode='replicate', value=0.0)
        return y.reshape(y.shape[0], -1).flatten()
def make_model():
    return Model().double().eval()
