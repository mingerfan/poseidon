"""Owned explicit-trust inference source."""
import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.operation=nn.InstanceNorm2d(2, affine=True, track_running_stats=True, eps=1e-5)
    def forward(self, x):
        y=F.instance_norm(x, running_mean=self.operation.running_mean, running_var=self.operation.running_var, weight=self.operation.weight, bias=self.operation.bias, use_input_stats=False, momentum=.1, eps=1e-5)
        return y.reshape(y.size(0), -1).flatten()
def make_model():
    return Model().double().eval()
