import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self, x):
        y = F.instance_norm(x)
        return y.reshape(y.numel())

def make_model():
    return Model().double().eval()
