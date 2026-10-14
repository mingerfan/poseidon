import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool1d(x, 2).flatten()

def make_model():
    return Model().double().eval()
