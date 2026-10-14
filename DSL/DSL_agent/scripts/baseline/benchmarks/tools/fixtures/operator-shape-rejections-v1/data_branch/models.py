import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 2)
        if y.sum() > 0:
            return y
        return -y

def make_model():
    return Model().double().eval()
