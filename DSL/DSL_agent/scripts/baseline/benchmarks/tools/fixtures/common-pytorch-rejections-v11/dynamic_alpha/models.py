import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return torch.add(x,x,alpha=x)
def make_model():
    return Model().double().eval()
