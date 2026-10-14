import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return torch.flip(x,(0,))
def make_model():
    return Model().double().eval()
