import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        if x.sum()>0:
            return x
        return -x
def make_model():
    return Model().double().eval()
