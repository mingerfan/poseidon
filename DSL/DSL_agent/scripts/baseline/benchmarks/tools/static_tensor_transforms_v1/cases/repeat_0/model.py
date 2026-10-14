import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return x.repeat(2)
def make():
    return Model().eval()
