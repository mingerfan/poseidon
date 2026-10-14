import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return torch.tile(x,(2,)).mean(dim=-1)
def make():
    return Model().eval()
