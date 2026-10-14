import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return x.repeat(2,1).mean(dim=0)
def make():
    return Model().eval()
