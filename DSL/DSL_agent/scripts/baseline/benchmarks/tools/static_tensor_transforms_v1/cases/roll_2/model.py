import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return x.roll(shifts=(1,2),dims=(-1,-1)).mean(dim=1)
def make():
    return Model().eval()
