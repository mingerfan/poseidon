import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return x.repeat_interleave(repeats=2,dim=-1).mean(dim=-1)
def make():
    return Model().eval()
