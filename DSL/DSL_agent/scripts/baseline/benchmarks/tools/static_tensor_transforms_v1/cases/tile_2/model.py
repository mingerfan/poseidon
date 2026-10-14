import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return torch.tile(input=x,dims=(1,2,1)).square()
def make():
    return Model().eval()
