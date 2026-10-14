import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return torch.repeat_interleave(x,2,output_size=4)
def make():
    return Model().eval()
