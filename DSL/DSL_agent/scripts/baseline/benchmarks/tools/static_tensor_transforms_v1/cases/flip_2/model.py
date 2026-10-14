import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return x.flip((-2,)).reshape(2,2)-x.reshape(2,2)
def make():
    return Model().eval()
