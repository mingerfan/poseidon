import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x,y):
        v=x.repeat((2,1))+y
        return {'result':v,'summary':v.mean(dim=0)}
def make():
    return Model().eval()
