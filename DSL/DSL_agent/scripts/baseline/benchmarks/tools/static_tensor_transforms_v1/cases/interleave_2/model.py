import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self,x,y):
        v=torch.repeat_interleave(x,repeats=2,dim=0,output_size=2)+y
        return {'result':v,'summary':v.mean(dim=0)}
def make():
    return Model().eval()
