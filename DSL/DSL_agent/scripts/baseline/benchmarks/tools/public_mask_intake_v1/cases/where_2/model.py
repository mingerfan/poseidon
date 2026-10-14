import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.3128140411415007, 0.08300443650928369], [0.3974819949902413, -0.15477491056353065]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
    def forward(self,x,*,z):
        snapshot=x.clone()
        c=torch.where(self.mask.flip(0).reshape(2,1),-.125,z+x)
        return {'y':c+z+self.w,'saved':[snapshot,x,None]}
def make():
    return Model().eval()
