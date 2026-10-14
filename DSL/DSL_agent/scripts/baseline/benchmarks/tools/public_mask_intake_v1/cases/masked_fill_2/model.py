import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.13801227236067481, -0.08907335815368411], [0.24829012211740353, 0.12499327641617497]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
    def forward(self,x,*,z):
        snapshot=x.clone()
        c=(x+z).masked_fill(mask=self.mask.reshape(2,1),value=-.25)
        return {'y':c+z+self.w,'saved':[snapshot,x,None]}
def make():
    return Model().eval()
