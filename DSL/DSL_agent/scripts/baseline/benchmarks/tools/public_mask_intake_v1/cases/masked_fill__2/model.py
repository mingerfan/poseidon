import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.3128686129822872, -0.15309293820245742], [0.0536970558096489, 0.3115895706185865]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
    def forward(self,x,*,z):
        snapshot=x.clone()
        base=x.view(2,2)
        c=base.masked_fill_(mask=self.mask.reshape(2,1),value=-.25)
        return {'y':c+z+self.w,'saved':[snapshot,x,None]}
def make():
    return Model().eval()
