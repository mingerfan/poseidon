import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[1, -2]],dtype=torch.int64))
    def forward(self,x,*,z):
        w=self.public.repeat_interleave(2,dim=0).double()
        return {'y':x*w+z,'saved':[x,None]}
def make():
    return Model().eval()
