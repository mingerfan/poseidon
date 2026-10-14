import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[True, False], [True, False]],dtype=torch.bool))
    def forward(self,x,*,z):
        w=self.public.flip((0,)).double()
        return {'y':x*w+z,'saved':[x,None]}
def make():
    return Model().eval()
