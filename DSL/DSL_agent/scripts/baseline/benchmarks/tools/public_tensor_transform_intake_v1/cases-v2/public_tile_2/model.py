import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[True], [False]],dtype=torch.bool))
    def forward(self,x,*,z):
        w=torch.tile(self.public,(1,2)).double()
        return {'y':x*w+z,'saved':[x,None]}
def make():
    return Model().eval()
