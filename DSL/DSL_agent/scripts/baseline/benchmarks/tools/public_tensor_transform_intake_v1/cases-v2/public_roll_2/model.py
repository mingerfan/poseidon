import torch
from torch.nn import functional as F
from helpers import reorder
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[0.037180275227654325, -0.10498261963045996], [0.4830005552411748, 0.16799121547629314]],dtype=torch.float64))
    def forward(self,x,*,z):
        w=reorder(self.public)
        return {'y':x.roll(1,-1)*w+z,'saved':[x,None]}
def make():
    return Model().eval()
