import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[-0.4921951139409858, -0.4965565597332485], [-0.2088244187625874, -0.3289189453212431]],dtype=torch.float64))
    def forward(self,x):
        w=torch.flip(self.public,(-2,-1))
        return x.square()+w
def make():
    return Model().eval()
