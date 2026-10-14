import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.02401810856696268, 0.09732892314059455], [0.1237278263564891, 0.31048266562131527]],dtype=torch.float64))
    def forward(self,x,*,z):
        c=torch.zeros_like(input=x,dtype=torch.float32).double()
        return {'y':x*(c+.125)+z+self.w,'saved':[x,None]}
def make():
    return Model().eval()
