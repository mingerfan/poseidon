import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.40196431146227046, -0.11414682304362506], [0.1984942311385987, -0.23903020816185486]],dtype=torch.float64))
    def forward(self,x,*,z):
        c=torch.full_like(x,True,dtype=torch.bool).double()
        return {'y':x*(c+.125)+z+self.w,'saved':[x,None]}
def make():
    return Model().eval()
