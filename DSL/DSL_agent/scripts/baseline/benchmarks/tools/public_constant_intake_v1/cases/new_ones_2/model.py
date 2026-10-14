import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.16884309695228755, -0.42295693442174626], [0.06367582720094439, 0.018046256400283123]],dtype=torch.float64))
    def forward(self,x,*,z):
        c=x.new_ones(())
        return {'y':x*(c+.125)+z+self.w,'saved':[x,None]}
def make():
    return Model().eval()
