import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([0.3519157962841883, 0.0933799910729442, 0.4471672443640215, -0.17412177438397197],dtype=torch.float64))
    def forward(self,x):
        c=x.new_ones((1,4)).reshape(4)
        return x*(c+.125)+self.w
def make():
    return Model().eval()
