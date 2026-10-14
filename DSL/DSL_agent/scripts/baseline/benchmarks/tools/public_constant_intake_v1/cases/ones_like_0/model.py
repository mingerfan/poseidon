import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([0.13587779117607918, 0.30851211057257183, 0.0632657655179813, -0.25501638385071035],dtype=torch.float64))
    def forward(self,x):
        c=torch.ones_like(x)
        return x*(c+.125)+self.w
def make():
    return Model().eval()
