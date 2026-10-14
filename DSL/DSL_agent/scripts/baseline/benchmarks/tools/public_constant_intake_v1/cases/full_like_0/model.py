import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([-0.25153454605527137, 0.42628296969811413, 0.20022663171926575, -0.3878335353194888],dtype=torch.float64))
    def forward(self,x):
        c=torch.full_like(x,.125000017,dtype=torch.float32).double()
        return x*(c+.125)+self.w
def make():
    return Model().eval()
