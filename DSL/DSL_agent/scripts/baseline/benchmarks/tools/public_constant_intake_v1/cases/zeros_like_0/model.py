import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([-0.2051434855412952, 0.4249731436519821, 0.44474236468418, -0.33727390106473365],dtype=torch.float64))
    def forward(self,x):
        c=torch.zeros_like(x)
        return x*(c+.125)+self.w
def make():
    return Model().eval()
