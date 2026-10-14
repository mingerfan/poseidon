import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([-0.47083916049982466, 0.17188732893095948],dtype=torch.float64))
    def forward(self,x):
        w=self.public.repeat(2)
        return x+w
def make():
    return Model().eval()
