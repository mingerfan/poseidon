import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([-0.36569168899737337, 0.46975048036484013],dtype=torch.float64))
    def forward(self,x):
        w=torch.tile(self.public,(2,))
        return x*w
def make():
    return Model().eval()
