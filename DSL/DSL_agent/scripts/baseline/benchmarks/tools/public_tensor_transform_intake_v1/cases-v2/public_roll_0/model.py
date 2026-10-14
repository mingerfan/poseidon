import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([-0.22223017902355124, -0.3133112531032525, 0.37237268658153266, -0.2881319424636041],dtype=torch.float64))
    def forward(self,x):
        w=torch.roll(self.public,shifts=1)
        return x+w
def make():
    return Model().eval()
