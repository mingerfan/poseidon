import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([-0.258197054212857, -0.30646182259300725],dtype=torch.float64))
    def forward(self,x):
        w=torch.repeat_interleave(self.public,2)
        return x+w
def make():
    return Model().eval()
