import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([-0.08062016145656059, -0.19928419678943987, -0.37332817902307164, -0.15028207825918927],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False, False, True],dtype=torch.bool))
    def forward(self,x):
        snapshot=x.clone()
        c=x.masked_fill(self.mask,.125)
        return c*self.w+snapshot
def make():
    return Model().eval()
