import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("std",torch.tensor([.5,-2.,3.],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([[True],[False]],dtype=torch.bool))
    def forward(self,x):
        divided=x/self.std
        return {"masked":divided.mul(self.mask),"residual":divided+x}
def make():
    return Model().eval()
