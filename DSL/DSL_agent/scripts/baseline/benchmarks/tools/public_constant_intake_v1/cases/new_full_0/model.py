import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([-0.33994352710802084, -0.19961538099760884, -0.41592362620153733, 0.44869415547415775],dtype=torch.float64))
    def forward(self,x):
        c=x.new_full((1,),-.375)
        return x*(c+.125)+self.w
def make():
    return Model().eval()
