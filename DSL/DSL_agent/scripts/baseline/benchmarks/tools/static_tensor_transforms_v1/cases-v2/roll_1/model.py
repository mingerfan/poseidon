import torch
import torch.nn.functional as F
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("weight",torch.tensor([[.17,-.29],[-.37,.41]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([.11,-.13],dtype=torch.float64))
    def forward(self,x):
        return F.linear(torch.roll(x,(1,-1),(0,-1)),self.weight,self.bias)
def make():
    return Model().eval()
