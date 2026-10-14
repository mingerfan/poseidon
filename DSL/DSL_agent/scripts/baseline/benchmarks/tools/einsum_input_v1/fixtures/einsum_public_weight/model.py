import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight",torch.zeros(2,2,dtype=torch.float64))
    def forward(self,x):
        return torch.einsum("ij,jk->ik",x,self.weight)
def make():
    return Model().eval()
