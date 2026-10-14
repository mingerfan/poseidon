import torch
class Model(torch.nn.Module):
    def forward(self,x,y,z):
        return torch.einsum("ij,jk,k->i",x,y,z)
def make():
    return Model().eval()
