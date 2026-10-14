import torch
class Model(torch.nn.Module):
    def forward(self,x,y,z,w):
        return torch.einsum("ij,jk,kl,l->i",x,y,z,w)
def make():
    return Model().eval()
