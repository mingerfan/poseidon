import torch
class Model(torch.nn.Module):
    def forward(self,x,y):
        return torch.einsum("ij,jk->ik",x,y)
def make():
    return Model().eval()
