import torch
class Model(torch.nn.Module):
    def forward(self,x,y):
        return torch.einsum("bhqd,bhkd->bhqk",x,y)
def make():
    return Model().eval()
