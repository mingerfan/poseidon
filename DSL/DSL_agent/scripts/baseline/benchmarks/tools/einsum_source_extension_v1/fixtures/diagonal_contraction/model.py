import torch
class Model(torch.nn.Module):
 def forward(self,x,y):
  return torch.einsum("bii,ij->bj",x,y)
def make():return Model().eval()
