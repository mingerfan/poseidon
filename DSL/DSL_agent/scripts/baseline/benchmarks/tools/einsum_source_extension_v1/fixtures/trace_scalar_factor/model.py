import torch
class Model(torch.nn.Module):
 def forward(self,x,y):
  s=torch.einsum("ii->",x)
  return torch.einsum(",ij->ij",s,y)
def make():return Model().eval()
