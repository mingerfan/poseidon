import torch
class Model(torch.nn.Module):
 def forward(self,x):
  s=x[0]
  s.mul_(.25)
  return torch.einsum(",i->i",s,x)
def make():return Model().eval()
