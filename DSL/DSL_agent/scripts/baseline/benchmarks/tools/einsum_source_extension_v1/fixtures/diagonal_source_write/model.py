import torch
class Model(torch.nn.Module):
 def forward(self,x):
  d=torch.einsum("ii->i",x)
  x.mul_(.25)
  return d
def make():return Model().eval()
