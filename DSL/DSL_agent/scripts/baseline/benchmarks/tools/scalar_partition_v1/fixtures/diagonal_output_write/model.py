import torch
class Model(torch.nn.Module):
 def forward(self,x):
  d=torch.einsum("ii->i",x)
  d.add_(.125)
  return d,x
def make():return Model().eval()
