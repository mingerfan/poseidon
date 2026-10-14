import torch
class Model(torch.nn.Module):
 def forward(self,x):
  d=torch.einsum("ii->i",x)
  d.pow_(2)
  d.div_(2)
  return x,d
def make():return Model().eval()
