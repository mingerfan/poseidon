import torch
class Model(torch.nn.Module):
 def forward(self,x):
  p=x.unbind()
  return torch.stack([p[-1],p[1],p[1]])
def make():return Model()
