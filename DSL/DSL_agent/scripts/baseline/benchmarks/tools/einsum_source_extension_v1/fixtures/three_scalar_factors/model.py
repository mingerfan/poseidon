import torch
class Model(torch.nn.Module):
 def forward(self,x,y,z):
  return torch.einsum(",,->",x.sum(),y.sum(),z.sum())
def make():return Model().eval()
