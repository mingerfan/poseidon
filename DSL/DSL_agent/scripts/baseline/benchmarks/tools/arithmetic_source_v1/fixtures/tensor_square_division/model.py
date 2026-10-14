import torch
class Model(torch.nn.Module):
 def forward(self,x):
  return torch.pow(x,2)/2
def make():return Model().eval()
