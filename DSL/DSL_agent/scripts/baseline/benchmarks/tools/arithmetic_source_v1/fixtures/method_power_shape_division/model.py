import torch
class Model(torch.nn.Module):
 def forward(self,x):
  a=x.pow(exponent=4)
  return a.divide(2).reshape(a.numel())
def make():return Model().eval()
