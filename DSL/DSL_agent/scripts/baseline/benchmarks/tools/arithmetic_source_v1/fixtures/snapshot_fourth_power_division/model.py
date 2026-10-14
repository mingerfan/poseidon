import torch
class Model(torch.nn.Module):
 def forward(self,x):
  a=x.clone()
  x.pow_(4)
  x.div_(2)
  return a,x
def make():return Model().eval()
