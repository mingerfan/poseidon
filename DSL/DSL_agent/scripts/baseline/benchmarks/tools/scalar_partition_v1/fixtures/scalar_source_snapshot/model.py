import torch
class Model(torch.nn.Module):
 def forward(self,x):
  a,b=x.unbind()
  s=a.clone()
  x.mul_(.25)
  return s,a+b
def make():return Model()
