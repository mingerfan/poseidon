import torch
class Model(torch.nn.Module):
 def forward(self,x):
  a,b=x.unbind()
  a.pow_(4)
  b.div_(2)
  return x,a
def make():return Model().eval()
