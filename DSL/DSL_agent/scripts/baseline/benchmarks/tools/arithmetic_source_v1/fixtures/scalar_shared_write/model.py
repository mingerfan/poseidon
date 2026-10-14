import torch
class Model(torch.nn.Module):
 def forward(self,x):
  a,b=x.unbind()
  a.add_(b*.125)
  return a,x
def make():return Model()
