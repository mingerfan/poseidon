import torch
class Model(torch.nn.Module):
 def forward(self,x):
  a,b=torch.unbind(input=x,dim=-1)
  return a+b*.25
def make():return Model()
