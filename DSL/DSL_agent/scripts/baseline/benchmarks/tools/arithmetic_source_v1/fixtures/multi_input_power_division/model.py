import torch
class Model(torch.nn.Module):
 def forward(self,x,y):
  a=torch.true_divide(x,2)
  b=y.pow(2)
  return dict(main=a+b,secondary=torch.stack([a,b]))
def make():return Model().eval()
