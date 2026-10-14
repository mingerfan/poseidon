import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer("w1",torch.zeros((4,4),dtype=torch.float64))
 def forward(self,x):
  x=torch.nn.functional.linear(x,self.w1)
  return x
def make():return Model().eval()
