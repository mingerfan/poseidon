import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('weight',torch.zeros((2, 2),dtype=torch.float64))
 def forward(self,x):
  a=torch.nn.functional.linear(x,self.weight)
  return torch.pow(a,4)/2
def make():return Model().eval()
