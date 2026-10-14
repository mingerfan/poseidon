import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('weight',torch.zeros(2,dtype=torch.float64))
 def forward(self,x):
  a,b=self.weight.unbind()
  return x*a+b
def make():return Model()
