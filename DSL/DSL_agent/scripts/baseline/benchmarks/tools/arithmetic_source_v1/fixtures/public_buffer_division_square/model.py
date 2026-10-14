import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('gain',torch.zeros((),dtype=torch.float64))
 def forward(self,x):
  x.div_(self.gain)
  return x.pow(2)
def make():return Model().eval()
