import torch
class Model(torch.nn.Module):
 def forward(self,x):
  return x.unbind()[0]
def make():return Model()
