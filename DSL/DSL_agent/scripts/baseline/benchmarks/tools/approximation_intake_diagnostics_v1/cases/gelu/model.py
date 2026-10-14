import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.act=torch.nn.GELU()
 def forward(self,x): return self.act(x)
def build(): return Model().double().eval()
