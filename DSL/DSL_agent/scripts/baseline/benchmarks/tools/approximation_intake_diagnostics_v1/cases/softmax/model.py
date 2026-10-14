import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.sm=torch.nn.Softmax(dim=-1)
 def forward(self,x): return self.sm(x)
def build(): return Model().double().eval()
