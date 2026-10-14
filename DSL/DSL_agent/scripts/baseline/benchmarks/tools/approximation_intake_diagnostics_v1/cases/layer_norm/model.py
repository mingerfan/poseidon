import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.norm=torch.nn.LayerNorm((2,),elementwise_affine=False)
 def forward(self,x): return self.norm(x)
def build(): return Model().double().eval()
