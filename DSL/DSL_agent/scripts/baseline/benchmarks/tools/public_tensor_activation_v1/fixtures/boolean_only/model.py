import torch
class BooleanOnly(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer("mask",torch.tensor([True,False,True],dtype=torch.bool))
 def forward(self,x):return x.mul(self.mask)
class Ordinary(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer("weight",torch.tensor([[.2,-.1,.3],[.4,.1,-.2],[-.3,.2,.5]],dtype=torch.float64))
 def forward(self,x):return x@self.weight+x
def boolean_only():return BooleanOnly().eval()
def ordinary():return Ordinary().eval()
