"""Owned deterministic arithmetic module factories."""
import torch
from torch import nn
class Base(nn.Module):
    def __init__(self,public_shape,side,style):
        super().__init__()
        self.side,self.style=side,style
        if public_shape is None:self.public=0.375
        else:self.register_buffer("public",torch.zeros(public_shape,dtype=torch.float64))
    def operands(self,x):
        return (self.public,x) if self.side=="left" else (x,self.public)
class Add(Base):
    def forward(self,x):
        a,b=self.operands(x)
        if self.style=="method":return a.add(other=b,alpha=1)
        if self.style=="function":return torch.add(input=a,other=b,alpha=1)
        return a+b
class Subtract(Base):
    def forward(self,x):
        a,b=self.operands(x)
        if self.style=="method":return a.sub(other=b,alpha=1)
        if self.style=="function":return torch.sub(input=a,other=b,alpha=1)
        return a-b
class Multiply(Base):
    def forward(self,x):
        a,b=self.operands(x)
        if self.style=="method":return a.mul(other=b)
        if self.style=="function":return torch.mul(input=a,other=b)
        return a*b
def make_model(op,public_shape,side,style="operator",float32=False):
    module={"add":Add,"subtract":Subtract,"multiply":Multiply}[op](public_shape,side,style).eval()
    return module.float() if float32 else module.double()
