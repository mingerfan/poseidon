from torch import nn
class Block(nn.Module):
    def forward(self,x): return x.square()
