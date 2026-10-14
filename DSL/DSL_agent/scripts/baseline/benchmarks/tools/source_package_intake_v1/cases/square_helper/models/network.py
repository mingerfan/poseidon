from torch import nn
from .layers import Block
class Model(nn.Module):
    def __init__(self,gain):
        super().__init__()
        self.block=Block()
        self.gain=gain
    def forward(self,x): return self.block(x)+self.gain
def build(gain=0.25): return Model(gain)
