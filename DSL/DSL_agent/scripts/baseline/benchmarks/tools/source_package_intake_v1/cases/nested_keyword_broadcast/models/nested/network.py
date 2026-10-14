from torch import nn
from ..ops import combine
class Model(nn.Module):
    def forward(self,first,*,second):
        return {'fused':combine(first,second),'summary':second.mean(0)}
def build(): return Model()
