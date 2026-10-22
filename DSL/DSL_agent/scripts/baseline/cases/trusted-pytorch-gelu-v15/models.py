import torch
from torch.nn import functional as F

class Model(torch.nn.Module):
    def __init__(self,variant,style):
        super().__init__()
        self.variant=variant
        self.style=style
        self.activation=torch.nn.GELU(approximate=variant)
    def forward(self,x):
        y=x*.125
        if self.style=="module":
            return self.activation(y)+y
        if self.style=="functional":
            return F.gelu(y,approximate=self.variant).square()
        return F.gelu(input=y,approximate=self.variant),y

def make(variant,style):
    return Model(variant,style).double().eval()
