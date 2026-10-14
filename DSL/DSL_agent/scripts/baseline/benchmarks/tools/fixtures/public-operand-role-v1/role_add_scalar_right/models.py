"""Owned deterministic arithmetic models for isolated intake testing."""
import torch
from torch import nn

class Arithmetic(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "tensor":
            self.register_buffer("public", torch.zeros((2,2),dtype=torch.float64))
        else:
            self.public = 0.375

class AddLeft(Arithmetic):
    def forward(self, x):
        return self.public + x
class AddRight(Arithmetic):
    def forward(self, x):
        return x + self.public
class SubtractLeft(Arithmetic):
    def forward(self, x):
        return self.public - x
class SubtractRight(Arithmetic):
    def forward(self, x):
        return x - self.public
class MultiplyLeft(Arithmetic):
    def forward(self, x):
        return self.public * x
class MultiplyRight(Arithmetic):
    def forward(self, x):
        return x * self.public

def make_model(op, kind, side):
    classes = {
        ("add","left"): AddLeft, ("add","right"): AddRight,
        ("subtract","left"): SubtractLeft, ("subtract","right"): SubtractRight,
        ("multiply","left"): MultiplyLeft, ("multiply","right"): MultiplyRight,
    }
    return classes[(op,side)](kind).eval().double()
