"""Owned small static matrix models for cross-input equivalence checks."""
import torch
from torch import nn

class CipherMatrix(nn.Module):
    def forward(self, left, right):
        return torch.matmul(left, right)

class PublicRightMatrix(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("weight", torch.zeros(shape, dtype=torch.float64))

    def forward(self, left):
        return torch.matmul(left, self.weight)

class PublicLeftMatrix(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("weight", torch.zeros(shape, dtype=torch.float64))

    def forward(self, right):
        return torch.matmul(self.weight, right)

def make_model(public_side=None, public_shape=None):
    if public_side == "right":
        return PublicRightMatrix(public_shape).eval().double()
    if public_side == "left":
        return PublicLeftMatrix(public_shape).eval().double()
    return CipherMatrix().eval().double()
