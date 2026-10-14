"""Trusted static PyTorch reference: encrypted vector doubled."""
import torch


class Add(torch.nn.Module):
    def forward(self, x):
        return x + x


def build_model():
    return Add().eval().double()
