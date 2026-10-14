"""Asymmetric oracle: positive Hecate rotation is tested against left roll."""
import torch


class Rotate(torch.nn.Module):
    def forward(self, x):
        return torch.roll(x, shifts=-1, dims=-1)


def build_model():
    return Rotate().eval()
