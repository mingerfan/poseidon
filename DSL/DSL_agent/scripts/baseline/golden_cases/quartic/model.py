"""Fourth power forces a compiler-generated rescale at waterline 40."""
import torch


class Quartic(torch.nn.Module):
    def forward(self, x):
        return torch.pow(x, 4)


def build_model():
    return Quartic().eval()
