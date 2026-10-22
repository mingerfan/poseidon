import torch


class Rotate(torch.nn.Module):
    def forward(self, x):
        return torch.roll(x, shifts=-2, dims=-1)


def build_model():
    return Rotate().eval()
