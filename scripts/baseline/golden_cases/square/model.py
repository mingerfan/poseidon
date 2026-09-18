import torch


class Square(torch.nn.Module):
    def forward(self, x):
        return torch.square(x)


def build_model():
    return Square().eval()
