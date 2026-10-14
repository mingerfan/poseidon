import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ConvTranspose1d(
            2, 2, kernel_size=2, stride=2, output_padding=1,
            groups=2, dtype=torch.float64
        )

    def forward(self, x):
        return self.layer(x)

def make():
    return Model().eval()
