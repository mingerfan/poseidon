import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.tensor([True, False, True, True, False]))

    def forward(self, batch):
        y = F.interpolate(batch["value"], size=5, mode="linear", align_corners=False)
        return {"result": y * self.mask, "aux": [y.square(), None]}

def make():
    return Model().eval()
