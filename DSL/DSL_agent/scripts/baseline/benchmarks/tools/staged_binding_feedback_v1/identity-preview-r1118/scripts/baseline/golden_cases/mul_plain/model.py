"""Independent PyTorch reference; public fixed elementwise weights."""
import json
from pathlib import Path
import torch


class MultiplyPlain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        values = json.loads((Path(__file__).parents[1] / "fixtures.json").read_text())
        self.register_buffer("weight", torch.tensor(values["mul_weights"], dtype=torch.float64))

    def forward(self, x):
        return x * self.weight


def build_model():
    return MultiplyPlain().eval()
