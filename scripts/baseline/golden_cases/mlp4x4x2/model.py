"""Independent PyTorch Linear(4,4) -> square -> Linear(4,2) reference."""
import json
from pathlib import Path
import torch


class PolynomialMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(4, 4, dtype=torch.float64)
        self.output = torch.nn.Linear(4, 2, dtype=torch.float64)

    def forward(self, x):
        return self.output(torch.square(self.hidden(x)))


def build_model():
    fixed = json.loads((Path(__file__).parents[1] / "fixtures.json").read_text())
    with torch.random.fork_rng(devices=[]):
        model = PolynomialMLP()
    with torch.no_grad():
        for layer, weight, bias in ((model.hidden, "mlp_hidden_weight", "mlp_hidden_bias"),
                                    (model.output, "linear_weight", "linear_bias")):
            layer.weight.copy_(torch.tensor(fixed[weight], dtype=torch.float64))
            layer.bias.copy_(torch.tensor(fixed[bias], dtype=torch.float64))
    return model.eval()
