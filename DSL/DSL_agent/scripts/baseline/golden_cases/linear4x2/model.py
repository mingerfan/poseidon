"""Independent torch.nn.Linear reference, not a slot-algorithm reimplementation."""
import json
from pathlib import Path
import torch


def build_model():
    values = json.loads((Path(__file__).parents[1] / "fixtures.json").read_text())
    # Isolate constructor RNG use; forward has no randomness or side effects.
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(4, 2, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(values["linear_weight"], dtype=torch.float64))
        model.bias.copy_(torch.tensor(values["linear_bias"], dtype=torch.float64))
    return model.eval()
