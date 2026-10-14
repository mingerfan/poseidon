"""Trusted reproducible PyTorch model factory; data-only JSON input, no pickle.

Eight families x six configurations. Translator never receives family names.
Four logical input elements only; flatten is a declared row-major view, not
arbitrary batching. Seeds fix model construction; actual arrays are also saved.
"""
import math
import torch
import numpy as np

from seal_artifact_gate import require

FAMILIES = ("affine", "polynomial", "linear", "mlp2", "mlp3", "fanout", "residual", "flatten_linear")
WIDTHS = ((1, 2, 1), (2, 3, 2), (3, 2, 3), (4, 2, 4), (3, 4, 2), (4, 4, 4))


def descriptors():
    return [dict(schema=1, id=f"{family}-{index}", family=family, configuration=index)
            for family in FAMILIES for index in range(6)]


def validate_descriptor(data):
    if type(data) is dict and type(data.get("schema")) is int and data["schema"] in (2, 3, 4, 5):
        from model_graph import validate_graph
        validate_graph(data)
        return "custom_graph", None
    require(type(data) is dict and set(data) == {"schema", "id", "family", "configuration"}, "Unexpected case fields")
    require(type(data["schema"]) is int and data["schema"] == 1, "Unknown descriptor schema")
    family, index = data["family"], data["configuration"]
    require(type(family) is str and family in FAMILIES and type(index) is int and 0 <= index < 6,
            "Unsupported family/configuration")
    require(data["id"] == f"{family}-{index}", "Case id does not match descriptor")
    return family, index


class CatalogModel(torch.nn.Module):
    def __init__(self, family, index):
        super().__init__()
        self.family = family
        a, b, out = WIDTHS[index]
        dims = {"linear": (4, out), "flatten_linear": (4, out),
                "mlp2": (4, a, out), "mlp3": (4, a, b, out)}.get(family, ())
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o, dtype=torch.float64)
                                          for i, o in zip(dims, dims[1:])])
        # Independent RNG streams: definition changes in one family don't alter others.
        rng = np.random.default_rng(20260905 + 100 * FAMILIES.index(family) + index)
        choices = np.array([-.75, -.5, -.25, -.125, .125, .25, .5, .75])
        for name in ("w", "b", "c"):
            self.register_buffer(name, torch.tensor(rng.choice(choices, 4), dtype=torch.float64))
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.tensor(rng.choice(choices, tuple(layer.weight.shape)), dtype=torch.float64))
                layer.bias.copy_(torch.tensor(rng.choice(choices / 4, tuple(layer.bias.shape)), dtype=torch.float64))

    def forward(self, x):
        if self.family == "affine":
            return x * self.w + self.b
        if self.family == "polynomial":
            return torch.square(x) * self.w + x * self.b + self.c
        if self.family == "fanout":
            branch = x * self.w + self.b
            return torch.square(branch) + torch.square(x) * self.c
        if self.family == "residual":
            branch = x * self.w + self.b
            return x + torch.square(branch) * self.c
        if self.family == "flatten_linear":
            x = torch.flatten(x)
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index != len(self.layers) - 1:
                x = torch.square(x)
        return x


def build_model(descriptor):
    family, index = validate_descriptor(descriptor)
    if family == "custom_graph":
        from model_graph import build_graph_model
        return build_graph_model(descriptor)
    with torch.random.fork_rng(devices=[]):
        model = CatalogModel(family, index)
    shape = ([2, 2] if index % 2 == 0 else [1, 4]) if family == "flatten_linear" else [4]
    return model.eval(), shape


def test_inputs(shape):
    require(math.prod(shape) == 4, "Only four input elements allowed")
    flat = np.array([[0, 0, 0, 0], [.5, -1, .25, -.75],
                     np.random.default_rng(42).uniform(-1, 1, 4), [-1, 1, -1, 1]], dtype=np.float64)
    return flat.reshape(4, *shape)
