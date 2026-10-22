"""Deterministic, ordinary PyTorch InstanceNorm source contexts."""
import torch
from torch import nn
from torch.nn import functional as F

SHAPES = {
    "module1_input": [[1, 1, 3], [2, 2, 2], [1, 2, 3]],
    "module2_input": [[1, 1, 2, 2], [1, 2, 2, 2], [2, 1, 2, 2]],
    "module1_frozen": [[1, 1, 3], [2, 2, 2], [1, 2, 3]],
    "module2_frozen": [[1, 1, 2, 2], [1, 2, 2, 2], [2, 1, 2, 2]],
    "unbatched1": [[1, 3], [2, 2], [2, 3]],
    "unbatched2": [[1, 2, 2], [2, 2, 2], [2, 2, 3]],
    "unbatched1_frozen": [[1, 3], [2, 2], [2, 3]],
    "unbatched2_frozen": [[1, 2, 2], [2, 2, 2], [2, 2, 3]],
    "functional_input": [[1, 1, 3], [2, 2, 2], [1, 2, 2, 2]],
    "functional_frozen": [[1, 1, 3], [2, 2, 2], [1, 2, 2, 2]],
}

def source_specs(family, context):
    return [{"name": "x", "shape": SHAPES[family][context]}]

def source_outputs(context):
    return ["out"] if context == 0 else ["out", "summary" if context == 1 else "skip"]

def source_approximations(family):
    return dict(format="poseidon-pytorch-approximations-v1",
                defaults={} if "frozen" in family else
                dict(rsqrt=dict(method="newton", domain=[.5, 1.], iterations=1)),
                modules=[], nodes=[])

class Model(nn.Module):
    def __init__(self, family, context):
        super().__init__()
        self.family, self.context = family, context
        dims = SHAPES[family][context]
        self.functional = family.startswith("functional")
        self.frozen = "frozen" in family
        self.unbatched = family.startswith("unbatched")
        channels = dims[0] if self.unbatched else dims[1]
        affine = context != 0
        self.eps = 1e-5 if self.frozen else .5
        channel = torch.arange(channels, dtype=torch.float64)
        if self.functional:
            self.register_buffer("gain", .8 + .2 * channel if affine else None)
            self.register_buffer("offset", -.1 + .05 * channel if affine else None)
            self.register_buffer("running_mean", -.1 + .15 * channel if self.frozen else None)
            self.register_buffer("running_var", .75 + .2 * channel if self.frozen else None)
        else:
            cls = nn.InstanceNorm1d if "1" in family else nn.InstanceNorm2d
            self.norm = cls(channels, eps=self.eps, affine=affine,
                            track_running_stats=self.frozen).double().eval()
            with torch.no_grad():
                if affine:
                    self.norm.weight.copy_(.8 + .2 * channel)
                    self.norm.bias.copy_(-.1 + .05 * channel)
                if self.frozen:
                    self.norm.running_mean.copy_(-.1 + .15 * channel)
                    self.norm.running_var.copy_(.75 + .2 * channel)

    def forward(self, x):
        z = x * .125
        if self.functional:
            y = F.instance_norm(input=z, running_mean=self.running_mean,
                running_var=self.running_var, weight=self.gain, bias=self.offset,
                use_input_stats=not self.frozen, momentum=.1, eps=self.eps)
        else:
            y = self.norm(input=z)
        if self.context == 0:
            return y
        if self.context == 1:
            return y, y.mean(dim=-1, keepdim=True)
        return {"normalized": y, "residual": y + z * .125}

def make_model(family, context):
    return Model(family, context).double().eval()
