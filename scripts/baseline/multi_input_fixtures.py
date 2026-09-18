"""Independent arithmetic oracles for multi-input golden bring-up, not Agent answers.

These fixtures do not yet implement user graph schema 3. They establish the
multi-input compiler/runtime ABI before the public model loader is expanded.
"""
import math

CASES = {
    "dual_add": (2, 4), "ordered_subtract": (2, 4), "dual_product": (2, 4),
    "weighted_merge": (2, 4), "dot_difference": (2, 1),
    "triple_merge": (3, 4), "quad_merge": (4, 4), "dual_outputs": (2, 2),
}
NAMES = ("x", "y", "z", "t")
CONSTANTS = {"weight": [0.5, -0.75, 0.125, 1.0], "bias": 0.125}


def reference(case, inputs, constants):
    """Pure Python scalar arithmetic; no DSL/FX/runtime/decrypted values used."""
    arity, _ = CASES[case]
    if len(inputs) != arity or any(len(v) != 4 for v in inputs):
        raise ValueError("Independent reference input shape mismatch")
    if any(not math.isfinite(v) for row in inputs for v in row):
        raise ValueError("Non-finite reference input")
    x, y = inputs[:2]
    weight, bias = constants["weight"], constants["bias"]
    if case == "dual_add":
        return [a+b for a, b in zip(x, y)]
    if case == "ordered_subtract":
        return [a-b for a, b in zip(x, y)]
    if case == "dual_product":
        return [a*b for a, b in zip(x, y)]
    if case == "weighted_merge":
        return [a*w+b+bias for a, b, w in zip(x, y, weight)]
    if case == "dot_difference":
        return [math.fsum((a-b)*w for a, b, w in zip(x, y, weight))+bias]
    if case == "triple_merge":
        return [a+b-c for a, b, c in zip(*inputs)]
    if case == "quad_merge":
        return [(a+b)-(c+d) for a, b, c, d in zip(*inputs)]
    return [x[0]+y[0], x[0]-y[0]]


def fixture_inputs(arity):
    """Zero, asymmetric signed, seeded random and +/-1 boundary; no aliasing."""
    import random
    rng = random.Random(20260907)
    signed = [[0.5, -1.0, 0.25, -0.75], [-0.125, 0.5, -0.5, 0.25],
              [0.75, 0.125, -0.25, -0.5], [-0.5, 0.75, 0.125, 0.5]]
    boundary = [[-1., 1., -1., 1.], [1., 1., -1., -1.],
                [1., -1., -1., 1.], [-1., -1., 1., 1.]]
    return [[[0.]*4 for _ in range(arity)], signed[:arity],
            [[rng.uniform(-1., 1.) for _ in range(4)] for _ in range(arity)], boundary[:arity]]


def torch_model(case, constants):
    """Secondary cross-check using actual torch ops, independent of DSL lowering."""
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.tensor(constants["weight"], dtype=torch.float64))
            self.register_buffer("bias", torch.tensor(constants["bias"], dtype=torch.float64))

        def forward(self, *args):
            x, y = args[:2]
            if case == "dual_add":
                return x+y
            if case == "ordered_subtract":
                return x-y
            if case == "dual_product":
                return x*y
            if case == "weighted_merge":
                return x*self.weight+y+self.bias
            if case == "dot_difference":
                return (torch.dot(x-y, self.weight)+self.bias).reshape(1)
            if case == "triple_merge":
                return x+y-args[2]
            if case == "quad_merge":
                return x+y-(args[2]+args[3])
            return torch.stack((x[0]+y[0], x[0]-y[0]))

    return Model().eval()
