"""R1 shape inference over static tuples and public weight-shape metadata."""

import builtins
from math import prod
from typing import Callable

from .ops_spec import OPS


class ShapeError(Exception):
    pass


class WeightError(Exception):
    def __init__(self, code, msg):
        self.code, self.msg = code, msg
        super().__init__(msg)


def _axis(axis, rank):
    axis = axis + rank if axis < 0 else axis
    if not 0 <= axis < rank:
        raise ShapeError(f"axis {axis} outside rank {rank}")
    return axis


def _pair(value):
    return (value, value) if type(value) is int else value


def infer(kind: str, in_shapes: list[tuple], attrs: dict,
          wshape: Callable[[str], tuple]) -> list[tuple]:
    a = {key: attrs.get(key, field.default) for key, field in OPS[kind].attrs.items()}
    x = in_shapes[0]
    rank = len(x)

    def weight(ref):
        return tuple(wshape(ref.name))

    def expected(ref, shape):
        actual = weight(ref)
        if actual != shape:
            raise WeightError("E_WEIGHT_SHAPE", f"{ref.name}: expected {shape}, got {actual}")

    def bn(prefix, channels):
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            name = prefix + "." + suffix
            actual = tuple(wshape(name))
            if actual != (channels,):
                raise WeightError("E_WEIGHT_SHAPE", f"{name}: expected {(channels,)}, got {actual}")

    if kind in ("pack", "relayout", "act", "poly", "scale", "refresh"):
        return [x]
    if kind == "softmax":
        _axis(a["axis"], rank)
        return [x]
    if kind == "conv2d":
        if rank != 4 or x[0] != 1:
            raise ShapeError("conv2d requires [1,C,H,W]")
        w = weight(a["w"])
        if len(w) != 4:
            raise WeightError("E_WEIGHT_SHAPE", f"{a['w'].name}: expected rank 4, got {w}")
        co, ci, kh, kw = w
        groups = a["groups"]
        if x[1] % groups or co % groups or ci != x[1] // groups:
            raise WeightError("E_WEIGHT_SHAPE", f"{a['w'].name}: channels/groups incompatible with {x}")
        sh, sw = _pair(a["stride"])
        ph, pw = _pair(a["pad"])
        dh, dw = _pair(a["dilation"])
        ho = (x[2] + 2*ph - dh*(kh-1) - 1)//sh + 1
        wo = (x[3] + 2*pw - dw*(kw-1) - 1)//sw + 1
        if min(ho, wo) < 1:
            raise ShapeError("conv2d produces an empty spatial dimension")
        if a["b"] is not None:
            expected(a["b"], (co,))
        if a["bn"] is not None:
            bn(a["bn"], co)
        return [(1, co, ho, wo)]
    if kind == "linear":
        if rank < 1:
            raise ShapeError("linear requires rank >= 1")
        w = weight(a["w"])
        if len(w) != 2 or w[1] != x[-1]:
            raise WeightError("E_WEIGHT_SHAPE", f"{a['w'].name}: expected [out,{x[-1]}], got {w}")
        if a["b"] is not None:
            expected(a["b"], (w[0],))
        return [x[:-1] + (w[0],)]
    if kind == "matmul":
        y = in_shapes[1]
        if rank < 2 or len(y) < 2 or x[:-2] != y[:-2] or x[-1] != y[-2]:
            raise ShapeError("matmul requires equal batch shapes and matching contracted dimensions")
        return [x[:-2] + (x[-2], y[-1])]
    if kind in ("avgpool", "maxpool"):
        if rank != 4:
            raise ShapeError("avgpool requires rank 4")
        kh, kw = _pair(a["k"])
        sh, sw = _pair(a["stride"] or a["k"])
        ph, pw = _pair(a["pad"])
        ho, wo = (x[2]+2*ph-kh)//sh+1, (x[3]+2*pw-kw)//sw+1
        if min(ho, wo) < 1:
            raise ShapeError("avgpool produces an empty spatial dimension")
        return [x[:2] + (ho, wo)]
    if kind in ("mean", "sum"):
        axes = a["axis"] if isinstance(a["axis"], tuple) else (a["axis"],)
        axes = tuple(_axis(axis, rank) for axis in axes)
        if len(set(axes)) != len(axes):
            raise ShapeError("Reduction axes must be distinct")
        result = tuple(1 if i in axes else n for i, n in enumerate(x)) if a["keepdim"] else tuple(
            n for i, n in enumerate(x) if i not in axes)
        if not result:
            raise ShapeError("Scalar reductions are outside PT-DSL v1")
        return [result]
    if kind in ("add", "sub", "mul"):
        if x != in_shapes[1]:
            raise ShapeError(f"{kind} requires equal tensor shapes")
        return [x]
    if kind == "bias":
        expected(a["b"], (x[_axis(a["axis"], rank)],))
        return [x]
    if kind == "batchnorm":
        if rank < 2:
            raise ShapeError("batchnorm requires rank >= 2")
        bn(a["bn"], x[1])
        return [x]
    if kind == "layernorm":
        if rank < 1:
            raise ShapeError("layernorm requires rank >= 1")
        for key in ("w", "b"):
            if a[key] is not None:
                expected(a[key], (x[-1],))
        return [x]
    if kind == "flatten":
        start = _axis(a["start_dim"], rank)
        return [x[:start] + (prod(x[start:]),)]
    if kind == "reshape":
        target = list(a["shape"])
        if target.count(-1) > 1 or any(n < 1 and n != -1 for n in target):
            raise ShapeError("Invalid reshape dimensions")
        if -1 in target:
            known = prod(n for n in target if n != -1)
            if prod(x) % known:
                raise ShapeError("Inferred reshape dimension must divide the element count")
            target[target.index(-1)] = prod(x)//known
        if prod(target) != prod(x):
            raise ShapeError("reshape changes the element count")
        return [tuple(target)]
    if kind == "permute":
        dims = tuple(_axis(axis, rank) for axis in a["dims"])
        if sorted(dims) != list(range(rank)):
            raise ShapeError("permute dims must be a permutation of all axes")
        return [tuple(x[i] for i in dims)]
    if kind == "concat":
        axis = _axis(a["axis"], rank)
        if any(len(y) != rank or any(y[i] != x[i] for i in range(rank) if i != axis) for y in in_shapes):
            raise ShapeError("concat shapes differ outside the concatenated axis")
        return [x[:axis] + (sum(y[axis] for y in in_shapes),) + x[axis+1:]]
    if kind == "slice":
        axis = _axis(a["axis"], rank)
        length = len(range(*builtins.slice(a["start"], a["end"], a["step"]).indices(x[axis])))
        if length < 1:
            raise ShapeError("slice produces an empty dimension")
        return [x[:axis] + (length,) + x[axis+1:]]
    if kind == "split":
        axis = _axis(a["axis"], rank)
        if sum(a["sizes"]) != x[axis]:
            raise ShapeError("split sizes must sum to the split dimension")
        return [x[:axis] + (size,) + x[axis+1:] for size in a["sizes"]]
    if kind == "pad":
        pads = a["pads"]
        if len(pads)//2 > rank:
            raise ShapeError("pad has more pairs than tensor dimensions")
        result = list(x)
        for i in range(len(pads)//2):
            result[rank-1-i] += pads[2*i] + pads[2*i+1]
        return [tuple(result)]
    raise ShapeError(f"Unknown operator {kind}")
