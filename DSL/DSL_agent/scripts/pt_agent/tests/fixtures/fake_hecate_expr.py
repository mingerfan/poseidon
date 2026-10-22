"""Test-only Hecate API for trusted emitted arithmetic programs."""

import os
from pathlib import Path

import numpy as np


_REGISTERED = None
_SIGNATURE = ""


class Expr:
    def __init__(self, data):
        self.value = np.array(data, dtype=np.float64, copy=True)

    def _binary(self, other, operation, reverse=False):
        if not isinstance(other, Expr):
            raise TypeError("Fake Hecate operations require Expr operands")
        left, right = (other.value, self.value) if reverse else (self.value, other.value)
        return Expr(operation(left, right))

    def __add__(self, other):
        return self._binary(other, np.add)

    def __radd__(self, other):
        return self._binary(other, np.add, True)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        return self._binary(other, np.subtract, True)

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    def __rmul__(self, other):
        return self._binary(other, np.multiply, True)

    def __neg__(self):
        return Expr(-self.value)

    def rotate(self, k):
        return Expr(np.roll(self.value, -k))


class Plain(Expr):
    pass


def bootstrap(x):
    return Expr(x.value)


def func(sig):
    def register(fn):
        global _REGISTERED, _SIGNATURE
        _REGISTERED, _SIGNATURE = fn, sig
        return fn
    return register


def save(dirs, cst_dirs):
    with np.load(os.environ["FAKE_HECATE_INPUTS"]) as data:
        inputs = [Expr(data[f"x{i}"]) for i in range(_SIGNATURE.split(",").count("c"))]
    outputs = _REGISTERED(*inputs)
    outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    np.savez(Path(dirs) / "fake_outputs.npz", **{f"o{i}": value.value for i, value in enumerate(outputs)})
