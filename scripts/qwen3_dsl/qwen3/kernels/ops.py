"""Small tracing helpers; CKKS parameter management stays in Dacapo.

Elementwise operations work on physical slots, not NumPy broadcasting rules.
"""

import operator

import numpy as np

from .expr import Expr, Plain


__all__ = ["add", "mul", "square", "polynomial", "block_sum_repeated"]


def _expression(value, name):
    if not isinstance(value, Expr):
        raise TypeError(f"{name} must be a Hecate expression; use hc.Plain for constants")


def _ciphertext(value, name):
    _expression(value, name)
    if isinstance(value, Plain):
        raise TypeError(f"{name} must be a ciphertext expression")


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def add(x, y):
    """Add aligned slots. x is ciphertext; y is ciphertext or hc.Plain."""
    _ciphertext(x, "x")
    _expression(y, "y")
    return x + y


def mul(x, y):
    """Multiply aligned slots (not matrix multiplication)."""
    _ciphertext(x, "x")
    _expression(y, "y")
    return x * y


def square(x):
    """Exact elementwise square activation; one ciphertext multiplication."""
    return mul(x, x)


def block_sum_repeated(x, *, block_size, blocks, slots):
    """Repeat each contiguous block's sum in that block; zero every tail slot.

    Hecate rotate(s) reads slot i+s at output slot i. At each butterfly
    step, the matching bit chooses the other half of the same block. Masked
    +/- power-of-two rotations cannot mix adjacent blocks or physical tails.
    """
    _ciphertext(x, "x")
    dimensions = []
    for name, value in (("block_size", block_size), ("blocks", blocks), ("slots", slots)):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a positive integer")
        value = operator.index(value)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
        dimensions.append(value)
    block_size, blocks, slots = dimensions
    size = block_size * blocks
    if block_size & (block_size - 1) or slots & (slots - 1) or size > slots:
        raise ValueError("block_size/slots must be powers of two and all blocks must fit slots")
    result = mul(x, Plain(np.ones(size)))
    indices = np.arange(size)
    shift = 1
    while shift < block_size:
        lower_half = (indices & shift) == 0
        left = mul(result.rotate(shift), Plain(lower_half.astype(np.float64)))
        right = mul(result.rotate(-shift), Plain((~lower_half).astype(np.float64)))
        result = add(result, add(left, right))
        shift *= 2
    return result


def polynomial(x, coefficients, *, slots):
    """Evaluate c[0] + c[1]*x + ... by Horner's rule in the first slots slots.

    coefficients must describe a nonconstant real polynomial. This does not
    choose a ReLU/GELU approximation or certify its interval/error. Other slots
    are unspecified. slots must fit the selected CKKS context.
    """
    _ciphertext(x, "x")
    coefficients = _real_array(coefficients, "coefficients")
    if coefficients.ndim != 1 or coefficients.size < 2:
        raise ValueError("coefficients must be a 1D array with at least two entries")
    if coefficients[-1] == 0:
        raise ValueError("the highest-degree coefficient must be nonzero")
    if isinstance(slots, (bool, np.bool_)):
        raise TypeError("slots must be a positive integer")
    slots = operator.index(slots)
    if slots < 1:
        raise ValueError("slots must be a positive integer")

    # A one-element Plain is zero-padded by Runtime, not broadcast. Expand
    # each scalar coefficient explicitly over the requested prefix.
    result = Plain(np.full(slots, coefficients[-1]))
    for coefficient in coefficients[-2::-1]:
        result = mul(x, result)
        if coefficient != 0:
            result = add(result, Plain(np.full(slots, coefficient)))
    return result
