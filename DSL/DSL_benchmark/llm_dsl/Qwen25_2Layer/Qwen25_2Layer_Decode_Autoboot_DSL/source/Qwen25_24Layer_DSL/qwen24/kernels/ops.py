"""Small tracing helpers; CKKS parameter management stays in Dacapo.

Elementwise operations work on physical slots, not NumPy broadcasting rules.
The two linear layouts are the ones used by examples/benchmarks/MLP.py.
"""

import operator

import numpy as np

from .expr import Expr, Plain


__all__ = ["add", "mul", "mul_add", "square", "polynomial", "linear"]


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


def mul_add(x, y, bias):
    """Elementwise x * y + bias; emits a multiply and an add, not fused FMA."""
    _ciphertext(x, "x")
    _expression(y, "y")
    _expression(bias, "bias")
    return add(mul(x, y), bias)


def square(x):
    """Exact elementwise square activation; one ciphertext multiplication."""
    return mul(x, x)


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


def linear(x, weight, bias, *, layout):
    """Compute W @ x + b using one of the existing MLP slot layouts.

    layout='mnist_input': W=(100,784), b=(100,). Input is eight blocks of
    100 values, each duplicated consecutively, with the last block padded to
    100 (1600 slots total). Output is valid in the first 100 slots.

    layout='prefix': W=(10,100), b=(10,). Input is valid in the first 100
    slots. Output is valid in the first 10 slots.

    Requires at least 1600 physical slots. No other shapes/layouts are
    supported. The caller supplies an already packed/encrypted input;
    Hecate Expr does not carry enough metadata to check its actual layout.
    """
    _ciphertext(x, "x")
    shapes = {"mnist_input": (100, 784), "prefix": (10, 100)}
    if layout not in shapes:
        raise ValueError(f"unsupported linear layout: {layout!r}")
    weight = _real_array(weight, "weight")
    bias = _real_array(bias, "bias")
    expected = shapes[layout]
    if weight.shape != expected or bias.shape != (expected[0],):
        raise ValueError(
            f"layout {layout!r} requires weight {expected} and bias {(expected[0],)}; "
            f"got {weight.shape} and {bias.shape}"
        )
    if layout == "mnist_input":
        result = _mnist_input_matmul(x, weight)
    else:
        result = _mnist_output_matmul(x, weight)
    return add(result, Plain(bias))


def _mnist_input_matmul(x, weight):
    diagonals = np.zeros((100, 1600))
    for n in range(100):
        for c in range(8):
            for k in range(100):
                index = c * 100 + (k + n) % 100
                if index < 784:
                    diagonals[n, c * 200 + k] = weight[k, index]
    diagonals = [Plain(row) for row in diagonals]
    for n in range(100):
        term = mul(x.rotate(n), diagonals[n])
        result = term if n == 0 else add(result, term)
    for shift in (800, 400, 200):
        result = add(result, result.rotate(shift))
    return result


def _mnist_output_matmul(x, weight):
    normal = np.zeros((10, 100))
    wrap = np.zeros((10, 100))
    for n in range(10):
        for c in range(10):
            for k in range(10):
                index = c * 10 + k
                if k + n < 10:
                    normal[n, index] = weight[k, c * 10 + k + n]
                else:
                    wrap[n, index] = weight[k, c * 10 + k + n - 10]
    normal = [Plain(row) for row in normal]
    wrap = [Plain(row) for row in wrap]
    for n in range(10):
        term = mul(x.rotate(n), normal[n])
        if n != 0:
            term = add(term, mul(x.rotate(n - 10), wrap[n]))
        result = term if n == 0 else add(result, term)
    result = add(result, result.rotate(50))
    for i in range(5):
        term = result.rotate(i * 10)
        output = term if i == 0 else add(output, term)
    return output
