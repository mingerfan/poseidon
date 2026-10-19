"""Arithmetic-only nonlinear kernels, shared by calibration and Hecate lowering.

Only +, *, and public constants occur in evaluation. Domain validation here is
metadata validation; callers must establish encrypted input bounds separately.
"""

from dataclasses import dataclass
import math
import operator

import numpy as np


def _domain(domain, positive=False):
    values = np.asarray(domain, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError("domain must be a finite increasing pair")
    if positive and values[0] <= 0:
        raise ValueError("inverse square root domain must be positive")
    return tuple(values)


def _count(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or operator.index(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return operator.index(value)


@dataclass(frozen=True)
class ChebyshevApproximation:
    function: str
    coefficients: tuple
    domain: tuple
    evaluation: str = "balanced"
    block_size: int = 8

    def __post_init__(self):
        if self.function not in {"silu", "exp", "rsqrt"}:
            raise ValueError("Chebyshev kernel supports silu, exp and rsqrt")
        c = np.asarray(self.coefficients, dtype=np.float64)
        if c.ndim != 1 or len(c) < 2 or not np.isfinite(c).all() or c[-1] == 0:
            raise ValueError("finite nonconstant Chebyshev coefficients required")
        object.__setattr__(self, "coefficients", tuple(c))
        object.__setattr__(self, "domain", _domain(self.domain, positive=self.function == "rsqrt"))
        if self.evaluation not in {"balanced", "blocked"}:
            raise ValueError("unknown Chebyshev evaluation method")
        _count(self.block_size, "block_size", 2)


@dataclass(frozen=True)
class InverseSqrtApproximation:
    domain: tuple
    iterations: int
    function: str = "rsqrt"
    evaluation: str = "newton"
    refinement_steps: int = 2

    def __post_init__(self):
        if self.function != "rsqrt":
            raise ValueError("inverse square root kernel requires rsqrt")
        object.__setattr__(self, "domain", _domain(self.domain, positive=True))
        _count(self.iterations, "iterations")
        if self.evaluation not in {"newton", "residual"}:
            raise ValueError("unknown inverse square root evaluation method")
        _count(self.refinement_steps, "refinement_steps", 0)


@dataclass(frozen=True)
class NormalizeSquareSoftmax:
    centered_bound: float
    squarings: int
    reciprocal_iterations: int
    max_length: int
    exp_degree: int = 12
    intermediate_iterations: int | None = None
    exp_bound: float = .5

    def __post_init__(self):
        if not math.isfinite(self.centered_bound) or self.centered_bound <= 0:
            raise ValueError("centered_bound must be finite and positive")
        _count(self.squarings, "squarings", 0)
        _count(self.reciprocal_iterations, "reciprocal_iterations")
        _count(self.max_length, "max_length")
        _count(self.exp_degree, "exp_degree")
        if not math.isfinite(self.exp_bound) or not 0 < self.exp_bound <= 2:
            raise ValueError("exp_bound must be in (0, 2]")
        if self.centered_bound / 2 ** self.squarings > self.exp_bound:
            raise ValueError("scaled centered scores must fit the exp interval")
        if self.intermediate_iterations is not None:
            _count(self.intermediate_iterations, "intermediate_iterations")
            # With row mass in [.9, 1], sum(square) >= .81/max_length.
            # This bound makes reciprocal relative error <= .1, preserving
            # that invariant for the next round (in exact arithmetic).
            required = math.ceil(math.log2(math.log(10) * 1.01 * self.max_length / .81))
            if not required <= self.intermediate_iterations <= self.reciprocal_iterations:
                raise ValueError("intermediate iterations must preserve row mass and not exceed final iterations")


def chebyshev(x, profile, constant):
    """Balanced T_2m=2*T_m^2-1, T_2m+1=2*T_m*T_m+1-T_1.

    O(degree) ciphertext products, O(log degree) ciphertext product depth.
    Keep the Chebyshev basis: conversion to large raw powers is unstable.
    """
    lo, hi = profile.domain
    z = x * constant(2 / (hi - lo)) + constant(-(hi + lo) / (hi - lo))
    terms = {1: z}

    def term(n):
        if n not in terms:
            half = n // 2
            if n % 2:
                terms[n] = term(half) * term(half + 1) * constant(2) + z * constant(-1)
            else:
                terms[n] = term(half) * term(half) * constant(2) + constant(-1)
        return terms[n]

    def evaluate(coefficients):
        # Splitting at m uses T_(m+j)=2*T_m*T_j-T_(m-j). Public
        # coefficient transformations leave only one ciphertext product
        # per block, while the small T_j terms are shared by all leaves.
        while len(coefficients) > 1 and coefficients[-1] == 0:
            coefficients = coefficients[:-1]
        degree = len(coefficients) - 1
        if degree == 0:
            return constant(coefficients[0])
        if profile.evaluation == "blocked" and degree > profile.block_size:
            split = 1 << (degree.bit_length() - 1)
            low = list(coefficients[:split])
            high = [coefficients[split]] + [2 * c for c in coefficients[split + 1:]]
            for j in range(1, len(high)):
                low[split - j] -= coefficients[split + j]
            return term(split) * evaluate(high) + evaluate(low)
        result = term(1) * constant(coefficients[1]) + constant(coefficients[0])
        for n, coefficient in enumerate(coefficients[2:], 2):
            if coefficient != 0:
                result = result + term(n) * constant(coefficient)
        return result

    result = evaluate(profile.coefficients)
    # The recursive closure forms a Python reference cycle. Release cached
    # GPU tensors now instead of retaining them until cyclic GC runs.
    terms.clear()
    return result


def inverse_sqrt(x, profile, constant):
    """Newton iteration with public upper bound B; y approaches 1/sqrt(x/B)."""
    scaled = x * constant(1 / profile.domain[1])
    if profile.evaluation == "residual":
        residual = scaled * constant(-1) + constant(1)
        y = residual * constant(.5) + constant(1)
        for _ in range(profile.iterations - 1):
            residual = residual * residual * (residual * constant(.25) + constant(.75))
            y = y * (residual * constant(.5) + constant(1))
        # Recompute from the original scaled variance. This corrects the
        # loss of significance in 1-scaled when scaled is small in float32.
        for _ in range(profile.refinement_steps):
            y = y * (scaled * y * y * constant(-.5) + constant(1.5))
        return y * constant(1 / math.sqrt(profile.domain[1]))
    y = scaled * constant(-.5) + constant(1.5)
    for _ in range(profile.iterations - 1):
        y = y * (scaled * y * y * constant(-.5) + constant(1.5))
    return y * constant(1 / math.sqrt(profile.domain[1]))


def reciprocal(x, upper, iterations, constant):
    """Fixed Goldschmidt product; error after k steps is (1-x/upper)^(2^k).

    Valid for 0 < x <= upper. The iteration count is public. Maintaining the
    residual separately shortens the graph compared with recomputing 1-x*y.
    """
    residual = x * constant(-1 / upper) + constant(1)
    result = (residual + constant(1)) * constant(1 / upper)
    for _ in range(iterations - 1):
        residual = residual * residual
        result = result * (residual + constant(1))
    return result


def small_exp(x, degree, constant):
    """Taylor exp: degree 12 on [-.5,.5], or degree 20 on [-2,2]."""
    powers = {1: x}

    def power(n):
        if n not in powers:
            powers[n] = power(n // 2) * power(n - n // 2)
        return powers[n]

    result = x + constant(1)
    for n in range(2, degree + 1):
        result = result + power(n) * constant(1 / math.factorial(n))
    powers.clear()
    return result


def normalize_square(centered, profile, length, constant, total, broadcast, scalar_constant=None):
    """Softmax via small exponential, then repeated square and normalization.

    length is public (or an array of public causal row lengths). total and
    broadcast preserve the backend's scalar-slot convention. No max or clip.
    """
    scalar_constant = constant if scalar_constant is None else scalar_constant
    intermediate = (profile.reciprocal_iterations if profile.intermediate_iterations is None
                    else profile.intermediate_iterations)
    numerator = small_exp(centered * constant(2.0 ** -profile.squarings),
                          profile.exp_degree, constant)
    initial_upper = 2 ** math.ceil(math.log2(math.exp(profile.exp_bound) * 1.01))
    inverse = reciprocal(total(numerator), initial_upper * length,
                         intermediate if profile.squarings else profile.reciprocal_iterations, scalar_constant)
    probability = numerator * broadcast(inverse)
    for step in range(profile.squarings):
        numerator = probability * probability
        # Exact normalized nonnegative probabilities have sum(square) in
        # [1/length, 1]. A small public margin tolerates floating point error.
        inverse = reciprocal(total(numerator), 1.01,
                             profile.reciprocal_iterations if step == profile.squarings - 1 else intermediate,
                             scalar_constant)
        probability = numerator * broadcast(inverse)
    return probability
