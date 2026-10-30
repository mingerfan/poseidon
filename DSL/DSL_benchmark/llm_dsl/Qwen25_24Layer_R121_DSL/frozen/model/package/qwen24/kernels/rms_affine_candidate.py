"""Equivalent RMS layouts for compiler selection; no coefficient refitting."""
import math
import numpy as np
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import ops, qwen_nonlinear as nonlinear
from qwen24.kernels.expr import Plain
from qwen24.kernels import zero_tail_contract as tail

STRATEGIES = ('mean_end', 'power2', 'power6', 'prepower2', 'prepower3', 'prepower4', 'prepower6', 'prepower13')
CURRENT = 'mean_end'


def select(strategy):
    global CURRENT
    if strategy not in STRATEGIES:
        raise ValueError('Unknown public RMS optimization strategy')
    CURRENT = strategy


def rms_norm(x, weight, *, rsqrt, eps=1e-6, strategy=None):
    strategy = CURRENT if strategy is None else strategy
    if strategy not in STRATEGIES: raise ValueError('Unknown public RMS strategy')
    weight = ops._real_array(weight, 'RMSNorm weight')
    if weight.shape != (x.size,) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('RMSNorm requires weight [hidden] and finite eps>0')
    if getattr(rsqrt, 'function', None) != 'rsqrt':
        raise ValueError('RMSNorm requires an rsqrt approximation')
    cheb = isinstance(rsqrt, nonlinear.ChebyshevApproximation)
    affine = False
    if affine:
        lo, hi = rsqrt.domain
        factor = 2.0 / (hi - lo)
        shift = factor * eps - (hi + lo) / (hi - lo)
    else: factor, shift = 1.0, eps
    mask = np.zeros(x.slots)
    if strategy.startswith('prepower'):
        mask[:x.size] = math.sqrt(factor / x.size)
        scaled = ops.mul(x.expr, Plain(mask))
        reduced = ops.mul(scaled, scaled)
    else:
        mask[:x.size] = factor / x.size
        reduced = ops.mul(ops.mul(x.expr, x.expr), Plain(mask))
    step = 1
    while step < x.slots:
        reduced = ops.add(reduced, reduced.rotate(step)); step *= 2
    constant = lambda value: Plain(np.full(x.slots, value))
    argument = ops.add(reduced, constant(shift))
    if affine:
        inverse = nonlinear.chebyshev_unit(argument, rsqrt, constant)
    elif cheb:
        if 'power' in strategy:
            from qwen24.kernels import rms_power_polynomial
            inverse = rms_power_polynomial.evaluate(argument, rsqrt, constant, int(strategy.split('power')[1]))
        else:
            inverse = nonlinear.chebyshev(argument, rsqrt, constant)
    elif isinstance(rsqrt, ql.InverseSqrtApproximation):
        inverse = nonlinear.inverse_sqrt(argument, rsqrt, constant)
    else:
        inverse = ql.approximate(ql.PackedVector(argument, x.slots, x.slots),
                                 rsqrt, function='rsqrt').expr
    if strategy.endswith('_early'):
        result = ops.mul(ops.mul(x.expr, Plain(weight)), inverse)
    else:
        result = ops.mul(ops.mul(x.expr, inverse), Plain(weight))
    return tail._proven_prefix(ql.PackedVector(result, x.size, x.slots))


def install(strategy=None):
    if strategy is not None: select(strategy)
    ql.rms_norm = rms_norm
