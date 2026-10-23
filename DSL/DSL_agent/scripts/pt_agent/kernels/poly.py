"""R5 Chebyshev interpolation and its specified small-step/giant-step evaluation.

Exact GELU uses math.erf through np.vectorize, as in R3's NumPy analyzer.
"""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .layout import KernelError
from .slots import accumulate
from ..paths import DACAPO_ROOT


MINIMAX_DATA = DACAPO_ROOT / "python/poly/poly/data/sgn151527.txt"
MINIMAX_SETS = {(15, 15, 27): ((0, 16, 2.0), (16, 32, 1.7), (32, 60, 2.0))}


@lru_cache()
def minimax_coeffs(degrees: tuple) -> list[np.ndarray]:
    if degrees not in MINIMAX_SETS:
        raise KernelError(f"minimax_comp supports degrees {sorted(MINIMAX_SETS)}")
    values = np.asarray([float(line.strip()) for line in MINIMAX_DATA.read_text().splitlines()])
    return [values[start:end]/divisor for start, end, divisor in MINIMAX_SETS[degrees]]


def act_minimax(b, x, degrees: tuple, domain: tuple[float, float]):
    lo, hi = domain
    if not (hi > 0 and lo == -hi):
        raise KernelError("minimax_comp requires a symmetric domain (-B, B)")
    s = x*(1.0/hi)
    for c in minimax_coeffs(degrees):
        s = cheb_eval(b, s, c)
    return x*(s+0.5)


def _gelu(x):
    return 0.5*x*(1+np.vectorize(math.erf, otypes=[np.float64])(x/math.sqrt(2)))


EXACT = {
    "relu": lambda x: np.maximum(x, 0),
    "silu": lambda x: x/(1+np.exp(-x)),
    "gelu": _gelu,
    "sigmoid": lambda x: 1/(1+np.exp(-x)),
    "tanh": np.tanh,
    "exp": np.exp,
}


def cheb_fit(fn: str, deg: int, lo: float, hi: float) -> np.ndarray:
    if fn not in EXACT or type(deg) is not int or not 1 <= deg <= 255 or not lo < hi:
        raise KernelError("Chebyshev fit requires a supported function, degree 1..255 and lo < hi")
    return np.polynomial.chebyshev.chebinterpolate(lambda u: EXACT[fn](((hi-lo)*u+(hi+lo))/2), deg)


def cheb_eval(b, u, coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim != 1 or len(coeffs) < 2:
        raise KernelError("Chebyshev evaluation requires degree >= 1")
    d = len(coeffs)-1
    l = max(1, (d+1).bit_length()//2)
    g = 2**l
    T = {1: u}
    for j in range(2, g+1):
        if j % 2 == 0:
            s = T[j//2]*T[j//2]
            T[j] = s+s-1.0
        else:
            h = j//2
            q = T[h]*T[h+1]
            T[j] = q+q-T[1]
    m = 0
    while g*2**m <= d:
        m += 1
    G = {g: T[g]}
    for i in range(1, m):
        prev = G[g*2**(i-1)]
        s = prev*prev
        G[g*2**i] = s+s-1.0

    def base(c):
        out = None
        for j in range(1, len(c)):
            if c[j] != 0:
                out = accumulate(out, T[j]*float(c[j]))
        c0 = float(c[0]) if len(c) else 0.0
        if out is None:
            return c0
        return out+c0 if c0 != 0.0 else out

    def rec(c, level):
        c = np.trim_zeros(np.asarray(c, float), 'b')
        if len(c) <= g:
            return base(c)
        k = g*2**level
        while k >= len(c):
            level -= 1
            k = g*2**level
        q, r = np.polynomial.chebyshev.chebdiv(c, [0]*k+[1])
        hq = rec(q, level-1)
        hr = rec(r, level-1)
        t = G[k]*hq if isinstance(hq, float) else hq*G[k]
        return t+hr if not isinstance(hr, float) or hr != 0.0 else t

    result = rec(coeffs, m-1)
    if isinstance(result, float):
        raise KernelError("Chebyshev coefficients reduce to a constant")
    return result


def act_cheb(b, x, fn: str, deg: int, domain: tuple[float, float]):
    lo, hi = domain
    coeffs = cheb_fit(fn, deg, lo, hi)
    a = 2/(hi-lo)
    c = -(hi+lo)/(hi-lo)
    u = x*a+c
    return cheb_eval(b, u, coeffs)
