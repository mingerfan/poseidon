"""R5 padded flat linear transform with the prescribed diagonal or BSGS schedule."""

import numpy as np

from .layout import FlatLayout, KernelError
from .slots import accumulate, sum_slots


def linear_flat(b, x, lin: FlatLayout, weight: np.ndarray, bias: np.ndarray | None, *, algo="bsgs", n1=None):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2 or min(weight.shape) < 1 or weight.shape[1] != lin.size:
        raise KernelError("linear weight must have shape (O, lin.size)")
    O, I = weight.shape
    Ip, Op = 1 << (I-1).bit_length(), 1 << (O-1).bit_length()
    if Op > Ip or 2*Ip > lin.n:
        raise KernelError("linear requires Op <= Ip and 2*Ip <= n")
    if algo not in ("diag", "bsgs"):
        raise KernelError("linear algo must be diag or bsgs")
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (O,):
            raise KernelError("linear bias must have length O")
    Wp = np.zeros((Op, Ip), dtype=np.float64)
    Wp[:O, :I] = weight
    xr = x+x.rotate(-Ip)
    slots = np.arange(Ip)

    def diagonal(d):
        out = np.zeros(lin.n, dtype=np.float64)
        out[:Ip] = Wp[slots % Op, (slots+d) % Ip]
        return out

    z = None
    if algo == "diag":
        for d in range(Op):
            u = diagonal(d)
            if np.any(u):
                z = accumulate(z, xr.rotate(d)*b.const(u))
    else:
        g = n1 if n1 is not None else 2**((Op.bit_length()-1+1)//2)
        if type(g) is not int or g < 1 or g & (g-1) or g > Op:
            raise KernelError("BSGS n1 must be a power of two <= Op")
        R = [xr.rotate(i) for i in range(g)]
        for j in range((Op+g-1)//g):
            inner = None
            for i in range(g):
                d = g*j+i
                if d < Op:
                    u = np.roll(diagonal(d), g*j)
                    if np.any(u):
                        inner = accumulate(inner, R[i]*b.const(u))
            if inner is not None:
                z = accumulate(z, inner.rotate(g*j))
    y = sum_slots(b, z, Ip//Op, Op)
    mask = np.zeros(lin.n, dtype=np.float64)
    mask[:O] = 1
    y = y*b.const(mask)
    if bias is not None:
        offset = np.zeros(lin.n, dtype=np.float64)
        offset[:O] = bias
        y = y+b.const(offset)
    return y, FlatLayout(lin.n, O)
