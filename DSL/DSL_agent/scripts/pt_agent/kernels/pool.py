"""R5 global spatial mean from mux slots to a single flat copy."""

import numpy as np

from .layout import FlatLayout, KernelError, MuxLayout
from .slots import accumulate, sum_slots


def mean_mux(b, x, lin: MuxLayout):
    if lin.H & (lin.H-1) or lin.W & (lin.W-1):
        raise KernelError("mean_mux requires power-of-two H and W")
    B = sum_slots(b, x, lin.W, lin.k)
    B = sum_slots(b, B, lin.H, lin.k*lin.k*lin.W)
    out = None
    for t in range(lin.T):
        for a in range(lin.k):
            src = t*lin.k*lin.k*lin.H*lin.W+a*lin.k*lin.W
            dst = t*lin.k*lin.k+a*lin.k
            mask = np.zeros(lin.n, dtype=np.float64)
            indices = np.arange(dst, min(dst+lin.k, lin.C))
            mask[indices] = 1/(lin.H*lin.W)
            if not np.any(mask):
                continue
            out = accumulate(out, B.rotate(src-dst)*b.const(mask))
    return out, FlatLayout(lin.n, lin.C)
