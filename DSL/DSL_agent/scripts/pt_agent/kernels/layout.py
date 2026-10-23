"""Single-ciphertext mux slot layout from the R4 formulas."""

from dataclasses import dataclass

import numpy as np


class KernelError(Exception):
    pass


@dataclass(frozen=True)
class MuxLayout:
    n: int
    C: int
    H: int
    W: int
    k: int

    @property
    def T(self):
        return (self.C+self.k*self.k-1)//(self.k*self.k)

    @property
    def used(self):
        return self.T*self.k*self.k*self.H*self.W

    @property
    def p(self):
        return 1 << ((self.n//self.used).bit_length()-1)

    @property
    def block(self):
        return self.n//self.p

    def slot(self, c: int, y: int, x: int, copy: int = 0) -> int:
        t, r = divmod(c, self.k*self.k)
        a, b = divmod(r, self.k)
        return copy*self.block+t*(self.k*self.k*self.H*self.W)+(self.k*y+a)*(self.k*self.W)+(self.k*x+b)

    def index_grid(self) -> np.ndarray:
        copy = np.arange(self.p, dtype=np.int64)[:, None, None, None]
        c = np.arange(self.C, dtype=np.int64)[None, :, None, None]
        y = np.arange(self.H, dtype=np.int64)[None, None, :, None]
        x = np.arange(self.W, dtype=np.int64)[None, None, None, :]
        t, r = np.divmod(c, self.k*self.k)
        a, b = np.divmod(r, self.k)
        return copy*self.block+t*(self.k*self.k*self.H*self.W)+(self.k*y+a)*(self.k*self.W)+(self.k*x+b)


@dataclass(frozen=True)
class FlatLayout:
    """Flat values occupy [:size]; every slot in [size:n] must remain zero."""

    n: int
    size: int


def mux_layout(n, C, H, W, k) -> MuxLayout:
    for name, value in (("n", n), ("k", k)):
        if type(value) is not int or value < 1 or value & (value-1):
            raise KernelError(f"{name} must be a positive power of two")
    if any(type(value) is not int or value < 1 for value in (C, H, W)):
        raise KernelError("C, H and W must be positive integers")
    layout = MuxLayout(n, C, H, W, k)
    if layout.used > n:
        raise KernelError("tensor does not fit in a single ciphertext")
    return layout


def channel_slots(lay: MuxLayout, c: int) -> np.ndarray:
    return lay.index_grid()[0, c].reshape(-1)


def pack_mux(x: np.ndarray, lay: MuxLayout) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (lay.C, lay.H, lay.W):
        raise KernelError("input shape differs from mux layout")
    result = np.zeros(lay.n, dtype=np.float64)
    result[lay.index_grid()] = x[None, ...]
    return result


def unpack_mux(v: np.ndarray, lay: MuxLayout, copy: int = 0) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (lay.n,) or not 0 <= copy < lay.p:
        raise KernelError("slot vector or copy index differs from mux layout")
    return v[lay.index_grid()[copy]]


def pack_flat(v: np.ndarray, lay: FlatLayout) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (lay.size,) or not 1 <= lay.size <= lay.n:
        raise KernelError("input shape differs from flat layout")
    out = np.zeros(lay.n, dtype=np.float64)
    out[:lay.size] = v
    return out


def unpack_flat(x: np.ndarray, lay: FlatLayout) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != lay.n or not 1 <= lay.size <= lay.n:
        raise KernelError("slot array differs from flat layout")
    return x[..., :lay.size]
