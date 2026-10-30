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


@dataclass(frozen=True)
class MultiMuxLayout:
    """Consecutive channel intervals, each with its own mux replication."""

    n: int
    C: int
    H: int
    W: int
    k: int

    @property
    def blocks(self):
        capacity = self.n//(self.k*self.k*self.H*self.W)*self.k*self.k
        if not capacity:
            raise KernelError("spatial tiling not supported: k²·H·W > n")
        return tuple((start, min(start+capacity, self.C),
                      mux_layout(self.n, min(capacity, self.C-start), self.H, self.W, self.k))
                     for start in range(0, self.C, capacity))

    def index_grid(self):
        return [lay.index_grid() for _, _, lay in self.blocks]


@dataclass(frozen=True)
class ConcatMuxLayout(MultiMuxLayout):
    """An ordered concatenation of valid, possibly unequal channel blocks."""

    partition_sizes: tuple[int, ...]

    @property
    def blocks(self):
        if not self.partition_sizes or sum(self.partition_sizes) != self.C:
            raise KernelError("concat partitions must sum to C")
        result, start = [], 0
        for size in self.partition_sizes:
            block = mux_layout(self.n, size, self.H, self.W, self.k)
            result.append((start, start+size, block))
            start += size
        return tuple(result)


@dataclass(frozen=True)
class MultiFlatLayout:
    n: int
    size: int

    @property
    def blocks(self):
        capacity = self.n//2
        return tuple((start, min(start+capacity, self.size),
                      FlatLayout(self.n, min(capacity, self.size-start)))
                     for start in range(0, self.size, capacity))

    def index_grid(self):
        return [np.arange(lay.size) for _, _, lay in self.blocks]


def multi_mux_layout(n, C, H, W, k):
    # Validate through the original factory without changing its behavior.
    mux_layout(n, 1, 1, 1, 1)
    if type(k) is not int or k < 1 or k & (k-1):
        raise KernelError("k must be a positive power of two")
    if any(type(value) is not int or value < 1 for value in (C, H, W)):
        raise KernelError("C, H and W must be positive integers")
    groups = n//(k*k*H*W)
    if not groups:
        raise KernelError("spatial tiling not supported: k²·H·W > n")
    return (mux_layout(n, C, H, W, k) if C <= groups*k*k
            else MultiMuxLayout(n, C, H, W, k))


def multi_flat_layout(n, size):
    mux_layout(n, 1, 1, 1, 1)
    if type(size) is not int or size < 1:
        raise KernelError("flat size must be a positive integer")
    return FlatLayout(n, size) if size <= n//2 else MultiFlatLayout(n, size)


def layout_blocks(lay):
    if isinstance(lay, (MultiMuxLayout, MultiFlatLayout, TokenLayout, ScoreLayout)):
        return lay.blocks
    return ((0, lay.C if isinstance(lay, MuxLayout) else lay.size, lay),)


def pack_multi(x, lay):
    x = np.asarray(x, dtype=np.float64)
    expected = (lay.C, lay.H, lay.W) if isinstance(lay, MultiMuxLayout) else (lay.size,)
    if x.shape != expected:
        raise KernelError("input shape differs from multi layout")
    pack = pack_mux if isinstance(lay, MultiMuxLayout) else pack_flat
    return [pack(x[start:end], block) for start, end, block in lay.blocks]


def unpack_multi(values, lay):
    if len(values) != len(lay.blocks):
        raise KernelError("ciphertext count differs from multi layout")
    decoded = []
    for value, (_, _, block) in zip(values, lay.blocks):
        value = np.asarray(value, dtype=np.float64)
        if isinstance(block, MuxLayout):
            decoded.append(unpack_mux(value, block) if value.ndim == 1 else
                           np.stack([unpack_mux(sample, block) for sample in value]))
        else:
            decoded.append(unpack_flat(value, block))
    axis = -3 if isinstance(lay, MultiMuxLayout) else -1
    return np.concatenate(decoded, axis=axis)


def layout_record(lay):
    from dataclasses import asdict

    kind = ('token' if isinstance(lay, TokenLayout) else
            'score' if isinstance(lay, ScoreLayout) else
            'multi_mux' if isinstance(lay, MultiMuxLayout) else
            'multi_flat' if isinstance(lay, MultiFlatLayout) else
            'mux' if isinstance(lay, MuxLayout) else 'flat')
    record = dict(kind=kind, **asdict(lay))
    if isinstance(lay, (MultiMuxLayout, MultiFlatLayout, TokenLayout, ScoreLayout)):
        record['blocks'] = [dict(start=start, end=end, layout=layout_record(block))
                            for start, end, block in lay.blocks]
    return record


def layout_from_record(record):
    fields = {key: value for key, value in record.items() if key not in ('kind', 'blocks')}
    if record['kind'] == 'multi_mux' and 'partition_sizes' in fields:
        fields['partition_sizes'] = tuple(fields['partition_sizes'])
        return ConcatMuxLayout(**fields)
    return {'mux': MuxLayout, 'flat': FlatLayout, 'token': TokenLayout, 'score': ScoreLayout,
            'multi_mux': MultiMuxLayout, 'multi_flat': MultiFlatLayout}[record['kind']](**fields)


@dataclass(frozen=True)
class TokenLayout:
    """Each token owns consecutive flat blocks, each of capacity n/2."""

    n: int
    L: int
    D: int

    def __post_init__(self):
        if self.n < 2 or self.n & (self.n-1) or min(self.L, self.D) < 1:
            raise KernelError('invalid TokenLayout dimensions')

    @property
    def token(self):
        return multi_flat_layout(self.n, self.D)

    @property
    def blocks(self):
        return tuple((t*self.D+start, t*self.D+end, block)
                     for t in range(self.L) for start, end, block in layout_blocks(self.token))


@dataclass(frozen=True)
class ScoreLayout:
    """One query ciphertext, with consecutive head/key intervals."""

    n: int
    L: int
    H: int
    Lk: int

    def __post_init__(self):
        if (self.n < 2 or self.n & (self.n-1) or min(self.L, self.H, self.Lk) < 1
                or self.H*self.Lk > self.n):
            raise KernelError('attention heads/keys do not fit ScoreLayout')

    @property
    def blocks(self):
        size = self.H*self.Lk
        return tuple((t*size, (t+1)*size, FlatLayout(self.n, size)) for t in range(self.L))


def pack_sequence(value, layout):
    value = np.asarray(value, dtype=np.float64)
    if isinstance(layout, TokenLayout):
        if value.shape != (layout.L, layout.D):
            raise KernelError('input shape differs from TokenLayout')
        return [pack_flat(value[t, start:end], block) for t in range(layout.L)
                for start, end, block in layout_blocks(layout.token)]
    if value.shape != (layout.H, layout.L, layout.Lk):
        raise KernelError('input shape differs from ScoreLayout')
    return [pack_flat(value[:, t, :].reshape(-1), FlatLayout(layout.n, layout.H*layout.Lk))
            for t in range(layout.L)]


def unpack_sequence(values, layout):
    if len(values) != len(layout.blocks):
        raise KernelError('ciphertext count differs from sequence layout')
    if isinstance(layout, TokenLayout):
        count = len(layout_blocks(layout.token))
        return np.stack([np.concatenate([np.asarray(values[t*count+i])[..., :block.size]
                        for i, (_, _, block) in enumerate(layout_blocks(layout.token))], axis=-1)
                        for t in range(layout.L)], axis=-2)
    return np.stack([np.asarray(value)[..., :layout.H*layout.Lk].reshape(
                     np.asarray(value).shape[:-1]+(layout.H, layout.Lk))
                     for value in values], axis=-2)
