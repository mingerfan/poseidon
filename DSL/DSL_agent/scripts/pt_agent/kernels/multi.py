"""Channel/length tiling using only the existing backend arithmetic interface."""

import numpy as np

from .layout import (FlatLayout, KernelError, MuxLayout, MultiFlatLayout, MultiMuxLayout,
                     layout_blocks, multi_flat_layout, multi_mux_layout)
from .conv import conv2d_mux
from .pool import mean_mux
from .linear import linear_flat
from .slots import accumulate, sum_slots


MULTI = (MultiMuxLayout, MultiFlatLayout)


def values_list(value, layout):
    return value if isinstance(layout, MULTI) else [value]


def _result(values, layout):
    return (values if isinstance(layout, MULTI) else values[0]), layout


def act_blocks(b, value, layout, apply):
    return _result([apply(item, block) for item, (_, _, block) in
                    zip(values_list(value, layout), layout_blocks(layout))], layout)


def add_blocks(b, x, lx, y, ly):
    if lx != ly:
        raise KernelError("residual layouts must match")
    return _result([left+right for left, right in
                    zip(values_list(x, lx), values_list(y, ly))], lx)


def conv2d_blocks(b, x, lin, weight, *, stride, scale=None, shift=None):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 4:
        raise KernelError("weight must have shape (Co, Ci, fh, fw)")
    Co, Ci, fh, fw = weight.shape
    if Ci != lin.C or fh != fw or fh % 2 != 1:
        raise KernelError("convolution requires matching input channels and an odd square kernel")
    if type(stride) is not int or stride not in (1, 2) or lin.H % stride or lin.W % stride:
        raise KernelError("stride must be 1 or 2 and divide H and W")
    lout = multi_mux_layout(lin.n, Co, lin.H//stride, lin.W//stride, lin.k*stride)
    if not isinstance(lin, MULTI) and not isinstance(lout, MULTI):
        return conv2d_mux(b, x, lin, weight, stride=stride, scale=scale, shift=shift)
    for name, vector in (('scale', scale), ('shift', shift)):
        if vector is not None and np.asarray(vector).shape != (Co,):
            raise KernelError(f"{name} must have length Co")
    n, k, H, W = lin.n, lin.k, lin.H, lin.W
    center = (fh-1)//2
    inputs = list(zip(values_list(x, lin), layout_blocks(lin)))
    # Step 1 is owned by the input block, independent of output projection.
    translated = [[[value.rotate(k*k*W*(iy-center)+k*(ix-center))
                    for ix in range(fw)] for iy in range(fh)] for value, _ in inputs]
    ys, xs = np.arange(H)[:, None], np.arange(W)[None, :]
    outputs = []
    for out_start, out_end, lo in layout_blocks(lout):
        D = None
        output_grid = lo.index_grid()
        for block_index, (_, (in_start, in_end, li)) in enumerate(inputs):
            p, grid = li.p, li.index_grid()
            for group in range((lo.C+p-1)//p):
                B = None
                active = min(p, lo.C-p*group)
                for iy in range(fh):
                    for ix in range(fw):
                        dy, dx = iy-center, ix-center
                        valid = (0 <= ys+dy) & (ys+dy < H) & (0 <= xs+dx) & (xs+dx < W)
                        u = np.zeros(n, dtype=np.float64)
                        w = weight[out_start+p*group:out_start+p*group+active,
                                   in_start:in_end, iy, ix, None, None]
                        u[grid[:active]] = w*valid[None, None, :, :]
                        if np.any(u):
                            B = accumulate(B, translated[block_index][iy][ix]*b.const(u))
                if B is None:
                    continue
                Cs = sum_slots(b, B, k, 1)
                Cs = sum_slots(b, Cs, k, k*W)
                Cs = sum_slots(b, Cs, li.T, k*k*H*W)
                for copy in range(active):
                    o = p*group+copy
                    t, r = divmod(o, lo.k*lo.k)
                    a, c = divmod(r, lo.k)
                    delta = t*(lo.k*lo.k*lo.H*lo.W)+a*(lo.k*lo.W)+c-copy*li.block
                    mask = np.zeros(n, dtype=np.float64)
                    mask[output_grid[0, o]] = 1.0 if scale is None else scale[out_start+o]
                    D = accumulate(D, Cs.rotate(-delta)*b.const(mask))
        if D is None:
            D = inputs[0][0]*b.const(np.zeros(n))
        # Fill and bias only after all input-block contributions have accumulated.
        for copy in range(lo.p.bit_length()-1):
            D = D+D.rotate(-(2**copy)*lo.block)
        if shift is not None and np.any(shift[out_start:out_end]):
            offset = np.zeros(n, dtype=np.float64)
            offset[output_grid] = np.asarray(shift)[None, out_start:out_end, None, None]
            D = D+b.const(offset)
        outputs.append(D)
    return _result(outputs, lout)


def downsample_blocks(b, x, lin, before, after):
    weight = np.zeros((lin.C+before+after, lin.C, 1, 1), dtype=np.float64)
    for c in range(lin.C):
        weight[c+before, c, 0, 0] = 1.0
    return conv2d_blocks(b, x, lin, weight, stride=2)


def _merge_flat(b, pieces, n, size):
    lout = multi_flat_layout(n, size)
    outputs = []
    for start, end, _ in layout_blocks(lout):
        value = None
        for source, source_start, source_end in pieces:
            lo, hi = max(start, source_start), min(end, source_end)
            if lo >= hi:
                continue
            term = source
            if lo != source_start or hi != source_end:
                mask = np.zeros(n)
                mask[lo-source_start:hi-source_start] = 1
                term = term*b.const(mask)
            term = term.rotate(start-source_start)
            value = accumulate(value, term)
        outputs.append(value)
    return _result(outputs, lout)


def mean_blocks(b, x, lin):
    pieces = [(mean_mux(b, value, block)[0], start, end)
              for value, (start, end, block) in zip(values_list(x, lin), layout_blocks(lin))]
    return _merge_flat(b, pieces, lin.n, lin.C)


def relayout_blocks(b, x, lin):
    if not isinstance(lin, (MuxLayout, MultiMuxLayout)) or lin.H != 1 or lin.W != 1:
        raise KernelError("multi mux to flat relayout requires H = W = 1; spatial tiling not supported")
    return mean_blocks(b, x, lin)


def linear_blocks(b, x, lin, weight, bias, *, algo='bsgs', n1=None):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2 or min(weight.shape) < 1 or weight.shape[1] != lin.size:
        raise KernelError("linear weight must have shape (O, lin.size)")
    O = weight.shape[0]
    lout = multi_flat_layout(lin.n, O)
    if not isinstance(lin, MULTI) and not isinstance(lout, MULTI):
        return linear_flat(b, x, lin, weight, bias, algo=algo, n1=n1)
    if algo not in ('diag', 'bsgs'):
        raise KernelError("linear algo must be diag or bsgs")
    if bias is not None and np.asarray(bias).shape != (O,):
        raise KernelError("linear bias must have length O")
    inputs = list(zip(values_list(x, lin), layout_blocks(lin)))
    # Cache each translated input for all output blocks with the same square size.
    replicas, rotations = {}, {}
    outputs = []
    for out_start, out_end, lo in layout_blocks(lout):
        y = None
        for index, (value, (in_start, in_end, li)) in enumerate(inputs):
            side = 1 << (max(li.size, lo.size)-1).bit_length()
            if 2*side > lin.n:
                raise KernelError("multi linear requires 2*square side <= n")
            key = index, side
            if key not in replicas:
                replicas[key] = value+value.rotate(-side)
            xr = replicas[key]
            matrix = np.zeros((side, side))
            matrix[:lo.size, :li.size] = weight[out_start:out_end, in_start:in_end]
            slots = np.arange(side)

            def diagonal(d):
                u = np.zeros(lin.n)
                u[:side] = matrix[slots, (slots+d) % side]
                return u

            def rotate_input(step):
                rotation_key = (*key, step)
                if rotation_key not in rotations:
                    rotations[rotation_key] = xr.rotate(step)
                return rotations[rotation_key]

            if algo == 'diag':
                for d in range(side):
                    u = diagonal(d)
                    if np.any(u):
                        y = accumulate(y, rotate_input(d)*b.const(u))
            else:
                g = n1 if n1 is not None else 2**((side.bit_length()-1+1)//2)
                if type(g) is not int or g < 1 or g & (g-1) or g > side:
                    raise KernelError("BSGS n1 must be a power of two <= Op")
                R = [rotate_input(i) for i in range(g)]
                for j in range((side+g-1)//g):
                    inner = None
                    for i in range(g):
                        d = g*j+i
                        if d < side:
                            u = np.roll(diagonal(d), g*j)
                            if np.any(u):
                                inner = accumulate(inner, R[i]*b.const(u))
                    if inner is not None:
                        y = accumulate(y, inner.rotate(g*j))
        if y is None:
            y = inputs[0][0]*b.const(np.zeros(lin.n))
        mask = np.zeros(lin.n)
        mask[:lo.size] = 1
        y = y*b.const(mask)
        if bias is not None:
            offset = np.zeros(lin.n)
            offset[:lo.size] = bias[out_start:out_end]
            y = y+b.const(offset)
        outputs.append(y)
    return _result(outputs, lout)
