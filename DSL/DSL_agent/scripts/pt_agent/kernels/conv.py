"""R4's six-step mux convolution, BN folding, option A shortcut and residual add."""

import numpy as np

from .layout import KernelError, MuxLayout, mux_layout
from .slots import accumulate, sum_slots


def conv2d_mux(b, x, lin: MuxLayout, weight: np.ndarray, *, stride: int,
               scale: np.ndarray | None = None, shift: np.ndarray | None = None):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 4:
        raise KernelError("weight must have shape (Co, Ci, fh, fw)")
    Co, Ci, fh, fw = weight.shape
    if Ci != lin.C or fh != fw or fh % 2 != 1:
        raise KernelError("convolution requires matching input channels and an odd square kernel")
    if type(stride) is not int or stride not in (1, 2) or lin.H % stride or lin.W % stride:
        raise KernelError("stride must be 1 or 2 and divide H and W")
    lout = mux_layout(lin.n, Co, lin.H//stride, lin.W//stride, lin.k*stride)
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float64)
        if scale.shape != (Co,):
            raise KernelError("scale must have length Co")
    if shift is not None:
        shift = np.asarray(shift, dtype=np.float64)
        if shift.shape != (Co,):
            raise KernelError("shift must have length Co")
    n, k, H, W, T, p, block = lin.n, lin.k, lin.H, lin.W, lin.T, lin.p, lin.block
    ko, Ho, Wo, po, blocko = lout.k, lout.H, lout.W, lout.p, lout.block
    center = (fh-1)//2

    # Step 1: emit every spatial translation before inspecting weight vectors.
    translated = [[x.rotate(k*k*W*(i1-center)+k*(i2-center)) for i2 in range(fw)] for i1 in range(fh)]
    grid = lin.index_grid()
    output_grid = lout.index_grid()
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    D = None
    for group in range((Co+p-1)//p):
        # Step 2: only all-zero weight vectors and their empty groups are skipped.
        B = None
        for i1 in range(fh):
            for i2 in range(fw):
                dy, dx = i1-center, i2-center
                valid = (0 <= ys+dy) & (ys+dy < H) & (0 <= xs+dx) & (xs+dx < W)
                u = np.zeros(n, dtype=np.float64)
                active = min(p, Co-p*group)
                values = weight[p*group:p*group+active, :, i1, i2, None, None]
                u[grid[:active]] = values*valid[None, None, :, :]
                if not np.any(u):
                    continue
                B = accumulate(B, translated[i1][i2]*b.const(u))
        if B is None:
            continue

        # Step 3: fold the two interleaved channel axes, then channel groups.
        Cs = sum_slots(b, B, k, 1)
        Cs = sum_slots(b, Cs, k, k*W)
        Cs = sum_slots(b, Cs, T, k*k*H*W)

        # Step 4: project each copy's convolution into output copy zero.
        for j in range(p):
            o = p*group+j
            if o >= Co:
                continue
            to, r = divmod(o, ko*ko)
            ao, bo = divmod(r, ko)
            delta = to*(ko*ko*Ho*Wo)+ao*(ko*Wo)+bo-j*block
            mask = np.zeros(n, dtype=np.float64)
            mask[output_grid[0, o]] = 1.0 if scale is None else scale[o]
            D = accumulate(D, Cs.rotate(-delta)*b.const(mask))

    # Step 5: fill all power-of-two output copies.
    for j in range(po.bit_length()-1):
        D = accumulate(D, D.rotate(-(2**j)*blocko))

    # Step 6: apply the public shift to every output copy.
    if shift is not None and np.any(shift):
        offset = np.zeros(n, dtype=np.float64)
        offset[output_grid] = shift[None, :, None, None]
        D = accumulate(D, b.const(offset))
    return D, lout


def fold_bn(gamma, beta, mean, var, eps, conv_bias=None) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(gamma, dtype=np.float64)/np.sqrt(np.asarray(var, dtype=np.float64)+eps)
    shift = np.asarray(beta, dtype=np.float64)-scale*np.asarray(mean, dtype=np.float64)
    if conv_bias is not None:
        shift = shift+scale*np.asarray(conv_bias, dtype=np.float64)
    return scale, shift


def downsample_pad_mux(b, x, lin: MuxLayout, pad_before: int, pad_after: int):
    Co = lin.C+pad_before+pad_after
    weight = np.zeros((Co, lin.C, 1, 1), dtype=np.float64)
    for c in range(lin.C):
        weight[c+pad_before, c, 0, 0] = 1.0
    return conv2d_mux(b, x, lin, weight, stride=2)


def add_mux(b, x, lx: MuxLayout, y, ly: MuxLayout):
    if lx != ly:
        raise KernelError("residual layouts must match")
    return accumulate(x, y), lx
