"""CNN arithmetic over mux channel blocks, using the public backend interface."""

from dataclasses import dataclass

import numpy as np

from .layout import (ConcatMuxLayout, FlatLayout, KernelError, MuxLayout, MultiMuxLayout,
                     layout_blocks, multi_flat_layout, mux_layout)
from .multi import MULTI, _result, values_list
from .slots import accumulate, sum_slots


@dataclass(frozen=True)
class DifferenceLayout:
    """An observer record holding extrema of the valid internal comparisons."""


class DifferenceStats:
    def __init__(self):
        self.minimum = self.maximum = None

    def add(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.size:
            low, high = float(data.min()), float(data.max())
            self.minimum = low if self.minimum is None else min(self.minimum, low)
            self.maximum = high if self.maximum is None else max(self.maximum, high)

    @property
    def value(self):
        return np.asarray([self.minimum, self.maximum], dtype=np.float64)


def concat_blocks(values, layouts):
    first = layouts[0]
    if not all(isinstance(lay, (MuxLayout, MultiMuxLayout)) and
               (lay.n, lay.H, lay.W, lay.k) == (first.n, first.H, first.W, first.k)
               for lay in layouts):
        raise KernelError('concat requires mux blocks with equal n, H, W and k')
    blocks = [block for lay in layouts for _, _, block in layout_blocks(lay)]
    out = ConcatMuxLayout(first.n, sum(lay.C for lay in layouts), first.H, first.W,
                          first.k, tuple(block.C for block in blocks))
    return [item for value, lay in zip(values, layouts) for item in values_list(value, lay)], out


def pool_shape(lay, kernel, stride, pad):
    pair = lambda value: (value, value) if type(value) is int else tuple(value)
    kernel, stride, pad = pair(kernel), pair(stride), pair(pad)
    if (kernel[0] != kernel[1] or stride[0] != stride[1] or pad[0] != pad[1]
            or kernel[0] < 1 or stride[0] not in (1, 2, kernel[0])
            or not 0 <= pad[0] <= kernel[0]//2):
        raise KernelError('pool requires a square kernel, stride 1 or 2 and valid symmetric padding')
    k, s, p = kernel[0], stride[0], pad[0]
    h, w = (lay.H+2*p-k)//s+1, (lay.W+2*p-k)//s+1
    if min(h, w) < 1:
        raise KernelError('pool output is empty')
    return k, s, p, h, w


def projected_layout(lin, height, width, stride):
    gap = lin.k*(stride if stride & (stride-1) == 0 else 1)
    blocks = [mux_layout(lin.n, block.C, height, width, gap)
              for _, _, block in layout_blocks(lin)]
    if not isinstance(lin, MultiMuxLayout):
        return blocks[0]
    if stride == 1 and (height, width) == (lin.H, lin.W):
        return lin
    return ConcatMuxLayout(lin.n, lin.C, height, width, gap,
                           tuple(block.C for block in blocks))


def project(b, value, lin, lout, stride):
    if lin == lout and stride == 1:
        return value
    # At fixed channel, strides alter the base offset and leave the sampled
    # pixel spacing intact for the usual halving case. Group equal offsets.
    ys = np.minimum(np.arange(lout.H)*stride, lin.H-1)
    xs = np.minimum(np.arange(lout.W)*stride, lin.W-1)
    source = lin.index_grid()[0][:, ys][:, :, xs]
    target = lout.index_grid()[0]
    differences = source-target
    result = None
    for delta in np.unique(differences):
        mask = np.zeros(lin.n)
        mask[target[differences == delta]] = 1
        result = accumulate(result, value.rotate(int(delta))*b.const(mask))
    for copy in range(lout.p.bit_length()-1):
        result = result+result.rotate(-(2**copy)*lout.block)
    return result


def depthwise_blocks(b, x, lin, weight, *, stride=1, pad=None, scale=None, shift=None):
    weight = np.asarray(weight, dtype=np.float64)
    if (not isinstance(lin, (MuxLayout, MultiMuxLayout)) or weight.ndim != 4
            or weight.shape[:2] != (lin.C, 1) or weight.shape[2] != weight.shape[3]):
        raise KernelError('depthwise requires weight (C,1,k,k) and a mux input')
    kernel = weight.shape[2]
    pad = (kernel-1)//2 if pad is None else pad
    _, stride, pad, height, width = pool_shape(lin, kernel, stride, pad)
    for vector in (scale, shift):
        if vector is not None and np.asarray(vector).shape != (lin.C,):
            raise KernelError('depthwise scale and shift must have length C')
    lout = projected_layout(lin, height, width, stride)
    ys, xs = np.arange(lin.H)[:, None], np.arange(lin.W)[None, :]
    outputs = []
    for value, (start, end, li), (_, _, lo) in zip(values_list(x, lin), layout_blocks(lin), layout_blocks(lout)):
        result, grid = None, li.index_grid()
        for iy in range(kernel):
            for ix in range(kernel):
                dy, dx = iy-pad, ix-pad
                valid = (ys+dy >= 0) & (ys+dy < lin.H) & (xs+dx >= 0) & (xs+dx < lin.W)
                if (height-1)*stride >= lin.H and dy != 0:
                    valid = valid & (ys != lin.H-1)
                if (width-1)*stride >= lin.W and dx != 0:
                    valid = valid & (xs != lin.W-1)
                u = np.zeros(lin.n)
                weights = weight[start:end, 0, iy, ix]
                if scale is not None:
                    weights = weights*np.asarray(scale)[start:end]
                u[grid] = weights[None, :, None, None]*valid[None, None]
                if np.any(u):
                    term = value.rotate(li.k*li.k*li.W*dy+li.k*dx)*b.const(u)
                    result = accumulate(result, term)
        if result is None:
            result = value*b.const(np.zeros(lin.n))
        result = project(b, result, li, lo, stride)
        if shift is not None and np.any(np.asarray(shift)[start:end]):
            offset = np.zeros(lin.n)
            offset[lo.index_grid()] = np.asarray(shift)[None, start:end, None, None]
            result = result+b.const(offset)
        outputs.append(result)
    return _result(outputs, lout)


def avgpool_blocks(b, x, lin, *, kernel, stride, pad):
    k, s, p, _, _ = pool_shape(lin, kernel, stride, pad)
    return depthwise_blocks(b, x, lin, np.full((lin.C, 1, k, k), 1/(k*k)), stride=s, pad=p)


def maxpool_blocks(b, x, lin, *, kernel, stride, pad, apply, observer=None):
    k, s, p, height, width = pool_shape(lin, kernel, stride, pad)
    lout = projected_layout(lin, height, width, s)
    offsets = [i-p for i in range(k) if i != p]
    outputs = []
    for value, (_, _, li), (_, _, lo) in zip(values_list(x, lin), layout_blocks(lin), layout_blocks(lout)):
        result = value
        for axis in (3, 2):
            source = result
            for offset in offsets:
                positions = np.arange(li.W if axis == 3 else li.H)
                valid_axis = (positions+offset >= 0) & (positions+offset < len(positions))
                target_length = width if axis == 3 else height
                if (target_length-1)*s >= len(positions):
                    valid_axis[-1] = False
                valid = np.broadcast_to(valid_axis[None, :] if axis == 3 else valid_axis[:, None], (li.H, li.W))
                difference = source.rotate(offset*(li.k if axis == 3 else li.k*li.k*li.W))-result
                if observer is not None:
                    observer(difference, li, valid)
                mask = np.zeros(li.n)
                mask[li.index_grid()] = valid[None, None]
                result = result+apply(difference)*b.const(mask)
        outputs.append(project(b, result, li, lo, s))
    return _result(outputs, lout)


def maxpool_tensor(x, *, kernel, stride, pad, apply, observer=None):
    """Same separable merge order on logical NCHW float64 tensors."""
    k = kernel if type(kernel) is int else kernel[0]
    s = stride if type(stride) is int else stride[0]
    p = pad if type(pad) is int else pad[0]
    height, width = x.shape[-2:]
    oh, ow = (height+2*p-k)//s+1, (width+2*p-k)//s+1
    result = x
    for axis in (3, 2):
        source = result
        for offset in (i-p for i in range(k) if i != p):
            positions = np.arange(x.shape[axis])
            valid = (positions+offset >= 0) & (positions+offset < len(positions))
            target_length = ow if axis == 3 else oh
            if (target_length-1)*s >= len(positions):
                valid[-1] = False
            shape = [1]*4
            shape[axis] = len(positions)
            mask = valid.reshape(shape)
            difference = np.roll(source, -offset, axis=axis)-result
            if observer is not None:
                observer(np.take(difference, np.flatnonzero(valid), axis=axis))
            result = result+apply(difference)*mask
    result = np.take(result, np.minimum(np.arange(oh)*s, height-1), axis=2)
    return np.take(result, np.minimum(np.arange(ow)*s, width-1), axis=3)


def batchnorm_blocks(b, x, lin, scale, shift):
    outputs = []
    for value, (start, end, block) in zip(values_list(x, lin), layout_blocks(lin)):
        multiplier, offset = np.zeros(lin.n), np.zeros(lin.n)
        if isinstance(block, MuxLayout):
            grid = block.index_grid()
            multiplier[grid] = np.asarray(scale)[None, start:end, None, None]
            offset[grid] = np.asarray(shift)[None, start:end, None, None]
        else:
            multiplier[:block.size] = scale[start:end]
            offset[:block.size] = shift[start:end]
        outputs.append(value*b.const(multiplier)+b.const(offset))
    return _result(outputs, lin)


def flatten_linear(b, x, lin, weight, bias, *, algo='bsgs', n1=None):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[1] != lin.C*lin.H*lin.W:
        raise KernelError('flatten linear weight must have C*H*W input columns')
    if algo not in ('diag', 'bsgs'):
        raise KernelError('linear algo must be diag or bsgs')
    output = multi_flat_layout(lin.n, weight.shape[0])
    if bias is not None and np.asarray(bias).shape != (weight.shape[0],):
        raise KernelError('linear bias must have length O')
    inputs = list(zip(values_list(x, lin), layout_blocks(lin)))
    rotations = {}
    results = []
    for out_start, out_end, lo in layout_blocks(output):
        size = 1 << (lo.size-1).bit_length()
        total = None
        for block_index, (value, (start, end, li)) in enumerate(inputs):
            columns = li.index_grid()[0].reshape(-1)
            features = np.full(li.n, -1, dtype=np.int64)
            features[columns] = np.arange(len(columns))
            slots = np.arange(li.n)
            rows = slots % size

            def diagonal(d):
                cols = features[(slots+d) % li.n]
                valid = (rows < lo.size) & (cols >= 0)
                u = np.zeros(li.n)
                u[valid] = weight[out_start+rows[valid], start*li.H*li.W+cols[valid]]
                return u

            def rotate(step):
                key = (block_index, step)
                if key not in rotations:
                    rotations[key] = value.rotate(step)
                return rotations[key]

            partial = None
            if algo == 'diag':
                for d in range(size):
                    u = diagonal(d)
                    if np.any(u):
                        partial = accumulate(partial, rotate(d)*b.const(u))
            else:
                g = n1 if n1 is not None else 2**((size.bit_length()-1+1)//2)
                if type(g) is not int or g < 1 or g & (g-1) or g > size:
                    raise KernelError('BSGS n1 must be a power of two <= Op')
                values = [rotate(i) for i in range(g)]
                for j in range((size+g-1)//g):
                    inner = None
                    for i in range(g):
                        d = g*j+i
                        if d < size:
                            u = np.roll(diagonal(d), g*j)
                            if np.any(u):
                                inner = accumulate(inner, values[i]*b.const(u))
                    if inner is not None:
                        partial = accumulate(partial, inner.rotate(g*j))
            if partial is not None:
                total = accumulate(total, sum_slots(b, partial, li.n//size, size))
        if total is None:
            total = inputs[0][0]*b.const(np.zeros(lin.n))
        mask = np.zeros(lin.n)
        mask[:lo.size] = 1
        total = total*b.const(mask)
        if bias is not None:
            offset = np.zeros(lin.n)
            offset[:lo.size] = bias[out_start:out_end]
            total = total+b.const(offset)
        results.append(total)
    return _result(results, output)


def fold_input_bn(weight, bias, scale, shift, *, groups=None):
    """Fold an input affine map into a linear or an unpadded convolution."""
    weight = np.asarray(weight, dtype=np.float64)
    scale, shift = np.asarray(scale), np.asarray(shift)
    if weight.ndim == 2:
        offset = weight@shift
        folded = weight*scale[None, :]
    elif groups == 1:
        offset = weight.sum(axis=(2, 3))@shift
        folded = weight*scale[None, :, None, None]
    else:
        if weight.shape[:2] != (len(scale), 1) or groups != len(scale):
            raise KernelError('input BN folding requires ordinary or depthwise convolution')
        offset = weight.sum(axis=(1, 2, 3))*shift
        folded = weight*scale[:, None, None, None]
    return folded, offset if bias is None else offset+np.asarray(bias)
