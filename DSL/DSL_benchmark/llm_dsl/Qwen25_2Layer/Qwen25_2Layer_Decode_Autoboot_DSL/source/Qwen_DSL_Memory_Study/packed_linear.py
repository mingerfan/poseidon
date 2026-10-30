"""Row-packed Linear candidate: full slot arithmetic, no decrypt/CPU matmul.

Input and output retain the existing prefix ABI, including arbitrary input tail.
Masks add three ct-pt multiplications to the longest path versus the old Linear.
No changes to the frozen, previously delivered Qwen kernels.
"""
import numpy as np
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import ops
from qwen24.kernels.expr import Plain


def plan(in_size, out_size, slots):
    if any(type(v) is not int or v <= 0 for v in (in_size, out_size, slots)):
        raise ValueError('Positive integer dimensions required')
    if slots & (slots - 1) or max(in_size, out_size) > slots:
        raise ValueError('Dimensions must fit a power-of-two slot capacity')
    period = 1 << (in_size - 1).bit_length()
    # B <= P avoids circular aliasing in the final unmasked gather tree.
    rows = min(slots // period, period)
    return dict(period=period, rows_per_plain=rows,
                batches=(out_size + rows - 1)//rows, in_size=in_size,
                out_size=out_size, slots=slots)


def linear(x, weight, bias=None):
    w = ops._real_array(weight, 'weight')
    if w.ndim != 2 or w.shape[1] != x.size:
        raise ValueError('weight must have shape [out, input prefix]')
    p = plan(x.size, w.shape[0], x.slots)
    period, rows = p['period'], p['rows_per_plain']
    if bias is not None:
        bias = ops._real_array(bias, 'bias')
        if bias.shape != (w.shape[0],):
            raise ValueError('bias must have shape [out]')
    # Clear dirty tail before copying input into disjoint row blocks.
    mask = np.zeros(x.slots)
    mask[:x.size] = 1
    replicated = ops.mul(x.expr, Plain(mask))
    step = 1
    while step < rows:
        replicated = ops.add(replicated, replicated.rotate(-step * period))
        step *= 2
    result = None
    for start in range(0, w.shape[0], rows):
        count = min(rows, w.shape[0] - start)
        values = np.zeros(x.slots)
        values[:rows*period].reshape(rows, period)[:count, :x.size] = w[start:start+count]
        part = ops.mul(replicated, Plain(values))
        # At row starts, this sums precisely the P elements of that row.
        step = 1
        while step < period:
            part = ops.add(part, part.rotate(step))
            step *= 2
        select = np.zeros(x.slots)
        select[np.arange(count) * period] = 1
        part = ops.mul(part, Plain(select))
        # After selecting row starts, add rotations to move r*P to r.
        # Other replicas cannot enter [0,B) when B<=P; final mask removes them.
        step = 1
        while step < rows:
            part = ops.add(part, part.rotate(step * (period - 1)))
            step *= 2
        keep = np.zeros(x.slots)
        keep[:count] = 1
        part = ops.mul(part, Plain(keep))
        if start:
            part = part.rotate(-start)
        result = part if result is None else ops.add(result, part)
    if bias is not None:
        values = np.zeros(x.slots)
        values[:len(bias)] = bias
        result = ops.add(result, Plain(values))
    return ql.PackedVector(result, w.shape[0], x.slots)
