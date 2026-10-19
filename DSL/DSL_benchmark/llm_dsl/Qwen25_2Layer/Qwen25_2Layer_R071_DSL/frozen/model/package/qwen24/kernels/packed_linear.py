"""Candidate: skip an input mask only for structurally proven zero tails.

Unmarked external inputs keep the validated dense-stride Linear path.
All output gather masks remain; a clean-prefix certificate follows them.
This candidate still requires dedicated CKKS validation.
"""
import numpy as np
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import ops
from qwen24.kernels.expr import Plain
from qwen24.kernels import zero_tail_contract as tail


def plan(in_size, out_size, slots):
    if any(type(v) is not int or v <= 0 for v in (in_size, out_size, slots)):
        raise ValueError('Positive integer dimensions required')
    if slots & (slots - 1) or max(in_size, out_size) > slots:
        raise ValueError('Dimensions must fit a power-of-two slot capacity')
    period = in_size
    rows = min(slots // period, period)
    gather_copies=1<<(rows-1).bit_length()
    # The final rotate/add tree contains all shifts k*(P-1). With a dense
    # stride, S is not necessarily divisible by P; explicitly prove that no
    # other selected row wraps into the retained prefix. This is public shape
    # arithmetic, never dependent on encrypted values.
    def safe(p,b):
        return all((r*p-k*(p-1))%slots>=b or ((r*p-k*(p-1))%slots==r and k==r)
                   for r in range(b) for k in range(1<<(b-1).bit_length()))
    if not safe(period,rows):
        period=1<<(in_size-1).bit_length()
        rows=min(slots//period,period)
        assert safe(period,rows)
    return dict(period=period, rows_per_plain=rows,
                batches=(out_size + rows - 1)//rows, in_size=in_size,
                out_size=out_size, slots=slots,dense_stride=(period==in_size))


def _repeat_exact(clean, width, count):
    # Build only powers of two that fit; combine disjoint blocks according to
    # the bits of count. Do not temporarily create an overflowing next power.
    powers={1:clean};copies=1
    while copies*2<=count:
        powers[copies*2]=ops.add(powers[copies],powers[copies].rotate(-copies*width))
        copies*=2
    result=None;offset=0
    for block in reversed(sorted(powers)):
        if count & block:
            part=powers[block]
            if offset:part=part.rotate(-offset*width)
            result=part if result is None else ops.add(result,part)
            offset+=block
    assert offset==count
    return result


def _sum_exact(vector, length):
    # Each retained row start gets exactly length consecutive entries.
    # Example 4864 = 4096 + 512 + 256, avoiding padding every row to 8192.
    powers={1:vector};width=1
    while width*2<=length:
        powers[width*2]=ops.add(powers[width],powers[width].rotate(width))
        width*=2
    result=None;offset=0
    for block in reversed(sorted(powers)):
        if length & block:
            part=powers[block]
            if offset:part=part.rotate(offset)
            result=part if result is None else ops.add(result,part)
            offset+=block
    assert offset==length
    return result


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
    replicated = x.expr if tail.known_zero_tail(x) else ops.mul(x.expr, Plain(mask))
    replicated = _repeat_exact(replicated,period,rows)
    result = None
    for start in range(0, w.shape[0], rows):
        count = min(rows, w.shape[0] - start)
        values = np.zeros(x.slots)
        values[:rows*period].reshape(rows, period)[:count, :x.size] = w[start:start+count]
        part = ops.mul(replicated, Plain(values))
        part = _sum_exact(part,period)
        select = np.zeros(x.slots)
        select[np.arange(count) * period] = 1
        part = ops.mul(part, Plain(select))
        # Public plan() checked all cyclic gather aliases before this tree.
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
    return tail._proven_prefix(ql.PackedVector(result, w.shape[0], x.slots))
