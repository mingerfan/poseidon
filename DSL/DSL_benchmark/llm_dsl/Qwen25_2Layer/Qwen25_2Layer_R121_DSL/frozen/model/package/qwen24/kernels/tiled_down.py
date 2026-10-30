"""Candidate column-tiled projection with wider output row groups.

Not installed automatically; official compile/CKKS evidence still required.
"""
import numpy as np

from qwen24.kernels import ops
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import zero_tail_contract as tail
from qwen24.kernels.expr import Plain


def linear(x, weight, bias=None, *, tile=256):
    weight = ops._real_array(weight, "weight")
    if weight.ndim != 2 or weight.shape[1] != x.size or not 0 < len(weight) <= x.slots:
        raise ValueError("Weight shape must be [positive output, input prefix]")
    if type(tile) is not int or tile <= 0 or tile & (tile-1) or tile > x.slots:
        raise ValueError("Tile must be a power of two fitting the slot capacity")
    if bias is not None:
        bias = ops._real_array(bias, "bias")
        if bias.shape != (len(weight),):
            raise ValueError("Bias shape must match output")
    rows = x.slots // tile
    gather_width = min(rows, tile)
    # All dimensions below are public. Masks keep dirty physical tails out of
    # the replicated tile; a structural prefix proof does not remove slices.
    sources = []
    for start in range(0, x.size, tile):
        mask = np.zeros(x.slots)
        mask[:min(tile, x.size-start)] = 1
        shifted = x.expr.rotate(start) if start else x.expr
        source = ops.mul(shifted, Plain(mask))
        copies = 1
        while copies < rows:
            source = ops.add(source, source.rotate(-copies*tile))
            copies *= 2
        sources.append(source)
    result = None
    for start in range(0, len(weight), rows):
        count = min(rows, len(weight)-start)
        partial = None
        for column, source in enumerate(sources):
            begin = column*tile
            length = min(tile, x.size-begin)
            values = np.zeros(x.slots)
            values[:rows*tile].reshape(rows, tile)[:count, :length] = weight[
                start:start+count, begin:begin+length]
            term = ops.mul(source, Plain(values))
            partial = term if partial is None else ops.add(partial, term)
        step = 1
        while step < tile:
            partial = ops.add(partial, partial.rotate(step))
            step *= 2
        mask = np.zeros(x.slots)
        mask[np.arange(count)*tile] = 1
        partial = ops.mul(partial, Plain(mask))
        # Compact at most tile rows at a time. Larger gathers would mix
        # true values into neighboring chunks before the final numeric mask.
        step = 1
        while step < gather_width:
            partial = ops.add(partial, partial.rotate(step*(tile-1)))
            step *= 2
        for row_base in range(0, count, gather_width):
            take = min(gather_width, count-row_base)
            keep = np.zeros(x.slots)
            keep[row_base*tile:row_base*tile+take] = 1
            piece = ops.mul(partial, Plain(keep))
            shift = row_base*tile-(start+row_base)
            if shift:
                piece = piece.rotate(shift)
            result = piece if result is None else ops.add(result, piece)
    if bias is not None:
        values = np.zeros(x.slots)
        values[:len(bias)] = bias
        result = ops.add(result, Plain(values))
    return tail._proven_prefix(ql.PackedVector(result, len(weight), x.slots))
