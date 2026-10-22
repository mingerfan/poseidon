"""Candidate two-known-token gate/up/SwiGLU batching; not yet CKKS accepted.

Only encrypted add/multiply/rotate and existing nonlinear/down operators.
No model adapter is installed by importing this module.
"""
import numpy as np

from qwen24.kernels import ops
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import zero_tail_contract as tail
from qwen24.kernels.expr import Plain


def geometry(width, rows, slots):
    period = 1 << (width - 1).bit_length()
    half = slots // 2
    return dict(period=period, half=half,
                supported=slots >= 2 and half % period == 0 and rows + period <= half)


def swiglu_pair(inputs, gate_weight, up_weight, down_weight, *, silu,
                baby_count=16, require_batched=False):
    inputs = tuple(inputs)
    if len(inputs) != 2:
        raise ValueError("Exactly two already-known inputs are required")
    x, y = inputs
    ql._matching(x, y)
    gate_weight = ops._real_array(gate_weight, "gate_weight")
    up_weight = ops._real_array(up_weight, "up_weight")
    down_weight = ops._real_array(down_weight, "down_weight")
    if (gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape or
            gate_weight.shape[1] != x.size or
            down_weight.shape != (x.size, gate_weight.shape[0])):
        raise ValueError("Invalid SwiGLU weight shapes")
    if silu.function != "silu":
        raise ValueError("A SiLU approximation is required")
    if type(baby_count) is not int or baby_count <= 0 or baby_count & (baby_count - 1):
        raise ValueError("Baby count must be a positive power of two")
    rows = len(gate_weight)
    if rows <= 0:
        raise ValueError("Intermediate width must be positive")
    layout = geometry(x.size, rows, x.slots)
    if not layout["supported"]:
        if require_batched:
            raise ValueError("Two-region geometry does not fit")
        return tuple(ql.swiglu(v, gate_weight, up_weight, down_weight, silu=silu)
                     for v in inputs)
    period, half = layout["period"], layout["half"]
    baby_count = min(baby_count, period)
    mask = np.zeros(x.slots)
    mask[:x.size] = 1
    # Match the admitted linear lowering's structural prefix contract.
    # Raw external inputs still keep numerical masks. This proof does not
    # assert zero CKKS noise; the revised graph requires new real validation.
    left = x.expr if tail.known_zero_tail(x) else ops.mul(x.expr, Plain(mask))
    right = y.expr if tail.known_zero_tail(y) else ops.mul(y.expr, Plain(mask))
    right = right.rotate(-half)
    repeated = ops.add(left, right)
    step = period
    while step < half:
        repeated = ops.add(repeated, repeated.rotate(-step))
        step *= 2
    babies = [repeated] + [repeated.rotate(b) for b in range(1, baby_count)]
    row_ids = np.arange(rows)

    def project(weight):
        total = None
        for giant in range(0, period, baby_count):
            subtotal = None
            for baby, source in enumerate(babies):
                columns = (row_ids + giant + baby) % period
                valid = columns < x.size
                selected = row_ids[valid]
                diagonal = np.zeros(x.slots)
                values = weight[selected, columns[valid]]
                diagonal[selected] = diagonal[half + selected] = values
                if giant:
                    diagonal = np.roll(diagonal, giant)
                term = ops.mul(source, Plain(diagonal))
                subtotal = term if subtotal is None else ops.add(subtotal, term)
            if giant:
                subtotal = subtotal.rotate(giant)
            total = subtotal if total is None else ops.add(total, subtotal)
        return total

    gate, up = project(gate_weight), project(up_weight)
    # Physical full-slot metadata is intentional: there are two disjoint
    # activation regions. Polynomial values in the gaps are multiplied by up;
    # each down input receives an explicit numerical boundary mask afterwards.
    activated = ql.approximate(ql.PackedVector(gate, x.slots, x.slots),
                               silu, function="silu")
    product = ops.mul(activated.expr, up)
    keep = np.zeros(x.slots)
    keep[:rows] = 1
    outputs = []
    for offset in (0, half):
        shifted = product.rotate(offset) if offset else product
        # Tiled-down masks every input slice numerically. Combine the
        # old unpack mask with that slice boundary; do not claim a zero tail
        # for this raw window, which still contains the other token/noise.
        packed = ql.PackedVector(shifted, rows, x.slots)
        from qwen24.kernels import tiled_down
        outputs.append(tiled_down.linear(packed, down_weight, tile=min(256, packed.slots)))
    return tuple(outputs)
