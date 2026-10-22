"""Pack vocabulary rows across all slots and share baby-step input rotations.

The returned chunks retain the original LM-head prefix ABI. This changes only
linear algebra lowering; it uses ciphertext rotations, addition and plaintext
multiplication, with no decryption or ciphertext inspection.
"""
import numpy as np
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import ops
from qwen24.kernels.expr import Plain
from qwen24.kernels import zero_tail_contract as tail


def lm_head(x, weight, *, chunk_size, baby_policy='baseline', baby_count_override=None):
    w = ops._real_array(weight, 'weight')
    if w.ndim != 2 or w.shape[1] != x.size or w.shape[0] == 0:
        raise ValueError('lm_head weight must have shape [vocabulary, hidden]')
    if type(chunk_size) is not int or chunk_size <= 0 or x.slots % chunk_size:
        raise ValueError('chunk_size must be a positive divisor of the slot capacity')
    period = 1 << (x.size-1).bit_length()
    baby_count = 1 << ((period.bit_length()-1)//2)
    if baby_policy not in ('baseline','chunks'):
        raise ValueError('Unknown LM-head grouping policy')
    if baby_policy == 'chunks':
        # Baby rotations are shared by every full-slot output chunk. Account
        # for this reuse instead of optimizing a single output chunk.
        chunks = (len(w)+x.slots-1)//x.slots
        candidates = [1 << k for k in range(period.bit_length())]
        baby_count = min(candidates,
            key=lambda b:(b-1 + chunks*(period//b-1), b))
    if baby_count_override is not None:
        if type(baby_count_override) is not int or baby_count_override<=0 or baby_count_override & (baby_count_override-1):
            raise ValueError('Baby count override must be a positive power of two')
        # The same small-shape tests use a smaller period. Both are powers of
        # two, so this preserves exact coverage without an incomplete group.
        baby_count=min(baby_count_override,period)
        assert period % baby_count == 0
    mask = np.zeros(x.slots)
    mask[:x.size] = 1
    repeated = x.expr if tail.known_zero_tail(x) else ops.mul(x.expr, Plain(mask))
    step = period
    while step < x.slots:
        repeated = ops.add(repeated, repeated.rotate(-step))
        step *= 2
    babies = [repeated] + [repeated.rotate(b) for b in range(1, baby_count)]
    row_ids = np.arange(x.slots)
    outputs = []
    for start in range(0, len(w), x.slots):
        count = min(x.slots, len(w)-start)
        total = None
        for giant in range(0, period, baby_count):
            subtotal = None
            for baby, source in enumerate(babies):
                # A_j[i] = W[i,(i+j) mod period]. Rotating each group by g
                # requires the compensating plaintext D_j[k] = A_j[k-g].
                columns = (row_ids[:count]+giant+baby) % period
                valid = columns < x.size
                diagonal = np.zeros(x.slots)
                selected_rows = row_ids[:count][valid]
                diagonal[selected_rows] = w[start+selected_rows, columns[valid]]
                if giant:
                    diagonal = np.roll(diagonal, giant)
                term = ops.mul(source, Plain(diagonal))
                subtotal = term if subtotal is None else ops.add(subtotal, term)
            if giant:
                subtotal = subtotal.rotate(giant)
            total = subtotal if total is None else ops.add(total, subtotal)
        packed = ql.PackedVector(total, count, x.slots)
        # After compensating each giant-step rotation, every diagonal is
        # supported only on this chunk's prefix. No extraction is necessary
        # when returning the entire already-clean block. Split chunks keep it.
        if count <= chunk_size:
            outputs.append(tail._proven_prefix(packed))
        else:
            outputs.extend(ql.extract(packed, offset, min(chunk_size, count-offset))
                           for offset in range(0, count, chunk_size))
    return tuple(outputs)
