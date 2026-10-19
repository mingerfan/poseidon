"""Pack vocabulary rows across all slots and share baby-step input rotations.

The returned chunks retain the original LM-head prefix ABI. This changes only
linear algebra lowering; it uses ciphertext rotations, addition and plaintext
multiplication, with no decryption or ciphertext inspection.
"""
import numpy as np
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import ops
from qwen24.kernels.expr import Plain


def lm_head(x, weight, *, chunk_size):
    w = ops._real_array(weight, 'weight')
    if w.ndim != 2 or w.shape[1] != x.size or w.shape[0] == 0:
        raise ValueError('lm_head weight must have shape [vocabulary, hidden]')
    if type(chunk_size) is not int or chunk_size <= 0 or x.slots % chunk_size:
        raise ValueError('chunk_size must be a positive divisor of the slot capacity')
    period = 1 << (x.size-1).bit_length()
    baby_count = 1 << ((period.bit_length()-1)//2)
    mask = np.zeros(x.slots)
    mask[:x.size] = 1
    repeated = ops.mul(x.expr, Plain(mask))
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
        outputs.extend(ql.extract(packed, offset, min(chunk_size, count-offset))
                       for offset in range(0, count, chunk_size))
    return tuple(outputs)
