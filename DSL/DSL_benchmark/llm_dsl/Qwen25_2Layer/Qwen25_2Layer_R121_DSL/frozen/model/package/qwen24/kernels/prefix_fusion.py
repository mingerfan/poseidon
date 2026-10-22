"""Reuse structural prefix proofs across adjacent DSL operators.

No ciphertext inspection, range inference, or client-provided certificates.
The proof describes exact slot algebra; it does not assert zero CKKS noise.
External inputs and partial slices keep explicit masks.
"""
from . import qwen_layers as ql
from . import zero_tail_contract as tail


def install():
    if getattr(ql, '_prefix_fusion_installed', False):
        return
    original_extract = ql.extract
    original_rope = ql.rope
    original_add = ql.add
    original_concatenate = ql.concatenate

    def extract(x, start, size):
        start = ql._integer(start, 'start', 0)
        size = ql._integer(size, 'size')
        if start + size > x.size:
            raise ValueError('slice exceeds vector size')
        if start == 0 and size == x.size and tail.known_zero_tail(x):
            # Same logical slice and physical layout: no operation is needed.
            return x
        # Partial slices still mask, even when the enclosing vector is clean.
        return tail._proven_prefix(original_extract(x, start, size))

    def rope(*args, **kwargs):
        # Both branches of the original formula end in prefix-only plaintext
        # multiplies. This is true even for a dirty external input.
        return tail._proven_prefix(original_rope(*args, **kwargs))

    def add(x, y):
        result = original_add(x, y)
        # One clean operand is insufficient: the other may have dirty tails.
        if tail.known_zero_tail(x) and tail.known_zero_tail(y):
            return tail._proven_prefix(result)
        return result

    def concatenate(vectors):
        # Original code validates size and uses ql.extract for every operand.
        # Only the full slices of certified operands are now identity edges.
        return tail._proven_prefix(original_concatenate(vectors))

    ql.extract = extract
    ql.rope = rope
    ql.add = add
    ql.concatenate = concatenate
    ql._prefix_fusion_installed = True
