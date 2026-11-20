"""Rectangular block linear transforms using the existing flat reduction.

This is an explicitly selected schedule variant.  Original transformer and
multi-block entry points remain unchanged.  Every partial output uses the
original linear_flat arithmetic, then combines the original input blocks.
"""

from contextlib import contextmanager


VARIANT = 'compact-rectangular-linear-v1'
POLICY = dict(variant=VARIANT, original_weights=True,
    arithmetic='original linear_flat per input block and output chunk, then ciphertext sums',
    public_vector_cache=False, generated_vector_identity_cache=False,
    native_trace_verified=False, full_model_math_verified=False)


def _power(size):
    return 1 << (size-1).bit_length()


def inventory(slots, input_size, output_size, *, bias=False):
    """Geometry-only upper bounds; dense native sizes remain unmeasured."""
    if (type(slots) is not int or slots < 2 or slots & (slots-1)
            or type(input_size) is not int or input_size < 1
            or type(output_size) is not int or output_size < 1):
        raise ValueError('Power-of-two slots and positive integer dimensions are required')
    capacity = slots//2
    inputs = [min(capacity, input_size-start) for start in range(0, input_size, capacity)]
    outputs = [min(capacity, output_size-start) for start in range(0, output_size, capacity)]
    diagonal_sites = mask_sites = original_sites = 0
    for source in inputs:
        input_power = _power(source)
        for target in outputs:
            side = _power(max(source, target))
            original_sites += min(side, source+target-1)
            for start in range(0, target, input_power):
                width = min(input_power, target-start)
                diagonal_sites += _power(width)
                mask_sites += 1
    return dict(variant=VARIANT, slots=slots, input_size=input_size, output_size=output_size,
        input_block_sizes=inputs, output_block_sizes=outputs,
        compact_diagonal_sites=diagonal_sites, compact_output_mask_sites=mask_sites,
        original_square_geometrically_nonempty_diagonal_sites=original_sites,
        bias_sites=len(outputs) if bias else 0,
        compact_diagonal_dense_bytes=diagonal_sites*slots*8,
        original_diagonal_dense_bytes=original_sites*slots*8,
        dense_bytes_per_operand=slots*8,
        distinct_constants_measured=False, native_bytes_measured=False,
        note='All nonzero weight values retain their original coordinates. Zero operands and deduplication can reduce distinct constants; shared masks and metadata are separate.')


def compact_linear_blocks(backend, value, layout, weight, bias=None, *, algo='bsgs', n1=None):
    """Same flat/multi-flat interface, with output-sized flat diagonals."""
    import numpy as np
    from .layout import FlatLayout, MultiFlatLayout, KernelError, layout_blocks, multi_flat_layout
    from .linear import linear_flat
    from . import slots
    from .slots import accumulate
    if not isinstance(layout, (FlatLayout, MultiFlatLayout)) or layout.n != backend.slots:
        raise KernelError('Compact linear requires the original flat input layout and matching slots')
    matrix = np.asarray(weight)
    if matrix.dtype != np.float64 or matrix.ndim != 2 or min(matrix.shape) < 1 or matrix.shape[1] != layout.size:
        raise KernelError('Compact linear requires original float64 weights of shape (O, I)')
    if algo not in ('diag', 'bsgs'):
        raise KernelError('Compact linear algo must be diag or bsgs')
    if bias is not None:
        bias = np.asarray(bias)
        if bias.dtype != np.float64 or bias.shape != (matrix.shape[0],):
            raise KernelError('Compact linear requires the original float64 output bias')
    values = list(value) if isinstance(layout, MultiFlatLayout) else [value]
    inputs = list(layout_blocks(layout))
    if len(values) != len(inputs):
        raise KernelError('Compact linear input block count differs from the original layout')
    output = multi_flat_layout(backend.slots, matrix.shape[0])
    result = []
    # linear_flat's public vectors stream when no enclosing cache is active.
    # This also avoids retaining short-lived weight views through owner keys.
    token = slots._current.set(None)
    try:
        for out_start, out_stop, out_layout in layout_blocks(output):
            total = None
            for source, (in_start, in_stop, in_layout) in zip(values, inputs):
                input_power = _power(in_layout.size)
                for offset in range(0, out_layout.size, input_power):
                    stop = min(out_layout.size, offset+input_power)
                    part = matrix[out_start+offset:out_start+stop, in_start:in_stop]
                    if not np.any(part):
                        continue
                    # Op <= Ip even for a short final input block.  Each term
                    # is masked by the original flat kernel before placement.
                    term, _ = linear_flat(backend, source, in_layout, part, None,
                                          algo=algo, n1=n1)
                    total = accumulate(total, term.rotate(-offset))
            if total is None:
                total = values[0]*backend.const(np.zeros(backend.slots, dtype=np.float64))
            if bias is not None:
                offset = np.zeros(backend.slots, dtype=np.float64)
                offset[:out_stop-out_start] = bias[out_start:out_stop]
                total = total+backend.const(offset)
            result.append(total)
    finally:
        slots._current.reset(token)
    return (result if isinstance(output, MultiFlatLayout) else result[0]), output


def compact_projection(backend, values, layout, weight, bias=None, *, algo='bsgs', n1=None,
                       tall_min_ratio=4):
    """Use compact shrinking rectangles and preserve the existing tall route."""
    import numpy as np
    from .layout import FlatLayout, MultiFlatLayout, TokenLayout, KernelError, layout_blocks
    from .streaming_linear import streaming_projection
    from .transformer import values as values_list
    matrix = np.asarray(weight)
    if matrix.ndim != 2 or matrix.dtype != np.float64:
        raise KernelError('Compact projection requires the original float64 matrix')
    if matrix.shape[0] > matrix.shape[1]:
        return streaming_projection(backend, values, layout, matrix, bias, algo=algo,
                                    n1=n1, tall_min_ratio=tall_min_ratio)
    source = layout.token if isinstance(layout, TokenLayout) else layout
    if not isinstance(source, (FlatLayout, MultiFlatLayout)):
        raise KernelError('Compact projection requires a flat or token layout')
    count = len(layout_blocks(source))
    groups = layout.L if isinstance(layout, TokenLayout) else 1
    if len(values) != groups*count:
        raise KernelError('Compact projection block count differs from its original layout')
    result, output = [], None
    for group in range(groups):
        chunk = values[group*count:(group+1)*count]
        value, output = compact_linear_blocks(backend,
            chunk if isinstance(source, MultiFlatLayout) else chunk[0], source, matrix, bias,
            algo=algo, n1=n1)
        result.extend(values_list(value, output))
    if isinstance(layout, TokenLayout):
        return result, TokenLayout(backend.slots, layout.L, matrix.shape[0])
    return (result if isinstance(output, MultiFlatLayout) else result[0]), output


def embedding_from_selection(backend, selection, original_embedding):
    """Original embedding values with the explicitly selected compact route."""
    import numpy as np
    from ..backends.count import CountBackend
    from ..backends.streaming_mock import StreamingMockBackend
    from ..backends.streaming_emit import StreamingEmitBackend
    from .transformer import values
    if not isinstance(backend, (CountBackend, StreamingMockBackend, StreamingEmitBackend)):
        raise ValueError('Compact embedding requires a streaming diagnostic or source backend')
    embedding = np.asarray(original_embedding)
    vocabulary = selection['metadata']['vocabulary']
    if (embedding.dtype != np.float64 or embedding.ndim != 2
            or embedding.shape[0] != vocabulary or embedding.shape[1] < 1):
        raise ValueError('Compact embedding requires the unchanged float64 [vocabulary, hidden] matrix')
    for start in range(0, vocabulary, 128):
        if not np.isfinite(embedding[start:start+128]).all():
            raise ValueError('The original embedding matrix must remain finite')
    output, layout = compact_linear_blocks(backend, selection['one_hot'], selection['layout'], embedding.T)
    return list(values(output, layout)), layout


@contextmanager
def compact_transformer_projections():
    """Explicit per-run variant selection, restored even after a failure."""
    from . import transformer
    original = transformer.linear
    compact_projection.__wrapped__ = getattr(original, '__wrapped__', original)
    transformer.linear = compact_projection
    try:
        yield
    finally:
        transformer.linear = original
