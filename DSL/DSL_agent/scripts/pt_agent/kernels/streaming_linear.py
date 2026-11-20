"""Stream diagnostic flat linear constants using the existing arithmetic.

    The existing wrapped transformer linear routine keeps a scope-wide public
    vector cache.  For a shrinking flat projection its undecorated function
    calls the same original multi/flat kernels without opening another scope.
    An already active public-vector scope is suspended and restored here.
    Legacy EmitBackend is outside this helper.  StreamingEmitBackend writes
    indexed constants as the same operands are encountered.
"""

from contextlib import contextmanager


POLICY = dict(public_vector_cache=False, generated_vector_identity_cache=False,
              retained_constant_payload=False, emit_supported=False,
              streaming_emit_artifact_supported=True, native_trace_verified=False,
              arithmetic='original flat/multi BSGS or diagonal schedule',
              scope='Count, ephemeral Mock and streamed-source shrinking flat projections')


def streaming_linear(backend, values, layout, weight, bias=None, *, algo='bsgs', n1=None):
    import numpy as np
    from ..backends.count import CountBackend
    from ..backends.streaming_mock import StreamingMockBackend
    from ..backends.streaming_emit import StreamingEmitBackend
    from . import slots, transformer
    from .layout import FlatLayout, MultiFlatLayout

    if not isinstance(backend, (CountBackend, StreamingMockBackend, StreamingEmitBackend)):
        raise ValueError('Streaming diagnostic linear requires CountBackend or StreamingMockBackend or StreamingEmitBackend')
    matrix = np.asarray(weight)
    if (matrix.dtype != np.float64 or matrix.ndim != 2
            or not isinstance(layout, (FlatLayout, MultiFlatLayout))
            or matrix.shape[1] != layout.size or not 1 <= matrix.shape[0] <= matrix.shape[1]):
        raise ValueError('Streaming diagnostic linear requires a float64 shrinking flat projection')
    token = slots._current.set(None)
    try:
        # O <= I excludes the separate decorated tall-linear path.  Generated
        # vectors remain writable, so Count's identity cache stores no arrays.
        return transformer.linear.__wrapped__(backend, values, layout, matrix, bias,
                                              algo=algo, n1=n1)
    finally:
        slots._current.reset(token)


def streaming_projection(backend, xs, layout, weight, bias=None, *, algo='bsgs', n1=None, tall_min_ratio=4):
    """Original projection branches, including token and tall output layouts."""
    import numpy as np
    from ..backends.count import CountBackend
    from ..backends.streaming_mock import StreamingMockBackend
    from ..backends.streaming_emit import StreamingEmitBackend
    from . import slots,transformer,multi
    from .layout import FlatLayout,MultiFlatLayout,TokenLayout,layout_blocks
    from .linear import linear_flat
    if not isinstance(backend,(CountBackend,StreamingMockBackend,StreamingEmitBackend)):
        raise ValueError('Streaming projections require a diagnostic or streamed-source backend')
    matrix=np.asarray(weight)
    if matrix.dtype != np.float64 or matrix.ndim != 2:
        raise ValueError('Streaming projections require the original float64 matrix')
    output_size,input_size=matrix.shape
    padded_input=1 << (input_size-1).bit_length()
    source=layout.token if isinstance(layout,TokenLayout) else layout
    count=len(layout_blocks(source))
    groups=layout.L if isinstance(layout,TokenLayout) else 1
    result,output=[],None
    token=slots._current.set(None)
    try:
        for group in range(groups):
            chunk=xs[group*count:(group+1)*count]
            value=chunk if isinstance(source,MultiFlatLayout) else chunk[0]
            if isinstance(source,FlatLayout) and 2*padded_input<=backend.slots and output_size/padded_input>=tall_min_ratio:
                value,output=transformer.tall_linear.__wrapped__(backend,value,source,matrix,bias)
            else:
                run=multi.linear_blocks if isinstance(source,MultiFlatLayout) or output_size>input_size else linear_flat
                if isinstance(source,FlatLayout) and output_size>input_size:
                    value=[value];source=MultiFlatLayout(source.n,source.size)
                value,output=run(backend,value,source,matrix,bias,algo=algo,n1=n1)
            result.extend(transformer.values(value,output))
    finally:
        slots._current.reset(token)
    if isinstance(layout,TokenLayout):return result,TokenLayout(backend.slots,layout.L,output_size)
    return (result if isinstance(output,MultiFlatLayout) else result[0]),output


@contextmanager
def streaming_transformer_projections():
    """Use only the new streaming implementation during one trusted lowering."""
    from . import transformer
    original=transformer.linear
    original_unwrapped=original.__wrapped__
    streaming_projection.__wrapped__=original_unwrapped
    transformer.linear=streaming_projection
    try:
        yield
    finally:
        transformer.linear=original
