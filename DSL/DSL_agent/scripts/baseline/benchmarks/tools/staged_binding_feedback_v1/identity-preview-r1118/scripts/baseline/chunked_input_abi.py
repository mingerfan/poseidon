"""Versioned C-order logical input binding, independent of model/DSL evaluation."""
import math

from seal_artifact_gate import require

TASK = 'hecate-chunked-input-synthesis-v1'


def binding(shape):
    require(type(shape) is list and 1 <= len(shape) <= 4 and
            all(type(n) is int and 1 <= n <= 16 for n in shape),
            'Chunked input requires rank 1..4 and positive integer dimensions <=16')
    size = math.prod(shape)
    require(5 <= size <= 16, 'Chunked input v1 supports 5..16 logical elements')
    chunks = [dict(name=f'chunk{i}', dsl_name=n, shape=[4],
                   flat_start=4*i, flat_stop=min(4*i+4, size))
              for i, n in enumerate(('x', 'y', 'z', 't')[:(size+3)//4])]
    return dict(schema=1, kind='contiguous_period4_chunks', logical_name='x',
                logical_shape=list(shape), logical_elements=size, order='C-row-major',
                padding=0, slot_period=4, chunks=chunks)


def physical_specs(value):
    return [{k: c[k] for k in ('name', 'dsl_name', 'shape')} for c in value['chunks']]


def validate_request_binding(request):
    model, layout = request.get('model'), request.get('layout')
    require(type(model) is dict and type(model.get('schema')) is int and model['schema'] == 4,
            'Chunked request requires original schema-4 model')
    expected = binding(model.get('input_shape'))
    # JSON identity also excludes bool/float substitutions in ABI integer fields.
    import json
    encode = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(type(layout) is dict and encode(layout.get('model_input_binding')) == encode(expected),
            'Logical-to-physical input binding changed')
    require(encode(layout.get('inputs')) == encode(physical_specs(expected)) and
            type(layout.get('input_slot_period')) is int and layout['input_slot_period'] == 4 and
            layout.get('input_order') == 'C-row-major' and 'input_shape' not in layout,
            'Chunked physical input layout changed')
    return expected


def pack_inputs(logical, shape):
    """Client side only: preserve every element, pad only the final physical block."""
    import numpy as np
    plan = binding(shape)
    require(type(logical) is np.ndarray and logical.shape == (4, *shape) and
            logical.dtype == np.float64 and np.isfinite(logical).all(), 'Invalid logical input arrays')
    flat = logical.reshape(4, plan['logical_elements'])
    packed = np.zeros((4, len(plan['chunks']), 4), dtype=np.float64)
    for i, chunk in enumerate(plan['chunks']):
        start, stop = chunk['flat_start'], chunk['flat_stop']
        packed[:, i, :stop-start] = flat[:, start:stop]
    return packed
