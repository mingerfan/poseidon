"""Dispatch the U4 packed operations without changing the older CNN path."""

from types import SimpleNamespace
import numpy as np

from .config import LINEAR_TALL_MIN_RATIO
from .kernels import transformer as tr
from .kernels.cnn import DifferenceLayout
from .kernels.layout import (FlatLayout, MultiFlatLayout, TokenLayout, ScoreLayout,
                             layout_blocks, KernelError, unpack_sequence)


SEQUENCE = (TokenLayout, ScoreLayout)
POINT_OPS = frozenset(('layernorm', 'rmsnorm', 'softmax'))


def uses_transformer(program):
    return any(op.kind in POINT_OPS | {'matmul', 'shuffle', 'select'} or
               op.kind == 'pack' and op.attrs['layout'] in ('token', 'score') for op in program.ops)


def dispatch(op, env, weights, b, *, exact=False, observer=None, consumers=None):
    a = op.attrs
    value, layout = env[op.inputs[0]]
    xs = tr.values(value, layout)
    weight = lambda key: None if a.get(key) is None else weights[a[key].name]
    observations = {}

    def observe(name, item, lay, attrs, mask=None):
        if observer is None or not hasattr(item, 'value'):
            return
        data = item.value[..., :lay.size]
        active = None if mask is None else mask.value[..., :lay.size] != 0
        sample_event = getattr(observer, 'sample_event', None)
        if sample_event is not None:
            sample_event(op.id, name, data, active)
        if mask is not None:
            data = data[active]
        observations.setdefault(name, [attrs, []])[1].append(np.asarray(data).reshape(-1))

    if observer is not None and hasattr(observer, 'inverse_event'):
        def inverse_result(role, total, inverse):
            observer.inverse_event(op.id, role, total.value[..., :1], inverse.value[..., :1])
        observe.inverse_result = inverse_result

    if op.kind == 'linear':
        result = tr.linear(b, xs, layout, weight('w'), weight('b'),
                           algo=a['algo'], n1=a['n1'], tall_min_ratio=LINEAR_TALL_MIN_RATIO)
    elif op.kind == 'act':
        if observer is not None:
            data = unpack_sequence([x.value for x in xs], layout)
            sample_event = getattr(observer, 'sample_event', None)
            if sample_event is not None:
                sample_event(op.id, 'act', data, None)
            observer(op.inputs[0], SimpleNamespace(value=data),
                     DifferenceLayout(), ('exact',) if exact else tr.configuration(a))
        result = tr.activation(b, xs, layout, a, exact=exact), layout
    elif op.kind in ('layernorm', 'rmsnorm'):
        attrs = tr.approximation(a, 'rsqrt', 'rsqrt')
        result = tr.normalize(b, xs, layout, weight('w'), weight('b'), a['eps'], attrs,
                              rms=op.kind == 'rmsnorm', exact=exact,
                              observe=lambda item, lay, cfg: observe('rsqrt', item, lay, cfg))
    elif op.kind == 'softmax':
        mask, mask_layout = None, None
        if a['mask'] is not None:
            mask_value, mask_layout = env[a['mask'].name]
            mask = tr.values(mask_value, mask_layout)
        result = tr.softmax(b, xs, layout, tr.approximation(a, 'exp', 'exp'),
                            tr.approximation(a, 'inv', 'inv'), keep=a['keep'], mask=mask, mask_layout=mask_layout,
                            shift=a.get('shift', 0.0), inv_scale=a.get('inv_scale', 1.0),
                            steps=a.get('steps', 0),
                            step_inv=tr.approximation(a,'step_inv','inv') if a.get('steps',0) else None,
                            final_inv=tr.approximation(a,'final_inv','inv') if a.get('final_inv_method') else None,
                            exact=exact, observe=observe)
    elif op.kind == 'matmul':
        other, other_layout = env[op.inputs[1]]
        keep = a.get('__keep__','all')
        if consumers:
            uses = consumers[op.outputs[0]]
            if len(uses) == 1 and uses[0].kind == 'softmax':
                keep = uses[0].attrs['keep']
            source = next((producer for producer in consumers.get('__ops__', ())
                           if producer.outputs[0] == op.inputs[0] and producer.kind == 'softmax'), None)
            if source is not None:
                keep = source.attrs['keep']
        result = tr.matmul(b, xs, layout, tr.values(other, other_layout), other_layout,
            transpose_b=a['transpose_b'], heads=a['heads'], kv_heads=a['kv_heads'], scale=a['scale'], keep=keep)
    elif op.kind == 'shuffle':
        result = tr.shuffle(b, xs, layout, a['index'], a['sign'])
    elif op.kind == 'select':
        other, other_layout = env[op.inputs[1]]
        result = tr.select(b, xs, layout, tr.values(other, other_layout), other_layout)
    elif op.kind in ('add', 'sub', 'mul', 'scale', 'bias'):
        if op.kind == 'scale':
            ys = [a['c']]*len(xs)
        elif op.kind == 'bias' or 'constant' in a:
            constant = weight('b' if op.kind == 'bias' else 'constant')
            if isinstance(layout, TokenLayout):
                constant = np.broadcast_to(constant, (1, layout.L, layout.D)).reshape(-1)
            elif isinstance(layout, ScoreLayout):
                constant = np.broadcast_to(constant, (1, layout.H, layout.L, layout.Lk)).transpose(0,2,1,3).reshape(-1)
            else:
                constant = np.broadcast_to(constant, (1, layout.size)).reshape(-1)
            ys = []
            for lo, hi, block in layout_blocks(layout):
                mask = np.zeros(b.slots); mask[:hi-lo] = constant[lo:hi]
                ys.append(b.const(mask))
        else:
            other, other_layout = env[op.inputs[1]]
            if layout != other_layout:
                raise KernelError('elementwise operands require matching layouts')
            ys = tr.values(other, other_layout)
        fn = (lambda x,y:x+y) if op.kind in ('add', 'bias') else (lambda x,y:x-y) if op.kind == 'sub' else (lambda x,y:x*y)
        result = [fn(x, y) for x, y in zip(xs, ys)], layout
    elif op.kind == 'slice' and isinstance(layout, TokenLayout):
        if a['axis'] != 1 or a['step'] != 1:
            raise KernelError('only static token-position selection is supported')
        start, stop, step = slice(a['start'], a['end'], a['step']).indices(layout.L)
        count = len(layout_blocks(layout.token))
        result = xs[start*count:stop*count], TokenLayout(layout.n, stop-start, layout.D)
    elif op.kind == 'flatten' and isinstance(layout, TokenLayout) and layout.L == 1:
        result = (xs if isinstance(layout.token, MultiFlatLayout) else xs[0]), layout.token
    elif op.kind == 'refresh':
        result = value, layout
    else:
        raise KernelError(f'暂不支持 token operation {op.kind}')
    if observer is not None:
        for name, (attrs, arrays) in observations.items():
            data = np.concatenate(arrays).reshape(1, -1)
            observer(op.outputs[0]+'::'+name, SimpleNamespace(value=data), DifferenceLayout(),
                     ('exact',) if exact else tr.configuration(attrs))
    value, layout = result
    if isinstance(layout, (FlatLayout,)) and isinstance(value, list):
        value = value[0]
    return value, layout
