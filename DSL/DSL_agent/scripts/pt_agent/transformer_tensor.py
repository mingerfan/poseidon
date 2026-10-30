"""Independent tensor semantics for token/score PT-DSL programs."""

import numpy as np

from .kernels import approx
from .kernels.poly import act_cheb, act_minimax
from .kernels import transformer as tr


def polynomial(value, attrs, exact=False):
    if exact or attrs['method'] == 'exact':
        return approx.function(attrs['fn'], value)
    if attrs['method'] in approx.NEW_METHODS:
        return approx.activation(None, value, attrs)
    if attrs['method'] == 'cheb':
        return act_cheb(None, value, attrs['fn'], attrs['deg'], attrs['domain'])
    return act_minimax(None, value, tuple(attrs['degrees']), attrs['domain'])


def run(program, weights, arrays, *, observer=None, act_mode='program',
        point_evaluator=None, step_observer=None):
    if not isinstance(arrays, dict):
        if len(program.params) != 1:
            raise ValueError('token programs with multiple inputs require a mapping')
        arrays = {program.params[0].name: arrays}
    env = {p.name: np.asarray(arrays[p.name], dtype=np.float64) for p in program.params}
    exact = act_mode == 'exact'

    def observe(op, suffix, data, attrs, mask=None):
        if observer is not None:
            sample_event = getattr(observer, 'sample_event', None)
            if sample_event is not None:
                sample_event(op.id, suffix or 'act', np.asarray(data), mask)
            if mask is not None:
                data = data[mask]
            observer(op.outputs[0]+'::'+suffix if suffix else op.inputs[0],
                     np.asarray(data).reshape(1,-1) if suffix else data,
                     ('exact',) if exact else tr.configuration(attrs))

    def apply_point(op, role, value, attrs, mask=None):
        if point_evaluator is None:
            return polynomial(value, attrs, exact)
        return point_evaluator(op, role, value, attrs, mask)

    def inverse_result(op, role, value, inverse):
        callback = getattr(observer, 'inverse_event', None)
        if callback is not None:
            callback(op.id, role, value, inverse)

    for op in program.ops:
        a=op.attrs; x=env[op.inputs[0]]
        w=lambda key: None if a.get(key) is None else weights[a[key].name]
        if op.kind in ('pack','relayout','refresh'):
            value=x
        elif op.kind == 'linear':
            value=x@w('w').T
            if w('b') is not None:
                value=value+w('b')
        elif op.kind in ('add','sub','mul','scale','bias'):
            y=(a['c'] if op.kind=='scale' else w('b') if op.kind=='bias' else
               w('constant') if 'constant' in a else env[op.inputs[1]])
            value=x+y if op.kind in ('add','bias') else x-y if op.kind=='sub' else x*y
        elif op.kind == 'act':
            observe(op,'',x,a)
            value=apply_point(op,'act',x,a)
        elif op.kind in ('layernorm','rmsnorm'):
            centered=x if op.kind=='rmsnorm' else x-x.mean(axis=-1,keepdims=True)
            variance=(centered*centered).mean(axis=-1,keepdims=True)+a['eps']
            attrs=tr.approximation(a,'rsqrt','rsqrt')
            observe(op,'rsqrt',variance,attrs)
            value=centered*apply_point(op,'rsqrt',variance,attrs)
            if w('w') is not None:value=value*w('w')
            if w('b') is not None:value=value+w('b')
        elif op.kind == 'matmul':
            y=env[op.inputs[1]];H,Hkv=a['heads'],a['kv_heads']
            if a['transpose_b']:
                N,L,D=x.shape;dh=D//H;K=y.shape[1]
                q=x.reshape(N,L,H,dh).transpose(0,2,1,3)
                k=y.reshape(N,K,Hkv,dh).transpose(0,2,1,3)
                value=q@np.repeat(k,H//Hkv,axis=1).swapaxes(-1,-2)
                if a['scale'] is not None:value=value*a['scale']
            else:
                N,H,L,K=x.shape;dh=y.shape[-1]//Hkv
                v=y.reshape(N,K,Hkv,dh).transpose(0,2,1,3)
                value=(x@np.repeat(v,H//Hkv,axis=1)).transpose(0,2,1,3).reshape(N,L,H*dh)
        elif op.kind == 'softmax':
            keep=np.ones(x.shape,dtype=bool)
            if a['keep']=='causal':
                keep &= np.arange(x.shape[-1])[None,:] <= np.arange(x.shape[-2])[:,None]
            mask=None if a['mask'] is None else env[a['mask'].name]
            if mask is not None:
                keep &= mask.reshape((len(mask),)+(1,)*(x.ndim-2)+(x.shape[-1],)).astype(bool)
            exp=tr.approximation(a,'exp','exp');inv=tr.approximation(a,'inv','inv')
            shifted=x if a.get('shift',0.0)==0 else x-a['shift']
            if a.get('steps',0):
                shifted=shifted*(2.0**-a['steps'])
            observe(op,'exp',shifted,exp,keep)
            e=apply_point(op,'exp',shifted,exp,keep)*keep
            total=e.sum(axis=-1,keepdims=True)
            if a.get('inv_scale',1.0)!=1:
                total=total*(1/a['inv_scale'])
            observe(op,'inv',total,inv)
            inverse=apply_point(op,'inv',total,inv)
            inverse_result(op,'inv',total,inverse)
            if a.get('inv_scale',1.0)!=1:
                inverse=inverse*(1/a['inv_scale'])
            value=e*inverse
            if a.get('steps',0):
                step_inv=tr.approximation(a,'step_inv','inv')
                for index in range(a['steps']):
                    squares=value*value
                    total=squares.sum(axis=-1,keepdims=True)
                    last=a.get('final_inv_method') is not None and index==a['steps']-1
                    role='final_inv' if last else f'step_inv[{index}]'
                    attrs=tr.approximation(a,'final_inv','inv') if last else step_inv
                    observe(op,role,total,attrs)
                    inverse=apply_point(op,role,total,attrs)
                    inverse_result(op,role,total,inverse)
                    value=squares*inverse
        elif op.kind == 'shuffle':
            value=x[...,a['index']]*np.asarray(a['sign'])
        elif op.kind == 'select':
            value=(x*env[op.inputs[1]][...,None]).sum(axis=1,keepdims=True)
        elif op.kind == 'slice':
            slices=[slice(None)]*x.ndim
            slices[a['axis']]=slice(a['start'],a['end'],a['step'])
            value=x[tuple(slices)]
        elif op.kind == 'flatten':
            value=x.reshape(x.shape[:a['start_dim']]+(-1,))
        else:
            raise ValueError(f'{op.kind}: unsupported token tensor operation')
        if step_observer is not None:
            step_observer(op,value)
        if not np.isfinite(value).all():
            from .backends import BackendError
            raise BackendError('Mock arithmetic produced a nonfinite value')
        env[op.outputs[0]]=value
    result=[env[name] for name in program.outputs]
    return result[0] if len(result)==1 else result
