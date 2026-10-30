"""Actual kernel function support, used by r8-v8 checks and capability text."""

from ..kernels import approx, poly


def method_functions():
    return dict(cheb=frozenset(poly.EXACT), remez=frozenset(approx.REMEZ_FUNCTIONS),
                newton=approx.NEWTON_FUNCTIONS, composite=approx.COMPOSITE_FUNCTIONS,
                minimax_comp=frozenset(('relu',)), exact=frozenset(('square',)))


def supported_methods(fn):
    return sorted(method for method, functions in method_functions().items() if fn in functions)


def method_table():
    names = sorted(set().union(*method_functions().values()))
    return '\n'.join(f'{fn}: '+', '.join(supported_methods(fn)) for fn in names)


def approximation_roles(op):
    if op.kind in ('act', 'maxpool'):
        yield '', op.attrs.get('fn', 'relu'), op.attrs['method']
    elif op.kind in ('layernorm', 'rmsnorm'):
        yield 'rsqrt', 'rsqrt', op.attrs['rsqrt_method']
    elif op.kind == 'softmax':
        for role, fn in (('exp', 'exp'), ('inv', 'inv'), ('step_inv', 'inv'), ('final_inv', 'inv')):
            method = op.attrs.get(role+'_method')
            if method is not None:
                yield role, fn, method
