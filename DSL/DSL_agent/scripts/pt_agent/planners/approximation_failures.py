"""A1 numerical candidate failures; implementation errors remain exceptions."""

import math
import numpy as np

from ..kernels import approx
from ..kernels.layout import KernelError


NUMERICAL_ERRORS=(KernelError,approx.ApproximationError,FloatingPointError)


def require_finite(value,label):
    if isinstance(value,(list,tuple)):
        for item in value:require_finite(item,label)
    elif not np.isfinite(value).all():
        raise FloatingPointError('nonfinite '+label)


def failure(exc,**fields):
    nonfinite=isinstance(exc,FloatingPointError) or 'nonfinite' in str(exc).lower()
    for name in ('max_error','error_amplitude'):
        if hasattr(exc,name) and not math.isfinite(getattr(exc,name)):nonfinite=True
    result=dict(fields,reason='nonfinite' if nonfinite else 'degenerate',
                exception_type=type(exc).__name__,message=str(exc))
    if isinstance(exc,approx.ApproximationError):
        result.update(iterations=exc.iterations,max_error=exc.max_error,
                      abs_E=abs(exc.error_amplitude),history=exc.history)
    return result


def point_fields(point):
    a=point.attrs;op=point.op;k=op.attrs.get('steps');rho=op.attrs.get('inv_budget')
    role=point.role or 'act'
    target=(1e-3/(2**k) if role=='exp' else
            (rho or .25) if point.prefix in ('inv','step_inv') and k and op.attrs.get('final_inv_method') else
            1e-3 if role!='act' else None)
    return dict(line=op.lineno,call_lines=list(op.call_lines),role=role,fn=a['fn'],
                fit=a.get('fit') or 'absolute',method=a['method'],degree=a.get('deg'),
                deg0=a.get('deg0'),iters=a.get('iters'),domain=a['domain'],target=target,k=k,rho=rho)


def program_failures(program):
    """Fit and lower each distinct approximation before numerical execution.

    No samples are used here. Successful coefficients and evaluation order are
    unchanged, and only the A1 caller opts into structured rejection.
    """
    from ..backends import CountBackend
    from ..kernels import transformer as tr
    from ..kernels.poly import cheb_fit,minimax_coeffs
    from ..transformer_validation import points
    errors=[];visited=set()
    for point in points(program):
        identity=point.op.id,point.prefix
        if identity in visited:continue
        visited.add(identity)
        a=point.attrs;method=a['method']
        if method=='exact':continue
        try:
            coefficients=(approx.fitted_coefficients(a) if method in approx.NEW_METHODS else
                          cheb_fit(a['fn'],a['deg'],*a['domain']) if method=='cheb' else
                          minimax_coeffs(tuple(a['degrees'])))
            require_finite(coefficients,'coefficients')
            backend=CountBackend(8)
            tr.apply(backend,backend.input(0),a)
        except NUMERICAL_ERRORS as exc:
            errors.append(failure(exc,**point_fields(point)))
    return errors
