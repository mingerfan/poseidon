"""Checked public Chebyshev data backed by the project's existing NumPy.

Only whitelisted operations on bounded real data; no candidate Python or
arbitrary NumPy attributes. Immutable wrappers never expose a native object.
"""
import ast
from dataclasses import dataclass
import operator
import public_numeric as numeric
from seal_artifact_gate import require


@dataclass(frozen=True)
class Float64Type:
    pass


FLOAT64 = Float64Type()


@dataclass(frozen=True)
class Polynomial:
    coefficients: tuple
    domain: tuple = (-1.,1.)
    window: tuple = (-1.,1.)
    symbol: str = 'x'


OPS = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
       ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod, ast.Pow:operator.pow}
UFUNCS = frozenset(('floor','ceil','log2'))


def vector(value):
    arr = numeric.array(value,floating=True)
    require(len(arr.shape) <= 1 and 1 <= len(arr.values) <= 128,
            'Chebyshev requires a nonempty real coefficient vector')
    return tuple(float(numeric.number(v)) for v in arr.values)


def create(coef, domain=None, window=None, symbol='x'):
    import numpy as np
    require(type(symbol) is str and symbol.isidentifier() and len(symbol) <= 64,
            'Polynomial symbol must be a bounded public identifier')
    domain = (-1.,1.) if domain is None else vector(domain)
    window = (-1.,1.) if window is None else vector(window)
    require(len(domain) == len(window) == 2,'Polynomial domain/window require two endpoints')
    result = Polynomial(vector(coef),domain,window,symbol)
    # Check pinned NumPy's own constructor invariants; do not retain its object.
    np.polynomial.Chebyshev(result.coefficients,domain=domain,window=window,symbol=symbol)
    return result


def native(value):
    import numpy as np
    if type(value) is Polynomial:
        return np.polynomial.Chebyshev(value.coefficients,domain=value.domain,
                                      window=value.window,symbol=value.symbol)
    if type(value) in (int,float):
        return numeric.number(value)
    return np.asarray(vector(value),dtype=np.float64)


def wrap(value):
    import numpy as np
    require(type(value) is np.polynomial.Chebyshev,'Expected a Chebyshev result')
    return create(value.coef.tolist(),value.domain.tolist(),value.window.tolist(),value.symbol)


def binary(op,left,right):
    import numpy as np
    require(type(op) in OPS,'Unsupported public polynomial operator')
    require(type(left) is Polynomial or type(right) is Polynomial,'Expected a public polynomial')
    def length(value):
        return len(value.coefficients) if type(value) is Polynomial else len(vector(value))
    if type(op) is ast.Mult:
        require(length(left)+length(right)-1 <= 128,'Polynomial product degree limit')
    if type(op) is ast.Pow:
        require(type(left) is Polynomial and type(right) in (int,float) and
                right == int(right) and 0 <= right <= 16,'Polynomial exponent must be an integer from 0 to 16')
        require((length(left)-1)*int(right)+1 <= 128,'Polynomial power degree limit')
    try:
        with np.errstate(all='raise'):
            return wrap(OPS[type(op)](native(left),native(right)))
    except (TypeError,ValueError,ZeroDivisionError,OverflowError,FloatingPointError) as error:
        raise ValueError('Invalid public Chebyshev arithmetic/domain') from error


def ufunc(name,value):
    import numpy as np
    require(name in UFUNCS,'Unsupported public numeric function')
    scalar = type(value) in (int,float)
    data = numeric.array(value,floating=True)
    methods = {'floor':np.floor,'ceil':np.ceil,'log2':np.log2}
    try:
        with np.errstate(all='raise'):
            out = methods[name](np.asarray(data.values,dtype=np.float64).reshape(data.shape))
    except (TypeError,ValueError,OverflowError,FloatingPointError) as error:
        raise ValueError('Invalid public numeric function domain') from error
    values = tuple(float(numeric.number(float(v))) for v in out.reshape(-1))
    return values[0] if scalar else numeric.Array(data.shape,values,True)
