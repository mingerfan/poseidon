"""Trusted finite c/p heap for native object-array views and inplace operators.

Only abstract cell kinds and NumPy integer index grids live here, never candidate
Python or ciphertext values. Old native contracts keep immutable tuple typing.
"""
import ast
from dataclasses import dataclass
import native_array_core as base
from seal_artifact_gate import require

METHODS=base.METHODS
constructor=base.constructor
check_constructor=base.check_constructor


@dataclass(eq=False)
class ArrayState:
    kinds: list
    indices: object

    def __getitem__(self,key):
        if key==0:return 'array'
        if key==1:return tuple(self.kinds[int(i)] for i in self.indices.flat)
        if key==2:return tuple(self.indices.shape)
        raise IndexError(key)


def is_array(value):
    return type(value) is ArrayState


def freeze(value):
    if is_array(value):return ('array',value[1],value[2])
    if type(value) is tuple and len(value)==2 and value[0] in ('list','tuple'):
        return (value[0],tuple(freeze(v) for v in value[1]))
    return value


def fresh(value):
    """Native array returns reconstruct fresh C-order storage per Func call."""
    import numpy as np
    if base.is_array(value):
        base.dimensions(value[2])
        return ArrayState(list(value[1]),np.arange(len(value[1]),dtype=np.intp).reshape(value[2]))
    if type(value) is tuple and len(value)==2 and value[0] in ('list','tuple'):
        return (value[0],tuple(fresh(v) for v in value[1]))
    return value


def array_type(value):
    # np.array(..., dtype=object) allocates new cells, even when given an ndarray.
    # Its default order='K' can preserve a transposed array's Fortran layout.
    if is_array(value):
        import numpy as np
        return result_type(value,np.array(value.indices,dtype=np.intp,copy=True,order='K'))
    return fresh(base.array_type(freeze(value)))


def allocate_with_layout(kinds,prototype):
    import numpy as np
    base.dimensions(prototype.shape)
    indices=np.empty_like(prototype,dtype=np.intp,order='K')
    indices.flat[:]=np.arange(len(kinds),dtype=np.intp)
    return ArrayState(list(kinds),indices)


def grid(value):
    require(is_array(value),'Expected alias-aware native object array')
    return value.indices


def result_type(original,result):
    import numpy as np
    if type(result) is np.ndarray:
        base.dimensions(result.shape)
        if np.shares_memory(original.indices,result):return ArrayState(original.kinds,result)
        return allocate_with_layout(tuple(original.kinds[int(i)] for i in result.flat),result)
    return original.kinds[int(result)]


def subscript_type(value,node):
    try:return result_type(value,grid(value)[base.index(node)])
    except (IndexError,TypeError,ValueError) as error:raise ValueError('Invalid native array index') from error


def method_type(value,node):
    return result_type(value,base.apply_method(grid(value),node))


def arithmetic_type(left,right=None):
    import numpy as np
    result=base.arithmetic_type(freeze(left),freeze(right))
    if not base.is_array(result):return result
    # Only finite integer address grids, never FHE/plaintext values. NumPy ufunc
    # allocation follows operand strides/order='K', not unconditionally C order.
    prototype=np.negative(left.indices) if right is None else np.add(left.indices,right.indices if is_array(right) else 0)
    return allocate_with_layout(result[1],prototype)


def inplace_type(left,right):
    """Validate every pair before changing any cell kind; no shape expansion."""
    require(is_array(left),'Inplace target must be an object ndarray')
    result=base.arithmetic_type(freeze(left),freeze(right))
    shape=result[2] if base.is_array(result) else ()
    require(shape==left[2],'Inplace broadcast cannot change target shape')
    for i in left.indices.flat:left.kinds[int(i)]='c'
    # Object ndarray __iop__ preserves even a zero-dimensional array object.
    return left


def bindings(symbols,target):
    import numpy as np
    return [dict(name=name,shape=list(v[2]),kinds=list(v[1]),same_object=v is target,
                 shares_target_storage=bool(np.shares_memory(v.indices,target.indices)))
            for name,v in sorted(symbols.items()) if is_array(v)]


def runtime_bindings(symbols,target,frontend):
    np=frontend.np
    # Native p/c parameters and call results are BOTH Python Expr wrappers.
    # Do not infer IR type from the wrapper class or copy the static prediction.
    require(all(isinstance(c,frontend.Expr) for v in symbols.values() if type(v) is np.ndarray for c in v.flat),
            'Unexpected non-Expr native storage cell')
    return [dict(name=name,shape=list(v.shape),same_object=v is target,
                 shares_target_storage=bool(np.shares_memory(v,target)))
            for name,v in sorted(symbols.items()) if type(v) is np.ndarray]


def changed_cells(symbols,target):
    affected=set(int(i) for i in target.indices.flat)
    return [dict(name=name,indices=[i for i,index in enumerate(v.indices.flat)
                                   if v.kinds is target.kinds and int(index) in affected])
            for name,v in sorted(symbols.items()) if is_array(v)]


def trace_projection(sites):
    """Static c/p labels are deliberately NOT a runtime observation claim."""
    return [dict(site,**{phase:[{k:v for k,v in r.items() if k!='kinds'} for r in site[phase]]
                         for phase in ('before','after')}) for site in sites]
