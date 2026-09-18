"""Bounded object storage for symbolic Hecate construction, not ciphertext slots.

NumPy owns indexing/view/overlap semantics. Only checked immutable symbolic IDs,
Empty sentinels, None and bounded real scalars may enter the storage. Candidate
Python never receives the underlying ndarray or an arbitrary Python object.
"""
import ast
from dataclasses import dataclass
import math

from seal_artifact_gate import require
from public_numeric import number


class Empty:
    """Construction sentinel; deliberately has no invented zero arithmetic."""


class ObjectDtype:
    pass


OBJECT = ObjectDtype()


@dataclass(eq=False)
class ObjectArray:
    data: object

    @property
    def shape(self):
        return self.data.shape


def scalar(value):
    from function_construction import Value
    require(type(value) in (Value, Empty, type(None), int, float, bool),
            'Object array cell must be a symbolic value, Empty, None or bounded real scalar')
    if type(value) in (int, float):
        number(value)
    return value


def shape(value, *, infer=False):
    dimensions = tuple(value) if type(value) in (tuple, list) else (value,)
    require(len(dimensions) <= 4 and all(type(d) is int and
            (-1 if infer else 0) <= d <= 128 for d in dimensions),
            'Object array requires bounded public integer dimensions, rank at most four')
    require(dimensions.count(-1) <= (1 if infer else 0), 'Invalid inferred object array shape')
    require(math.prod(d for d in dimensions if d >= 0) <= 128, 'Object array element limit')
    return dimensions


def wrap(data):
    import numpy as np
    require(type(data) is np.ndarray and data.dtype == np.dtype(object), 'Expected object storage')
    shape(data.shape)
    require(data.size <= 128, 'Object array element limit')
    for item in data.flat:
        scalar(item)
    return ObjectArray(data)


def native(value):
    """Validate recursively before NumPy sees anything with coercion hooks."""
    visits = 0
    def visit(item, depth):
        nonlocal visits
        visits += 1
        require(visits <= 4096 and depth <= 4, 'Object data traversal resource limit')
        if type(item) is ObjectArray:
            return wrap(item.data).data
        if type(item) in (list, tuple):
            require(len(item) <= 128, 'Object data length limit')
            return [visit(child, depth+1) for child in item]
        return scalar(item)
    return visit(value, 0)


def array(value, *, copy=True):
    import numpy as np
    if type(value) is ObjectArray:
        return wrap(value.data.copy() if copy else value.data)
    data = native(value)
    # Validate rectangular layout with a bounded walk before any allocation.
    def layout(item, depth=0):
        require(depth <= 4, 'Object array rank limit')
        if type(item) is np.ndarray:
            return item.shape
        if type(item) is list:
            children = [layout(child, depth+1) for child in item]
            tail = children[0] if children else ()
            require(all(child == tail for child in children), 'Ragged object arrays not supported')
            return (len(item),)+tail
        return ()
    shape(layout(data))
    try:
        return wrap(np.array(data, dtype=object))
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid object array constructor') from error


def empty(dimensions):
    import numpy as np
    # NumPy object storage is initialized with None, never an encrypted zero.
    return wrap(np.empty(shape(dimensions), dtype=object))


def full(dimensions, fill):
    import numpy as np
    dimensions = shape(dimensions)
    fill = array(fill).data  # checked rectangular values, bounded allocation
    try:
        return wrap(np.full(dimensions, fill, dtype=object))
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid object fill broadcast') from error


def index_key(key):
    parts = key if type(key) is tuple else (key,)
    require(len(parts) <= 4, 'Object index rank limit')
    for part in parts:
        if type(part) is slice:
            require(all(x is None or type(x) is int and abs(x) <= 1048576
                        for x in (part.start, part.stop, part.step)) and part.step != 0,
                    'Object slice requires public integer bounds and nonzero step')
        else:
            require(type(part) is int and abs(part) <= 1048576,
                    'Object indices must be public integers; no boolean/advanced indexing')
    return key


def get(value, key):
    import numpy as np
    try:
        result = value.data[index_key(key)]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError('Invalid object array index') from error
    return wrap(result) if type(result) is np.ndarray else scalar(result)


def put(value, key, item):
    key = index_key(key)
    # Validate destination before RHS coercion; all cells remain safe even if
    # native assignment rejects a shape. Do not deep-copy symbolic expressions.
    destination = get(value, key)
    if type(destination) is not ObjectArray:
        # Native NumPy permits nested mutable objects in scalar cells; this
        # contract does not. Reject BEFORE mutation so no storage cycle forms.
        scalar(item)
    data = array(item).data if type(item) in (list, tuple) else native(item)
    try:
        value.data[key] = data
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError('Invalid object array assignment/broadcast') from error
    wrap(value.data)


def reshape(value, dimensions):
    try:
        return wrap(value.data.reshape(shape(dimensions, infer=True)))
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid object array reshape') from error


def flatten(value):
    return wrap(value.data.flatten())


def copy(value):
    return wrap(value.data.copy())


def transpose(value, axes=None):
    if axes is not None:
        require(type(axes) in (tuple, list) and len(axes) == len(value.shape) and
                all(type(a) is int for a in axes), 'Transpose axes must be public integer permutation')
    try:
        return wrap(value.data.transpose(None if axes is None else tuple(axes)))
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid object array transpose') from error


def concatenate(values, axis=0):
    import numpy as np
    require(type(values) in (tuple, list) and 1 <= len(values) <= 128 and
            all(type(v) is ObjectArray for v in values), 'Concatenate requires object arrays')
    require(axis is None or type(axis) is int and -4 <= axis <= 3, 'Invalid public concatenate axis')
    require(sum(v.data.size for v in values) <= 128, 'Concatenate element limit')
    try:
        return wrap(np.concatenate([v.data for v in values], axis=axis))
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid object concatenate shapes') from error


def iterate(value):
    require(bool(value.shape), 'Zero-dimensional object array is not iterable')
    for i in range(value.shape[0]):
        yield get(value, i)  # read lazily: iteration observes subsequent writes


def arithmetic_facts(left, right):
    """Trusted observation only; no candidate hooks or changes to semantics."""
    import numpy as np
    from public_numeric import Array
    from function_construction import Value
    def dimensions(value):
        return list(value.shape) if type(value) in (ObjectArray,Array) else []
    def has(value, kind):
        return type(value) is ObjectArray and any(
            type(v) is Value and v.kind == kind for v in value.data.flat)
    return dict(left_shape=dimensions(left),right_shape=dimensions(right),
        overlapping=type(left) is ObjectArray and type(right) is ObjectArray and
                    bool(np.shares_memory(left.data,right.data)),
        empty_left=type(left) is ObjectArray and any(type(v) is Empty for v in left.data.flat),
        cipher_pair=has(left,'cipher') and has(right,'cipher'))


def elementwise(op, left, right, apply, *, inplace=False):
    """Bounded ndarray broadcasting; apply is a trusted symbolic scalar callback.

    Never invoke a NumPy ufunc on candidate objects. Snapshot both operands
    before writes so overlapping slices read the original values. A 0-D object
    ufunc returns a scalar, whereas an in-place operation retains the ndarray.
    """
    import numpy as np
    from public_numeric import Array
    require(type(op) in (ast.Add, ast.Sub, ast.Mult), 'Unverified object array operator')
    require(type(left) is ObjectArray or type(right) is ObjectArray,
            'Object arithmetic requires an object array')
    def operand(value):
        if type(value) is Array:
            shape(value.shape)
            return np.array([scalar(x) for x in value.values], dtype=object).reshape(value.shape)
        return array(value).data
    a, b = operand(left), operand(right)
    try:
        dimensions = np.broadcast_shapes(a.shape, b.shape)
    except ValueError as error:
        raise ValueError('Incompatible object broadcast shapes') from error
    shape(dimensions)  # Bound allocation BEFORE broadcasting/materializing.
    if inplace:
        require(type(left) is ObjectArray and dimensions == left.shape,
                'In-place object broadcast cannot expand destination shape')
    aa = np.broadcast_to(a, dimensions).copy()
    bb = np.broadcast_to(b, dimensions).copy()
    result = np.empty(dimensions, dtype=object)
    for index in np.ndindex(dimensions):
        result[index] = scalar(apply(op, scalar(aa[index]), scalar(bb[index])))
    if inplace:
        left.data[...] = result
        return left
    return scalar(result.item()) if not dimensions else wrap(result)


def empty_binary(op, left, right, resolve):
    """Dispatch exactly the verified scalar subset of upstream Empty methods.

Expr +/- Empty raises inside Expr's resolveType, rather than returning
NotImplemented. Empty - Expr returns Expr unchanged. Other operand families
need their own dispatch validation and must not be treated as scalar zero.
"""
    from function_construction import Value
    require(type(op) in (ast.Add, ast.Sub), 'Empty supports only addition/subtraction')
    if type(left) is Empty:
        require(type(right) in (Value, int, float, bool, list), 'Unsupported Empty RHS conversion')
        return resolve(right)
    require(type(right) is Empty and type(left) in (int, float, bool, list),
            'Expr/array +/- Empty is not a supported reverse Empty operation')
    return resolve(left)

def unary(op, value, apply):
    """Fresh checked object result; NumPy-like 0-D scalar return, no input writes."""
    import numpy as np
    require(type(op) in (ast.USub, ast.UAdd), 'Unknown object unary operator')
    require(type(value) is ObjectArray, 'Object unary operation requires object storage')
    wrap(value.data)  # validate all cells and allocation limits first
    result=np.empty(value.shape,dtype=object)
    for index in np.ndindex(value.shape):
        result[index]=scalar(apply(op,scalar(value.data[index])))
    return scalar(result.item()) if not value.shape else wrap(result)
