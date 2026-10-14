"""Bounded native Expr object-array typing and literal-only storage operations.

Arrays contain separate Hecate Expr values, never packed ciphertext slots.
Static inference uses an integer index grid, not candidate Python or FHE values.
Actual dispatch uses the same checked NumPy operations on real frontend objects.
"""
import ast
import math

from hecate_contract import rotation_literal
from seal_artifact_gate import require

MAX_CELLS = 16
MAX_RANK = 4
METHODS = ('reshape','flatten','transpose','copy','item')


def is_array(value):
    return type(value) is tuple and len(value) == 3 and value[0] == 'array'


def dimensions(values, *, infer=False):
    values = tuple(values)
    require(len(values) <= MAX_RANK and all(type(d) is int and
            (-1 if infer else 0) <= d <= MAX_CELLS for d in values), 'Native array rank/dimension bound')
    require(values.count(-1) <= int(infer), 'Native array inferred shape bound')
    require(math.prod(d for d in values if d >= 0) <= MAX_CELLS, 'Native array cell bound')
    return values


def shape_and_cells(value, depth=0):
    require(depth <= MAX_RANK, 'Native array nesting bound')
    if value in ('c','p'):
        return (), (value,)
    if is_array(value):
        return value[2],value[1]
    require(type(value) is tuple and len(value) == 2 and value[0] in ('list','tuple'),
            'Native array data must be Expr cells or rectangular containers')
    parts = [shape_and_cells(v,depth+1) for v in value[1]]
    tail = parts[0][0] if parts else ()
    require(all(s == tail for s,_ in parts),'Ragged native array data')
    shape = dimensions((len(parts),)+tail)
    return shape,tuple(c for _,cells in parts for c in cells)


def constructor(node):
    return (type(node) is ast.Call and type(node.func) is ast.Attribute and
            node.func.attr == 'array' and type(node.func.value) is ast.Name and node.func.value.id == 'np')


def check_constructor(node):
    require(constructor(node) and len(node.args) == 1 and type(node.args[0]) is not ast.Starred and
            len(node.keywords) == 1 and node.keywords[0].arg == 'dtype' and
            type(node.keywords[0].value) is ast.Name and node.keywords[0].value.id == 'object',
            'Native arrays require exactly np.array(data, dtype=object)')


def array_type(value):
    shape,cells = shape_and_cells(value)
    dimensions(shape)
    require(len(cells) <= MAX_CELLS and all(c in ('c','p') for c in cells),'Invalid native array cells')
    return ('array',cells,shape)


def integer(node):
    result = rotation_literal(node)
    require(abs(result) <= 1048576,'Native storage integer bound')
    return result


def index(node):
    def part(n):
        if type(n) is ast.Slice:
            return slice(*(None if value is None else integer(value) for value in (n.lower,n.upper,n.step)))
        if type(n) is ast.Constant and n.value is Ellipsis: return Ellipsis
        if type(n) is ast.Constant and n.value is None: return None
        return integer(n)
    if type(node) is ast.Tuple:
        require(len(node.elts) <= MAX_RANK+1,'Native array index rank bound')
        return tuple(part(n) for n in node.elts)
    return part(node)


def grid(value):
    import numpy as np
    require(is_array(value),'Expected native object array')
    dimensions(value[2])
    return np.arange(len(value[1]),dtype=np.int64).reshape(value[2])


def result_type(original, result):
    import numpy as np
    if type(result) is np.ndarray:
        dimensions(result.shape)
        return ('array',tuple(original[1][int(i)] for i in result.flat),tuple(result.shape))
    return original[1][int(result)]


def subscript_type(value, node):
    try:
        return result_type(value,grid(value)[index(node)])
    except (IndexError,TypeError,ValueError) as error:
        raise ValueError('Invalid native array index') from error


def literal_tuple(nodes):
    if len(nodes) == 1 and type(nodes[0]) in (ast.Tuple,ast.List):
        nodes = nodes[0].elts
    return tuple(integer(n) for n in nodes)


def apply_method(value, node):
    """Only trusted NumPy ndarray receivers; all arguments are public literals."""
    require(type(node) is ast.Call and type(node.func) is ast.Attribute and not node.keywords
            and not any(type(a) is ast.Starred for a in node.args),'Native storage method signature')
    method = node.func.attr
    require(method in METHODS,'Unsupported native array method')
    try:
        if method == 'reshape':
            # reshape(()) is valid for size-one storage, reshape() is not.
            require(bool(node.args),'reshape requires dimensions')
            result = value.reshape(dimensions(literal_tuple(node.args),infer=True))
        elif method == 'transpose':
            axes = literal_tuple(node.args) if node.args else None
            if axes is not None:
                require(len(axes) <= MAX_RANK,'Native transpose rank bound')
            result = value.transpose(axes)
        elif method == 'flatten':
            require(not node.args,'Only C-order flatten() supported')
            result = value.flatten()
        elif method == 'copy':
            require(not node.args,'Only copy() without arguments supported')
            result = value.copy()
        else:
            require(len(node.args) <= MAX_RANK,'Native item argument count bound')
            if len(node.args) == 1 and type(node.args[0]) is ast.Tuple:
                args = (literal_tuple(node.args),)
            else:
                args = literal_tuple(node.args)
            result = value.item(*args)
        # item() returns a single Expr at runtime, an integer during inference.
        if method != 'item':
            dimensions(result.shape)
        return result
    except (IndexError,TypeError,ValueError) as error:
        raise ValueError('Invalid native array '+method) from error


def method_type(value,node):
    return result_type(value,apply_method(grid(value),node))


def arithmetic_type(left, right=None):
    """NumPy object ufunc semantics, not packed-slot broadcasting.

    Scalar Expr on the left dispatches to Hecate before NumPy and is deliberately
    rejected. A scalar Expr on the right is NumPy's zero-dimensional operand.
    NumPy unwraps zero-dimensional object ufunc results to the contained Expr.
    """
    import numpy as np
    require(is_array(left), 'Native array arithmetic requires array on the left')
    if right is None:
        require(all(k == 'c' for k in left[1]), 'Native array negation requires ciphertext cells')
        shape = left[2]
    else:
        require(is_array(right) or right in ('c','p'), 'Invalid native arithmetic operand')
        right_shape = right[2] if is_array(right) else ()
        try:
            shape = dimensions(np.broadcast_shapes(left[2],right_shape))
        except ValueError as error:
            raise ValueError('Incompatible or oversized native broadcast') from error
        a = np.asarray(left[1],dtype=object).reshape(left[2])
        b = np.asarray(right[1] if is_array(right) else [right],dtype=object).reshape(right_shape)
        a,b = np.broadcast_arrays(a,b)
        require(all('c' in (x,y) for x,y in zip(a.flat,b.flat)),
                'Every native arithmetic cell needs a ciphertext operand')
    return ('array',('c',)*math.prod(shape),tuple(shape)) if shape else 'c'
