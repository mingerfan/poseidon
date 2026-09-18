"""Bounded public real arithmetic and immutable C-order arrays; no candidate code.

This is a checked value implementation, not a general NumPy interpreter. The
caller owns name resolution, AST evaluation and conversion to Hecate constants.
"""
import ast
from dataclasses import dataclass
import itertools
import math
import operator
from seal_artifact_gate import require

OPS = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
       ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv,
       ast.Mod:operator.mod, ast.Pow:operator.pow}
LIMIT = 1048576


def number(value):
    require(type(value) in (int,float) and abs(value) <= LIMIT and math.isfinite(value),
            'Public number must be a finite bounded real')
    return value


def scalar_binary(op, left, right):
    number(left); number(right)
    require(type(op) in OPS, 'Unsupported public numeric operator')
    if type(op) in (ast.Div,ast.FloorDiv,ast.Mod):
        require(right != 0, 'Public division by zero')
    if type(op) is ast.Pow:
        require(abs(right) <= 16, 'Public exponent resource limit')
    try:
        value = OPS[type(op)](left,right)
    except (ValueError,ZeroDivisionError,OverflowError) as error:
        raise ValueError('Invalid public numeric domain') from error
    return number(value)


@dataclass(frozen=True)
class Array:
    shape: tuple
    values: tuple
    floating: bool = False


def array(value, *, floating=False):
    visits = 0
    elements = 0
    kinds = []
    def collect(item, depth):
        nonlocal visits, elements
        visits += 1
        require(visits <= 4096, 'Public array traversal resource limit')
        require(depth <= 4, 'Public array rank limit')
        if type(item) is Array:
            kinds.append(item.floating or any(type(x) is float for x in item.values))
            elements += len(item.values)
            require(elements <= 128, 'Public array element limit')
            return item.shape, list(item.values)
        if type(item) in (list,tuple):
            require(len(item) <= 128, 'Public array element limit')
            children = [collect(x,depth+1) for x in item]
            shape = children[0][0] if children else ()
            require(all(s == shape for s,v in children), 'Ragged public array')
            require(len(item)*math.prod(shape) <= 128, 'Public array element limit')
            return (len(item),)+shape, [x for s,v in children for x in v]
        elements += 1
        require(elements <= 128, 'Public array element limit')
        kinds.append(type(item) is float)
        return (), [number(item)]
    shape, values = collect(value,0)
    require(len(shape) <= 4 and len(values) <= 128, 'Public array shape/element limit')
    floats = floating or any(kinds) or not kinds
    return Array(shape, tuple(float(x) if floats else int(x) for x in values),floats)


def offset(shape, indices):
    out = 0
    for dim,index in zip(shape,indices):
        out = out*dim+index
    return out


def binary(op, left, right):
    if type(left) is not Array and type(right) is not Array:
        return scalar_binary(op,left,right)
    a = left if type(left) is Array else array(left)
    b = right if type(right) is Array else array(right)
    rank = max(len(a.shape),len(b.shape))
    sa,sb = (1,)*(rank-len(a.shape))+a.shape, (1,)*(rank-len(b.shape))+b.shape
    require(all(x == y or x == 1 or y == 1 for x,y in zip(sa,sb)),
            'Incompatible public array broadcast shapes')
    shape = tuple(y if x == 1 else x for x,y in zip(sa,sb))
    require(math.prod(shape) <= 128, 'Public array broadcast element limit')
    integral = not (a.floating or b.floating) and all(type(x) is int for x in (*a.values,*b.values))
    if type(op) is ast.Pow and integral:
        require(all(x >= 0 for x in b.values), 'Integer array cannot have negative integer powers')
    values=[]
    for index in itertools.product(*(range(d) for d in shape)):
        ai = tuple(0 if d == 1 else i for d,i in zip(sa,index))
        bi = tuple(0 if d == 1 else i for d,i in zip(sb,index))
        values.append(scalar_binary(op,a.values[offset(sa,ai)],b.values[offset(sb,bi)]))
    floats = not integral or type(op) is ast.Div
    return Array(shape,tuple(float(v) if floats else int(v) for v in values),floats)


def index(value, indices):
    indices = indices if type(indices) is tuple else (indices,)
    require(len(indices) <= len(value.shape), 'Too many public array indices')
    indices += (slice(None),)*(len(value.shape)-len(indices))
    selections,shape=[],[]
    for dim,item in zip(value.shape,indices):
        if type(item) is slice:
            selected=range(*item.indices(dim))
            selections.append(selected)
            shape.append(len(selected))
        else:
            require(type(item) is int and -dim <= item < dim, 'Public array index out of bounds/type')
            selections.append((item % dim,))
    values=tuple(value.values[offset(value.shape,i)] for i in itertools.product(*selections))
    return Array(tuple(shape),values,value.floating) if shape else values[0]


def reshape(value, shape):
    shape = shape if type(shape) in (tuple,list) else (shape,)
    require(len(shape) <= 4 and all(type(d) is int and -1 <= d <= 128 for d in shape),
            'Invalid public reshape dimensions')
    require(shape.count(-1) <= 1, 'Only one inferred reshape dimension')
    shape=list(shape)
    if -1 in shape:
        known=math.prod(d for d in shape if d != -1)
        require(known > 0 and len(value.values) % known == 0, 'Ambiguous/incompatible inferred shape')
        shape[shape.index(-1)]=len(value.values)//known
    require(math.prod(shape) == len(value.values), 'Reshape must preserve element count')
    return Array(tuple(shape),value.values,value.floating)


def encoding(value):
    """Explicit boundary: current verified Hecate ABI is scalar/1D period-four."""
    if type(value) is Array:
        require(len(value.shape) <= 1 and len(value.values) in (1,4),
                'Encoding needs scalar/1D length one or four; explicitly reshape, do not flatten silently')
        data=list(value.values)
    else:
        data=number(value)
    require(all(abs(x) <= 1024 for x in (data if type(data) is list else [data])),
            'Derived constant exceeds encoding magnitude bound')
    return data
