"""Static logical axis permutations, with explicit periodic CKKS slot routing."""
import math
from seal_artifact_gate import require


def axes(shape,dims):
    require(type(shape) in (tuple,list) and 1<=len(shape)<=4 and
            all(type(n) is int and n>0 for n in shape) and math.prod(shape)<=256,
            'Permutation requires bounded positive rank1..4 shape')
    rank=len(shape)
    require(type(dims) in (tuple,list) and len(dims)==rank and
            all(type(d) is int and -rank<=d<rank for d in dims),'Invalid permutation axes')
    normalized=tuple(d%rank for d in dims)
    require(len(set(normalized))==rank,'Permutation axes must be unique')
    return normalized


def transpose_axes(shape,dim0,dim1):
    rank=len(shape);identity=axes(shape,list(range(rank)))
    require(type(dim0) is int and type(dim1) is int and -rank<=dim0<rank and -rank<=dim1<rank,
            'Invalid transpose axes')
    order=list(identity);a,b=dim0%rank,dim1%rank;order[a],order[b]=order[b],order[a]
    return tuple(order)


def source_order(shape,dims):
    """Lowering map: scatter original flat positions into output C-order positions."""
    dims=axes(shape,dims);out=tuple(shape[d] for d in dims);order=[None]*math.prod(shape)
    for flat in range(len(order)):
        original=[(flat//math.prod(shape[k+1:]))%shape[k] for k in range(len(shape))]
        destination=0
        for axis in dims:destination=destination*shape[axis]+original[axis]
        order[destination]=flat
    return out,order


def routing(shape,dims,period):
    from packed_input_abi import rotations
    rotations(period)
    require(math.prod(shape)<=period,'Permutation shape exceeds slot period')
    out,order=source_order(shape,dims);groups={}
    for destination,source in enumerate(order):
        # rotate(delta)[j] reads x[j+delta]; masks are in destination coordinates.
        delta=(source-destination)%period
        groups.setdefault(delta,[0.]*period)[destination]=1.
    return out,order,groups


def reference(value,shape,dims):
    """Independent nested gather, not the lowering map or routing masks."""
    dims=axes(shape,dims);out=tuple(shape[d] for d in dims)
    def visit(coordinates):
        if len(coordinates)<len(out):
            return [visit(coordinates+[i]) for i in range(out[len(coordinates)])]
        source=[0]*len(shape)
        for output_axis,input_axis in enumerate(dims):source[input_axis]=coordinates[output_axis]
        item=value
        for index in source:item=item[index]
        return item
    return visit([])
