"""Static tensor split normalization; no tensor data or backend operations."""
import math
from benchmark_graph import require
from model_resources import resolve

VERSION="poseidon-pytorch-static-split-v1"

def arguments(args,kwargs,*,method=False):
    names=("input","split_size","dim") if method else ("tensor","split_size_or_sections","dim")
    require(len(args)<=3 and not(set(kwargs)-set(names)),"Static split arguments")
    require(not method or "input" not in kwargs,"Tensor split input arguments")
    values=dict(zip(names,(None,None,0)))
    for key,value in zip(names,args):
        require(key not in kwargs,"Duplicate static split argument: "+key)
        values[key]=value
    values.update(kwargs)
    require(values[names[0]] is not None and values[names[1]] is not None,"Static split operands")
    return dict(input=values[names[0]],sections=values[names[1]],axis=values[names[2]])

def plan(dims,sections,axis=0,*,resource_profile="general-tensors-v1"):
    limits=resolve(resource_profile)
    require(type(dims) in (tuple,list) and 1<=len(dims)<=limits.rank and
        all(type(n) is int and 0<n<=limits.axis_size for n in dims) and
        math.prod(dims)<=limits.tensor_elements,"Static split input shape budget")
    require(type(axis) is int and -len(dims)<=axis<len(dims),"Static split axis")
    axis%=len(dims);width=dims[axis]
    if type(sections) is int:
        require(sections>0,"Static split size must be positive")
        count=(width+sections-1)//sections
        require(count<=limits.original_nodes,"Static split expansion budget")
        parts=[min(sections,width-i*sections) for i in range(count)]
    else:
        require(type(sections) in (tuple,list) and sections and
                len(sections)<=limits.original_nodes,"Static split sections budget")
        require(all(type(n) is int and n>0 for n in sections),"Static split sections must be positive")
        require(sum(sections)==width,"Static split sections must cover axis")
        parts=list(sections)
    shapes=[[*dims[:axis],n,*dims[axis+1:]] for n in parts]
    offset=0;slices=[]
    for n in parts:
        slices.append(dict(axis=axis,start=offset,stop=offset+n,step=1));offset+=n
    return dict(version=VERSION,source_shape=list(dims),axis=axis,sections=parts,result_shapes=shapes,
                lowering="split" if 2<=len(parts)<=4 else "slice",slices=slices)
