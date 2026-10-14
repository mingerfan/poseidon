"""Data-only fixed Torch tensor assembly signatures and exact public values."""
import math
import numpy as np
from benchmark_graph import require,digest
from model_resources import resolve,identity
from pytorch_public_value_ops import checked_array,materialize,signature
FUNCTIONS=frozenset(("torch.cat","torch.concat","torch.concatenate","torch.stack"))
DTYPES=("|b1","<i4","<i8","<f4","<f8")
def operation(kind,target):
    return ("stack"if target=="torch.stack"else"concat")if kind=="call_function"and target in FUNCTIONS else None
def target_name(kind,target):
    if kind!="call_function":return None
    import torch
    return next(("torch."+name for name in("cat","concat","concatenate","stack")if target is getattr(torch,name)),None)
def arguments(kind,target,args,kwargs):
    op=operation(kind,target)
    require(op is not None and type(args)is tuple and type(kwargs)is dict and len(args)<=3
        and not set(kwargs)-{"tensors","dim","out"},"public_join_signature")
    values=dict(zip(("tensors","dim","out"),args))
    for key,value in kwargs.items():
        require(key not in values,"public_join_duplicate_argument");values[key]=value
    require("tensors"in values and values.get("out")is None,"public_join_out_mutation")
    tensors=values["tensors"];axis=values.get("dim",0)
    require(type(tensors)in(tuple,list)and 1<=len(tensors)<=8,"public_join_tensor_sequence")
    require(type(axis)is int,"public_join_static_axis")
    return op,list(tensors),axis
def checked_shape(value,*,resource_profile,constant=False,remaining=None):
    cap=resolve(resource_profile)
    require(type(value)in(list,tuple)and len(value)<=cap.rank and
        all(type(d)is int and 0<d<=cap.axis_size for d in value),"public_join_shape")
    size=math.prod(value)
    require(size<=(cap.constant_elements if constant else cap.tensor_elements),"public_join_tensor_capacity")
    if remaining is not None:require(type(remaining)is int and 0<=size<=remaining,"public_join_total_constant_capacity")
    return list(value)
def result_shape(op,shapes,axis,*,resource_profile,constant=False,remaining=None):
    require(op in("stack","concat")and type(shapes)is list and 1<=len(shapes)<=8,"public_join_shape_sequence")
    ss=[checked_shape(s,resource_profile=resource_profile)for s in shapes];rank=len(ss[0])
    require(all(len(s)==rank for s in ss),"public_join_rank")
    upper=rank+1 if op=="stack"else rank
    require(type(axis)is int and -upper<=axis<upper,"public_join_axis");axis%=upper
    if op=="stack":
        require(all(s==ss[0]for s in ss),"public_join_stack_shape")
        dims=list(ss[0]);dims.insert(axis,len(ss))
    else:
        require(rank>0 and all(all(v==ss[0][i]for i,v in enumerate(s)if i!=axis)for s in ss),
                "public_join_concat_shape")
        dims=list(ss[0]);dims[axis]=sum(s[axis]for s in ss)
    checked_shape(dims,resource_profile=resource_profile,constant=constant,remaining=remaining)
    return dims,axis
def public_plan(kind,target,args,kwargs,*,resolve_value,resource_profile,remaining):
    op,tensors,axis=arguments(kind,target,args,kwargs);arrays=[];operands=[]
    for value in tensors:
        bound=resolve_value(value)
        require(type(bound)is tuple and len(bound)==2 and type(bound[0])is str,"public_join_bound_tensor")
        data=checked_array(bound[1],resource_profile=resource_profile);arrays.append(data)
        operands.append(dict(name=bound[0],value=signature(data,resource_profile=resource_profile)))
    dims,axis=result_shape(op,[list(v.shape)for v in arrays],axis,
        resource_profile=resource_profile,constant=True,remaining=remaining)
    # Torch concatenation preserves its floating category even with integer
    # operands; np.result_type(int64,float32) would incorrectly widen to f64.
    dtype=max((v.dtype.str for v in arrays),key=DTYPES.index)
    cast=[v.astype(dtype,copy=False)for v in arrays]
    with np.errstate(over="ignore",invalid="ignore"):
        logical=np.stack(cast,axis=axis)if op=="stack"else np.concatenate(cast,axis=axis)
    logical=checked_array(logical,resource_profile=resource_profile)
    require(list(logical.shape)==dims and logical.dtype.str==dtype,"public_join_result")
    physical=materialize(logical,resource_profile=resource_profile)
    proof=dict(format="poseidon-public-tensor-join-values-v1",operation=op,axis=axis,
        operands=operands,logical_output=signature(logical,resource_profile=resource_profile),
        materialized_output=signature(physical,resource_profile=resource_profile),resource_profile=identity(resource_profile))
    proof["binding"]=digest(proof);return logical,physical,proof
