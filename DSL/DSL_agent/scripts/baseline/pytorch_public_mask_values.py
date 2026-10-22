"""Closed public mask signatures, shape rules and independent NumPy values."""
import math
import numpy as np
from benchmark_graph import require,digest
from model_resources import resolve,identity as resource_identity
from pytorch_public_value_ops import checked_array,materialize,signature

METHODS=frozenset(("where","masked_fill","masked_fill_"))
DTYPES={"bool":"|b1","int32":"<i4","int64":"<i8","float32":"<f4","float64":"<f8"}
CATEGORIES={"|b1":0,"<i4":1,"<i8":1,"<f4":2,"<f8":2}
def operation(kind,target):
    if kind=="call_function"and target=="torch.where":return "where"
    if kind=="call_method"and target in METHODS:return target
    return None
def arguments(kind,target,args,kwargs):
    op=operation(kind,target)
    require(op is not None and type(args)is tuple and type(kwargs)is dict,"public_mask_signature")
    names=("condition","input","other")if kind=="call_function"else(
          ("input","condition","other")if op=="where"else("input","mask","value"))
    require(len(args)<=3 and not set(kwargs)-set(names),"public_mask_arguments")
    values=dict(zip(names,args))
    for key,value in kwargs.items():
        require(key not in values,"public_mask_duplicate_argument");values[key]=value
    require(set(values)==set(names),"public_mask_arguments")
    return op,values["condition"]if op=="where"else values["mask"],values["input"],values["other"]if op=="where"else values["value"]
def transform_arguments(kind,target,args,kwargs):
    """V33 public Tensor.flip scalar axis matches the fixed Torch method."""
    if kind=="call_method"and target=="flip":
        if len(args)==2 and not kwargs and type(args[1])is int:
            return (args[0],(args[1],)),kwargs
        if len(args)==1 and set(kwargs)=={"dims"}and type(kwargs["dims"])is int:
            return args,dict(dims=(kwargs["dims"],))
    return args,kwargs
def shape(value,*,resource_profile,constant=False,remaining=None):
    cap=resolve(resource_profile)
    require(type(value)in(tuple,list)and len(value)<=cap.rank and
            all(type(d)is int and 0<d<=cap.axis_size for d in value),"public_mask_shape")
    size=math.prod(value)
    require(size<=(cap.constant_elements if constant else cap.tensor_elements),"public_mask_tensor_budget")
    if remaining is not None:require(type(remaining)is int and size<=remaining,"public_mask_total_constant_budget")
    return list(value)
def result_shape(mask,left,right,*,resource_profile,inplace=False,constant=False,remaining=None):
    dims=[shape(v,resource_profile=resource_profile)for v in(mask,left,right)]
    try:out=list(np.broadcast_shapes(*dims))
    except ValueError:raise ValueError("public_mask_broadcast")from None
    shape(out,resource_profile=resource_profile,constant=constant,remaining=remaining)
    require(not inplace or out==dims[1],"public_mask_inplace_shape")
    return out
def scalar(value):
    require(type(value)in(bool,int,float),"public_mask_finite_scalar")
    try:valid=math.isfinite(value)
    except OverflowError:valid=False
    require(valid,"public_mask_finite_scalar")
    if type(value)is int:require(-(2**63)<=value<2**63,"public_mask_integer_scalar_range")
    return value
def operand(value,resolve_value,*,resource_profile):
    selected=resolve_value(value)
    if selected is not None:
        require(type(selected)is tuple and len(selected)==2 and type(selected[0])is str,"public_mask_public_binding")
        array=checked_array(selected[1],resource_profile=resource_profile)
        return array,dict(kind="public_node",name=selected[0],value=signature(array,resource_profile=resource_profile)),2 if array.ndim else 1
    value=scalar(value)
    dtype="|b1"if type(value)is bool else"<i8"if type(value)is int else"<f4"
    # Scalars are weak operands. A float scalar only gets the default float32
    # dtype if no higher-priority tensor in its category participates.
    return value,dict(kind="scalar",value=value,dtype=dtype),0
def promoted_dtype(values):
    choices=[]
    for value,record,priority in values:
        dtype=record["value"]["dtype"]if record["kind"]=="public_node"else record["dtype"]
        choices.append((CATEGORIES[dtype],priority,dtype))
    category=max(v[0]for v in choices);pool=[v for v in choices if v[0]==category]
    priority=max(v[1]for v in pool)
    return max((v[2]for v in pool if v[1]==priority),key=lambda d:np.dtype(d).itemsize)
def fill(value,dtype):
    scalar(value)
    if dtype in("<i4","<i8"):
        info=np.iinfo(np.dtype(dtype));require(info.min<=value<=info.max,"public_mask_fill_range")
    elif dtype in("<f4","<f8"):
        require(abs(value)<=np.finfo(np.dtype(dtype)).max,"public_mask_fill_range")
    return value
def public_plan(kind,target,args,kwargs,*,resolve_value,resource_profile="general-tensors-v1",remaining=None):
    op,condition,left,right=arguments(kind,target,args,kwargs)
    require(op!="masked_fill_","public_mask_public_inplace")
    mask,mask_record,_=operand(condition,resolve_value,resource_profile=resource_profile)
    require(type(mask)is np.ndarray and mask.dtype.str=="|b1","public_mask_boolean_condition")
    a=operand(left,resolve_value,resource_profile=resource_profile)
    b=operand(right,resolve_value,resource_profile=resource_profile)
    adims=list(a[0].shape)if type(a[0])is np.ndarray else[]
    bdims=list(b[0].shape)if type(b[0])is np.ndarray else[]
    if op!="where":
        require(type(a[0])is np.ndarray and not bdims,"public_mask_scalar_fill")
        dtype=a[0].dtype.str
        fill(b[0].item()if type(b[0])is np.ndarray else b[0],dtype)
    else:dtype=promoted_dtype((a,b))
    dims=result_shape(list(mask.shape),adims,bdims,resource_profile=resource_profile,constant=True,remaining=remaining)
    with np.errstate(over="ignore",invalid="ignore"):
        av=np.asarray(a[0],dtype=dtype);bv=np.asarray(b[0],dtype=dtype)
        logical=np.where(mask,av,bv)if op=="where"else np.where(mask,bv,av)
    logical=checked_array(np.asarray(logical),resource_profile=resource_profile)
    require(list(logical.shape)==dims,"public_mask_result_shape")
    physical=materialize(logical,resource_profile=resource_profile)
    proof=dict(format="poseidon-public-mask-value-v1",source_kind=kind,source_target=target,operation=op,
        operands=[mask_record,a[1],b[1]],output=signature(logical,resource_profile=resource_profile),
        materialized_output=signature(physical,resource_profile=resource_profile),
        resource_profile=resource_identity(resource_profile),encrypted_condition=False)
    proof["binding"]=digest(proof)
    return logical,physical,proof
