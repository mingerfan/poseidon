"""Closed shape-only constructors, independently implemented without Torch."""
import math
import numpy as np
from benchmark_graph import require,digest
from model_resources import resolve

OPERATIONS = ("zeros_like","ones_like","full_like","new_zeros","new_ones","new_full")
METHODS = frozenset(OPERATIONS[3:])
FUNCTIONS = frozenset("torch."+name for name in OPERATIONS[:3])
DTYPES = {"float64":"<f8","float32":"<f4","int32":"<i4","int64":"<i8","bool":"|b1"}

def operation(kind,target):
    if kind=="call_method" and target in METHODS:return target
    if kind=="call_function" and target in FUNCTIONS:return target.split(".")[-1]
    return None

def shape(value,limits):
    require(type(value) in (tuple,list) and len(value)<=limits.rank and
            all(type(v) is int and 0<v<=limits.axis_size for v in value),
            "public_constant_static_shape")
    result=list(value)
    require(math.prod(result)<=limits.constant_elements,"public_constant_element_budget")
    return result

def arguments(kind,target,args,kwargs):
    op=operation(kind,target)
    require(op is not None and type(args) in (tuple,list) and type(kwargs) is dict,
            "public_constant_registered_call")
    names=("input","size","fill_value") if op=="new_full" else ("input","size") if op.startswith("new_") else (
        ("input","fill_value") if op=="full_like" else ("input",))
    options={"dtype","device","requires_grad","layout"} | ({"pin_memory"} if op.startswith("new_") else {"memory_format"})
    require(len(args)<=len(names) and not(set(kwargs)-set(names)-options),"public_constant_signature")
    bound=dict(zip(names,args))
    require(not(set(bound)&set(kwargs)),"public_constant_duplicate_argument")
    bound.update(kwargs)
    require(all(n in bound for n in names),"public_constant_missing_argument")
    require(bound.get("requires_grad",False) is False,"public_constant_requires_grad")
    require(bound.get("pin_memory",False) is False,"public_constant_pin_memory")
    require(bound.get("device") in (None,"cpu"),"public_constant_cpu_device")
    # Unspecified layout/memory format preserves a dense constant's value;
    # explicit unsupported enum objects never enter the data transport.
    require(bound.get("layout") is None and bound.get("memory_format") is None,
            "public_constant_layout")
    return op,bound

def plan(kind,target,args,kwargs,*,parent_name,parent_metadata,resource_profile="general-tensors-v1",remaining_elements=None):
    from pytorch_public_value_ops import checked_array,materialize,signature
    from pytorch_static_trace import ALIAS_POLICY
    limits=resolve(resource_profile);op,bound=arguments(kind,target,args,kwargs)
    require(type(parent_name) is str and parent_name and bound["input"]==parent_name,
            "public_constant_shape_source")
    require(type(parent_metadata) is dict and set(parent_metadata)=={"policy","shape","dtype"} and
            parent_metadata["policy"]==ALIAS_POLICY and parent_metadata["dtype"] in DTYPES,
            "public_constant_source_metadata")
    source_shape=shape(parent_metadata["shape"],limits)
    dims=shape(bound["size"],limits) if op.startswith("new_") else source_shape
    dtype=bound.get("dtype")
    require(dtype is None or type(dtype) is str and dtype.startswith("torch.") and
            dtype.removeprefix("torch.") in DTYPES,"public_constant_dtype")
    dtype=DTYPES[parent_metadata["dtype"] if dtype is None else dtype.removeprefix("torch.")]
    fill=bound["fill_value"] if op in ("full_like","new_full") else 0 if op in ("zeros_like","new_zeros") else 1
    require(type(fill) in (bool,int,float) and math.isfinite(fill),"public_constant_finite_scalar")
    if dtype in ("<i4","<i8"):
        info=np.iinfo(np.dtype(dtype))
        require(info.min<=fill<=info.max,"public_constant_integer_range")
    elif dtype in ("<f4","<f8"):
        require(abs(fill)<=np.finfo(np.dtype(dtype)).max,"public_constant_float_range")
    require(remaining_elements is None or type(remaining_elements) is int and
            0<=remaining_elements<=limits.total_constant_elements and math.prod(dims)<=remaining_elements,
            "public_constant_total_element_budget")
    logical=checked_array(np.full(dims,fill,dtype=dtype),resource_profile=resource_profile)
    physical=materialize(logical,resource_profile=resource_profile)
    proof=dict(format="poseidon-public-constant-plan-v1",operation=op,source_node=parent_name,
        source_metadata=parent_metadata,shape=list(dims),dtype=dtype,fill_value=fill,
        dependency="static_shape_and_declared_dtype_only",
        logical_output=signature(logical,resource_profile=resource_profile),
        physical_output=signature(physical,resource_profile=resource_profile),
        resource_profile=resource_profile)
    proof["binding"]=digest(proof)
    return logical,physical,proof
