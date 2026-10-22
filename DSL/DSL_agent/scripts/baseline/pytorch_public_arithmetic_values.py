"""Bounded closed public arithmetic; no Torch/model execution or DSL lowering."""
import copy,math
import numpy as np
from benchmark_graph import require,digest
from model_resources import resolve,identity as resource_identity
from pytorch_public_value_ops import checked_array,signature

POLICY="poseidon-public-value-expressions-v3"
METHODS=frozenset(("add","sub","subtract","mul","multiply","div","divide","true_divide","neg","negative"))
FUNCTIONS=frozenset(("operator.add","operator.sub","operator.mul","operator.truediv","operator.neg",
    "_operator.add","_operator.sub","_operator.mul","_operator.truediv","_operator.neg",
    "torch.add","torch.sub","torch.subtract","torch.mul","torch.multiply",
    "torch.div","torch.divide","torch.true_divide","torch.neg","torch.negative"))
NAMES={"add":"add","sub":"subtract","subtract":"subtract","mul":"multiply","multiply":"multiply",
       "truediv":"divide","div":"divide","divide":"divide","true_divide":"divide","neg":"negate","negative":"negate"}

def registered(kind,target):
    return kind=="call_method" and target in METHODS or kind=="call_function" and target in FUNCTIONS

def describe(kind,target,args,kwargs):
    require(type(kind)is str and type(target)is str and registered(kind,target)
        and type(args)in(tuple,list)and type(kwargs)is dict and all(type(k)is str for k in kwargs),
        "public_arithmetic_call")
    name=target.rsplit(".",1)[-1];operation=NAMES[name]
    unary=operation=="negate";names=("input",)if unary else("input","other")
    require(len(args)<=len(names),"public_arithmetic_arguments")
    values=dict(zip(names,args));attributes=dict(kwargs)
    for field in names:
        if field in attributes:
            require(field not in values,"public_arithmetic_duplicate_argument")
            values[field]=attributes.pop(field)
    require(set(values)==set(names),"public_arithmetic_arguments")
    if operation in ("add","subtract"):
        allowed={"alpha"} if kind=="call_method"or target.startswith("torch.") else set()
        require(not set(attributes)-allowed,"public_arithmetic_arguments")
        alpha=attributes.get("alpha",1)
        require(type(alpha)in(int,float)and math.isfinite(alpha)and abs(alpha)<=1024,
                "public_arithmetic_alpha")
        parameters=dict(alpha=float(alpha))
    elif operation=="divide":
        allowed={"rounding_mode"}if kind=="call_method"or target.startswith("torch.")else set()
        require(not set(attributes)-allowed and attributes.get("rounding_mode")is None,
                "public_arithmetic_rounding")
        parameters={}
    else:
        require(not attributes,"public_arithmetic_arguments");parameters={}
    return operation,parameters,[values[field]for field in names]

def _operand(value,resource_profile):
    if type(value)is np.ndarray:
        array=checked_array(value,resource_profile=resource_profile)
        require(array.dtype.str=="<f8","public_arithmetic_explicit_float64")
        require(np.all(np.abs(array)<=1024),"public_arithmetic_operand_limit")
        return array
    require(type(value)in(int,float)and math.isfinite(value)and abs(value)<=1024,
            "public_arithmetic_static_number")
    return np.asarray(value,dtype="<f8")

def evaluate(values,operation,attributes,*,resource_profile):
    limits=resolve(resource_profile)
    require(type(values)is list and operation in("add","subtract","multiply","divide","negate")
        and len(values)==(1 if operation=="negate"else 2),"public_arithmetic_operands")
    arrays=[_operand(v,resource_profile)for v in values]
    try:dimensions=np.broadcast_shapes(*(v.shape for v in arrays))
    except ValueError:raise ValueError("public_arithmetic_broadcast")from None
    require(len(dimensions)<=limits.rank and math.prod(dimensions)<=limits.constant_elements
        and all(0<d<=limits.axis_size for d in dimensions),"public_arithmetic_broadcast_capacity")
    if operation in("add","subtract"):
        require(type(attributes)is dict and set(attributes)=={"alpha"}and type(attributes["alpha"])is float
            and math.isfinite(attributes["alpha"])and abs(attributes["alpha"])<=1024,"public_arithmetic_alpha")
    else:require(type(attributes)is dict and not attributes,"public_arithmetic_attributes")
    if operation=="divide":
        require(np.all(arrays[1]!=0)and np.all(np.abs(1.0/arrays[1])<=1024),
                "public_arithmetic_divisor_limit")
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        if operation=="add":result=arrays[0]+attributes["alpha"]*arrays[1]
        elif operation=="subtract":result=arrays[0]-attributes["alpha"]*arrays[1]
        elif operation=="multiply":result=arrays[0]*arrays[1]
        elif operation=="divide":result=arrays[0]/arrays[1]
        else:result=-arrays[0]
    result=checked_array(np.asarray(result,dtype="<f8"),resource_profile=resource_profile)
    require(np.all(np.abs(result)<=1024),"public_arithmetic_result_limit")
    return result

def plan(kind,target,args,kwargs,*,resolve_value,resource_profile):
    """References are resolved only from previously authenticated public nodes."""
    operation,attributes,operands=describe(kind,target,args,kwargs)
    values=[];bindings=[]
    for operand in operands:
        resolved=resolve_value(operand)
        if resolved is None:
            require(type(operand)in(int,float),"public_arithmetic_unbound_operand")
            values.append(operand);bindings.append(dict(kind="literal",value=operand))
        else:
            name,value=resolved
            require(type(name)is str and name,"public_arithmetic_parent")
            array=_operand(value,resource_profile);values.append(array)
            bindings.append(dict(kind="public_node",name=name,
                                 signature=signature(array,resource_profile=resource_profile)))
    require(any(v["kind"]=="public_node"for v in bindings),"public_arithmetic_public_dependency")
    result=evaluate(values,operation,attributes,resource_profile=resource_profile)
    record=dict(format=POLICY,source_kind=kind,source_target=target,operation=operation,
        attributes=copy.deepcopy(attributes),operands=bindings,
        logical_output=signature(result,resource_profile=resource_profile),
        materialized_output=signature(result,resource_profile=resource_profile),
        resource_profile=resource_identity(resource_profile),compute_dtype="float64",
        rounding_mode=None,uploaded_code_executed=False)
    record["binding"]=digest(record)
    return result,result,record
