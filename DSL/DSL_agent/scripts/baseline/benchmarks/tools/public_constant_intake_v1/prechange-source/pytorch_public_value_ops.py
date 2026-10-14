"""Registered, bounded public tensor expressions. No model import or execution."""
import copy,hashlib,json,math
import numpy as np
from benchmark_graph import require,digest
from model_resources import resolve,identity as resource_identity

POLICY="poseidon-public-value-expressions-v1"
CAST_POLICY="poseidon-public-value-expressions-v2"
ARITHMETIC_POLICY="poseidon-public-value-expressions-v3"
TRANSFORM_POLICY="poseidon-public-value-expressions-v4"
ARITHMETIC_POLICIES=frozenset((ARITHMETIC_POLICY,TRANSFORM_POLICY))
POLICIES=frozenset((POLICY,CAST_POLICY,*ARITHMETIC_POLICIES))

def checked_policy(value):
    require(type(value)is str and value in POLICIES,"public_expression_policy")
    return value

DATA_TYPES=("<f8","<f4","|b1","<i4","<i8")
OPERATIONS=frozenset(("reshape","permute","slice","bool_cast","identity"))

def checked_array(value,*,resource_profile):
    limits=resolve(resource_profile)
    require(type(value)is np.ndarray and value.dtype.str in DATA_TYPES,"public_expression_array_type")
    require(value.ndim<=limits.rank and 0<value.size<=limits.constant_elements and
            all(0<d<=limits.axis_size for d in value.shape),"public_expression_value_capacity")
    require(np.isfinite(value).all(),"public_expression_nonfinite")
    result=np.array(value,copy=True,order="C");result.flags.writeable=False
    return result

def signature(value,*,resource_profile):
    value=checked_array(value,resource_profile=resource_profile)
    return dict(dtype=value.dtype.str,shape=list(value.shape),elements=int(value.size),
                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())

def axis(value,rank):
    require(type(value)is int and -rank<=value<rank,"public_expression_axis")
    return value%rank

def shape(value,size,limits):
    require(type(value)is list and len(value)<=limits.rank and
            all(type(d)is int and (d>0 or d==-1)for d in value)and value.count(-1)<=1,"public_expression_shape")
    known=1
    for width in value:
        if width>0:
            require(width<=limits.axis_size,"public_expression_shape_capacity")
            known*=width
            require(known<=limits.constant_elements,"public_expression_shape_capacity")
    if -1 in value:
        require(size%known==0,"public_expression_shape_elements")
        dimensions=[size//known if d==-1 else d for d in value]
    else:
        require(known==size,"public_expression_shape_elements");dimensions=list(value)
    require(all(0<d<=limits.axis_size for d in dimensions),"public_expression_shape_capacity")
    return dimensions

def evaluate(value,operation,attributes,*,resource_profile="general-tensors-v1",policy=POLICY):
    limits=resolve(resource_profile);value=checked_array(value,resource_profile=resource_profile)
    policy=checked_policy(policy)
    operations=OPERATIONS|{"dtype_cast"} if policy in(CAST_POLICY,*ARITHMETIC_POLICIES) else OPERATIONS
    require(type(operation)is str and operation in operations and type(attributes)is dict and all(type(k)is str for k in attributes),"public_expression_operation")
    if operation=="reshape":
        require(set(attributes)=={"shape"},"public_expression_attributes")
        result=value.reshape(shape(attributes["shape"],value.size,limits))
    elif operation=="permute":
        require(set(attributes)=={"dims"}and type(attributes["dims"])is list and len(attributes["dims"])==value.ndim,"public_expression_attributes")
        dimensions=[axis(d,value.ndim)for d in attributes["dims"]]
        require(len(set(dimensions))==value.ndim,"public_expression_permutation")
        result=value.transpose(dimensions)
    elif operation=="slice":
        require(set(attributes)=={"selectors"}and type(attributes["selectors"])is list and
                len(attributes["selectors"])<=2*limits.rank+1,"public_expression_attributes")
        dimensions=0;selectors=[]
        for selector in attributes["selectors"]:
            if selector is None:selectors.append(None);continue
            require(dimensions<value.ndim,"public_expression_slice_axis");width=value.shape[dimensions];dimensions+=1
            if type(selector)is int:
                require(-width<=selector<width,"public_expression_index");selectors.append(selector)
            else:
                require(type(selector)is dict and all(type(k)is str for k in selector)and set(selector)=={"start","stop","step"}and
                        all(v is None or type(v)is int for v in selector.values()),"public_expression_slice")
                require(selector["step"]is None or selector["step"]>0,"public_expression_slice_step")
                selectors.append(slice(selector["start"],selector["stop"],selector["step"]))
        result=np.asarray(value[tuple(selectors)])
    elif operation=="dtype_cast":
        require(policy in(CAST_POLICY,*ARITHMETIC_POLICIES) and set(attributes)=={"dtype"} and
                type(attributes["dtype"])is str and attributes["dtype"]in("<f4","<f8"),
                "public_expression_dtype_cast")
        # Explicit public casting follows IEEE rounding; the original array stays bound.
        with np.errstate(over="ignore",invalid="ignore"):
            result=value.astype(attributes["dtype"])
    elif operation=="bool_cast":
        require(set(attributes)=={"dtype"}and type(attributes["dtype"])is str and
                attributes["dtype"]in("<f4","<f8")and value.dtype.str=="|b1","public_expression_bool_cast")
        result=value.astype(attributes["dtype"])
    else:
        require(not attributes,"public_expression_attributes");result=value
    result=checked_array(np.asarray(result),resource_profile=resource_profile)
    return result

def materialize(value,*,resource_profile):
    """Widen public float32 data exactly; retain its logical dtype separately."""
    value=checked_array(value,resource_profile=resource_profile)
    if value.dtype.str=="<f4":
        return checked_array(value.astype("<f8"),resource_profile=resource_profile)
    return value

def describe(kind,target,args,kwargs,*,parent_shape,resource_profile="general-tensors-v1",policy=POLICY):
    """Parse stable framework target names; all non-parent arguments stay static."""
    limits=resolve(resource_profile);policy=checked_policy(policy)
    require(type(kind)is str and kind in("call_method","call_function")and type(target)is str and
            type(args)in(tuple,list)and type(kwargs)is dict and all(type(k)is str for k in kwargs),"public_expression_call")
    require(type(parent_shape)is list and len(parent_shape)<=limits.rank and
            all(type(d)is int and 0<d<=limits.axis_size for d in parent_shape),"public_expression_parent_shape")
    rank=len(parent_shape);size=math.prod(parent_shape)
    require(args,"public_expression_input")
    tail=list(args[1:])
    def arguments(names,defaults=None):
        require(len(tail)<=len(names)and not(set(kwargs)-set(names)),"public_expression_arguments")
        values=dict(zip(names,tail))
        for name,value in kwargs.items():
            require(name not in values,"public_expression_duplicate_argument");values[name]=value
        for name,value in(defaults or {}).items():values.setdefault(name,value)
        require(set(values)==set(names),"public_expression_arguments")
        return values
    def sequence(value):
        require(type(value)in(list,tuple)and len(value)<=limits.rank and
                all(type(d)is int for d in value),"public_expression_static_sequence")
        return list(value)
    if kind=="call_method"and target in("reshape","view","permute"):
        require(not kwargs,"public_expression_arguments")
        dimensions=sequence(tail[0])if len(tail)==1 and type(tail[0])in(list,tuple)else sequence(tail)
        return ("permute",dict(dims=dimensions))if target=="permute"else("reshape",dict(shape=dimensions))
    if kind=="call_function"and target in("torch.reshape","torch.permute"):
        names=("shape",)if target=="torch.reshape"else("dims",)
        values=arguments(names);dimensions=sequence(values[names[0]])
        return ("reshape",dict(shape=dimensions))if target=="torch.reshape"else("permute",dict(dims=dimensions))
    if kind=="call_method"and target=="transpose"or kind=="call_function"and target=="torch.transpose":
        values=arguments(("dim0","dim1"));first=axis(values["dim0"],rank);second=axis(values["dim1"],rank)
        dimensions=list(range(rank));dimensions[first],dimensions[second]=dimensions[second],dimensions[first]
        return "permute",dict(dims=dimensions)
    if kind=="call_method"and target=="t"or kind=="call_function"and target=="torch.t":
        require(not tail and not kwargs and rank<=2,"public_expression_matrix_transpose")
        return "permute",dict(dims=list(reversed(range(rank))))
    if kind=="call_function"and target=="builtins.getattr":
        require(len(tail)==1 and type(tail[0])is str and tail[0]=="T"and not kwargs,"public_expression_attribute")
        return "permute",dict(dims=list(reversed(range(rank))))
    if kind=="call_method"and target=="unsqueeze"or kind=="call_function"and target=="torch.unsqueeze":
        dimension=arguments(("dim",))["dim"]
        require(type(dimension)is int and -rank-1<=dimension<=rank,"public_expression_axis")
        dimensions=list(parent_shape);dimensions.insert(dimension%(rank+1),1)
        require(len(dimensions)<=limits.rank,"public_expression_shape_capacity")
        return "reshape",dict(shape=dimensions)
    if kind=="call_method"and target=="squeeze"or kind=="call_function"and target=="torch.squeeze":
        dimension=arguments(("dim",),dict(dim=None))["dim"]
        if dimension is None:dimensions=[d for d in parent_shape if d!=1]
        else:
            index=axis(dimension,rank);dimensions=[d for i,d in enumerate(parent_shape)if i!=index or d!=1]
        return "reshape",dict(shape=dimensions)
    if kind=="call_method"and target=="flatten"or kind=="call_function"and target=="torch.flatten":
        values=arguments(("start_dim","end_dim"),dict(start_dim=0,end_dim=-1))
        require(all(type(v)is int for v in values.values()),"public_expression_flatten_axis")
        if rank==0:
            require(values==dict(start_dim=0,end_dim=-1),"public_expression_flatten_axis")
            dimensions=[1]
        else:
            first=axis(values["start_dim"],rank);last=axis(values["end_dim"],rank)
            require(first<=last,"public_expression_flatten_axis")
            dimensions=parent_shape[:first]+[math.prod(parent_shape[first:last+1])]+parent_shape[last+1:]
        return "reshape",dict(shape=dimensions)
    if kind=="call_method"and target=="contiguous":
        require(not tail and not kwargs,"public_expression_arguments")
        return "identity",{}
    if kind=="call_function"and target=="operator.getitem":
        selector=arguments(("key",))["key"];items=selector if type(selector)is tuple else(selector,)
        require(len(items)<=2*limits.rank+1,"public_expression_slice_capacity")
        ellipses=sum(v is Ellipsis for v in items);require(ellipses<=1,"public_expression_slice")
        if ellipses:
            consumed=sum(v is not None and v is not Ellipsis for v in items)
            require(consumed<=rank,"public_expression_slice_axis")
            expanded=[]
            for item in items:
                if item is Ellipsis:expanded.extend([slice(None)]*(rank-consumed))
                else:expanded.append(item)
            items=tuple(expanded)
        selectors=[]
        for item in items:
            if item is None or type(item)is int:selectors.append(item)
            else:
                require(type(item)is slice,"public_expression_static_slice")
                selectors.append(dict(start=item.start,stop=item.stop,step=item.step))
        return "slice",dict(selectors=selectors)
    if kind=="call_method"and target in("float","double"):
        require(not tail and not kwargs,"public_expression_arguments")
        return ("dtype_cast"if policy in(CAST_POLICY,*ARITHMETIC_POLICIES) else"bool_cast"),dict(dtype="<f4"if target=="float"else"<f8")
    if kind=="call_method"and target=="to":
        require((len(tail)==1 and not kwargs)or(not tail and set(kwargs)=={"dtype"}),"public_expression_cast_arguments")
        dtype=tail[0]if tail else kwargs["dtype"]
        require(type(dtype)is str and dtype in("torch.float32","torch.float64"),"public_expression_cast_dtype")
        return ("dtype_cast"if policy in(CAST_POLICY,*ARITHMETIC_POLICIES) else"bool_cast"),dict(dtype="<f4"if dtype=="torch.float32"else"<f8")
    raise ValueError("public_expression_operation_not_registered")

def plan(value,kind,target,args,kwargs,*,resource_profile="general-tensors-v1",policy=POLICY):
    value=checked_array(value,resource_profile=resource_profile);policy=checked_policy(policy)
    if policy==TRANSFORM_POLICY:
        from pytorch_public_tensor_transforms import registered,plan as transform_plan
        if registered(kind,target):
            return transform_plan(value,kind,target,args,kwargs,resource_profile=resource_profile)
    operation,attributes=describe(kind,target,args,kwargs,parent_shape=list(value.shape),resource_profile=resource_profile,policy=policy)
    result=evaluate(value,operation,attributes,resource_profile=resource_profile,policy=policy)
    physical=materialize(result,resource_profile=resource_profile)
    record=dict(format=policy,source_kind=kind,source_target=target,operation=operation,
                attributes=copy.deepcopy(attributes),input=signature(value,resource_profile=resource_profile),
                logical_output=signature(result,resource_profile=resource_profile),
                materialized_output=signature(physical,resource_profile=resource_profile),
                resource_profile=resource_identity(resource_profile),
                exact_float64_materialization=bool(result.dtype.str=="<f4"))
    record["binding"]=digest(record)
    return result,physical,record

def check_plan(record,value,kind,target,args,kwargs,*,expected_binding,resource_profile="general-tensors-v1",policy=POLICY):
    require(type(record)is dict and record.get("binding")==expected_binding==
            digest({k:v for k,v in record.items()if k!="binding"}),"public_expression_plan_binding")
    logical,physical,expected=plan(value,kind,target,args,kwargs,resource_profile=resource_profile,policy=policy)
    require(record==expected,"public_expression_plan_changed")
    return logical,physical
