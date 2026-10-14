"""Declared Softmax component projections; original execution stays unchanged."""
import copy
from source_calibration_contract import require
SOFT_MODULES={"torch.nn.Softmax":"softmax","torch.nn.LogSoftmax":"logsoftmax"}
SOFT_FUNCTIONS={f"{prefix}.{name}":("softmax" if name=="softmax" else "logsoftmax")
 for prefix in("torch","torch.nn.functional")for name in("softmax","log_softmax")}
SOFT_METHODS={"softmax":"softmax","log_softmax":"logsoftmax"}
ARITHMETIC="reference-float64"

def checked_projection(value):
    require(type(value) is dict and set(value)=={"kind","exp","arithmetic"}
      and value["kind"]=="softmax_exp_sum" and value["arithmetic"]==ARITHMETIC,
      "calibration_projection_rejected")
    from model_decomposition import profile
    try:profile("exp",value["exp"])
    except (ValueError,TypeError,KeyError):require(False,"calibration_projection_rejected")
    return copy.deepcopy(value)

def soft_parameters(node):
    """Require an explicit static axis; dtype is a native forward attribute."""
    if node["op"]=="call_module":
        from source_fx_metadata import checked_source
        item=checked_source(node,compound=True)["module"]
        if item["type"] not in SOFT_MODULES:return None
        a=item["attributes"]
        require(set(a)=={"dim"},"calibration_projection_rejected")
        operator=SOFT_MODULES[item["type"]];dim=a["dim"];dtype=None
    else:
        operator=(SOFT_FUNCTIONS.get(node["target"])if node["op"]=="call_function"
                  else SOFT_METHODS.get(node["target"])if node["op"]=="call_method"else None)
        if operator is None:return None
        args=node["args"];kwargs=node["kwargs"]
        require(type(args) is dict and set(args)=={"tuple"} and type(args["tuple"]) is list
          and type(kwargs) is dict and set(kwargs)=={"dict"} and type(kwargs["dict"]) is dict,
          "calibration_projection_rejected")
        functional=node["target"].startswith("torch.nn.functional.")
        keys=("input","dim","_stacklevel","dtype")if functional else("input","dim","dtype")
        require(len(args["tuple"])<=len(keys) and set(kwargs["dict"])<=set(keys),
          "calibration_projection_rejected")
        values=dict(zip(keys,args["tuple"]))
        for key,value in kwargs["dict"].items():
            require(key not in values,"calibration_projection_rejected");values[key]=value
        dim=values.get("dim");dtype=values.get("dtype")
        if "_stacklevel" in values:
            require(type(values["_stacklevel"]) is int and 1<=values["_stacklevel"]<=128,
              "calibration_projection_rejected")
        require(dtype is None or dtype in({"dtype":"torch.float32"},{"dtype":"torch.float64"}),
          "calibration_projection_rejected")
    require(type(dim) is int and -32<=dim<=31,"calibration_projection_rejected")
    return dict(operator=operator,dim=dim,dtype=copy.deepcopy(dtype))

def exp_sum(value,node,projection):
    """Measure the declared Taylor denominator, not native exp/Softmax output."""
    p=checked_projection(projection);params=soft_parameters(node)
    require(params is not None,"calibration_projection_rejected")
    import torch,math
    x=value.detach().to(dtype=torch.float64).clone()
    require(1<=x.ndim<=32 and -x.ndim<=params["dim"]<x.ndim,"calibration_projection_rejected")
    degree=p["exp"]["degree"]
    result=torch.full_like(x,1/math.factorial(degree))
    for i in range(degree-1,-1,-1):result=result*x+1/math.factorial(i)
    return result.sum(dim=params["dim"],keepdim=True)


def components(node,paths,resolver,rows):
    """Select and validate both approximation input domains independently."""
    params=soft_parameters(node)
    if params is None:return None
    from source_approximation_preparation import require as freeze_require
    operator=params["operator"]
    exp=resolver.select(node["name"],operator,"exp",paths)
    second="reciprocal" if operator=="softmax" else "log"
    following=resolver.select(node["name"],operator,second,paths)
    derived=dict(kind="softmax_exp_sum",exp=exp,arithmetic=ARITHMETIC)
    freeze_require(rows,"approximation_calibration_missing")
    freeze_require(all(r["projection"] in({"kind":"tensor"},derived) for r in rows),
      "approximation_calibration_location_unsupported")
    result=[]
    for function,projection,profile in(("exp",{"kind":"tensor"},exp),(second,derived,following)):
        selected=[r for r in rows if r["projection"]==projection]
        freeze_require(selected,"approximation_calibration_missing")
        freeze_require(all(r["op"]==node["op"] and r["target"]==node["target"] for r in selected),
          "approximation_preparation_binding_changed")
        lo=min(r["minimum"] for r in selected);hi=max(r["maximum"] for r in selected)
        freeze_require(profile["domain"][0]<=lo and hi<=profile["domain"][1],
          "approximation_calibration_domain_exceeded")
        from benchmark_graph import digest
        result.append(dict(source_node=node["name"],operator=operator,function=function,target=node["target"],
          profile_sha256=digest(profile),domain=list(profile["domain"]),projection=projection,
          observed_minimum=lo,observed_maximum=hi,samples=sum(r["samples"] for r in selected),
          values=sum(r["values"] for r in selected),source_parameters=params))
    return result
