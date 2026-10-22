"""Bound input-statistics projections for inference normalization."""
import copy,math
from source_calibration_contract import require
MODULES={"torch.nn.GroupNorm","torch.nn.InstanceNorm1d","torch.nn.InstanceNorm2d"}
FIELDS={
 "torch.nn.GroupNorm":{"num_groups","num_channels","eps","affine"},
 "torch.nn.InstanceNorm1d":{"num_features","eps","momentum","affine","track_running_stats","training"},
 "torch.nn.InstanceNorm2d":{"num_features","eps","momentum","affine","track_running_stats","training"}}
TARGETS={"torch.nn.functional.group_norm":"group_norm","torch.nn.functional.instance_norm":"instance_norm"}
PROJECTIONS={"group_norm_variance","instance_norm_variance"}
def checked_attributes(kind,values):
    require(kind in MODULES and type(values) is dict and set(values)==FIELDS[kind],"calibration_worker_protocol_error")
    require(type(values["eps"]) in(int,float) and math.isfinite(values["eps"]) and 0<=values["eps"]<=1
     and type(values["affine"]) is bool,"calibration_worker_protocol_error")
    if kind=="torch.nn.GroupNorm":
        require(all(type(values[k]) is int and 1<=values[k]<=65536 for k in("num_groups","num_channels"))
         and values["num_channels"]%values["num_groups"]==0,"calibration_worker_protocol_error")
    else:
        require(type(values["num_features"]) is int and 1<=values["num_features"]<=65536
          and all(type(values[k]) is bool for k in("track_running_stats","training"))
          and(values["momentum"] is None or type(values["momentum"]) in(int,float)
             and math.isfinite(values["momentum"]) and 0<=values["momentum"]<=1),"calibration_worker_protocol_error")
    return values
def describe(instance,kind):
    values={k:getattr(instance,k)for k in sorted(FIELDS[kind])}
    checked_attributes(kind,values);return values
def parameters(node):
    if node["op"]=="call_module":
        from source_fx_metadata import checked_source
        m=checked_source(node,compound=True)["module"];kind=m["type"]
        if kind not in MODULES:return None
        a=checked_attributes(kind,m["attributes"])
        if kind=="torch.nn.GroupNorm":
            return dict(operator="group_norm",groups=a["num_groups"],channels=a["num_channels"],eps=a["eps"],
              spatial_rank=None,input_statistics=True)
        return dict(operator="instance_norm",groups=None,channels=a["num_features"],eps=a["eps"],
          spatial_rank=1 if kind.endswith("1d") else 2,input_statistics=a["training"] or not a["track_running_stats"])
    op=TARGETS.get(node["target"])if node["op"]=="call_function"else None
    if op is None:return None
    args=node["args"];kwargs=node["kwargs"]
    require(type(args) is dict and set(args)=={"tuple"} and type(args["tuple"]) is list
      and type(kwargs) is dict and set(kwargs)=={"dict"} and type(kwargs["dict"]) is dict,"calibration_projection_rejected")
    keys=("input","num_groups","weight","bias","eps")if op=="group_norm"else(
      "input","running_mean","running_var","weight","bias","use_input_stats","momentum","eps")
    require(len(args["tuple"])<=len(keys) and set(kwargs["dict"])<=set(keys),"calibration_projection_rejected")
    values=dict(zip(keys,args["tuple"]))
    for key,value in kwargs["dict"].items():
        require(key not in values,"calibration_projection_rejected");values[key]=value
    eps=values.get("eps",1e-5)
    require(type(eps) in(int,float) and math.isfinite(eps) and 0<=eps<=1,"calibration_projection_rejected")
    if op=="group_norm":
        groups=values.get("num_groups")
        require(type(groups) is int and 1<=groups<=65536,"calibration_projection_rejected")
        return dict(operator=op,groups=groups,channels=None,eps=eps,spatial_rank=None,input_statistics=True)
    stats=values.get("use_input_stats",True)
    require(type(stats) is bool,"calibration_projection_rejected")
    return dict(operator=op,groups=None,channels=None,eps=eps,spatial_rank=None,input_statistics=stats)
def variance(value,node,projection):
    params=parameters(node)
    require(params is not None and params["input_statistics"] and
      projection=={"kind":params["operator"]+"_variance"},"calibration_projection_rejected")
    x=value.detach().clone()
    if params["operator"]=="group_norm":
        require(x.ndim in(3,4) and x.shape[1]%params["groups"]==0,"calibration_projection_rejected")
        if params["channels"] is not None:require(x.shape[1]==params["channels"],"calibration_projection_rejected")
        z=x.reshape(x.shape[0],params["groups"],-1)
        return z.var(dim=-1,unbiased=False,keepdim=True)+params["eps"]
    rank=params["spatial_rank"]
    if rank is not None:
        require(x.ndim in(rank+1,rank+2),"calibration_projection_rejected")
        if x.ndim==rank+1:x=x.unsqueeze(0)
    require(x.ndim in(3,4),"calibration_projection_rejected")
    if params["channels"] is not None:require(x.shape[1]==params["channels"],"calibration_projection_rejected")
    return x.var(dim=tuple(range(2,x.ndim)),unbiased=False,keepdim=True)+params["eps"]
def components(node,paths,resolver,rows):
    params=parameters(node)
    if params is None:return None
    if not params["input_statistics"]:return []
    from source_approximation_preparation import require as freeze_require
    projection={"kind":params["operator"]+"_variance"}
    freeze_require(rows,"approximation_calibration_missing")
    freeze_require(all(r["projection"]==projection for r in rows),"approximation_calibration_location_unsupported")
    freeze_require(all(r["op"]==node["op"] and r["target"]==node["target"] for r in rows),
      "approximation_preparation_binding_changed")
    p=resolver.select(node["name"],params["operator"],"rsqrt",paths)
    lo=min(r["minimum"]for r in rows);hi=max(r["maximum"]for r in rows)
    freeze_require(p["domain"][0]<=lo and hi<=p["domain"][1],"approximation_calibration_domain_exceeded")
    from benchmark_graph import digest
    return [dict(source_node=node["name"],operator=params["operator"],function="rsqrt",target=node["target"],
      profile_sha256=digest(p),domain=list(p["domain"]),projection=projection,source_parameters=params,
      observed_minimum=lo,observed_maximum=hi,samples=sum(r["samples"]for r in rows),values=sum(r["values"]for r in rows))]
