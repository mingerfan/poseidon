"""Closed original-FX source metadata; model code stays in its source sandbox."""
import re
import math
from source_calibration_contract import require
PATH=re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*")
ACTIVATIONS={"torch.nn.ReLU":"relu","torch.nn.GELU":"gelu","torch.nn.SiLU":"silu",
             "torch.nn.Sigmoid":"sigmoid","torch.nn.Tanh":"tanh"}
TRANSPARENT_ATTRIBUTES={
 "torch.nn.Conv1d":{"in_channels","out_channels","kernel_size","stride","padding","dilation","groups","padding_mode","bias"},
 "torch.nn.Conv2d":{"in_channels","out_channels","kernel_size","stride","padding","dilation","groups","padding_mode","bias"},
 "torch.nn.BatchNorm1d":{"num_features","eps","momentum","affine","track_running_stats","training"},
 "torch.nn.BatchNorm2d":{"num_features","eps","momentum","affine","track_running_stats","training"},
 "torch.nn.AvgPool1d":{"kernel_size","stride","padding","ceil_mode","count_include_pad","divisor_override"},
 "torch.nn.AvgPool2d":{"kernel_size","stride","padding","ceil_mode","count_include_pad","divisor_override"},
 "torch.nn.AdaptiveAvgPool1d":{"output_size"},
 "torch.nn.AdaptiveAvgPool2d":{"output_size"},
}
NEUTRAL={"torch.nn.Identity","torch.nn.Linear","torch.nn.Flatten","torch.nn.Dropout",*TRANSPARENT_ATTRIBUTES}

def pair_attribute(value,rank,*,zero=False):
    values=[value]*rank if type(value)is int else list(value)if type(value)in(list,tuple)else None
    require(type(values)is list and len(values)==rank and
            all(type(v)is int and (0 if zero else 1)<=v<=65536for v in values),
            "calibration_worker_protocol_error")
    return values

def checked_transparent(kind,attributes):
    if kind not in TRANSPARENT_ATTRIBUTES:return
    require(set(attributes)==TRANSPARENT_ATTRIBUTES[kind],"calibration_worker_protocol_error")
    rank=1 if kind.endswith("1d")else 2
    if "Conv"in kind:
        require(all(type(attributes[k])is int and 1<=attributes[k]<=65536for k in("in_channels","out_channels","groups"))and
                attributes["in_channels"]%attributes["groups"]==attributes["out_channels"]%attributes["groups"]==0and
                type(attributes["bias"])is bool and attributes["padding_mode"]in("zeros","reflect","replicate","circular"),
                "calibration_worker_protocol_error")
        for key in("kernel_size","stride","dilation","padding"):
            require(type(attributes[key])is list,"calibration_worker_protocol_error")
            pair_attribute(attributes[key],rank,zero=key=="padding")
    elif "BatchNorm"in kind:
        require(type(attributes["num_features"])is int and 1<=attributes["num_features"]<=65536and
                type(attributes["eps"])in(int,float)and math.isfinite(attributes["eps"])and 0<=attributes["eps"]<=1and
                all(type(attributes[k])is bool for k in("affine","track_running_stats","training"))and
                (attributes["momentum"]is None or type(attributes["momentum"])in(int,float)and
                 math.isfinite(attributes["momentum"])and 0<=attributes["momentum"]<=1),"calibration_worker_protocol_error")
    elif "Adaptive"in kind:
        require(type(attributes["output_size"])is list,"calibration_worker_protocol_error")
        pair_attribute(attributes["output_size"],rank)
    else:
        require(all(type(attributes[k])is bool for k in("ceil_mode","count_include_pad"))and
                (attributes["divisor_override"]is None or type(attributes["divisor_override"])is int and
                 1<=attributes["divisor_override"]<=65536),"calibration_worker_protocol_error")
        for key in("kernel_size","stride","padding"):
            require(type(attributes[key])is list,"calibration_worker_protocol_error")
            pair_attribute(attributes[key],rank,zero=key=="padding")

def describe_transparent(instance,kind):
    rank=1 if kind.endswith("1d")else 2
    result={k:getattr(instance,k,None)for k in sorted(TRANSPARENT_ATTRIBUTES[kind])}
    if "Conv"in kind:result["bias"]=instance.bias is not None
    for key in("kernel_size","stride","padding","dilation","output_size"):
        if key in result:
            if key=="stride"and result[key]is None:result[key]=instance.kernel_size
            result[key]=pair_attribute(result[key],rank,zero=key=="padding")
    checked_transparent(kind,result);return result
KNOWN=(*ACTIVATIONS,*NEUTRAL,"torch.nn.LayerNorm","torch.nn.Softmax","torch.nn.LogSoftmax")
def checked_source(node,*,compound=False):
    require(type(compound) is bool,"calibration_worker_protocol_error")
    source=node.get("source")
    require(type(source)is dict and set(source)=={"module_paths","module"},
            "calibration_worker_protocol_error")
    paths=source["module_paths"]
    require(type(paths)is list and len(paths)<=64 and len(paths)==len(set(paths))and
       all(type(p)is str and 0<len(p)<=256 and PATH.fullmatch(p)for p in paths),
       "calibration_worker_protocol_error")
    item=source["module"]
    if node["op"]!="call_module":
        require(item is None,"calibration_worker_protocol_error");return source
    require(node["target"]in paths and paths[-1]==node["target"]and
       type(item)is dict and set(item)=={"type","attributes"}and
       item["type"]in(*KNOWN,*(("torch.nn.GroupNorm","torch.nn.InstanceNorm1d","torch.nn.InstanceNorm2d")if compound else()),None)and type(item["attributes"])is dict,
       "calibration_worker_protocol_error")
    kind=item["type"];attributes=item["attributes"]
    if compound and kind in("torch.nn.GroupNorm","torch.nn.InstanceNorm1d","torch.nn.InstanceNorm2d"):
        from source_normalization_calibration import checked_attributes
        checked_attributes(kind,attributes);return source
    keys=({"dim"} if compound and kind in("torch.nn.Softmax","torch.nn.LogSoftmax") else
          TRANSPARENT_ATTRIBUTES[kind]if kind in TRANSPARENT_ATTRIBUTES else
          {"normalized_shape","eps"}if kind=="torch.nn.LayerNorm"and attributes else
          {"inplace"}if kind in("torch.nn.ReLU","torch.nn.SiLU")else
          {"approximate"}if kind=="torch.nn.GELU"else
          {"in_features","out_features","bias"}if kind=="torch.nn.Linear"else
          {"start_dim","end_dim"}if kind=="torch.nn.Flatten"else
          {"p","inplace"}if kind=="torch.nn.Dropout"else set())
    require(set(attributes)==keys,"calibration_worker_protocol_error")
    checked_transparent(kind,attributes)
    if compound and kind in("torch.nn.Softmax","torch.nn.LogSoftmax"):
        require(attributes["dim"] is None or type(attributes["dim"]) is int and -32<=attributes["dim"]<=31,
          "calibration_worker_protocol_error")
    if "inplace"in attributes:require(type(attributes["inplace"])is bool,"calibration_worker_protocol_error")
    if "approximate"in attributes:require(attributes["approximate"]in("none","tanh"),"calibration_worker_protocol_error")
    if kind=="torch.nn.Linear":
        require(all(type(attributes[k])is int and 1<=attributes[k]<=1000000for k in("in_features","out_features"))and
                type(attributes["bias"])is bool,"calibration_worker_protocol_error")
    if kind=="torch.nn.Flatten":
        require(all(type(attributes[k])is int and -32<=attributes[k]<=31for k in keys),"calibration_worker_protocol_error")
    if kind=="torch.nn.LayerNorm"and attributes:
        import math
        shape=attributes["normalized_shape"]
        require(type(shape)is list and 1<=len(shape)<=32 and all(type(v)is int and 1<=v<=65536for v in shape)
            and type(attributes["eps"])in(int,float)and math.isfinite(attributes["eps"])and 0<=attributes["eps"]<=1,
            "calibration_worker_protocol_error")
    if kind=="torch.nn.Dropout":
        require(type(attributes["p"])in(int,float)and 0<=attributes["p"]<=1,"calibration_worker_protocol_error")
    return source
def describe_source(graph_module,node,*,compound=False):
    from pytorch_approximation import module_paths
    paths=module_paths(node);item=None
    if node.op=="call_module":
        import torch
        instance=graph_module.get_submodule(node.target)
        candidates=(*KNOWN,*(("torch.nn.GroupNorm","torch.nn.InstanceNorm1d","torch.nn.InstanceNorm2d")if compound else()))
        kind=next((n for n in candidates if type(instance)is getattr(torch.nn,n.rsplit(".",1)[1])),None)
        attributes={}
        if compound and kind in("torch.nn.GroupNorm","torch.nn.InstanceNorm1d","torch.nn.InstanceNorm2d"):
            from source_normalization_calibration import describe
            attributes=describe(instance,kind)
        elif compound and kind in("torch.nn.Softmax","torch.nn.LogSoftmax"):attributes=dict(dim=instance.dim)
        elif kind in TRANSPARENT_ATTRIBUTES:attributes=describe_transparent(instance,kind)
        elif kind in("torch.nn.ReLU","torch.nn.SiLU"):attributes["inplace"]=instance.inplace
        elif kind=="torch.nn.GELU":attributes["approximate"]=instance.approximate
        elif kind=="torch.nn.Linear":
            attributes=dict(in_features=instance.in_features,out_features=instance.out_features,bias=instance.bias is not None)
        elif kind=="torch.nn.Flatten":attributes=dict(start_dim=instance.start_dim,end_dim=instance.end_dim)
        elif kind=="torch.nn.Dropout":attributes=dict(p=instance.p,inplace=instance.inplace)
        elif kind=="torch.nn.LayerNorm":attributes=dict(normalized_shape=list(instance.normalized_shape),eps=instance.eps)
        item=dict(type=kind,attributes=attributes)
    result=dict(module_paths=paths,module=item)
    checked_source(dict(op=node.op,target=str(node.target),source=result),compound=compound)
    return result

def layer_norm_parameters(node):
    """Bind the original module or literal functional LayerNorm parameters."""
    if node["op"]=="call_module":
        source=checked_source(node);item=source["module"]
        require(item["type"]=="torch.nn.LayerNorm"and set(item["attributes"])=={"normalized_shape","eps"},
                "calibration_projection_rejected")
        return item["attributes"]["normalized_shape"],item["attributes"]["eps"]
    require(node["op"]=="call_function"and node["target"]=="torch.nn.functional.layer_norm",
            "calibration_projection_rejected")
    args=node["args"];kwargs=node["kwargs"]
    require(type(args)is dict and set(args)=={"tuple"}and type(kwargs)is dict and set(kwargs)=={"dict"},
            "calibration_projection_rejected")
    values=dict(zip(("input","normalized_shape","weight","bias","eps"),args["tuple"]))
    for k,v in kwargs["dict"].items():
        require(k not in values,"calibration_projection_rejected");values[k]=v
    shape=values.get("normalized_shape");eps=values.get("eps",1e-5)
    require(type(shape)is dict and len(shape)==1and set(shape)<= {"list","tuple"},
            "calibration_projection_rejected")
    shape=next(iter(shape.values()))
    import math
    require(type(shape)is list and 1<=len(shape)<=32and all(type(v)is int and 1<=v<=65536for v in shape)
        and type(eps)in(int,float)and math.isfinite(eps)and 0<=eps<=1,"calibration_projection_rejected")
    return shape,eps
