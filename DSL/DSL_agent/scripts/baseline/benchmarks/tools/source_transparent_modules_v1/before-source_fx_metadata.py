"""Closed original-FX source metadata; model code stays in its source sandbox."""
import re
from source_calibration_contract import require
PATH=re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*")
ACTIVATIONS={"torch.nn.ReLU":"relu","torch.nn.GELU":"gelu","torch.nn.SiLU":"silu",
             "torch.nn.Sigmoid":"sigmoid","torch.nn.Tanh":"tanh"}
NEUTRAL={"torch.nn.Identity","torch.nn.Linear","torch.nn.Flatten","torch.nn.Dropout"}
KNOWN=(*ACTIVATIONS,*NEUTRAL,"torch.nn.LayerNorm","torch.nn.Softmax","torch.nn.LogSoftmax")
def checked_source(node):
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
       item["type"]in(*KNOWN,None)and type(item["attributes"])is dict,
       "calibration_worker_protocol_error")
    kind=item["type"];attributes=item["attributes"]
    keys=({"normalized_shape","eps"}if kind=="torch.nn.LayerNorm"and attributes else
          {"inplace"}if kind in("torch.nn.ReLU","torch.nn.SiLU")else
          {"approximate"}if kind=="torch.nn.GELU"else
          {"in_features","out_features","bias"}if kind=="torch.nn.Linear"else
          {"start_dim","end_dim"}if kind=="torch.nn.Flatten"else
          {"p","inplace"}if kind=="torch.nn.Dropout"else set())
    require(set(attributes)==keys,"calibration_worker_protocol_error")
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
def describe_source(graph_module,node):
    from pytorch_approximation import module_paths
    paths=module_paths(node);item=None
    if node.op=="call_module":
        import torch
        instance=graph_module.get_submodule(node.target)
        kind=next((n for n in KNOWN if type(instance)is getattr(torch.nn,n.rsplit(".",1)[1])),None)
        attributes={}
        if kind in("torch.nn.ReLU","torch.nn.SiLU"):attributes["inplace"]=instance.inplace
        elif kind=="torch.nn.GELU":attributes["approximate"]=instance.approximate
        elif kind=="torch.nn.Linear":
            attributes=dict(in_features=instance.in_features,out_features=instance.out_features,bias=instance.bias is not None)
        elif kind=="torch.nn.Flatten":attributes=dict(start_dim=instance.start_dim,end_dim=instance.end_dim)
        elif kind=="torch.nn.Dropout":attributes=dict(p=instance.p,inplace=instance.inplace)
        elif kind=="torch.nn.LayerNorm":attributes=dict(normalized_shape=list(instance.normalized_shape),eps=instance.eps)
        item=dict(type=kind,attributes=attributes)
    result=dict(module_paths=paths,module=item)
    checked_source(dict(op=node.op,target=str(node.target),source=result))
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
