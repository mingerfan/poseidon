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
    keys=({"inplace"}if kind in("torch.nn.ReLU","torch.nn.SiLU")else
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
        item=dict(type=kind,attributes=attributes)
    result=dict(module_paths=paths,module=item)
    checked_source(dict(op=node.op,target=str(node.target),source=result))
    return result
