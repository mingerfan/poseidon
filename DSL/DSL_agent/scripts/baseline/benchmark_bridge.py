"""Lossless format bridge, with explicit capability failures. Never choose by model ID."""
import copy
from benchmark_graph import FORMAT, validate


def from_legacy(old):
    if old["schema"] not in (2,3,4,5): raise ValueError("Export schema1 via catalog_graph_migration first")
    specs=copy.deepcopy(old["inputs"]) if old["schema"]==3 else [dict(name="x",shape=old["input_shape"])]
    g=dict(format=FORMAT,id=old["id"].replace("-","_"),inputs=specs,
           constants=copy.deepcopy(old["constants"]),nodes=[],
           outputs=[dict(name="output",value=old["output"])])
    for n in old["nodes"]:
        op=n["op"]; refs=list(n["inputs"]); a={}
        if op=="linear":
            refs.append(n["weight"])
            if n["bias"] is not None: refs.append(n["bias"])
        elif op=="batch_norm":
            count=len(old["constants"][n["running_mean"]])
            refs.extend([n["running_mean"],n["running_var"]])
            for field,default in (("weight",1.0),("bias",0.0)):
                key=n[field]
                if key is None:
                    key="generated_"+n["id"]+"_"+field
                    g["constants"][key]=[default]*count
                refs.append(key)
            a={"eps":n["eps"]}
        elif op.startswith("conv"):
            refs.append(n["weight"])
            if n["bias"] is not None: refs.append(n["bias"])
            rank=1 if op=="conv1d" else 2
            a={k:n[k] for k in ("stride","padding")}
            a.update(dilation=n.get("dilation",[1]*rank),groups=n.get("groups",1))
        else: a={k:v for k,v in n.items() if k not in ("id","op","inputs")}
        g["nodes"].append(dict(id=n["id"],op=op,inputs=refs,attrs=a,outputs=[n["id"]]))
    # Legacy IDs permit hyphens; graph names do not.
    names={}
    for inp in g["inputs"]: names[inp["name"]]=inp["name"].replace("-","_")
    names.update({n:n.replace("-","_") for n in g["constants"]})
    names.update({n["id"]:n["id"].replace("-","_") for n in g["nodes"]})
    for inp in g["inputs"]: inp["name"]=names[inp["name"]]
    g["constants"]={names[n]:v for n,v in g["constants"].items()}
    for n in g["nodes"]:
        n["id"]=names[n["id"]];n["outputs"]=[names[x] for x in n["outputs"]]
        n["inputs"]=[names[x] for x in n["inputs"]]
    g["outputs"][0]["value"]=names[g["outputs"][0]["value"]]
    validate(g)
    return g


def to_legacy(g):
    """Current compatible subset only. No truncation, ID dispatch, or silent approximation."""
    checked=validate(g)
    if len(g["outputs"])!=1: raise ValueError("Legacy backend has one logical output")
    if len(g["inputs"])==1:
        schema=5; old=dict(schema=schema,id=g["id"],input_shape=g["inputs"][0]["shape"])
        names={g["inputs"][0]["name"]:"x"}
    else:
        schema=3
        old=dict(schema=schema,id=g["id"],inputs=copy.deepcopy(g["inputs"]))
        names={x["name"]:x["name"] for x in g["inputs"]}
    old.update(constants=copy.deepcopy(g["constants"]),nodes=[],output=g["outputs"][0]["value"])
    for node in g["nodes"]:
        if len(node["outputs"])!=1: raise ValueError("Legacy backend cannot bind split outputs")
        op=node["op"];args=[names.get(r,r) for r in node["inputs"]];a=copy.deepcopy(node["attrs"])
        n=dict(id=node["outputs"][0],op=op,inputs=args)
        if op=="linear":
            n.update(inputs=args[:1],weight=args[1],bias=args[2] if len(args)==3 else None)
        elif op=="batch_norm":
            n.update(inputs=args[:1],running_mean=args[1],running_var=args[2],weight=args[3],bias=args[4])
        elif op.startswith("conv"):
            n.update(inputs=args[:1],weight=args[1],bias=args[2] if len(args)==3 else None)
        n.update(a);old["nodes"].append(n)
    old["output"]=names.get(old["output"],old["output"])
    from model_graph import validate_graph
    validate_graph(old)
    return old
