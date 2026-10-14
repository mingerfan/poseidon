"""Independent NumPy mathematical reference; never reads decomposition plans."""
import numpy as np
from benchmark_graph import require
from static_tensor_transform_plan import OPERATORS

def original(op, value, attrs):
    if op in ("repeat","tile"): return np.tile(value,attrs["repeats"])
    if op=="repeat_interleave": return np.repeat(value,attrs["repeats"],axis=attrs["axis"])
    if op=="flip": return np.flip(value,axis=tuple(attrs["axes"]))
    if op=="roll": return np.roll(value,shift=tuple(attrs["shifts"]) if attrs["axes"] is not None else attrs["shifts"][0],
                                 axis=None if attrs["axes"] is None else tuple(attrs["axes"]))
    raise ValueError("Tensor transform reference operation")

def evaluate(model, inputs, ideal=False):
    from operator_spatial_reference import evaluate as prior
    require(type(model) is dict and model.get("format")=="poseidon-operator-graph-v5","Tensor transform reference version")
    require(set(inputs)=={s["name"] for s in model["inputs"]},"Reference input names")
    values={n:np.asarray(v,dtype=np.float64) for n,v in model["constants"].items()}
    for spec in model["inputs"]:
        value=np.asarray(inputs[spec["name"]])
        require(value.dtype==np.float64 and list(value.shape)==spec["shape"] and np.isfinite(value).all(),"Reference input type/shape")
        values[spec["name"]]=value
    for node in model["nodes"]:
        op=node["op"]
        if op in OPERATORS:
            require(len(node["inputs"])==len(node["outputs"])==1,"Reference transform arity")
            result={node["outputs"][0]:original(op,values[node["inputs"][0]],node["attrs"])}
        else:
            refs=list(dict.fromkeys(node["inputs"]))
            local=dict(format="poseidon-operator-graph-v4",id=model["id"],
                inputs=[dict(name=r,shape=list(values[r].shape)) for r in refs],constants={},nodes=[node],
                outputs=[dict(name=r,value=r) for r in node["outputs"]])
            result=prior(local,{r:values[r] for r in refs},ideal=ideal)
        values.update(result)
    return {o["name"]:values[o["value"]] for o in model["outputs"]}
