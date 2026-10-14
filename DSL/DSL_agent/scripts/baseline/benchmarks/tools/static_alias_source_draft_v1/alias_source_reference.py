"""Independent array/storage reference; no normalizer or DSL conversion calls."""
import numpy as np

def polynomial(x,config):
    if config["method"]!="polynomial":raise ValueError("Explicit polynomial")
    lo,hi=config["domain"]
    if np.any(x<lo) or np.any(x>hi):raise ValueError("Approximation input outside domain")
    if config["basis"]=="power":return sum(float(c)*np.power(x,i) for i,c in enumerate(config["coefficients"]))
    if config["basis"]=="chebyshev":return np.polynomial.chebyshev.chebval(x,config["coefficients"])
    raise ValueError("Explicit basis")
def indices(spec):
    result=[]
    for x in spec:
        kind=x["kind"]
        if kind=="slice":result.append(slice(x["start"],x["stop"],x["step"]))
        elif kind=="int":result.append(x["value"])
        elif kind=="newaxis":result.append(None)
        elif kind=="ellipsis":result.append(Ellipsis)
        else:raise ValueError("Static index")
    return tuple(result)
def evaluate(request,inputs,*,approximate):
    source=request["source"];values={}
    for x in source["inputs"]:
        a=np.asarray(inputs[x["name"]],dtype=np.float64)
        if a.shape!=tuple(x["shape"]):raise ValueError("Input shape")
        values[x["name"]]=a.copy(order="C")
    for n,c in source["constants"].items():
        values[n]=np.asarray(c,dtype=np.float64);values[n].flags.writeable=False
    for node in source["nodes"]:
        op=node["op"];a=node["attrs"];args=[values[r] for r in node["inputs"]];x=args[0]
        if op=="alias":z=x
        elif op=="clone":z=x.copy(order="K")
        elif op=="contiguous":z=x if x.flags.c_contiguous else np.ascontiguousarray(x)
        elif op in ("reshape","view"):
            z=x.reshape(a["shape"])
            if op=="view" and not np.shares_memory(x,z):raise ValueError("Incompatible view")
        elif op=="flatten":
            start,end=a["start_dim"]%x.ndim,a["end_dim"]%x.ndim
            z=x.reshape(x.shape[:start]+(int(np.prod(x.shape[start:end+1])),)+x.shape[end+1:])
        elif op=="transpose":z=x.swapaxes(a["dim0"],a["dim1"])
        elif op=="permute":z=x.transpose(a["dims"])
        elif op=="slice":z=x[indices(a["index"])]
        elif op=="expand":z=np.broadcast_to(x,a["shape"])
        elif op in ("add","subtract","multiply"):
            other=args[1]*a["alpha"] if op!="multiply" else args[1]
            function={"add":np.add,"subtract":np.subtract,"multiply":np.multiply}[op]
            if a["inplace"]:function(x,other,out=x);z=x
            else:z=function(x,other)
        elif op=="square":z=np.square(x)
        elif op=="relu":
            z=polynomial(x,request["approximations"]["relu"]) if approximate else np.maximum(x,0)
            if a["inplace"]:x[...] = z;z=x
        else:raise ValueError("Reference operation")
        values[node["output"]]=z
    return {o["name"]:np.asarray(values[o["value"]],dtype=np.float64).copy() for o in source["outputs"]}
