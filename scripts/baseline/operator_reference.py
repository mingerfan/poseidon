"""Independent high-level NumPy references; never execute decomposition rules."""
import math
import numpy as np
from benchmark_graph import require
from benchmark_math import operation
from model_decomposition import OPS, profile

def interval(x,p):
    lo,hi=p["domain"]
    require(np.isfinite(x).all() and np.all(x>=lo) and np.all(x<=hi),
            "Reference input outside explicit approximation domain")

def approximate(op,x,p,ideal=False):
    profile(op,p);interval(x,p)
    if ideal:return {"exp":np.exp,"reciprocal":np.reciprocal,"rsqrt":lambda v:1/np.sqrt(v)}[op](x)
    if op=="exp":
        out=np.ones_like(x);term=np.ones_like(x)
        for k in range(1,p["degree"]+1):term=term*x/k;out=out+term
        return out
    if op=="reciprocal":
        # Closed-form geometric sum, independently from the emitted product recurrence.
        q=1-x/p["domain"][1]
        return (1-q**(2**p["iterations"]))/x
    out=np.full_like(x,1/math.sqrt(p["domain"][1]))
    for _ in range(p["iterations"]):out=out+(out*(1-x*out*out))*.5
    return out

def evaluate(model,inputs,ideal=False):
    require(set(inputs)=={i["name"] for i in model["inputs"]},"Reference input names")
    values={n:np.asarray(v,dtype=np.float64) for n,v in model["constants"].items()}
    for i in model["inputs"]:
        x=np.asarray(inputs[i["name"]])
        require(x.dtype==np.float64 and list(x.shape)==i["shape"] and np.isfinite(x).all(),"Reference input type/shape")
        values[i["name"]]=x
    for node in model["nodes"]:
        op=node["op"];xs=[values[r] for r in node["inputs"]];a=node["attrs"];x=xs[0]
        if op in OPS:z=operation(op,xs,a)
        elif op=="dot":z=np.asarray([np.dot(x,xs[1])])
        elif op=="matmul":z=x@xs[1]
        elif op in ("exp","reciprocal","rsqrt"):z=approximate(op,x,a["approximation"],ideal)
        elif op=="rms_norm":
            v=np.mean(x*x,axis=-1,keepdims=True)+a["eps"]
            z=x*approximate("rsqrt",v,a["approximation"],ideal)*xs[1]
        elif op=="softmax":
            # Validate approximate intermediates even when reporting ideal math.
            numerator=approximate("exp",x,a["exp"])
            denominator=numerator.sum(axis=a["axis"],keepdims=True)
            inverse=approximate("reciprocal",denominator,a["reciprocal"])
            if ideal:
                exact=np.exp(x-x.max(axis=a["axis"],keepdims=True))
                z=exact/exact.sum(axis=a["axis"],keepdims=True)
            else:z=numerator*inverse
        elif op=="rope":
            h=x.size//2;out=np.empty_like(x)
            for j in range(h):
                angle=a["position"]/a["theta"]**(2*j/x.size)
                out[j]=x[j]*math.cos(angle)-x[j+h]*math.sin(angle)
                out[j+h]=x[j]*math.sin(angle)+x[j+h]*math.cos(angle)
            z=out
        else:raise ValueError("No registered independent reference")
        parts=z if op=="split" else [z]
        require(len(parts)==len(node["outputs"]),"Reference output count")
        for n,v in zip(node["outputs"],parts):
            require(np.isfinite(v).all(),"Nonfinite operator reference")
            values[n]=np.asarray(v,dtype=np.float64)
    return {o["name"]:values[o["value"]] for o in model["outputs"]}
