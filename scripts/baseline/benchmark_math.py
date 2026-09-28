"""Independent NumPy/index-loop reference, never calls Torch, lowering or helpers."""
import itertools
import math
import numpy as np
from benchmark_graph import validate


def spatial(op, xs, a):
    x = xs[0]; rank = 1 if op.endswith("1d") else 2
    batched = x.ndim == rank+2
    if not batched: x = x[None]
    conv = op.startswith("conv")
    if conv:
        w = xs[1]; kernel=w.shape[2:]; co=w.shape[0]; dilation=a["dilation"]; groups=a["groups"]
    else:
        kernel=a["kernel"]; co=x.shape[1]; dilation=[1]*rank; groups=1
    outshape=tuple((n+2*p-d*(k-1)-1)//s+1 for n,p,d,k,s in
                   zip(x.shape[2:],a["padding"],dilation,kernel,a["stride"]))
    out=np.zeros((x.shape[0],co,*outshape),dtype=np.float64)
    for batch in range(x.shape[0]):
        for channel in range(co):
            for dst in np.ndindex(outshape):
                total=0.0; count=0
                ci=x.shape[1]//groups
                channels=range((channel//(co//groups))*ci,(channel//(co//groups)+1)*ci) if conv else [channel]
                for src_channel in channels:
                    for kk in np.ndindex(tuple(kernel)):
                        src=tuple(t*s-p+k*d for t,s,p,k,d in
                                  zip(dst,a["stride"],a["padding"],kk,dilation))
                        valid=all(0<=i<n for i,n in zip(src,x.shape[2:]))
                        if valid:
                            v=x[(batch,src_channel,*src)]
                            total+=v*w[(channel,src_channel%ci,*kk)] if conv else v
                            count+=1
                if conv:
                    if len(xs)==3: total+=xs[2][channel]
                else:
                    total/=math.prod(kernel) if a["count_include_pad"] else count
                out[(batch,channel,*dst)]=total
    return out if batched else out[0]


def operation(op, xs, a):
    x=xs[0]
    if op=="add": return x+xs[1]
    if op=="subtract": return x-xs[1]
    if op=="multiply": return x*xs[1]
    if op=="negate": return -x
    if op=="square": return x*x
    if op=="power": return np.power(x,a["exponent"])
    if op=="linear":
        w=xs[1]; result=np.empty((*x.shape[:-1],w.shape[0]),dtype=np.float64)
        for idx in np.ndindex(x.shape[:-1]):
            for j in range(w.shape[0]):
                result[(*idx,j)]=math.fsum(float(x[(*idx,k)])*float(w[j,k]) for k in range(w.shape[1]))
                if len(xs)==3: result[(*idx,j)]+=xs[2][j]
        return result
    if op=="flatten": return x.reshape(-1)
    if op=="reshape": return x.reshape(a["shape"])
    if op=="transpose": return np.swapaxes(x,a["dim0"],a["dim1"])
    if op=="permute": return np.transpose(x,a["dims"])
    if op=="rotate": return np.roll(x.reshape(-1),-a["step"]).reshape(x.shape)
    if op=="concat": return np.concatenate(xs,axis=a["axis"])
    if op=="stack": return np.stack(xs,axis=a["axis"])
    if op=="slice":
        idx=[slice(None)]*x.ndim; idx[a["axis"]]=slice(a["start"],a["stop"],a["step"])
        return x[tuple(idx)]
    if op=="split": return np.split(x,np.cumsum(a["sections"])[:-1],axis=a["axis"])
    if op in ("sum","mean"):
        fn=np.sum if op=="sum" else np.mean
        return np.asarray(fn(x,axis=tuple(a["axes"]),keepdims=a["keepdims"])).reshape(
            [1] if len(a["axes"])==x.ndim and not a["keepdims"] else
            tuple(1 if i in {j%x.ndim for j in a["axes"]} else n for i,n in enumerate(x.shape)
                  if a["keepdims"] or i not in {j%x.ndim for j in a["axes"]}))
    if op=="batch_norm":
        mean,var,gamma,beta=xs[1:]; s=[1,x.shape[1]]+[1]*(x.ndim-2)
        return (x-mean.reshape(s))*(gamma/np.sqrt(var+a["eps"])).reshape(s)+beta.reshape(s)
    if op=="polynomial":
        fn=np.polynomial.chebyshev.chebval if a["basis"]=="chebyshev" else np.polynomial.polynomial.polyval
        return fn(x,xs[1])
    if op.startswith(("conv","avg_pool")): return spatial(op,xs,a)
    raise ValueError("Unknown reference operator")


def evaluate(model, inputs):
    check=validate(model)
    if set(inputs)!={s["name"] for s in model["inputs"]}: raise ValueError("Reference inputs")
    values={k:np.asarray(v,dtype=np.float64) for k,v in model["constants"].items()}
    for spec in model["inputs"]:
        x=inputs[spec["name"]]
        if x.dtype!=np.float64 or list(x.shape)!=spec["shape"] or not np.isfinite(x).all():
            raise ValueError("Reference input shape/type")
        values[spec["name"]]=x.copy()
    for node in model["nodes"]:
        result=operation(node["op"],[values[r] for r in node["inputs"]],node["attrs"])
        parts=result if node["op"]=="split" else [result]
        for name,value in zip(node["outputs"],parts):
            value=np.asarray(value,dtype=np.float64)
            if list(value.shape)!=check["shapes"][name] or not np.isfinite(value).all():
                raise ValueError("Reference output shape/nonfinite")
            values[name]=value
    return {o["name"]:values[o["value"]].copy() for o in model["outputs"]}
