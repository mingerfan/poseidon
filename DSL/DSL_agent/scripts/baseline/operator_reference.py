"""Independent high-level NumPy references; never execute decomposition rules."""
import math
import numpy as np
from benchmark_graph import require
from benchmark_math import operation
from model_decomposition import OPS, profile, polynomial_profile

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


def absolute_polynomial(x,p):
    polynomial_profile(p);interval(x,p)
    return (np.polynomial.polynomial.polyval(x,p["coefficients"]) if p["basis"]=="power"
            else np.polynomial.chebyshev.chebval(x,p["coefficients"]))

def pair_extreme(lhs,rhs,p,maximum):
    d=absolute_polynomial(lhs-rhs,p)
    return (lhs+rhs+(d if maximum else -d))/2

def reduce_extreme(x,axes,p,maximum,keepdims):
    axes=tuple(sorted(i%x.ndim for i in axes))
    remaining=tuple(i for i in range(x.ndim) if i not in axes)
    values=x.transpose(remaining+axes).reshape(tuple(x.shape[i] for i in remaining)+(-1,))
    # Adjacent-pair tree is the declared approximation semantics, not a
    # commutative exact-max assumption. Odd leaves carry unchanged.
    while values.shape[-1]>1:
        pairs=values.shape[-1]//2
        combined=pair_extreme(values[...,:2*pairs:2],values[...,1:2*pairs:2],p,maximum)
        values=np.concatenate((combined,values[...,-1:]),axis=-1) if values.shape[-1]%2 else combined
    target=tuple(1 if i in axes else n for i,n in enumerate(x.shape) if keepdims or i not in axes) or (1,)
    return values.reshape(target)

def max_pool_reference(x,a,ideal):
    rank=x.ndim-2
    out=tuple((n+2*p-d*(k-1)-1)//st+1 for n,p,d,k,st in zip(x.shape[2:],a["padding"],a["dilation"],a["kernel"],a["stride"]))
    result=np.empty(x.shape[:2]+out)
    for coord in np.ndindex(*out):
        vectors=[]
        for offset in np.ndindex(*a["kernel"]):
            index=tuple(i*st-p+j*d for i,st,p,j,d in zip(coord,a["stride"],a["padding"],offset,a["dilation"]))
            if all(0<=i<n for i,n in zip(index,x.shape[2:])):vectors.append(x[(slice(None),slice(None))+index])
        require(vectors,"MaxPool empty window")
        stacked=np.stack(vectors,axis=-1)
        approx=reduce_extreme(stacked,[-1],a["approximation"],True,False)
        result[(slice(None),slice(None))+coord]=np.max(stacked,axis=-1) if ideal else approx
    return result

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


        elif op=="padding":
            if a["mode"]=="constant":z=np.pad(x,a["pads"],mode="constant",constant_values=a["value"])
            else:z=np.pad(x,a["pads"],mode="reflect" if a["mode"]=="reflect" else "edge")
        elif op in ("maximum","minimum"):
            z=pair_extreme(x,xs[1],a["approximation"],op=="maximum")
            if ideal:z=(np.maximum if op=="maximum" else np.minimum)(x,xs[1])
        elif op in ("max","min"):
            z=reduce_extreme(x,a["axes"],a["approximation"],op=="max",a["keepdims"])
            if ideal:
                z=np.asarray((np.max if op=="max" else np.min)(x,axis=tuple(a["axes"]),keepdims=a["keepdims"])).reshape(z.shape)
        elif op in ("max_pool1d","max_pool2d"):z=max_pool_reference(x,a,ideal)
        elif op=="broadcast":z=np.broadcast_to(x,a["shape"]).copy()
        elif op=="public_gather":z=np.take(x,a["indices"],axis=a["axis"])
        elif op=="adaptive_avg_pool":
            out=a["output_size"];z=np.empty(x.shape[:2]+tuple(out),dtype=np.float64)
            for coord in np.ndindex(*out):
                windows=tuple(slice(math.floor(i*m/n),math.ceil((i+1)*m/n))
                              for i,m,n in zip(coord,x.shape[2:],out))
                z[(slice(None),slice(None))+coord]=x[(slice(None),slice(None))+windows].mean(axis=tuple(range(2,x.ndim)))
        elif op in ("leaky_relu","prelu"):
            p=polynomial_profile(a["approximation"]);interval(x,p)
            alpha=a["negative_slope"] if op=="leaky_relu" else xs[1].reshape(
                (1,) if xs[1].size==1 else (1,x.shape[1])+((1,)*(x.ndim-2)))
            if ideal:z=np.where(x>=0,x,alpha*x)
            else:
                positive=(np.polynomial.polynomial.polyval(x,p["coefficients"]) if p["basis"]=="power"
                          else np.polynomial.chebyshev.chebval(x,p["coefficients"]))
                z=alpha*x+(1-alpha)*positive
        elif op=="logsoftmax":
            numerator=approximate("exp",x,a["exp"]);total=numerator.sum(axis=a["axis"],keepdims=True)
            p=polynomial_profile(a["log"]);interval(total,p)
            if ideal:
                largest=x.max(axis=a["axis"],keepdims=True)
                z=x-largest-np.log(np.exp(x-largest).sum(axis=a["axis"],keepdims=True))
            else:
                logarithm=(np.polynomial.polynomial.polyval(total,p["coefficients"]) if p["basis"]=="power"
                           else np.polynomial.chebyshev.chebval(total,p["coefficients"]))
                z=x-logarithm
        elif op=="dot":z=np.asarray([np.dot(x,xs[1])])
        elif op=="matmul":z=x@xs[1]
        elif op in ("exp","reciprocal","rsqrt"):z=approximate(op,x,a["approximation"],ideal)
        elif op=="variance":
            z=np.var(x,axis=tuple(a["axes"]),keepdims=a["keepdims"]).reshape(
                tuple(1 if i in [v%x.ndim for v in a["axes"]] else n for i,n in enumerate(x.shape)
                      if a["keepdims"] or i not in [v%x.ndim for v in a["axes"]]) or (1,))
        elif op in ("layer_norm","group_norm","instance_norm"):
            original_shape=x.shape
            if op=="layer_norm":
                axes=tuple(range(x.ndim-len(a["normalized_shape"]),x.ndim));gain,bias=xs[1:]
            else:
                groups=x.shape[1] if op=="instance_norm" else a["groups"]
                x=x.reshape(x.shape[0],groups,-1);axes=(2,)
                gain=xs[1].reshape((1,original_shape[1])+((1,)*(len(original_shape)-2)))
                bias=xs[2].reshape(gain.shape)
            centered=x-x.mean(axis=axes,keepdims=True)
            variance=x.var(axis=axes,keepdims=True)+a["eps"]
            z=(centered*approximate("rsqrt",variance,a["approximation"],ideal)).reshape(original_shape)*gain+bias
        elif op=="global_avg_pool":z=x.mean(axis=tuple(range(2,x.ndim)),keepdims=True)
        elif op=="dropout":z=x.copy()
        elif op=="sqrt":z=x*approximate("rsqrt",x,a["approximation"],ideal)
        elif op in ("relu","gelu","silu","sigmoid","tanh","log"):
            p=polynomial_profile(a["approximation"]);interval(x,p)
            if ideal:
                if op=="relu":z=np.maximum(x,0)
                elif op=="gelu":z=x*.5*(1+np.vectorize(math.erf)(x/math.sqrt(2)))
                elif op=="silu":z=x/(1+np.exp(-x))
                elif op=="sigmoid":z=1/(1+np.exp(-x))
                elif op=="tanh":z=np.tanh(x)
                else:z=np.log(x)
            elif p["basis"]=="power":z=np.polynomial.polynomial.polyval(x,p["coefficients"])
            else:z=np.polynomial.chebyshev.chebval(x,p["coefficients"])
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
