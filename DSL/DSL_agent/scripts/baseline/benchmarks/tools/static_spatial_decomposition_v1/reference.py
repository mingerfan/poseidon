"""Direct spatial mathematics and pinned Torch references, independent of lowering."""
import itertools, math
import numpy as np
import torch
import torch.nn.functional as F
from benchmark_math import operation as core_math
from benchmark_torch import evaluate as core_torch

def original_resize(x,a):
    factors=a["scale_factor"]
    target=tuple(a["size"]) if factors is None else tuple(math.floor(n*f) for n,f in zip(x.shape[2:],factors))
    scales=[n/m if factors is None or a["recompute_scale_factor"] else 1/f
            for n,m,f in zip(x.shape[2:],target,factors or [1.]*len(target))]
    vectorized=len(target)==2 and (sum(target)<=128 or (x.shape[1]>3 and math.prod(x.shape[2:])==1))
    result=np.empty((*x.shape[:2],*target),dtype=np.float64)
    for leading in np.ndindex(x.shape[:2]):
        for dst in np.ndindex(target):
            if a["mode"] in ("nearest","nearest-exact"):
                indices=[]
                for axis,(j,n,m,scale) in enumerate(zip(dst,x.shape[2:],target,scales)):
                    if vectorized and a["mode"]=="nearest" and n==m:k=j
                    elif vectorized and a["mode"]=="nearest" and m==2*n:k=j//2
                    elif vectorized:
                        k=int(np.floor(np.float32(np.float32(j+(.5 if a["mode"]=="nearest-exact" else 0.))*np.float32(scale))))
                    else:
                        point=j*scale if a["mode"]=="nearest" else max(scale*(j+.5)-.5,0.)+.5
                        k=int(np.floor(np.float32(point)))
                    indices.append(min(n-1,k))
                result[(*leading,*dst)]=x[(*leading,*indices)]
            else:
                neighbours=[]
                for j,n,m,scale in zip(dst,x.shape[2:],target,scales):
                    if n==m:
                        neighbours.append(((j,1.),));continue
                    position=(j*(n-1)/(m-1) if m>1 else 0.) if a["align_corners"] else max(scale*(j+.5)-.5,0.)
                    lo=min(n-1,int(math.floor(position)));frac=position-math.floor(position)
                    neighbours.append(((lo,1-frac),(min(lo+1,n-1),frac)))
                terms=[]
                for selected in itertools.product(*neighbours):
                    indices=[i for i,w in selected];weight=math.prod(w for i,w in selected)
                    terms.append(float(x[(*leading,*indices)])*weight)
                result[(*leading,*dst)]=math.fsum(terms)
    return result

def original_transpose(x,w,b,a):
    rank=w.ndim-2;unbatched=x.ndim==rank+1
    if unbatched:x=x[None]
    output=[(n-1)*s-2*p+d*(k-1)+1+o for n,s,p,d,k,o in
            zip(x.shape[2:],a["stride"],a["padding"],a["dilation"],w.shape[2:],a["output_padding"])]
    if a["output_size"] is not None:output=list(a["output_size"][-rank:])
    groups=a["groups"];inside=x.shape[1]//groups;per=w.shape[1]
    result=np.zeros((x.shape[0],per*groups,*output),dtype=np.float64)
    if b is not None:result+=b.reshape(1,-1,*([1]*rank))
    for batch in range(x.shape[0]):
        for cin in range(x.shape[1]):
            group=cin//inside
            for source in np.ndindex(x.shape[2:]):
                for cout in range(per):
                    for kernel in np.ndindex(w.shape[2:]):
                        dst=tuple(i*s-p+j*d for i,s,p,j,d in
                                  zip(source,a["stride"],a["padding"],kernel,a["dilation"]))
                        if all(0<=i<n for i,n in zip(dst,output)):
                            result[(batch,group*per+cout,*dst)]+=x[(batch,cin,*source)]*w[(cin,cout,*kernel)]
    return result[0] if unbatched else result

def torch_operation(op,xs,a):
    x=xs[0]
    if op=="resize":
        return F.interpolate(x,size=a["size"],scale_factor=a["scale_factor"],mode=a["mode"],
            align_corners=a["align_corners"],recompute_scale_factor=a["recompute_scale_factor"],
            antialias=a["antialias"])
    if op.startswith("conv_transpose"):
        rank=1 if op.endswith("1d") else 2
        constructor=torch.nn.ConvTranspose1d if rank==1 else torch.nn.ConvTranspose2d
        module=constructor(xs[1].shape[0],xs[1].shape[1]*a["groups"],xs[1].shape[2:],
             stride=a["stride"],padding=a["padding"],output_padding=a["output_padding"],
             groups=a["groups"],dilation=a["dilation"],bias=len(xs)==3,dtype=torch.float64).eval()
        with torch.no_grad():
            module.weight.copy_(xs[1])
            if len(xs)==3:module.bias.copy_(xs[2])
            return module(x,output_size=a["output_size"])
    if op=="square":return torch.square(x)
    if op=="add":return x+xs[1]
    if op=="mean":
        y=x.mean(tuple(a["axes"]),keepdim=a["keepdims"])
        return y.reshape(1) if y.ndim==0 else y
    if op=="flatten":return x.flatten()
    if op=="linear":return F.linear(x,xs[1],xs[2] if len(xs)==3 else None)
    raise ValueError("Unregistered reference "+op)

def evaluate(model,inputs,*,torch_reference=False):
    convert=(lambda x:torch.tensor(x,dtype=torch.float64)) if torch_reference else (lambda x:np.asarray(x,dtype=np.float64))
    values={k:convert(v) for k,v in model["constants"].items()}
    values.update({k:convert(v) for k,v in inputs.items()})
    for n in model["nodes"]:
        op,a=n["op"],n["attrs"];xs=[values[k]for k in n["inputs"]]
        if torch_reference:value=torch_operation(op,xs,a)
        elif op=="resize":value=original_resize(xs[0],a)
        elif op.startswith("conv_transpose"):value=original_transpose(xs[0],xs[1],xs[2] if len(xs)==3 else None,a)
        else:value=core_math(op,xs,a)
        values[n["outputs"][0]]=value
    return {o["name"]:(values[o["value"]].detach().numpy().copy() if torch_reference else values[o["value"]].copy())
            for o in model["outputs"]}
