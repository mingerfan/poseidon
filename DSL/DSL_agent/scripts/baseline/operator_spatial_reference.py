"""Direct spatial mathematics and pinned Torch references, independent of lowering."""
import itertools, math
import numpy as np
from benchmark_graph import require

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


def evaluate(model,inputs,ideal=False):
    from operator_reference import evaluate as existing
    from operator_graph import FORMAT as MATRIX_FORMAT
    require(type(model)is dict and model.get("format")=="poseidon-operator-graph-v4","Spatial reference version")
    require(set(inputs)=={s["name"]for s in model["inputs"]},"Reference input names")
    values={n:np.asarray(v,dtype=np.float64)for n,v in model["constants"].items()}
    for spec in model["inputs"]:
        value=np.asarray(inputs[spec["name"]])
        require(value.dtype==np.float64 and list(value.shape)==spec["shape"]and np.isfinite(value).all(),"Reference input type/shape")
        values[spec["name"]]=value
    for node in model["nodes"]:
        xs=[values[r]for r in node["inputs"]];op=node["op"]
        if op=="resize":outputs={node["outputs"][0]:original_resize(xs[0],node["attrs"])}
        elif op in("conv_transpose1d","conv_transpose2d"):
            outputs={node["outputs"][0]:original_transpose(xs[0],xs[1],xs[2]if len(xs)==3 else None,node["attrs"])}
        else:
            refs=list(dict.fromkeys(node["inputs"]))
            local=dict(format=MATRIX_FORMAT,id=model["id"],inputs=[dict(name=r,shape=list(values[r].shape))for r in refs],
                constants={},nodes=[node],outputs=[dict(name=r,value=r)for r in node["outputs"]])
            outputs=existing(local,{r:values[r]for r in refs},ideal=ideal)
        for name,value in outputs.items():
            values[name]=value.reshape(1)if op=="matmul"and value.ndim==0 else value
    return {o["name"]:values[o["value"]]for o in model["outputs"]}
