"""Independent ordered NumPy storage reference for the tested compositions.

No normalizer, decomposition, DSL lowering or candidate helper is used here.
The tested normalization operations evaluate their declared approximation.
"""
import math
import numpy as np

def rsqrt(value, profile):
    if profile["method"] != "newton": raise ValueError("Reference approximation")
    lo,hi=profile["domain"]
    if np.any(value<lo) or np.any(value>hi): raise ValueError("Reference approximation domain")
    y=np.full_like(value,1/math.sqrt(hi))
    for _ in range(profile["iterations"]): y=y*(1.5-.5*value*y*y)
    return y

def registered(op,args,a):
    x=args[0]
    if op=="linear": return np.matmul(x,args[1].T)+(args[2] if len(args)==3 else 0)
    if op=="matmul": return np.matmul(x,args[1])
    if op=="sum": return x.sum(axis=tuple(a["axes"]),keepdims=a["keepdims"])
    if op=="mean": return x.mean(axis=tuple(a["axes"]),keepdims=a["keepdims"])
    if op=="stack": return np.stack(args,axis=a["axis"])
    if op=="concat": return np.concatenate(args,axis=a["axis"])
    if op=="public_gather": return np.take(x,a["indices"],axis=a["axis"]).copy()
    if op=="layer_norm":
        axes=tuple(range(x.ndim-len(a["normalized_shape"]),x.ndim))
        mean=x.mean(axis=axes,keepdims=True)
        variance=((x-mean)**2).mean(axis=axes,keepdims=True)+a["eps"]
        return (x-mean)*rsqrt(variance,a["approximation"])*args[1]+args[2]
    if op=="group_norm":
        grouped=x.reshape(x.shape[0],a["groups"],-1)
        mean=grouped.mean(axis=-1,keepdims=True)
        variance=((grouped-mean)**2).mean(axis=-1,keepdims=True)+a["eps"]
        z=((grouped-mean)*rsqrt(variance,a["approximation"])).reshape(x.shape)
        public_shape=(1,x.shape[1])+((1,)*(x.ndim-2))
        return z*args[1].reshape(public_shape)+args[2].reshape(public_shape)
    if op in ("conv1d","conv2d"):
        spatial=x.ndim-2
        stride,padding,dilation=(a[key] for key in ("stride","padding","dilation"))
        w=args[1]
        padded=np.pad(x,[(0,0),(0,0)]+[(v,v) for v in padding])
        output_shape=tuple((x.shape[2+i]+2*padding[i]-dilation[i]*(w.shape[2+i]-1)-1)//stride[i]+1 for i in range(spatial))
        out=np.zeros((x.shape[0],w.shape[0],*output_shape),dtype=np.float64)
        per_in=x.shape[1]//a["groups"];per_out=w.shape[0]//a["groups"]
        for batch in range(x.shape[0]):
            for out_channel in range(w.shape[0]):
                group=out_channel//per_out
                for position in np.ndindex(output_shape):
                    value=0.
                    for in_channel in range(per_in):
                        for kernel in np.ndindex(w.shape[2:]):
                            where=tuple(position[i]*stride[i]+kernel[i]*dilation[i] for i in range(spatial))
                            value+=padded[(batch,group*per_in+in_channel,*where)]*w[(out_channel,in_channel,*kernel)]
                    out[(batch,out_channel,*position)]=value+(args[2][out_channel] if len(args)==3 else 0)
        return out
    raise ValueError("No independent reference registered for this test")

def index(spec):
    result=[]
    for item in spec:
        if item["kind"]=="slice":result.append(slice(item["start"],item["stop"],item["step"]))
        elif item["kind"]=="int":result.append(item["value"])
        elif item["kind"]=="ellipsis":result.append(Ellipsis)
        elif item["kind"]=="newaxis":result.append(None)
        else:raise ValueError("Reference index")
    return tuple(result)

def evaluate(source,inputs):
    values={item["name"]:np.asarray(inputs[item["name"]],dtype=np.float64).copy(order="C") for item in source["inputs"]}
    for name,value in source["constants"].items():
        values[name]=np.asarray(value,dtype=np.float64);values[name].flags.writeable=False
    for node in source["nodes"]:
        args=[values[name] for name in node["inputs"]];x=args[0];a=node["attrs"];op=node["op"]
        if op=="registered":z=registered(a["operator"],args,a["parameters"])
        elif op=="alias":z=x
        elif op=="clone":z=x.copy(order="K")
        elif op=="contiguous":z=x if x.flags.c_contiguous else np.ascontiguousarray(x)
        elif op=="reshape":z=x.reshape(a["shape"])
        elif op=="view":
            z=x.reshape(a["shape"])
            if not np.shares_memory(z,x):raise ValueError("Reference incompatible view")
        elif op=="transpose":z=x.swapaxes(a["dim0"],a["dim1"])
        elif op=="permute":z=x.transpose(a["dims"])
        elif op=="slice":z=x[index(a["index"])]
        elif op=="expand":z=np.broadcast_to(x,a["shape"])
        elif op in ("add","subtract","multiply"):
            y=args[1] if op=="multiply" else args[1]*a["alpha"]
            fn={"add":np.add,"subtract":np.subtract,"multiply":np.multiply}[op]
            if a["inplace"]:fn(x,y,out=x);z=x
            else:z=fn(x,y)
        elif op=="square":z=np.square(x)
        else:raise ValueError("No independent ordered reference")
        values[node["output"]]=z
    return {o["name"]:values[o["value"]].copy() for o in source["outputs"]}
