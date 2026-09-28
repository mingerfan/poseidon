"""Trusted chunked bindings for real virtual-ring helpers; not yet public DSL.

The fixed old binding supplies mathematical parameters and layout permutations.
Physical ciphertexts remain P-periodic; its larger Q is only an index space for
public maps, never a new physical period or rotation key requirement.
"""
import math
from benchmark_graph import validate,require,digest,canonical
from upstream_adapters import virtual_ring as vr
from upstream_adapters import virtual_node as old

ADAPTER="chunked-virtual-prefix-v1"
FAMILIES=old.FAMILIES

def _mapping(value,rows,q):
    """Lift a finite Q-index map to the full virtual ring without modulo-Q wrap."""
    groups={}
    for row in rows:
        require(len(row["mask"])==q,"Chunk map extent")
        for dst,factor in enumerate(row["mask"]):
            if factor!=0:
                src=(dst+row["step"])%q
                mask=groups.setdefault(src-dst,[0.]*vr.NT)
                mask[dst]+=factor
    require(groups,"Empty chunk map")
    out=None
    for step,mask in sorted(groups.items()):
        term=value.rotate(step)*mask
        out=term if out is None else out+term
    return out

def _reference_helper(value,call,family):
    """No payload: independent audit of pinned MPCB algorithms, never a reference answer."""
    def const(values,dtype="torch.float64"):
        return old._Constant(list(values)+[0.]*(vr.NT-len(values)),dtype)
    if family=="HE_MPBN":
        mean,var,gamma,beta=call["parameters"]
        gain=[g/math.sqrt(v+call["eps"]) for g,v in zip(gamma,var)]
        bias=[b-m*g for b,m,g in zip(beta,mean,gain)]
        return value*const(gain)+const(bias)
    weight,bias=call["parameters"];m=len(weight);n=len(weight[0])
    if family=="HE_ReshapeLinear":
        weight=[[row[src] for src in call["packed_to_original"]] for row in weight]
    x=value.binary(const([1.]*n,"torch.float32"),"multiply",True);x=x+x.rotate(-n)
    it=(n+m-1)//m;rows=[]
    for i,row in enumerate(weight):
        row=row[i%n:]+row[:i%n];rows.append(row+[0.]*(it*m-n))
    out=None
    for i in range(m):
        u=[rows[j][k*m+i] for k in range(it) for j in range(m)]
        term=x.rotate(i)*const(u);out=term if out is None else out+term
    for j in range((it-1).bit_length()):out=out+out.rotate((1<<j)*m)
    return out+const(bias)

def _actual_helper(value,call,family,helpers):
    import numpy as np
    import torch
    from types import SimpleNamespace
    array=np.array([value],dtype=object)
    if family=="HE_MPBN":
        mean,var,gamma,beta=[torch.tensor(v,dtype=torch.float64) for v in call["parameters"]]
        params=SimpleNamespace(running_mean=mean,running_var=var,weight=gamma,bias=beta,eps=call["eps"])
        return helpers.HE_MPBN(array,params)[0]
    w,b=[torch.tensor(v,dtype=torch.float64) for v in call["parameters"]]
    params=SimpleNamespace(weight=w,bias=b)
    if family=="HE_Linear":return helpers.HE_Linear(None,array,params)[0]
    return helpers.HE_ReshapeLinear(None,array,params,reshape=call["reshape"])[0]

def _run(binding,ciphers,zero,helpers=None):
    p=binding["period"];spec=binding["inner"];family=binding["helper"]
    context=vr.Context(p,limit=binding.get("work",1024))
    cls=vr.VirtualExpr if helpers is None else vr.proxy_type(helpers.hc.Expr)
    x=cls.input_chunks(context,ciphers,binding["input_count"]);out=None;calls=[]
    for call in spec["calls"]:
        value=_mapping(x,call["before"],spec["period"]) if "before" in call else x
        value=(_reference_helper(value,call,family) if helpers is None else
               _actual_helper(value,call,family,helpers))
        require(isinstance(value,vr.VirtualExpr) and value.context is context,"Chunk helper return context")
        if "after" in call:value=_mapping(value,call["after"],spec["period"])
        out=value if out is None else out+value
        calls.append(dict(helper=family,parameters_sha256=digest(call)))
    selected=binding["output_chunk"]
    result=out.export_chunk(selected["offset"],selected["elements"],binding["output_count"],zero)
    return result,dict(adapter=ADAPTER,helper=family,inner_calls=calls,virtual=context.report(),
                       unchanged_upstream_functions=True,bootstrap_removed=False,output_contribution_proven=False)

def bind_node(model,node_id,period,family,output_chunk):
    require(type(period) is int and period in (4,8,16,32,64,128,256),"Chunk helper period")
    checked=validate(model);nodes=[n for n in model["nodes"] if n["id"]==node_id]
    require(len(nodes)==1 and family in FAMILIES,"Chunk helper family/node")
    node=nodes[0];sizes=[math.prod(checked["shapes"][node["inputs"][0]]),
                       math.prod(checked["shapes"][node["outputs"][0]])]
    q=1<<max(2,(max(sizes)-1).bit_length())
    inner=old.bind_node(model,node_id,q,family)
    n=math.prod(inner["input_shape"]);m=math.prod(inner["output_shape"])
    require(max(n,m)<=256 and (n+period-1)//period<=4 and (m+period-1)//period<=4,
            "Chunk helper physical I/O budget")
    require(type(output_chunk) is int and 0<=output_chunk<(m+period-1)//period,"Chunk helper output index")
    offset=output_chunk*period
    binding=dict(schema=1,adapter=ADAPTER,helper=family,node_id=node_id,input_value=inner["input_value"],
        output_value=inner["output_value"],input_shape=inner["input_shape"],output_shape=inner["output_shape"],
        period=period,input_count=n,output_count=m,input_chunks=(n+period-1)//period,
        output_chunk=dict(index=output_chunk,offset=offset,elements=min(period,m-offset)),
        inner=inner,encrypted_zero_parameter=True,virtual_slots=65536,physical_slots=16384)
    _,record=_run(binding,[old._Opaque()] * binding["input_chunks"],old._Opaque())
    binding["work"]=max(1,len(record["virtual"]["physical_operations"]))
    require(binding["work"]<=1024,"Chunk helper work budget")
    return binding

def expected_record(binding):
    _,record=_run(binding,[old._Opaque()] * binding["input_chunks"],old._Opaque())
    return record

def verify_record(binding,record):
    require(type(record) is dict and canonical(record)==canonical(expected_record(binding)),
            "Chunk helper transcript differs from pinned public-structure replay")

def apply(binding,ciphers,zero,helpers):
    require(binding.get("adapter")==ADAPTER and binding.get("helper") in FAMILIES,"Chunk helper identity")
    require(type(ciphers) in (list,tuple) and len(ciphers)==binding["input_chunks"],"Chunk helper arguments")
    result,record=_run(binding,ciphers,zero,helpers)
    verify_record(binding,record)
    return result,record

def probe(binding,chunks,zero):
    """Finite contribution probe only; never the runner mathematical reference."""
    p=binding["period"];q=binding["inner"]["period"]
    values=tuple(v for chunk in chunks for v in chunk)[:binding["input_count"]]
    values+=tuple(0. for _ in range(q-len(values)))
    zeros=tuple(zero[i%p] for i in range(q))
    output=old.probe(binding["inner"],values,zeros)
    sel=binding["output_chunk"];part=output[sel["offset"]:sel["offset"]+sel["elements"]]
    return tuple(part)+tuple(0. for _ in range(p-len(part)))
