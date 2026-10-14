"""Fixed graph bindings to unchanged 65536-slot upstream MPBN/Linear helpers."""
import copy,math
from benchmark_graph import validate,require,canonical,digest
from upstream_adapters import virtual_ring as vr
from upstream_adapters.spatial_mapped import mapping,map_work,transform
from upstream_adapters.periodic_ring import proxy_type as mapping_proxy

ADAPTER="virtual-prefix-v1"
FAMILIES=("HE_MPBN","HE_Linear","HE_ReshapeLinear")

def bind_node(model,node_id,period,family):
    checked=validate(model);nodes=[n for n in model["nodes"] if n["id"]==node_id]
    require(len(nodes)==1 and family in FAMILIES,"Virtual helper node/family")
    node=nodes[0];shape=checked["shapes"][node["inputs"][0]];outshape=checked["shapes"][node["outputs"][0]]
    require(type(period) is int and period in (4,8,16,32,64,128,256),"Virtual helper period")
    require(max(math.prod(shape),math.prod(outshape))<=period,"Virtual helper intermediate exceeds P")
    require(all(n in model["constants"] for n in node["inputs"][1:]),"Virtual fixed public parameters")
    binding=dict(schema=1,adapter=ADAPTER,helper=family,node_id=node_id,input_value=node["inputs"][0],
                 output_value=node["outputs"][0],input_shape=shape,output_shape=outshape,period=period,
                 virtual_slots=vr.NT,physical_slots=16384,original_node=copy.deepcopy(node),
                 original_parameters=[copy.deepcopy(model["constants"][n]) for n in node["inputs"][1:]],
                 encrypted_zero_parameter=True)
    calls=[]
    if family=="HE_MPBN":
        require(node["op"]=="batch_norm","MPBN requires BatchNorm")
        count=math.prod(shape);spatial=math.prod(shape[2:]);channels=shape[1]
        expanded=[[values[(i//spatial)%channels] for i in range(count)] for values in binding["original_parameters"]]
        mean,var,gamma,beta=expanded;eps=node["attrs"]["eps"]
        gain=[g/math.sqrt(v+eps) for g,v in zip(gamma,var)]
        shift=[b-g*m for b,g,m in zip(beta,gain,mean)]
        require(all(math.isfinite(v) and abs(v)<=1024 for v in gain+shift),"MPBN derived constant bound")
        calls=[dict(input_count=count,output_count=count,parameters=expanded,eps=eps)]
        work=8
    else:
        require(node["op"]=="linear","Linear helper requires Linear")
        n=shape[-1];m=outshape[-1];rows=math.prod(shape[:-1])
        weight=binding["original_parameters"][0];bias=(binding["original_parameters"][1] if len(binding["original_parameters"])==2 else [0.]*m)
        perm=list(range(n));reshape=None;padded=max(n,m)
        if family=="HE_ReshapeLinear":
            producers={v:t for t in model["nodes"] for v in t["outputs"]};first=producers.get(node["inputs"][0])
            require(first is not None and first["op"]=="flatten","ReshapeLinear requires flatten producer")
            original_shape=checked["shapes"][first["inputs"][0]]
            require(rows==1 and math.prod(original_shape)==n,"ReshapeLinear whole tensor flatten")
            binding.update(input_value=first["inputs"][0],input_shape=original_shape,flatten_node=copy.deepcopy(first))
            h,w=(original_shape[-2:] if len(original_shape)>=3 else (1,n))
            c=n//(h*w);ko=2 if c>=4 and ((c+3)//4)*4*h*w<=period else 1
            to=max((c+ko*ko-1)//(ko*ko),(m+ko*ko*h*w-1)//(ko*ko*h*w))
            padded=to*ko*ko*h*w
            require(padded<=period,"ReshapeLinear padded geometry exceeds P")
            reshape=dict(ko=ko,ho=h,wo=w,to=to)
            # Packed position -> original flatten index, exactly matching upstream
            # rearrange of weight columns: (to ko1 ko2 ho wo) -> (to ho ko1 wo ko2).
            perm=[((((t*ko+a)*ko+b)*h+y)*w+x)
                  for t in range(to) for y in range(h) for a in range(ko) for x in range(w) for b in range(ko)]
        require(padded<=period and rows<=64,"Linear padded width/call budget")
        derived=[list(row)+[0.]*(padded-n) for row in weight]
        for row in range(rows):
            before=mapping([(dst,row*n+src,1.) for dst,src in enumerate(perm) if src<n],period)
            after=mapping([(row*m+j,j,1.) for j in range(m)],period)
            calls.append(dict(input_count=padded,output_count=m,parameters=[derived,list(bias)],
                              reshape=reshape,before=before,after=after,packed_to_original=perm))
        it=(padded+m-1)//m;reductions=(it-1).bit_length();bits=period.bit_length()-1
        # Conservative full-block accounting. Runtime independently enforces exact work.
        core=12+m*(2*bits+12)+reductions*(5*bits+25)
        work=sum(core+map_work(c["before"])+map_work(c["after"]) for c in calls)+len(calls)-1
    require(1<=work<=1024,"Virtual helper work exceeds 1024")
    binding.update(calls=calls,work=work,transformation="full sparse virtual ring; fixed public layout/padding; final logical prefix projection")
    return binding

def apply(binding,cipher,zero_cipher,helpers,mpcb):
    import numpy as np
    import torch
    from types import SimpleNamespace
    require(binding["adapter"]==ADAPTER and binding["helper"] in FAMILIES,"Virtual adapter identity")
    p=binding["period"];family=binding["helper"];records=[];mapping_record=dict(operations=[],rotations=[])
    mapped=mapping_proxy(helpers.hc.Expr);virtual=vr.proxy_type(helpers.hc.Expr)
    x=mapped(cipher,p,mapping_record,binding["work"]);out=None;count=0
    for call in binding["calls"]:
        before=transform(x,call["before"]).value if "before" in call else cipher
        context=vr.Context(p,limit=binding["work"])
        value=virtual.input(context,before,call["input_count"]);array=np.array([value],dtype=object)
        if family=="HE_MPBN":
            mean,var,gamma,beta=[torch.tensor(v,dtype=torch.float64) for v in call["parameters"]]
            params=SimpleNamespace(running_mean=mean,running_var=var,weight=gamma,bias=beta,eps=call["eps"])
            result=helpers.HE_MPBN(array,params)[0]
        else:
            weight,bias=[torch.tensor(v,dtype=torch.float64) for v in call["parameters"]]
            params=SimpleNamespace(weight=weight,bias=bias)
            if family=="HE_Linear":result=helpers.HE_Linear(None,array,params)[0]
            else:result=helpers.HE_ReshapeLinear(None,array,params,reshape=call["reshape"])[0]
        require(isinstance(result,vr.VirtualExpr) and result.context is context,"Virtual helper return context")
        value=result.export_prefix(call["output_count"],zero_cipher)
        records.append(dict(helper=family,parameters_sha256=digest(call),virtual=context.report()))
        count+=len(context.physical)
        term=transform(mapped(value,p,mapping_record,binding["work"]),call["after"]) if "after" in call else mapped(value,p,mapping_record,binding["work"])
        out=term if out is None else out+term
    count+=len(mapping_record["operations"]);require(count<=binding["work"],"Virtual total actual work")
    record=dict(adapter=ADAPTER,helper=family,binding_sha256=digest(binding),inner_calls=records,
                mapping_operations=len(mapping_record["operations"]),mapping_rotations=mapping_record["rotations"],
                operation_count=count,unchanged_upstream_functions=True,bootstrap_removed=False,output_contribution_proven=False)
    verify_record(binding,record)
    return out.value,record

def verify_record(binding,record):
    require(type(record) is dict and set(record)=={"adapter","helper","binding_sha256","inner_calls","mapping_operations",
            "mapping_rotations","operation_count","unchanged_upstream_functions","bootstrap_removed","output_contribution_proven"},"Virtual record fields")
    require(record["adapter"]==ADAPTER and record["helper"]==binding["helper"] and record["binding_sha256"]==digest(binding)
            and record["unchanged_upstream_functions"] is True and record["bootstrap_removed"] is False
            and record["output_contribution_proven"] is False,"Virtual record identity")
    calls=record["inner_calls"];require(type(calls) is list and len(calls)==len(binding["calls"]),"Virtual invocation count")
    expected=sum(map_work(c["before"])+map_work(c["after"]) for c in binding["calls"] if "before" in c)+len(calls)-1
    require(type(record["mapping_operations"]) is int and record["mapping_operations"]==expected,"Virtual mapping count")
    rotations=[dict(requested=r["step"],normalized=r["step"],steps=[1<<i for i in range(binding["period"].bit_length()-1) if r["step"]&(1<<i)])
               for c in binding["calls"] for side in ("before","after") for r in c.get(side,[])]
    require(canonical(record["mapping_rotations"])==canonical(rotations),"Virtual mapping rotations")
    count=expected
    for call,actual in zip(binding["calls"],calls):
        require(type(actual) is dict and set(actual)=={"helper","parameters_sha256","virtual"} and actual["helper"]==binding["helper"]
                and actual["parameters_sha256"]==digest(call),"Virtual call parameters")
        v=actual["virtual"]
        require(type(v) is dict and set(v)==set(vr.Context(binding["period"]).report()),"Virtual transcript fields")
        require(v["virtual_slots"]==65536 and v["physical_slots"]==16384 and v["block_period"]==binding["period"]
                and v["max_live_blocks"]==64 and type(v["peak_live_blocks"]) is int and 0<=v["peak_live_blocks"]<=64
                and v["secret_values_inspected"] is False and v["bootstrap_removed"] is False
                and v["nonperiodic_constant_truncation"] is False,"Virtual representation identity")
        require(type(v["physical_operations"]) is list and len(v["physical_operations"])<=binding["work"]
                and all(x in ("mask","add","subtract","multiply","negate","rotate","public_materialize") for x in v["physical_operations"]),"Virtual physical operations")
        predicted=expected_virtual_record(call,binding["helper"],binding["period"],binding["work"])
        differences={k:dict(actual=v[k],expected=predicted[k]) for k in v if canonical(v[k])!=canonical(predicted[k])}
        require(not differences,"Virtual transcript differs from pinned public-structure replay: "+repr(differences)[:12000])
        count+=len(v["physical_operations"])
        events=v["events"];require(type(events) is list and 2<=len(events)<=2048,"Virtual event budget")
        require(events[0]["operation"]=="input" and events[0]["count"]==call["input_count"] and
                events[-1]["operation"]=="output_projection" and events[-1]["selected_start"]==0 and
                events[-1]["selected_count"]==call["output_count"],"Virtual input/output transcript")
        require(events[-1]["encrypted_zero_argument_used"]==events[-1]["public_only"],"Virtual public output materialization")
        for constant in v["full_constants"]:
            if constant.get("kind")=="scalar_zero":
                require(constant==dict(kind="scalar_zero"),"Virtual scalar record");continue
            require(set(constant)=={"kind","entries","dtype","sha256","nonzero_blocks","nonzero_values"} and
                    constant["kind"]=="full_virtual_vector" and constant["entries"]==65536 and
                    constant["dtype"] in ("torch.float32","torch.float64") and
                    type(constant["sha256"]) is str and len(constant["sha256"])==64 and
                    all(c in "0123456789abcdef" for c in constant["sha256"]),"Virtual full constant record")
            indices=constant["nonzero_blocks"]
            require(type(indices) is list and indices==sorted(set(indices)) and len(indices)<=64
                    and all(type(i) is int and 0<=i<65536//binding["period"] for i in indices),"Virtual constant blocks")
    require(type(record["operation_count"]) is int and record["operation_count"]==count<=binding["work"],"Virtual total work")

def probe(binding,x,zero):
    """Finite public influence only; never the encrypted numerical reference."""
    family=binding["helper"];shape=binding["input_shape"];params=binding["original_parameters"]
    if family=="HE_MPBN":
        mean,var,gamma,beta=params;s=math.prod(shape[2:]);channels=shape[1];out=[];public=all(g==0 for g in gamma)
        for i in range(math.prod(shape)):
            c=(i//s)%channels;g=gamma[c]/math.sqrt(var[c]+binding["original_node"]["attrs"]["eps"])
            out.append((zero[i] if public else 0.)+x[i]*g+beta[c]-mean[c]*g)
    else:
        w=params[0];bias=params[1] if len(params)==2 else [0.]*len(w)
        n=len(w[0]);m=len(w);rows=math.prod(shape)//n;public=all(v==0 for r in w for v in r)
        out=[sum(x[r*n+i]*w[j][i] for i in range(n))+bias[j]+(zero[j] if public else 0.)
             for r in range(rows) for j in range(m)]
    return tuple(out+[0.]*(binding["period"]-len(out)))

class _Opaque:
    """No payload, no expected numerical answer; records Expr operation topology."""
    def __add__(self,x):return self
    __radd__=__add__
    __sub__=__add__
    __rsub__=__add__
    __mul__=__add__
    __rmul__=__add__
    def __neg__(self):return self
    def rotate(self,k):return self

class _Constant:
    def __init__(self,values,dtype):self.values=values;self.dtype=dtype
    def detach(self):return self
    def cpu(self):return self
    def reshape(self,*args):return self
    def tolist(self):return self.values

def expected_virtual_record(call,family,period,limit):
    """Independent public-structure replay of pinned MPCB.BN/Linear.

    This is an audit oracle for transcript consistency, never an execution path
    or mathematical reference. The actual helper alone produces the returned
    Hecate Expr. Constants are recomputed in full, including their zero tails.
    """
    c=vr.Context(period,limit=limit);x=vr.VirtualExpr.input(c,_Opaque(),call["input_count"])
    def constant(values,dtype="torch.float64"):
        require(len(values)<=65536,"Audit constant extent")
        return _Constant(list(values)+[0.]*(65536-len(values)),dtype)
    if family=="HE_MPBN":
        mean,var,gamma,beta=call["parameters"]
        gain=[g/math.sqrt(v+call["eps"]) for g,v in zip(gamma,var)]
        bias=[b-m*g for b,m,g in zip(beta,mean,gain)]
        out=x*constant(gain)+constant(bias)
    else:
        weight,bias=call["parameters"];m=len(weight);n=len(weight[0])
        if family=="HE_ReshapeLinear":
            weight=[[row[src] for src in call["packed_to_original"]] for row in weight]
        x=x.binary(constant([1.]*n,"torch.float32"),"multiply",True);x=x+x.rotate(-n)
        it=(n+m-1)//m;rows=[]
        for i,row in enumerate(weight):
            row=row[i%n:]+row[:i%n];rows.append(row+[0.]*(it*m-n))
        out=None
        for i in range(m):
            u=[rows[j][k*m+i] for k in range(it) for j in range(m)]
            term=x.rotate(i)*constant(u);out=term if out is None else out+term
        for j in range((it-1).bit_length()):out=out+out.rotate((1<<j)*m)
        out=out+constant(bias)
    out.export_prefix(call["output_count"],_Opaque())
    return c.report()
