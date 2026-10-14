"""Agent-owned MultiheadAttention graph draft; production intake is unchanged."""
import hashlib,math
from pathlib import Path
import numpy as np
from benchmark_graph import NAME,digest,canonical,dimensions,shape
from model_resources import resolve,identity
from model_decomposition import FORMAT as GRAPH_FORMAT,decompose,profile

FORMAT="poseidon-mha-source-plan-draft-v1"
MODULE_FORMAT="torch-2.0.1-mha-public-state-v1"
MESSAGES={
"mha_module_rejected":"Only the fixed native eval MultiheadAttention module is registered.",
"mha_state_rejected":"Public parameter names, shapes, precision and values must match the module configuration.",
"mha_arguments_rejected":"Arguments must match the fixed static MultiheadAttention signature.",
"mha_shape_rejected":"Static query/key/value shapes must agree with batch, sequence and projection dimensions.",
"mha_mask_rejected":"Public mask type, shape and finite-value representation must match the static attention signature.",
"mha_all_masked_unsupported":"An entirely masked attention row has no registered positive-denominator approximation.",
"mha_approximation_rejected":"Explicit exponential and reciprocal profiles are required.",
"mha_resource_rejected":"The graph and public data must fit the selected resource profile.",
"mha_graph_contract_rejected":"The normalized graph did not satisfy the existing operator/decomposition contract.",
"mha_integrity_rejected":"The plan must match its trusted source request and current implementation.",
}
class MhaError(ValueError):
    def __init__(self,code,parameter):
        if code not in MESSAGES or parameter not in {"module","weights","input","mask","approximation","resource_profile","call","plan"}:
            raise ValueError("Invalid trusted MHA rejection")
        self.code,self.parameter=code,parameter;super().__init__(MESSAGES[code])
    def feedback(self):
        return dict(format="poseidon-mha-source-rejection-draft-v1",code=self.code,
                    layer="integrity" if self.code=="mha_integrity_rejected" else "input",
                    location=dict(operator="multi_head_attention",parameter=self.parameter),
                    context=dict(constraint=MESSAGES[self.code]),retry_policy="never_automatic")
def fail(code,parameter):raise MhaError(code,parameter)
def require(condition,code,parameter):
    if not condition:fail(code,parameter)
def data(value,limits,parameter):
    count=0
    def visit(v,depth):
        nonlocal count
        count+=1
        require(count<=2*limits.total_constant_elements+limits.tensor_elements+256 and depth<=limits.rank+6,"mha_resource_rejected",parameter)
        if type(v) is dict:
            require(all(type(k) is str for k in v),"mha_arguments_rejected",parameter)
            return {k:visit(x,depth+1) for k,x in v.items()}
        if type(v) in (list,tuple):return [visit(x,depth+1) for x in v]
        if v is None or type(v) in (str,bool,int,float):return v
        fail("mha_arguments_rejected",parameter)
    return visit(value,0)
def implementation():
    base=Path(__file__).parents[3]
    paths=(Path(__file__),base/"benchmark_graph.py",base/"model_decomposition.py",base/"model_resources.py")
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def snapshot_module(module):
    import torch
    require(type(module) is torch.nn.MultiheadAttention and module.training is False,"mha_module_rejected","module")
    require(not module._forward_hooks and not module._forward_pre_hooks,"mha_module_rejected","module")
    config=dict(embed_dim=module.embed_dim,num_heads=module.num_heads,kdim=module.kdim,vdim=module.vdim,
                batch_first=module.batch_first,dropout=module.dropout,add_zero_attn=module.add_zero_attn,
                same_projection=module._qkv_same_embed_dim)
    parameters={}
    for name in ("in_proj_weight","in_proj_bias","q_proj_weight","k_proj_weight","v_proj_weight","bias_k","bias_v"):
        value=getattr(module,name)
        require(value is None or type(value) in (torch.Tensor,torch.nn.Parameter),"mha_state_rejected","weights")
        if value is not None:
            require(value.dtype is torch.float64 and value.device.type=="cpu","mha_state_rejected","weights")
        parameters[name]=None if value is None else value.detach().tolist()
    for name in ("weight","bias"):
        value=getattr(module.out_proj,name)
        require(value is None or type(value) in (torch.Tensor,torch.nn.Parameter),"mha_state_rejected","weights")
        if value is not None:require(value.dtype is torch.float64 and value.device.type=="cpu","mha_state_rejected","weights")
        parameters["out_proj."+name]=None if value is None else value.detach().tolist()
    return dict(format=MODULE_FORMAT,config=config,parameters=parameters)
def public_mask(tensor):
    import torch
    require(type(tensor) is torch.Tensor and tensor.device.type=="cpu","mha_mask_rejected","mask")
    if tensor.dtype is torch.bool:return dict(dtype="bool",values=tensor.tolist(),negative_infinity=None)
    require(tensor.dtype is torch.float64 and not torch.isnan(tensor).any().item() and not torch.isposinf(tensor).any().item(),"mha_mask_rejected","mask")
    blocked=torch.isneginf(tensor);finite=torch.where(blocked,torch.zeros_like(tensor),tensor)
    return dict(dtype="float64",values=finite.tolist(),negative_infinity=blocked.tolist() if blocked.any().item() else None)
def _array(value,expected,limits):
    try:
        s=shape(value,resource_profile=limits.name)
        require(tuple(expected)==s,"mha_state_rejected","weights")
        result=np.asarray(value,dtype=np.float64)
    except (ValueError,TypeError,OverflowError):fail("mha_state_rejected","weights")
    return result
def state_weights(module,limits):
    require(type(module) is dict and set(module)=={"format","config","parameters"} and module["format"]==MODULE_FORMAT,"mha_state_rejected","module")
    c=module["config"];p=module["parameters"]
    require(type(c) is dict and set(c)=={"embed_dim","num_heads","kdim","vdim","batch_first","dropout","add_zero_attn","same_projection"},"mha_state_rejected","module")
    require(all(type(c[k]) is int and 1<=c[k]<=limits.axis_size for k in ("embed_dim","num_heads","kdim","vdim")),"mha_state_rejected","module")
    e,h,k,v=(c[n] for n in ("embed_dim","num_heads","kdim","vdim"))
    require(e%h==0 and type(c["batch_first"]) is bool and type(c["add_zero_attn"]) is bool and type(c["same_projection"]) is bool and
            c["same_projection"]==(k==v==e) and type(c["dropout"]) in (int,float) and math.isfinite(c["dropout"]) and 0<=c["dropout"]<=1,"mha_state_rejected","module")
    names={"in_proj_weight","in_proj_bias","q_proj_weight","k_proj_weight","v_proj_weight","bias_k","bias_v","out_proj.weight","out_proj.bias"}
    require(type(p) is dict and set(p)==names,"mha_state_rejected","weights")
    if c["same_projection"]:
        require(all(p[n] is None for n in ("q_proj_weight","k_proj_weight","v_proj_weight")),"mha_state_rejected","weights")
        packed=_array(p["in_proj_weight"],[3*e,e],limits)
        q,kweight,vweight=np.split(packed,3,axis=0)
    else:
        require(p["in_proj_weight"] is None,"mha_state_rejected","weights")
        q=_array(p["q_proj_weight"],[e,e],limits);kweight=_array(p["k_proj_weight"],[e,k],limits);vweight=_array(p["v_proj_weight"],[e,v],limits)
    bias=None if p["in_proj_bias"] is None else _array(p["in_proj_bias"],[3*e],limits)
    qb,kb,vb=(None,None,None) if bias is None else np.split(bias,3)
    out=_array(p["out_proj.weight"],[e,e],limits)
    ob=None if p["out_proj.bias"] is None else _array(p["out_proj.bias"],[e],limits)
    require((p["bias_k"] is None)==(p["bias_v"] is None),"mha_state_rejected","weights")
    bk=bv=None
    if p["bias_k"] is not None:
        bk=_array(p["bias_k"],[1,1,e],limits).reshape(h,e//h)
        bv=_array(p["bias_v"],[1,1,e],limits).reshape(h,e//h)
    return c,dict(q=q,k=kweight,v=vweight,qb=qb,kb=kb,vb=vb,out=out,ob=ob,bk=bk,bv=bv)
def shapes(operands,c,limits):
    require(type(operands) is dict and set(operands)=={"query","key","value"},"mha_shape_rejected","input")
    declared={};role_shapes={}
    for role,x in operands.items():
        require(type(x) is dict and set(x)=={"kind","name","shape","dtype"} and x["kind"]=="cipher" and x["dtype"]=="float64" and type(x["name"]) is str and NAME.fullmatch(x["name"]),"mha_shape_rejected","input")
        try:s=dimensions(x["shape"],resource_profile=limits.name)
        except (ValueError,TypeError,OverflowError):fail("mha_shape_rejected","input")
        require(len(s) in (2,3),"mha_shape_rejected","input")
        if x["name"] in declared:require(declared[x["name"]]==s,"mha_shape_rejected","input")
        declared[x["name"]]=s;role_shapes[role]=s
    q,k,v=(role_shapes[r] for r in ("query","key","value"))
    require(len(q)==len(k)==len(v) and q[-1]==c["embed_dim"] and k[-1]==c["kdim"] and v[-1]==c["vdim"],"mha_shape_rejected","input")
    unbatched=len(q)==2
    if unbatched:n=1;l=q[0];s=k[0];require(v[0]==s,"mha_shape_rejected","input")
    else:
        batch_axis,seq_axis=(0,1) if c["batch_first"] else (1,0)
        n=q[batch_axis];l=q[seq_axis];s=k[seq_axis]
        require(k[batch_axis]==v[batch_axis]==n and v[seq_axis]==s,"mha_shape_rejected","input")
    require(len(declared)<=limits.inputs and sum(math.prod(v) for v in declared.values())<=limits.input_elements,"mha_resource_rejected","input")
    return declared,n,l,s,unbatched
def mask_parts(value,expected,limits):
    if value is None:return None
    require(type(value) is dict and set(value)=={"dtype","values","negative_infinity"},"mha_mask_rejected","mask")
    def check(v,kind):
        if type(v) is list:
            require(bool(v),"mha_mask_rejected","mask")
            for x in v:check(x,kind)
        else:
            require((type(v) is bool) if kind=="bool" else (type(v) in (int,float) and math.isfinite(v) and abs(v)<=1024),"mha_mask_rejected","mask")
    check(value["values"],value["dtype"])
    require(value["dtype"] in ("bool","float64"),"mha_mask_rejected","mask")
    try:a=np.asarray(value["values"],dtype=bool if value["dtype"]=="bool" else np.float64)
    except (TypeError,ValueError):fail("mha_mask_rejected","mask")
    require(tuple(a.shape) in expected and a.size<=limits.tensor_elements,"mha_mask_rejected","mask")
    if value["dtype"]=="bool":
        require(value["negative_infinity"] is None,"mha_mask_rejected","mask")
        return a,np.zeros(a.shape,dtype=np.float64)
    blocked=np.zeros(a.shape,dtype=bool)
    if value["negative_infinity"] is not None:
        check(value["negative_infinity"],"bool")
        blocked=np.asarray(value["negative_infinity"],dtype=bool)
        require(blocked.shape==a.shape and np.all(a[blocked]==0),"mha_mask_rejected","mask")
    return blocked,a
def effective_masks(options,c,n,l,s,unbatched,limits,extra):
    h=c["num_heads"];total=s+extra
    require(type(options) is dict and set(options)=={"key_padding_mask","need_weights","attn_mask","average_attn_weights","is_causal"},"mha_arguments_rejected","call")
    require(all(type(options[k]) is bool for k in ("need_weights","average_attn_weights","is_causal")),"mha_arguments_rejected","call")
    a=mask_parts(options["attn_mask"],{(l,s),(n*h,l,s)},limits)
    p=mask_parts(options["key_padding_mask"],{(s,)} if unbatched else {(n,s)},limits)
    require(not options["is_causal"] or a is not None,"mha_arguments_rejected","mask")
    blocked=np.zeros((n,h,l,total),dtype=bool);bias=np.zeros(blocked.shape,dtype=np.float64)
    if options["is_causal"]:
        ab,av=a
        expected=np.arange(s)[None,:]>np.arange(l)[:,None]
        require(np.all(ab==np.broadcast_to(expected,ab.shape)) and np.all(av==0),
                "mha_arguments_rejected","mask")
    implicit=options["is_causal"] and p is None and not options["need_weights"]
    if implicit:
        blocked[:]=np.arange(total)[None,:]>np.arange(l)[:,None]
    else:
        if a is not None:
            ab,av=a
            ab=np.broadcast_to(ab,(n*h,l,s)).reshape(n,h,l,s)
            av=np.broadcast_to(av,(n*h,l,s)).reshape(n,h,l,s)
            blocked[...,:s]|=ab;bias[...,:s]+=av
        if p is not None:
            pb,pv=p
            pb=pb.reshape(n,1,1,s);pv=pv.reshape(n,1,1,s)
            blocked[...,:s]|=pb;bias[...,:s]+=pv
    require(not np.any(np.all(blocked,axis=-1)),"mha_all_masked_unsupported","mask")
    return (~blocked).astype(np.float64).reshape(n*h,l,total),bias.reshape(n*h,l,total),implicit
def plan(module,operands,options,approximation,*,resource_profile="general-tensors-v1"):
    try:limits=resolve(resource_profile)
    except (TypeError,ValueError):fail("mha_resource_rejected","resource_profile")
    request=data(dict(module=module,operands=operands,options=options,approximation=approximation,resource_profile=resource_profile),limits,"input")
    module,operands,options,approximation=(request[k] for k in ("module","operands","options","approximation"))
    c,w=state_weights(module,limits);declared,n,l,s,unbatched=shapes(operands,c,limits)
    require(type(approximation) is dict and set(approximation)=={"exp","reciprocal"},"mha_approximation_rejected","approximation")
    try:profile("exp",approximation["exp"]);profile("reciprocal",approximation["reciprocal"])
    except (TypeError,ValueError,OverflowError):fail("mha_approximation_rejected","approximation")
    e,h=c["embed_dim"],c["num_heads"];d=e//h;nh=n*h
    extra=(1 if w["bk"] is not None else 0)+int(c["add_zero_attn"]);total=s+extra
    require(nh*l*total<=limits.tensor_elements and nh*l*s*d<=limits.tensor_elements,"mha_resource_rejected","input")
    allow,additive,implicit=effective_masks(options,c,n,l,s,unbatched,limits,extra)
    output_count=n*l*e+(n*l*total if options["average_attn_weights"] else nh*l*total) if options["need_weights"] else n*l*e
    require(output_count<=limits.output_elements,"mha_resource_rejected","input")
    graph=dict(format=GRAPH_FORMAT,id="static_mha_source",inputs=[dict(name=k,shape=list(v)) for k,v in declared.items()],constants={},nodes=[],outputs=[])
    used=set(declared);counter=0
    def fresh(base):
        nonlocal counter
        counter+=1;name=base+"_"+str(counter)
        while name in used:counter+=1;name=base+"_"+str(counter)
        used.add(name);return name
    def constant(v):
        name=fresh("public");graph["constants"][name]=np.asarray(v,dtype=np.float64).tolist();return name
    def emit(op,inputs,attrs=None):
        out=fresh("value");graph["nodes"].append(dict(id=fresh("node"),op=op,inputs=inputs,attrs=attrs or {},outputs=[out]));return out
    def scalar(x,k):return emit("multiply",[x,constant(k)])
    def heads(role,weight,bias,length):
        x=operands[role]["name"]
        if unbatched:x=emit("reshape",[x],dict(shape=[1,length,c[{"query":"embed_dim","key":"kdim","value":"vdim"}[role]]]))
        elif not c["batch_first"]:x=emit("permute",[x],dict(dims=[1,0,2]))
        refs=[x,constant(weight)]
        if bias is not None:refs.append(constant(bias))
        x=emit("linear",refs)
        x=emit("reshape",[x],dict(shape=[n,length,h,d]))
        x=emit("permute",[x],dict(dims=[0,2,1,3]))
        return emit("reshape",[x],dict(shape=[nh,length,d]))
    q=heads("query",w["q"],w["qb"],l);k=heads("key",w["k"],w["kb"],s);v=heads("value",w["v"],w["vb"],s)
    scaled_q=scalar(q,1/math.sqrt(d))
    score=emit("matmul",[scaled_q,emit("permute",[k],dict(dims=[0,2,1]))])
    base_bias=additive[:,:,:s];base_allow=allow[:,:,:s]
    if np.any(base_bias):score=emit("add",[score,constant(base_bias)])
    if not np.all(base_allow==1):score=emit("multiply",[score,constant(base_allow)])
    numerator=emit("exp",[score],dict(approximation=approximation["exp"]))
    if not np.all(base_allow==1):numerator=emit("multiply",[numerator,constant(base_allow)])
    denominator=emit("sum",[numerator],dict(axes=[-1],keepdims=True))
    bias_numerator=None;idx=s
    if w["bk"] is not None:
        bk=np.tile(w["bk"],(n,1))[:,None,:]
        bias_score=emit("sum",[emit("multiply",[scaled_q,constant(bk)])],dict(axes=[-1],keepdims=True))
        gate=allow[:,:,idx:idx+1]
        if not np.all(gate==1):bias_score=emit("multiply",[bias_score,constant(gate)])
        bias_numerator=emit("exp",[bias_score],dict(approximation=approximation["exp"]))
        if not np.all(gate==1):bias_numerator=emit("multiply",[bias_numerator,constant(gate)])
        denominator=emit("add",[denominator,bias_numerator]);idx+=1
    zero_gate=None
    if c["add_zero_attn"]:
        zero_gate=allow[:,:,idx:idx+1]
        denominator=emit("add",[denominator,constant(zero_gate)])
    inverse=emit("reciprocal",[denominator],dict(approximation=approximation["reciprocal"]))
    probabilities=emit("multiply",[numerator,inverse])
    attended=emit("matmul",[probabilities,v]);all_probabilities=[probabilities]
    if bias_numerator is not None:
        bias_prob=emit("multiply",[bias_numerator,inverse]);all_probabilities.append(bias_prob)
        expanded=emit("broadcast",[bias_prob],dict(shape=[nh,l,d]))
        contribution=emit("multiply",[expanded,constant(np.tile(w["bv"],(n,1))[:,None,:])])
        attended=emit("add",[attended,contribution])
    if c["add_zero_attn"] and options["need_weights"]:all_probabilities.append(inverse if np.all(zero_gate==1) else emit("multiply",[inverse,constant(zero_gate)]))
    attended=emit("reshape",[attended],dict(shape=[n,h,l,d]))
    attended=emit("permute",[attended],dict(dims=[0,2,1,3]))
    attended=emit("reshape",[attended],dict(shape=[n,l,e]))
    refs=[attended,constant(w["out"])]
    if w["ob"] is not None:refs.append(constant(w["ob"]))
    out=emit("linear",refs)
    if unbatched:out=emit("reshape",[out],dict(shape=[l,e]))
    elif not c["batch_first"]:out=emit("permute",[out],dict(dims=[1,0,2]))
    graph["outputs"].append(dict(name="attn_output",value=out))
    output_schema=dict(kind="tuple",items=[dict(kind="tensor",name="attn_output"),dict(kind="none")])
    if options["need_weights"]:
        probs=all_probabilities[0] if len(all_probabilities)==1 else emit("concat",all_probabilities,dict(axis=-1))
        probs=emit("reshape",[probs],dict(shape=[n,h,l,total]))
        if options["average_attn_weights"]:probs=emit("mean",[probs],dict(axes=[1],keepdims=False))
        if unbatched:probs=emit("reshape",[probs],dict(shape=[l,total] if options["average_attn_weights"] else [h,l,total]))
        graph["outputs"].append(dict(name="attn_weights",value=probs));output_schema["items"][1]=dict(kind="tensor",name="attn_weights")
    try:lowered,decomposition=decompose(graph,resource_profile=resource_profile)
    except (TypeError,ValueError,KeyError,IndexError,OverflowError):fail("mha_graph_contract_rejected","plan")
    result=dict(format=FORMAT,source_request=request,source_request_sha256=digest(request),mathematical_graph=graph,lowered_graph=lowered,decomposition=decomposition,
                implementation=implementation(),resource_profile=identity(resource_profile),output_schema=output_schema,
                mask_dispatch="implicit_causal" if implicit else "explicit_public_masks",effective_allow=allow.tolist(),
                effective_additive=additive.tolist(),production_intake_integrated=False,
                original_function_accuracy_certified=False,physical_layout_status="unverified",compiler_status="unverified",encrypted_status="unverified",agent_status="unverified")
    result["binding"]=digest(result);return result
def verify(value,*,expected_source_request_sha256):
    try:
        if type(value) is not dict or value.get("format")!=FORMAT or value["implementation"]!=implementation():raise ValueError("Identity")
        if value["binding"]!=digest({k:v for k,v in value.items() if k!="binding"}):raise ValueError("Binding")
        if value["source_request_sha256"]!=digest(value["source_request"]) or value["source_request_sha256"]!=expected_source_request_sha256:raise ValueError("Source")
        if canonical(plan(**value["source_request"]))!=canonical(value):raise ValueError("Semantics")
    except (ValueError,TypeError,KeyError,IndexError,OverflowError):fail("mha_integrity_rejected","plan")
    return value
def from_fx_call(node,module,operands,*,approximation,resource_profile="general-tensors-v1"):
    import torch
    require(type(node) is torch.fx.Node and node.op=="call_module" and type(operands) is dict,"mha_arguments_rejected","call")
    names=("query","key","value","key_padding_mask","need_weights","attn_mask","average_attn_weights","is_causal")
    require(len(node.args)<=len(names) and not(set(node.kwargs)-set(names)) and not(set(names[:len(node.args)])&set(node.kwargs)),"mha_arguments_rejected","call")
    args=dict(zip(names[3:],(None,True,None,True,False)))
    args.update(zip(names,node.args));args.update(node.kwargs)
    require(all(name in args for name in names),"mha_arguments_rejected","call")
    def bound(value):
        if type(value) is torch.fx.Node:
            require(value in operands,"mha_arguments_rejected","input");return operands[value]
        return value
    inputs={name:bound(args[name]) for name in names[:3]}
    options={name:bound(args[name]) for name in names[3:]}
    return plan(snapshot_module(module),inputs,options,approximation,resource_profile=resource_profile)
