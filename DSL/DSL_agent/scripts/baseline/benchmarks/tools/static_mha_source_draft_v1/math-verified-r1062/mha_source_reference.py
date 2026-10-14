"""Independent NumPy mathematical reference for the MHA source draft.

Uses the declared public state directly and does not call the graph planner.
"""
import math
import numpy as np

def evaluate(request,inputs,*,approximate):
    module=request["module"];c=module["config"];p=module["parameters"];options=request["options"]
    e,h=c["embed_dim"],c["num_heads"];d=e//h
    if c["same_projection"]:
        packed=np.asarray(p["in_proj_weight"],dtype=np.float64)
        qw,kw,vw=packed[:e],packed[e:2*e],packed[2*e:]
    else:
        qw=np.asarray(p["q_proj_weight"],dtype=np.float64)
        kw=np.asarray(p["k_proj_weight"],dtype=np.float64);vw=np.asarray(p["v_proj_weight"],dtype=np.float64)
    bias=None if p["in_proj_bias"] is None else np.asarray(p["in_proj_bias"],dtype=np.float64)
    raw=[np.asarray(inputs[request["operands"][role]["name"]],dtype=np.float64) for role in ("query","key","value")]
    unbatched=raw[0].ndim==2
    raw=[x[None] if unbatched else x if c["batch_first"] else x.swapaxes(0,1) for x in raw]
    q,k,v=[x@w.T+(0 if bias is None else bias[i*e:(i+1)*e]) for i,(x,w) in enumerate(zip(raw,(qw,kw,vw)))]
    n,l=q.shape[:2];s=k.shape[1]
    if p["bias_k"] is not None:
        k=np.concatenate((k,np.broadcast_to(np.asarray(p["bias_k"],dtype=np.float64),(n,1,e))),axis=1)
        v=np.concatenate((v,np.broadcast_to(np.asarray(p["bias_v"],dtype=np.float64),(n,1,e))),axis=1)
    if c["add_zero_attn"]:
        k=np.concatenate((k,np.zeros((n,1,e))),axis=1);v=np.concatenate((v,np.zeros((n,1,e))),axis=1)
    total=k.shape[1]
    q=q.reshape(n,l,h,d).transpose(0,2,1,3)
    k=k.reshape(n,total,h,d).transpose(0,2,1,3)
    v=v.reshape(n,total,h,d).transpose(0,2,1,3)
    logits=np.einsum("nhld,nhsd->nhls",q,k)/math.sqrt(d)
    blocked=np.zeros((n,h,l,total),dtype=bool)
    addition=np.zeros_like(logits)
    def mask_parts(mask):
        if mask["dtype"]=="bool":
            a=np.asarray(mask["values"],dtype=bool);return a,np.zeros_like(a,dtype=np.float64)
        a=np.asarray(mask["values"],dtype=np.float64)
        blocked=np.zeros_like(a,dtype=bool) if mask["negative_infinity"] is None else np.asarray(mask["negative_infinity"],dtype=bool)
        return blocked,a
    implicit=options["is_causal"] and options["key_padding_mask"] is None and not options["need_weights"]
    if implicit:
        for i in range(l):
            blocked[:,:,i,i+1:]=True
    else:
        if options["attn_mask"] is not None:
            m,b=mask_parts(options["attn_mask"])
            if m.ndim==2:
                m=np.broadcast_to(m,(n,h,l,s));b=np.broadcast_to(b,(n,h,l,s))
            else:m=m.reshape(n,h,l,s);b=b.reshape(n,h,l,s)
            blocked[...,:s]|=m;addition[...,:s]+=b
        if options["key_padding_mask"] is not None:
            m,b=mask_parts(options["key_padding_mask"])
            blocked[...,:s]|=m.reshape(n,1,1,s);addition[...,:s]+=b.reshape(n,1,1,s)
    logits+=addition
    if np.any(np.all(blocked,axis=-1)):raise ValueError("No defined positive attention denominator")
    if approximate:
        domain=request["approximation"]["exp"]["domain"];degree=request["approximation"]["exp"]["degree"]
        logits=np.where(blocked,0.,logits)
        if not (np.isfinite(logits).all() and np.all(logits>=domain[0]) and np.all(logits<=domain[1])):
            raise ValueError("Exponential reference interval")
        numerator=sum(logits**i/math.factorial(i) for i in range(degree+1))
        numerator=np.where(blocked,0.,numerator);total_exp=numerator.sum(axis=-1,keepdims=True)
        profile=request["approximation"]["reciprocal"];lo,hi=profile["domain"]
        if not (np.all(total_exp>=lo) and np.all(total_exp<=hi) and np.all(total_exp>0)):
            raise ValueError("Reciprocal reference interval")
        residual=1-total_exp/hi
        inverse=(1-residual**(2**profile["iterations"]))/total_exp
        probability=numerator*inverse
    else:
        logits=np.where(blocked,-np.inf,logits)
        numerator=np.exp(logits-np.max(logits,axis=-1,keepdims=True))
        probability=numerator/numerator.sum(axis=-1,keepdims=True)
    output=np.einsum("nhls,nhsd->nhld",probability,v).transpose(0,2,1,3).reshape(n,l,e)
    output=output@np.asarray(p["out_proj.weight"],dtype=np.float64).T
    if p["out_proj.bias"] is not None:output+=np.asarray(p["out_proj.bias"],dtype=np.float64)
    output=output[0] if unbatched else output if c["batch_first"] else output.swapaxes(0,1)
    result={"attn_output":output}
    if options["need_weights"]:
        weights=probability.mean(axis=1) if options["average_attn_weights"] else probability
        result["attn_weights"]=weights[0] if unbatched else weights
    return result
