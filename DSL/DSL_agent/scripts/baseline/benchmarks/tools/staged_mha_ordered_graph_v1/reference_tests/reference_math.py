"""Independent MHA loops over original public weights; never evaluates the DSL graph."""
import math

def independent_mha(module,query,key,value,options,approximation,*,approximate):
 import numpy as np
 e=module.embed_dim;heads=module.num_heads;dim=e//heads
 def array(p):return None if p is None else p.detach().numpy().copy()
 if module._qkv_same_embed_dim:
  packed=array(module.in_proj_weight);ws=[packed[i*e:(i+1)*e] for i in range(3)]
 else:ws=[array(module.q_proj_weight),array(module.k_proj_weight),array(module.v_proj_weight)]
 biases=[None]*3 if module.in_proj_bias is None else [array(module.in_proj_bias)[i*e:(i+1)*e] for i in range(3)]
 unbatched=query.ndim==2
 raw=[np.asarray(x,dtype=np.float64) for x in (query,key,value)]
 raw=[x[None] if unbatched else x if module.batch_first else x.swapaxes(0,1) for x in raw]
 projections=[]
 for x,w,bias in zip(raw,ws,biases):
  out=np.empty(x.shape[:-1]+(e,),dtype=np.float64)
  for index in np.ndindex(x.shape[:-1]):
   for o in range(e):out[index+(o,)]=math.fsum(float(x[index+(i,)])*float(w[o,i]) for i in range(x.shape[-1]))+(0. if bias is None else float(bias[o]))
  projections.append(out)
 q,k,v=projections;n,l=q.shape[:2];source=k.shape[1]
 if module.bias_k is not None:
  k=np.concatenate([k,np.broadcast_to(array(module.bias_k),(n,1,e))],axis=1);v=np.concatenate([v,np.broadcast_to(array(module.bias_v),(n,1,e))],axis=1)
 if module.add_zero_attn:
  k=np.concatenate([k,np.zeros((n,1,e))],axis=1);v=np.concatenate([v,np.zeros((n,1,e))],axis=1)
 total=k.shape[1];probabilities=np.zeros((n,heads,l,total));attended=np.zeros((n,l,e))
 am=options["attn_mask"];pm=options["key_padding_mask"]
 def mask_entry(mask,batch,head,row,col,padding=False):
  if mask is None or col>=source:return False,0.
  a=np.asarray(mask["values"]);index=(col,) if padding and unbatched else (batch,col) if padding else (row,col) if a.ndim==2 else (batch*heads+head,row,col)
  z=a[index]
  if mask["dtype"]=="bool":return bool(z),0.
  blocked=mask.get("negative_infinity");return (bool(np.asarray(blocked)[index]) if blocked is not None else False),float(z)
 for b in range(n):
  for h in range(heads):
   for row in range(l):
    blocked=[];logits=[]
    for col in range(total):
     ba,aa=mask_entry(am,b,h,row,col);bp,ap=mask_entry(pm,b,h,row,col,padding=True)
     masked=ba or bp
     if options.get("is_causal") and pm is None and not options["need_weights"]:masked=col>row
     blocked.append(masked);logits.append(math.fsum(float(q[b,row,h*dim+i])*float(k[b,col,h*dim+i]) for i in range(dim))/math.sqrt(dim)+aa+ap)
    if all(blocked):raise ValueError("all masked")
    if approximate:
     lo,hi=approximation["exp"]["domain"];degree=approximation["exp"]["degree"]
     if any(not math.isfinite(z) or z<lo or z>hi for z,m in zip(logits,blocked) if not m):raise ValueError("exp approximation domain")
     numer=[0. if m else math.fsum(z**i/math.factorial(i) for i in range(degree+1)) for z,m in zip(logits,blocked)]
     den=math.fsum(numer);lo,hi=approximation["reciprocal"]["domain"]
     if not lo<=den<=hi or den<=0:raise ValueError("reciprocal approximation domain")
     residual=1.-den/hi;inverse=(1.-residual**(2**approximation["reciprocal"]["iterations"]))/den
    else:
     maximum=max(z for z,m in zip(logits,blocked) if not m);numer=[0. if m else math.exp(z-maximum) for z,m in zip(logits,blocked)];inverse=1./math.fsum(numer)
    for col in range(total):probabilities[b,h,row,col]=numer[col]*inverse
    for i in range(dim):attended[b,row,h*dim+i]=math.fsum(float(probabilities[b,h,row,col])*float(v[b,col,h*dim+i]) for col in range(total))
 w=array(module.out_proj.weight);bias=array(module.out_proj.bias);output=np.empty_like(attended)
 for index in np.ndindex(attended.shape[:-1]):
  for o in range(e):output[index+(o,)]=math.fsum(float(attended[index+(i,)])*float(w[o,i]) for i in range(e))+(0. if bias is None else float(bias[o]))
 output=output[0] if unbatched else output if module.batch_first else output.swapaxes(0,1)
 result={"attn_output":output}
 if options["need_weights"]:
  weights=probabilities.mean(axis=1) if options["average_attn_weights"] else probabilities
  result["attn_weights"]=weights[0] if unbatched else weights
 return result

def linear(module,x):
 import numpy as np
 w=module.weight.detach().numpy();bias=None if module.bias is None else module.bias.detach().numpy();out=np.empty(x.shape[:-1]+(w.shape[0],))
 for index in np.ndindex(x.shape[:-1]):
  for o in range(w.shape[0]):out[index+(o,)]=math.fsum(float(x[index+(i,)])*float(w[o,i]) for i in range(w.shape[1]))+(0. if bias is None else float(bias[o]))
 return out

def composition(case,module,batch,approximation,*,approximate):
 from cases import masks
 projected=linear(module.pre,batch["q"]) if module.pre is not None else batch["q"]
 k=v=projected if case.get("self_attention") else None
 if not case.get("self_attention"):k,v=batch["k"],batch["v"]
 am,pm=masks(case);options=dict(attn_mask=am,key_padding_mask=pm,need_weights=case["need"],average_attn_weights=case["average"],is_causal=False)
 got=independent_mha(module.attention,projected,k,v,options,approximation,approximate=approximate)
 output=linear(module.post,got["attn_output"]) if module.post is not None else got["attn_output"]
 if case.get("residual"):output=output+projected
 result={"result":output}
 if case["need"]:result["weights"]=got["attn_weights"]
 if case.get("projected_output"):result["projected"]=projected
 return result
