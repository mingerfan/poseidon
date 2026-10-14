"""Plaintext composition fixtures. No provider, SDK, capture or backend startup."""
import copy, math
APPROXIMATION={"exp":{"method":"taylor","domain":[-.5,.5],"degree":6},
               "reciprocal":{"method":"goldschmidt","domain":[.5,8.],"iterations":8}}
CASES=(
 dict(id="self_post_linear_residual",e=4,h=2,n=1,l=2,s=2,batch_first=True,self_attention=True,
      need=True,average=False,post_width=4,residual=True,tree="named"),
 dict(id="linear_into_separate_cross_none",e=4,h=2,n=1,l=2,s=3,batch_first=True,kdim=3,vdim=2,
      pre_width=3,need=False,average=True,projected_output=True,tree="nested_none"),
 dict(id="boolean_padding_residual",e=2,h=1,n=1,l=2,s=3,batch_first=True,
      need=True,average=True,mask="bool_padding",residual=True,tree="tuple_none"),
 dict(id="additive_heads_post_linear",e=4,h=2,n=2,l=1,s=2,batch_first=True,
      need=True,average=False,mask="additive_heads_padding",post_width=2,tree="named"),
 dict(id="unbatched_cross_none",e=4,h=2,n=1,l=2,s=3,batch_first=True,unbatched=True,
      need=False,average=True,tree="nested_none"),
 dict(id="sequence_first_bias_zero",e=2,h=1,n=1,l=2,s=2,batch_first=False,
      bias_kv=True,zero_attn=True,need=True,average=True,post_width=2,residual=True,tree="tuple_none"),
)

def signatures(case):
 result=[]
 for role,name in (("query","q"),("key","k"),("value","v")):
  if case.get("self_attention") and role!="query":continue
  length=case["l"] if role=="query" else case["s"]
  width=case.get("pre_width",case["e"]) if role=="query" else case.get("kdim",case["e"]) if role=="key" else case.get("vdim",case["e"])
  shape=[length,width] if case.get("unbatched") else [case["n"],length,width] if case["batch_first"] else [length,case["n"],width]
  result.append(dict(name=name,shape=shape))
 return result

def output_names(case):
 return ["result"]+(["weights"] if case["need"] else [])+(["projected"] if case.get("projected_output") else [])

def validate_case(case):
 if not (type(case) is dict and type(case.get("e")) is int and type(case.get("h")) is int
         and case["e"]>0 and case["h"]>0 and case["e"]%case["h"]==0):raise ValueError("embed/head dimensions")
 if case.get("residual") and case.get("post_width",case["e"])!=case["e"]:raise ValueError("residual shape")
 if case.get("self_attention") and (case["l"]!=case["s"] or case.get("kdim",case["e"])!=case["e"] or case.get("vdim",case["e"])!=case["e"]):raise ValueError("self attention role shapes")
 if len(signatures(case))>4 or len(output_names(case))>4:raise ValueError("logical port budget")
 return case

def masks(case):
 if case.get("mask")=="bool_padding":
  return dict(dtype="bool",values=[[False,True,False],[True,False,False]],negative_infinity=None),dict(dtype="bool",values=[[False,True,False]],negative_infinity=None)
 if case.get("mask")=="additive_heads_padding":
  return dict(dtype="float64",values=[[[-.04*(i+1),-.01*i]] for i in range(case["n"]*case["h"])],negative_infinity=None),dict(dtype="float64",values=[[0.,-.025*(i+1)] for i in range(case["n"])],negative_infinity=None)
 return None,None

def public_matrix(rows,cols,offset=0):
 return [[math.sin(1+offset+i*cols+j)*.08 for j in range(cols)] for i in range(rows)]

def build(case):
 """Returns exact native eval source and declarations; preserves caller RNG."""
 import torch
 validate_case(case);previous=torch.get_rng_state().clone()
 try:
  attention=torch.nn.MultiheadAttention(case["e"],case["h"],batch_first=case["batch_first"],
      kdim=case.get("kdim"),vdim=case.get("vdim"),add_bias_kv=case.get("bias_kv",False),
      add_zero_attn=case.get("zero_attn",False),dtype=torch.float64).eval()
  pre=torch.nn.Linear(case["pre_width"],case["e"],dtype=torch.float64) if case.get("pre_width") else None
  post=torch.nn.Linear(case["e"],case["post_width"],dtype=torch.float64) if case.get("post_width") else None
  offset=0
  with torch.no_grad():
   for module in (attention,pre,post):
    if module is None:continue
    for parameter in module.parameters():
     values=[math.sin(offset+i+1)*.08 for i in range(parameter.numel())];offset+=parameter.numel()
     parameter.copy_(torch.tensor(values,dtype=torch.float64).reshape(parameter.shape))
  am,pm=masks(case)
  class Source(torch.nn.Module):
   def __init__(self):
    super().__init__();self.attention=attention;self.pre=pre;self.post=post
    self.register_buffer("attn_mask",None if am is None else torch.tensor(am["values"],dtype=torch.bool if am["dtype"]=="bool" else torch.float64))
    self.register_buffer("padding_mask",None if pm is None else torch.tensor(pm["values"],dtype=torch.bool if pm["dtype"]=="bool" else torch.float64))
   def forward(self,q,k,v):
    projected=self.pre(q) if case.get("pre_width") else q
    output,weights=self.attention(projected,k,v,need_weights=case["need"],average_attn_weights=case["average"],attn_mask=self.attn_mask,key_padding_mask=self.padding_mask)
    output=self.post(output) if case.get("post_width") else output
    output=output+projected if case.get("residual") else output
    return tree(case,output,weights,projected)
  class SelfSource(Source):
   def forward(self,q):
    output,weights=self.attention(q,q,q,need_weights=case["need"],average_attn_weights=case["average"],attn_mask=self.attn_mask,key_padding_mask=self.padding_mask)
    output=self.post(output) if case.get("post_width") else output
    output=output+q if case.get("residual") else output
    return tree(case,output,weights,q)
  return (SelfSource() if case.get("self_attention") else Source()).eval(),signatures(case),output_names(case)
 finally:torch.set_rng_state(previous)

def tree(case,output,weights,projected):
 if case["tree"]=="named":return {"result":output,"weights":weights}
 if case["tree"]=="tuple_none":return (output,[weights,None])
 return {"result":output,"aux":[None,projected]} if case.get("projected_output") else (output,None)

def samples(signature):
 import numpy as np
 for group in range(8):
  row={}
  for role,item in enumerate(signature):
   n=math.prod(item["shape"]);raw=[0. if group==0 else ((-1.)**i*.2 if group==1 else math.sin((group+1)*(i+1)+role)*.23) for i in range(n)]
   row[item["name"]]=np.asarray(raw,dtype=np.float64).reshape(item["shape"])
  yield row

# Separate, predeclared execution-feasibility configurations. They do not replace
# the six original-profile controls or claim accuracy against exact Softmax.
SHALLOW_APPROXIMATION={"exp":{"method":"taylor","domain":[-.5,.5],"degree":1},
                      "reciprocal":{"method":"goldschmidt","domain":[.5,8.],"iterations":1}}
def execution_candidates():
 result=[]
 for index in (0,1,2):
  item=copy.deepcopy(CASES[index]);item["id"]+="__declared_shallow_v1"
  item["approximation"]=copy.deepcopy(SHALLOW_APPROXIMATION);item["original_accuracy_certified"]=False
  result.append(item)
 return tuple(result)
