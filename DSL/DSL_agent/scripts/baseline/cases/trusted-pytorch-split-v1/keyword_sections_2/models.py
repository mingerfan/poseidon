"""Ordinary static split source contexts with immutable public state."""
import torch
from torch.nn import functional as F

SHAPES={
 "function_sections":[[2,4],[4,2],[1,2,6]],
 "method_size":[[5],[2,6],[1,2,7]],
 "keyword_sections":[[4],[1,6],[*[1]*31,8]],
 "tuple_output":[[4],[2,6],[1,2,8]],
 "public_loop":[[10],[2,12],[1,1,14]],
 "public_weight":[[1,3],[2,3],[1,2,3]]}

class Model(torch.nn.Module):
 def __init__(self,family,context):
  super().__init__();self.family=family;self.context=context
  self.axis=0 if family=="function_sections" and context==1 else -1
  width=SHAPES[family][context][self.axis]
  if family=="function_sections":self.sections=([width//2]*2 if context<2 else [2,2,2])
  elif family=="keyword_sections":self.sections=([2,2],[1,2,3],[2,2,4])[context]
  elif family=="tuple_output":self.sections=[2]*(context+2)
  else:self.sections=2
  if family=="public_weight":
   generator=torch.Generator().manual_seed(20261005+context)
   self.register_buffer("weight",torch.randn((6,3),generator=generator,dtype=torch.float64)*.125)
 def forward(self,x):
  family=self.family
  if family=="tuple_output":return torch.split(x*.125,self.sections,dim=-1)
  if family=="public_weight":
   sections=([3,3],[2,2,2],[1]*6)[self.context]
   parts=self.weight.split(split_size=sections,dim=0)
   y=torch.cat([F.linear(x,p) for p in parts],dim=-1)
  else:
   z=x*.125
   if family=="function_sections":parts=torch.split(z,self.sections,dim=self.axis)
   elif family=="keyword_sections":parts=torch.split(tensor=z,split_size_or_sections=self.sections,dim=self.axis)
   else:parts=z.split(split_size=2,dim=-1)
   if family=="public_loop":
    values=[]
    for i,p in enumerate(parts):values.append(p*(.125*(i+1)))
   else:values=[p.square()+p*.5 for p in parts]
   y=torch.cat(values,dim=self.axis)
  if self.context==0:return y
  aux=x.mean(dim=-1,keepdim=True)
  if self.context==1:return y,aux
  return dict(features=y,auxiliary=aux)

def make_model(family,context):
 if family not in SHAPES or type(context) is not int or not 0<=context<3:raise ValueError("Fixture selection")
 return Model(family,context).double().eval()

def source_specs(family,context):return [dict(name="x",shape=SHAPES[family][context])]
def source_outputs(family,context):
 if family=="tuple_output":return ["piece_"+str(i) for i in range(context+2)]
 return ["out"] if context==0 else ["out","aux"]
