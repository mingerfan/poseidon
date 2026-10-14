"""Twelve small source contexts, independent of DSL/compilation."""
import torch
class Dictionary(torch.nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant
  if variant==0:self.linear=torch.nn.Linear(3,3,bias=True).double()
  if variant==2:self.register_buffer("mask",torch.tensor([[True,False],[False,True]]))
 def forward(self,batch):
  x=batch["left"];y=batch["right"]
  if self.variant==0:return self.linear(x).square()+y
  if self.variant==1:return {"sum":(x+y).sum(dim=1),"extra":None}
  return {"masked":(x+y)*self.mask,"skip":[x,None]}
class Nested(torch.nn.Module):
 def __init__(self,variant):super().__init__();self.variant=variant
 def forward(self,batch,options):
  if self.variant==0:
   x=batch[0];y=batch[1][0]
   return (x+y*options["gain"],[x.square(),None])
  if self.variant==1:
   x=batch[0]["x"];y=batch[0]["y"]
   return (x.reshape(2,2)+y).transpose(0,1)
  x=batch[0]["x"];y=batch[0]["y"]
  return {"concat":torch.cat([x+y*options[0],x-y*options[1]],dim=1)}
class Keywords(torch.nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant
  if variant==0:self.linear=torch.nn.Linear(3,3,bias=False).double()
 def forward(self,x,/,*,other,gain=.25,mode="add"):
  if self.variant==0:return self.linear(x)+other*gain
  if self.variant==1:return x@other
  return x+other*gain if mode=="add"else x-other*gain
class Defaults(torch.nn.Module):
 def __init__(self,variant):super().__init__();self.variant=variant
 def forward(self,x,y,settings=None,include_aux=False):
  if self.variant==0:
   value=x+y
  elif self.variant==1:
   value=x*settings[0]+y*settings[1]
  else:value=(x+y*settings["gain"]).reshape(settings["shape"])
  return {"value":value,"aux":[x,None]}if include_aux else value
class ListDefault(Defaults):
 def forward(self,x,y,settings=[.5,-.25],include_aux=False):
  return super().forward(x,y,settings,include_aux)
class DictDefault(Defaults):
 def forward(self,x,y,settings={"gain":.5,"shape":[2,3]},include_aux=False):
  return super().forward(x,y,settings,include_aux)
def make_model(group,variant):
 return {"dictionary":Dictionary,"nested":Nested,"keywords":Keywords,
         "defaults":Defaults if variant==0 else ListDefault if variant==1 else DictDefault}[group](variant)
