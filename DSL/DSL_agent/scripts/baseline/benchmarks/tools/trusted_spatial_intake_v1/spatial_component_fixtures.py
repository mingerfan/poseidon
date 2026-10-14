"""Static spatial combinations with structured calls and public expression state."""
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_input_binding import CALL_FORMAT
from fixtures import public_weight

def tensor(name):return dict(kind="tensor",name=name)
def literal(value):return dict(kind="literal",value=value)
def call(args,kwargs=None):return dict(format=CALL_FORMAT,args=args,kwargs=kwargs or {})
class Dictionary(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer("mask",torch.tensor([True,False,True,True,False]))
    def forward(self,batch):
        y=F.interpolate(batch["value"],size=5,mode="linear",align_corners=False)
        return {"result":y*self.mask,"aux":[y.square(),None]}
class Nested(nn.Module):
    def forward(self,inputs,*,other):
        a=F.interpolate(inputs[0],size=5,mode="nearest-exact")
        b=F.interpolate(other["value"],size=5,mode="linear",align_corners=True)
        return a+b
class Keywords(nn.Module):
    def forward(self,x,*,length=5,mode="linear",aligned=False):
        return F.interpolate(x,size=length,mode=mode,align_corners=aligned)
class PublicState(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer("w",public_weight((2,),107).float())
        self.register_buffer("b",public_weight((1,),108).float())
    def forward(self,x):
        w=(self.w.double()+.125).reshape(1,1,2)
        b=self.b.double()*.5
        return F.conv_transpose1d(x,w,b,stride=2,output_padding=1)
class NamedAlias(nn.Module):
    def forward(self,x):
        y=F.interpolate(x,size=5,mode="nearest")
        before=y.clone();alias=y.reshape(1,5);alias.add_(.125)
        return {"before":before,"after":y,"aux":[None]}
class Upsampling2D(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.layer=nn.UpsamplingNearest2d(size=(3,4))if mode=="nearest"else nn.UpsamplingBilinear2d(size=(3,4))
    def forward(self,x):return self.layer(x)
def contexts():
    x=[dict(name="x",shape=[1,1,3])]
    return [
      ("structured_dict",Dictionary().eval(),x,["result","square"],call([dict(kind="dict",items=[["value",tensor("x")]])]),"strict-float64-v1"),
      ("nested_keywords",Nested().eval(),[dict(name="y",shape=[1,1,3]),*x],["output"],
       call([dict(kind="list",items=[tensor("x")])],dict(other=dict(kind="dict",items=[["value",tensor("y")]]))),"strict-float64-v1"),
      ("public_defaults",Keywords().eval(),x,["output"],call([tensor("x")]),"strict-float64-v1"),
      ("float32_public_weight",PublicState().eval(),x,["output"],call([tensor("x")]),"convert-float64-v1"),
      ("named_alias",NamedAlias().eval(),x,["before","after"],call([tensor("x")]),"strict-float64-v1"),
      ("nearest_2d_module",Upsampling2D("nearest").eval(),[dict(name="x",shape=[1,1,2,3])],["output"],call([tensor("x")]),"strict-float64-v1"),
      ("bilinear_2d_module",Upsampling2D("bilinear").eval(),[dict(name="x",shape=[1,1,2,3])],["output"],call([tensor("x")]),"strict-float64-v1"),
    ]
