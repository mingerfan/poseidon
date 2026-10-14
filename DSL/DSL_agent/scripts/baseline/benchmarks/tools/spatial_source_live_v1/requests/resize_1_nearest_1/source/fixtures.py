"""Trusted repository fixtures for draft spatial capture; fixed public state."""
import torch
from torch import nn
from torch.nn import functional as F

def public_weight(shape, seed):
    generator=torch.Generator().manual_seed(seed)
    return .3*torch.rand(shape,dtype=torch.float64,generator=generator)-.15

class Resize(nn.Module):
    def __init__(self,rank,mode,context):
        super().__init__()
        self.rank,self.mode,self.context=rank,mode,context
        self.align=bool(context%2) if mode in ("linear","bilinear") else None
        target=(5,) if rank==1 else (3,4)
        if context==0:
            self.layer=nn.Upsample(size=target,mode=mode,align_corners=self.align)
        self.target=target
    def forward(self,x):
        if self.context==0:
            return self.layer(x)
        if self.context==1:
            return F.interpolate(x,scale_factor=1.7,mode=self.mode,
                align_corners=self.align,recompute_scale_factor=False)
        y=F.interpolate(x,size=self.target,mode=self.mode,align_corners=self.align)
        return y.reshape(y.shape[0],-1)

class Transpose(nn.Module):
    def __init__(self,rank,context):
        super().__init__()
        self.rank,self.context=rank,context
        ci=2 if rank==1 else 4
        kernel=(2,) if rank==1 else (2,3)
        stride=(2,) if rank==1 else (2,1)
        padding=(0,) if rank==1 else (0,1)
        opad=(1,) if rank==1 else (1,0)
        co=2
        klass=nn.ConvTranspose1d if rank==1 else nn.ConvTranspose2d
        if context!=1:
            self.layer=klass(ci,co,kernel,stride=stride,padding=padding,
                output_padding=opad,groups=2,dtype=torch.float64)
            with torch.no_grad():
                self.layer.weight.copy_(public_weight(tuple(self.layer.weight.shape),41+rank))
                self.layer.bias.copy_(public_weight((co,),71+rank))
        else:
            self.register_buffer("weight",public_weight((ci,1,*kernel),101+rank))
            self.register_buffer("bias",public_weight((co,),151+rank))
        self.stride,self.padding,self.opad=stride,padding,opad
    def forward(self,x):
        if self.context==1:
            if self.rank==1:
                return F.conv_transpose1d(x,self.weight,self.bias,stride=self.stride,padding=self.padding,
                    output_padding=self.opad,groups=2,dilation=1)
            return F.conv_transpose2d(input=x,weight=self.weight,bias=self.bias,stride=self.stride,
                padding=self.padding,output_padding=self.opad,groups=2,dilation=1)
        if self.context==2:
            return self.layer(x,output_size=(7,) if self.rank==1 else (5,3))
        return self.layer(x)

class Composed(nn.Module):
    def __init__(self,context):
        super().__init__();self.context=context
        self.up=nn.Upsample(size=5,mode="linear",align_corners=False)
        self.project=nn.Linear(5,2,dtype=torch.float64)
        with torch.no_grad():
            self.project.weight.copy_(public_weight((2,5),201))
            self.project.bias.copy_(public_weight((2,),202))
    def forward(self,x):
        y=self.up(x)
        if self.context==0:return self.project(y.flatten(1))
        if self.context==1:return y+y.square()
        old=y.clone();view=y.view(1,-1);view.add_(.125)
        return (old,y)

def cases():
    result=[]
    for rank,modes in [(1,("nearest","nearest-exact","linear")),(2,("nearest","nearest-exact","bilinear"))]:
        for mode in modes:
            for context in range(3):
                result.append((f"resize_{rank}_{mode.replace('-','_')}_{context}",Resize(rank,mode,context).eval(),
                    [dict(name="x",shape=[1,1,3] if rank==1 else [1,2,2,3])],["output"]))
    for rank in (1,2):
        for context in range(3):
            result.append((f"transpose_{rank}_{context}",Transpose(rank,context).eval(),
                [dict(name="x",shape=[1,2,3] if rank==1 else [1,4,2,3])],["output"]))
    for context in range(3):
        result.append((f"composition_{context}",Composed(context).eval(),[dict(name="x",shape=[1,1,3])],
                       ["before","after"] if context==2 else ["output"]))
    return result
