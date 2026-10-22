"""Small inference fixtures for equivalent PyTorch module/function spellings."""
import torch
from torch.nn import functional as F

def fixed(shape, offset=0):
    count=1
    for n in shape:count*=n
    return (torch.arange(count,dtype=torch.float64).reshape(shape)+offset-count/2)/32

class Conv(torch.nn.Module):
    def __init__(self, rank, variant):
        super().__init__();self.rank=rank;self.variant=variant
        channels=2 if variant==2 else 1
        kernel=(1,) * rank if variant==2 else (2,) * rank
        self.register_buffer("weight",fixed((channels,1,*kernel),1))
        if variant!=1:self.register_buffer("bias",fixed((channels,),2))
    def forward(self,x):
        op=F.conv1d if self.rank==1 else F.conv2d
        if self.variant==0:return op(x,self.weight,self.bias)
        if self.variant==1:
            return op(input=x,weight=self.weight,bias=None,stride=2,padding=1,dilation=1,groups=1)
        return op(x,self.weight,self.bias,1,0,1,2)+x*.125

class Pool(torch.nn.Module):
    def __init__(self, rank, variant):
        super().__init__();self.rank=rank;self.variant=variant
    def forward(self,x):
        op=F.avg_pool1d if self.rank==1 else F.avg_pool2d
        if self.variant==0:return op(x,2)
        if self.variant==1:return op(input=x,kernel_size=2,stride=2,padding=1,
                                     ceil_mode=False,count_include_pad=False)
        return op(x,1,1,0,False,True)+x*.125

class Batch(torch.nn.Module):
    def __init__(self, variant):
        super().__init__();self.variant=variant
        self.register_buffer("mean",torch.tensor([.125,-.25],dtype=torch.float64))
        self.register_buffer("variance",torch.tensor([.75,1.25],dtype=torch.float64))
        if variant!=0:
            self.register_buffer("gain",torch.tensor([.5,.75],dtype=torch.float64))
            self.register_buffer("bias",torch.tensor([.0625,-.03125],dtype=torch.float64))
    def forward(self,x):
        if self.variant==0:return F.batch_norm(x,self.mean,self.variance,eps=.125)
        if self.variant==1:return F.batch_norm(input=x,running_mean=self.mean,running_var=self.variance,
            weight=self.gain,bias=self.bias,training=False,momentum=.2,eps=.125)
        return F.batch_norm(x,self.mean,self.variance,self.gain,self.bias,False,.1,.125)+x*.125

class Layer(torch.nn.Module):
    def __init__(self, variant):
        super().__init__();self.variant=variant
        self.layer=torch.nn.LayerNorm((2,2) if variant==2 else (2,),eps=1.,elementwise_affine=False)
    def forward(self,x):
        y=self.layer(x)
        return y+x*.125 if self.variant==1 else y.mean(-1,True) if self.variant==2 else y

class Group(torch.nn.Module):
    def __init__(self, variant):
        super().__init__();self.variant=variant
        self.layer=torch.nn.GroupNorm(2 if variant==1 else 1,2,eps=1.,affine=False)
    def forward(self,x):
        y=self.layer(x)
        return y+x.mean(-1,True)*.125 if self.variant==2 else y

class Transpose(torch.nn.Module):
    def __init__(self, variant):
        super().__init__();self.variant=variant
        if variant==0:
            self.register_buffer("weight",fixed((2,3),1))
            self.register_buffer("bias",fixed((2,),2))
    def forward(self,x):
        if self.variant==0:return x@self.weight.T+self.bias
        if self.variant==1:return x.T*x.T+x.T*.125
        return x.T+x.T*.125

def make(kind,variant=0):
    if kind=="conv1":return Conv(1,variant)
    if kind=="conv2":return Conv(2,variant)
    if kind=="pool1":return Pool(1,variant)
    if kind=="pool2":return Pool(2,variant)
    return {"batch":Batch,"layer":Layer,"group":Group,"transpose":Transpose}[kind](variant)
