import torch
from torch import nn
from torch.nn import functional as F

class ProjectedAttention(nn.Module):
    def __init__(self,mask_kind,context):
        super().__init__()
        self.mask_kind=mask_kind;self.context=context
        self.q=nn.Linear(4,4);self.k=nn.Linear(4,4);self.v=nn.Linear(4,4)
        length=3 if context==1 else 2
        if mask_kind=="boolean":
            mask=torch.tril(torch.ones(length,length,dtype=torch.bool))
            self.register_buffer("mask",mask)
        elif mask_kind=="additive":
            mask=torch.arange(length*length,dtype=torch.float64).reshape(length,length)*.025-.125
            self.register_buffer("mask",mask)
    def forward(self,x):
        return self.compute(x,x)
    def compute(self,x,other):
        q=self.q(x).reshape(x.shape[0],x.shape[1],2,2).transpose(1,2)
        k=F.linear(x,self.k.weight,self.k.bias).reshape(x.shape[0],x.shape[1],2,2).transpose(1,2)
        source=other if self.context==2 else x
        v=self.v(source).reshape(source.shape[0],source.shape[1],2,2).transpose(1,2)
        if self.context==1:q=F.dropout(q,p=.2,training=False)
        if self.context==2:v=F.dropout2d(v,p=.3,training=False)
        mask=self.mask if self.mask_kind in ("boolean","additive") else None
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,
                                         is_causal=self.mask_kind=="causal")
        y=y.transpose(1,2).reshape(y.shape[0],x.shape[1],4)
        if self.context==1:return (y+.125*x,)
        if self.context==2:return {"context":y,"skip":x[:,0,:]}
        return y

class ProjectedAttentionDual(ProjectedAttention):
    def forward(self,x,other):
        return self.compute(x,other)

class PublicAlpha(nn.Module):
    def __init__(self,context):
        super().__init__();self.context=context
        if context!=2:self.register_buffer("bias",torch.empty(3,dtype=torch.float64))
    def forward(self,x):
        if self.context==0:return torch.add(x,self.bias,alpha=.5)
        if self.context==1:return torch.sub(input=x,other=self.bias,alpha=-2.)
        return x.add(.25,alpha=-2.)

def make_model(family,context,mask_kind="none"):
    attention=ProjectedAttentionDual if context==2 else ProjectedAttention
    module=attention(mask_kind,context) if family=="projected_attention" else PublicAlpha(context)
    return module.double().eval()
