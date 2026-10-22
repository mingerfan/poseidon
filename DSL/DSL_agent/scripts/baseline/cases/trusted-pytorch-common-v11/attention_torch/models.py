import torch
from torch import nn
from torch.nn import functional as F
import math

class Square(nn.Module):
    def forward(self,x):
        return torch.square(x)

class MLP(nn.Module):
    def __init__(self,style):
        super().__init__()
        self.style=style
        blocks=[nn.Linear(4,3),Square(),nn.Linear(3,2)]
        self.layers=nn.Sequential(*blocks) if style=="sequential" else nn.ModuleList(blocks)
    def forward(self,x):
        if self.style=="sequential":
            return self.layers(x)
        if self.style=="loop":
            for layer in self.layers:
                x=layer(x)
            return x
        return F.linear(torch.square(F.linear(x,self.layers[0].weight,self.layers[0].bias)),
                        self.layers[2].weight,self.layers[2].bias)

class CNN(nn.Module):
    def __init__(self,style):
        super().__init__()
        self.style=style
        self.depth=nn.Conv2d(2,2,3,padding=1,groups=2)
        self.norm=nn.BatchNorm2d(2,eps=0.001)
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.project=nn.Linear(2,3)
    def forward(self,x):
        if self.style=="module":
            y=self.norm(self.depth(x))
            return self.project(self.pool(y).flatten(1))
        y=F.conv2d(x,self.depth.weight,self.depth.bias,padding=1,groups=2)
        if self.style=="mixed":
            y=self.norm(y)
        else:
            y=F.batch_norm(y,self.norm.running_mean,self.norm.running_var,
                           self.norm.weight,self.norm.bias,training=False,eps=0.001)
        return F.linear(F.adaptive_avg_pool2d(y,(1,1)).flatten(1),
                        self.project.weight,self.project.bias)

class Residual(nn.Module):
    def __init__(self,style):
        super().__init__()
        self.style=style
        self.project=nn.Linear(4,4)
    def forward(self,x):
        y=self.project(x)
        if self.style=="operator":
            return y+x+x
        if self.style=="functional":
            return torch.add(input=y,other=x,alpha=2)
        return y.add(x,alpha=2)

class Normalize(nn.Module):
    def __init__(self,style):
        super().__init__()
        self.style=style
        self.norm=nn.LayerNorm(4,eps=1.)
    def forward(self,x):
        if self.style=="module":
            return self.norm(x)
        if self.style=="functional":
            return F.layer_norm(x,(4,),self.norm.weight,self.norm.bias,eps=1.)
        centered=x-x.mean(dim=-1,keepdim=True)
        variance=torch.mean(centered*centered,dim=-1,keepdim=True)
        return centered*torch.rsqrt(variance+1.)*self.norm.weight+self.norm.bias

class Attention(nn.Module):
    def __init__(self,style):
        super().__init__()
        self.style=style
    def forward(self,q,k,v):
        if self.style=="fused":
            return F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
        logits=(q@k.transpose(-1,-2))*(1./math.sqrt(2.))
        probabilities=torch.softmax(logits,dim=-1) if self.style=="torch" else F.softmax(logits,dim=-1)
        return probabilities@v

class EvalDropout(nn.Module):
    def __init__(self,style):
        super().__init__()
        self.style=style
        self.drop=nn.Dropout(p=0.25)
        self.drop_spatial=nn.Dropout2d(p=0.25)
    def forward(self,x):
        if self.style=="module":
            y=self.drop(x)
        elif self.style=="functional":
            y=F.dropout(x,p=0.25,training=self.training)
        else:
            y=self.drop_spatial(x)
        return y+torch.square(y)

def make_model(family,style):
    classes={"mlp":MLP,"cnn":CNN,"residual":Residual,"normalization":Normalize,
             "attention":Attention,"dropout":EvalDropout}
    return classes[family](style).double().eval()
