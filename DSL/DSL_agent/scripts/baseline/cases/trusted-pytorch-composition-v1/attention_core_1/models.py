"""Small deterministic PyTorch composition contexts, supplementary to the frozen benchmark."""
import math
import torch
from torch import nn

SHAPES = {
    "cnn_norm_head": [[1,1,3,3],[1,2,3,3],[1,1,4,4]],
    "residual_layernorm": [[1,4],[2,3],[1,6]],
    "indexed_batched_matmul": [[1,2,2],[2,2,3],[1,3,2]],
    "public_matrix_fusion": [[1,2,3],[2,2,2],[1,3,4]],
    "attention_core": [[1,2,2],[1,3,2],[2,2,2]],
    "channel_broadcast_mix": [[1,1,3,3],[1,2,2,3],[2,1,2,2]],
    "static_split_stack": [[2,4],[1,6],[2,6]],
    "norm_depth_branches": [[1,4],[2,3],[1,6]],
}

def source_specs(family, context):
    dims=SHAPES[family][context]
    result=[dict(name="x",shape=dims)]
    if family=="indexed_batched_matmul":
        other=[[1,2,2],[1,3,2],[1,2,2]][context]
        result.append(dict(name="y",shape=other))
    elif family=="attention_core":
        result.extend([dict(name="y",shape=dims),dict(name="v",shape=dims)])
    elif family=="channel_broadcast_mix":
        result.append(dict(name="y",shape=[dims[0],dims[1],1,1]))
    return result

def source_outputs(context):
    return ["out"] if context==0 else ["out","aux"]

def source_approximations(family):
    defaults={}
    if family in ("cnn_norm_head","residual_layernorm","channel_broadcast_mix","norm_depth_branches"):
        defaults["rsqrt"]=dict(method="newton",domain=[.5,1.],iterations=1)
    elif family=="attention_core":
        defaults=dict(exp=dict(method="taylor",domain=[-.5,.5],degree=2),
                      reciprocal=dict(method="goldschmidt",domain=[1.,5.],iterations=1))
    return dict(format="poseidon-pytorch-approximations-v1",defaults=defaults,modules=[],nodes=[])

class Model(nn.Module):
    def __init__(self,family,context):
        super().__init__()
        self.family,self.context=family,context
        dims=SHAPES[family][context]
        if family=="cnn_norm_head":
            _,channels,height,width=dims
            self.features=nn.Sequential(nn.Conv2d(channels,2,1),
                nn.GroupNorm(2 if context==1 else 1,2,eps=.5),
                nn.AvgPool2d(2,stride=1),nn.Flatten(1))
            self.head=nn.Linear(2*(height-1)*(width-1),2)
        elif family=="residual_layernorm":
            width=dims[-1]
            self.projection=nn.Sequential(nn.Linear(width,width),nn.Identity(),nn.Dropout(.2))
            self.norm=nn.LayerNorm(width,eps=.5)
            self.head=nn.Linear(width,2)
        elif family=="indexed_batched_matmul":
            indices=[1,0] if dims[-1]==2 else [2,0,1]
            if context==2:indices=[0,0]
            self.register_buffer("indices",torch.tensor(indices,dtype=torch.int64))
        elif family=="public_matrix_fusion":
            batch,rows,width=dims
            gen=torch.Generator().manual_seed(5300+context)
            matrix=torch.rand([batch,width,2],generator=gen,dtype=torch.float64)*.2-.1
            back=torch.rand([rows,2],generator=gen,dtype=torch.float64)*.2-.1
            self.register_buffer("matrix",matrix)
            self.register_buffer("back",back)
        elif family=="attention_core":
            self.softmax=nn.Softmax(dim=-1)
        elif family=="channel_broadcast_mix":
            channels=dims[1]
            self.norm=nn.InstanceNorm2d(channels,eps=.5,affine=True,track_running_stats=False)
            self.pool=nn.AdaptiveAvgPool2d((1,1))
            self.head=nn.Linear(channels,2)
        elif family=="static_split_stack":
            self.width=dims[-1]
            self.head=nn.Linear(self.width,2)
        elif family=="norm_depth_branches":
            width=dims[-1]
            self.layers=nn.ModuleList([nn.Linear(width,width) for _ in range(context+1)])
            self.norm=nn.LayerNorm(width,eps=.5)
            self.head=nn.Linear(width,2)
        else:raise ValueError("Unknown test context")
        self.double()
        gen=torch.Generator().manual_seed(8100+list(SHAPES).index(family)*100+context)
        with torch.no_grad():
            for name,parameter in self.named_parameters():
                values=torch.rand(parameter.shape,generator=gen,dtype=torch.float64)
                if "norm.weight" in name or family=="cnn_norm_head" and name=="features.1.weight":
                    values=.7+.2*values
                elif "bias" in name:values=(values-.5)*.02
                else:values=(values-.5)*.24
                parameter.copy_(values)
        self.eval()

    def finish(self,out,aux):
        if self.context==0:return out
        if self.context==1:return out,aux
        return {"primary":out,"auxiliary":aux}

    def compute(self,x,y=None,v=None):
        aux=None
        if self.family=="cnn_norm_head":
            h=self.features(x*.125)
            out=self.head(h)
            if self.context:aux=h.mean(dim=-1,keepdim=True)
        elif self.family=="residual_layernorm":
            z=x*.125
            h=self.norm(self.projection(z)+z)
            out=self.head(h)
            if self.context:aux=h.mean(dim=-1,keepdim=True)
        elif self.family=="indexed_batched_matmul":
            h=(x*.125).index_select(dim=-1,index=self.indices) @ (y*.125)
            out=h.square()+h*.25
            if self.context:aux=h.mean(dim=-1,keepdim=True)
        elif self.family=="public_matrix_fusion":
            h=(x*.125) @ self.matrix
            out=h.transpose(-2,-1) @ self.back
            if self.context:aux=h.var(dim=-1,correction=0,keepdim=True)
        elif self.family=="attention_core":
            weights=self.softmax((x*.125) @ (y*.125).transpose(-2,-1))
            out=weights @ (v*.125)
            if self.context:aux=weights.mean(dim=-1,keepdim=True)
        elif self.family=="channel_broadcast_mix":
            h=self.norm(x*.0625+y*.125)
            pooled=self.pool(h.square()).flatten(1)
            out=self.head(pooled+y.flatten(1)*.125)
            if self.context:aux=h.mean(dim=(2,3),keepdim=False)+y.flatten(1)*.125
        elif self.family=="static_split_stack":
            z=x*.125
            first,second=torch.split(z,[self.width//2,self.width//2],dim=-1)
            h=torch.stack((first.square(),second*.5),dim=1)
            out=self.head(h.permute(0,2,1).reshape(x.shape[0],self.width))
            if self.context:aux=z.mean(dim=-1,keepdim=True)
        elif self.family=="norm_depth_branches":
            h=x*.125
            for layer in self.layers:h=layer(h).square()
            normalized=self.norm(h+x*.125)
            out=self.head(normalized)
            if self.context:aux=h.mean(dim=-1,keepdim=True)
        return self.finish(out,aux)

class OneInput(Model):
    def forward(self,x):return self.compute(x)

class TwoInputs(Model):
    def forward(self,x,y):return self.compute(x,y)

class ThreeInputs(Model):
    def forward(self,x,y,v):return self.compute(x,y,v)

def make_model(family,context):
    count=len(source_specs(family,context))
    return {1:OneInput,2:TwoInputs,3:ThreeInputs}[count](family,context)
