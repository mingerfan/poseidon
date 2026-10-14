import torch
from torch.nn import functional as F

class CommonModel(torch.nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind=kind
        if kind=="alpha":self.register_buffer("bias",torch.tensor([.125,-.25,.75],dtype=torch.float64))
        if kind=="boolean":self.register_buffer("mask",torch.tensor([[True,False],[True,True]]))
        if kind=="additive":self.register_buffer("mask",torch.tensor([[0.,-.2],[.1,0.]],dtype=torch.float64))
        if kind=="module_dropout":self.layer=torch.nn.Dropout(p=.3,inplace=False)

    def forward(self,x,y=None,z=None):
        if self.kind=="alpha":
            previous=x.clone()
            weighted=torch.add(x,self.bias,alpha=1.25)
            result=weighted*x+previous
            x.view(-1).add_(.125)
            return result,x,previous
        if self.kind in ("dropout","module_dropout"):
            previous=x.clone()
            alias=F.dropout(x,p=.3,training=False) if self.kind=="dropout" else self.layer(x)
            x.add_(.25)
            return alias,x,previous
        previous=z.clone()
        x.view(-1).add_(.125)
        value=F.dropout(z,p=.2,training=False)
        mask=self.mask if self.kind in ("boolean","additive") else None
        attention=F.scaled_dot_product_attention(x,y,value,attn_mask=mask,dropout_p=0.,
                                                  is_causal=self.kind=="causal")
        z.add_(.05)
        return dict(attention=attention,current=z,previous=previous)

class AlphaModel(CommonModel):
    def __init__(self):
        super().__init__("alpha")
    def forward(self,x):
        previous=x.clone()
        weighted=torch.add(x,self.bias,alpha=1.25)
        result=weighted*x+previous
        x.view(-1).add_(.125)
        return result,x,previous

class DropoutModel(CommonModel):
    def forward(self,x):
        previous=x.clone()
        alias=F.dropout(x,p=.3,training=False) if self.kind=="dropout" else self.layer(x)
        x.add_(.25)
        return alias,x,previous

def build(kind):
    if kind=="alpha":return AlphaModel().double().eval()
    if kind in ("dropout","module_dropout"):return DropoutModel(kind).double().eval()
    if kind=="optional_signature":return CommonModel("alpha").double().eval()
    return CommonModel(kind).double().eval()
