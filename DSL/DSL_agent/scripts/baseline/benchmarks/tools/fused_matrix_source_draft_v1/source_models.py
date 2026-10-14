import torch
from source_helpers import fused

def dimensions(op):
    return ([2],[2,3],[3,2]) if op=="addmm" else ([2],[2,2,3],[2,3,2])
class FusedModel(torch.nn.Module):
    def __init__(self,op,context,style):
        super().__init__();self.op,self.context,self.style=op,context,style
        index={"public_right":2,"public_bias":0}.get(context)
        if index is not None:
            dims=dimensions(op)[index]
            self.register_buffer("weight",torch.arange(torch.tensor(dims).prod().item(),dtype=torch.float64).reshape(dims)/13-.25)
    def forward(self,bias,left,right):
        previous=left.clone()
        b=self.weight if self.context=="public_bias" else bias
        r=self.weight if self.context=="public_right" else right
        value=fused(self.op,self.style,b,left,r)
        if self.context=="public_right":
            value=value.reshape(value.shape[0],-1).square().sum(dim=-1,keepdim=True)
        left.view(-1).add_(.125)
        return dict(result=value,current=left,previous=previous)
def build(op,context,style):
    return FusedModel(op,context,style).double().eval()
