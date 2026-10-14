import torch
class PublicViews(torch.nn.Module):
    def __init__(self,operation,context=0,float32=False,named=False,unused=False):
        super().__init__()
        self.operation=operation;self.context=context;self.named=named
        shape=((6,),(2,3),(2,1,3))[context]
        dtype=torch.float32 if float32 else torch.float64
        public=torch.arange(1,7,dtype=dtype).reshape(shape)
        if operation=="bool_cast":public=public.remainder(2)==0
        self.register_buffer("public",public)
        if unused:self.register_buffer("unused",torch.arange(6000,dtype=dtype))
    def forward(self,x):
        p=self.public
        if self.operation=="reshape":p=p.view(*((2,3),(3,2),(1,2,3))[self.context])
        elif self.operation=="permute":p=p.permute(*tuple(reversed(range(p.ndim))))
        elif self.operation=="slice":p=p[...,::2]
        elif self.operation=="bool_cast":p=p.float()
        else:p=p.contiguous()
        y=x*p if self.operation=="bool_cast"else x/p
        return {"primary":y,"saved":[x,None]}if self.named else y
def build(operation,context=0,float32=False,named=False,unused=False):
    return PublicViews(operation,context,float32,named,unused).eval()
