import torch
class Model(torch.nn.Module):
    def __init__(self,dtype,target,context=0,roundtrip=False):
        super().__init__()
        types={"<f4":torch.float32,"<f8":torch.float64,"<i4":torch.int32,"<i8":torch.int64,"|b1":torch.bool}
        self.register_buffer("public",torch.zeros((2,3),dtype=types[dtype]))
        self.target=target;self.context=context;self.roundtrip=roundtrip
    def forward(self,x):
        w=self.public
        if self.target=="<f4":w=w.float()
        else:w=w.to(dtype=torch.float64)
        if self.roundtrip:w=w.double()
        if self.context==1:return x+w.transpose(0,1)
        y=x*w
        if self.context==2:return {"primary":y+x,"saved":[x,None]}
        return y
def build(dtype,target,context=0,roundtrip=False):
    return Model(dtype,target,context,roundtrip)
