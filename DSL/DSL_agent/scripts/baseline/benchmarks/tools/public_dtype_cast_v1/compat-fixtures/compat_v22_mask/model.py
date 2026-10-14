import torch
class Model(torch.nn.Module):
    def __init__(self,dtype,target,context=0,roundtrip=False):
        super().__init__()
        types={"<f4":torch.float32,"<f8":torch.float64,"<i4":torch.int32,"<i8":torch.int64,"|b1":torch.bool}
        self.register_buffer("public",torch.zeros((2,3),dtype=types[dtype]))
        self.target=target;self.context=context;self.roundtrip=roundtrip
    def forward(self,x):
        return x*self.public

def build(dtype,target,context=0,roundtrip=False):
    return Model(dtype,target,context,roundtrip)
