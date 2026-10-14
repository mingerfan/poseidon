import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.26621147720245314, 0.315799748976334], [-0.24199862970552932, -0.009756490111800242]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
        self.register_buffer("proj",torch.tensor([[-0.06782593556244787, 0.29681581255599926], [0.2599816374701886, -0.025262029647644624]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([0.0498029072019959, 0.0972028206472704],dtype=torch.float64))
    def forward(self,x):
        snapshot=x.clone()
        c=self.w.float().masked_fill(self.mask.reshape(1,2),self.w.new_full((),.125000017,dtype=torch.float32)).double()
        return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x
def make():
    return Model().eval()
