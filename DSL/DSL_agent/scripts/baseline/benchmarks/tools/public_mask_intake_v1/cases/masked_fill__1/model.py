import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.3661109215685738, 0.33908093709950826], [-0.15634208025334032, -0.0005310448665894474]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
        self.register_buffer("proj",torch.tensor([[0.22827478802767825, -0.18403385605136], [-0.20011579077665415, 0.24866165414357627]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([-0.11170430703077106, -0.07680761910410266],dtype=torch.float64))
        self.register_buffer("fill",torch.tensor(.1875,dtype=torch.float64))
    def forward(self,x):
        snapshot=x.clone()
        base=x.view(2,2)
        c=base.masked_fill_(self.mask.reshape(1,2),self.fill)
        return F.linear(c*self.w,self.proj,self.bias)+snapshot
def make():
    return Model().eval()
