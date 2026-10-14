import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.12306582944111555, 0.2776576567169923], [0.029263975835348177, -0.18592669715761395]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
    def forward(self,x,*,z):
        snapshot=x.clone()
        c=torch.where(self.mask.reshape(2,1),self.w.new_full(self.w.shape,2,dtype=torch.int64),False).double()
        return {'y':x*(c+.125)+z+self.w,'saved':[snapshot,x,None]}
def make():
    return Model().eval()
