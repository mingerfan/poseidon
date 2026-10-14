import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.15008275284647776, -0.3759617194859967], [0.056600735289728366, 0.016041116800251665]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
        self.register_buffer("proj",torch.tensor([[-0.11104542900936876, 0.10076167121161128], [0.260591081441972, -0.09549464118179668]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([0.0017752748568258814, 0.06012373879040256],dtype=torch.float64))
    def forward(self,x):
        snapshot=x.clone()
        c=self.w.float().double().where(self.mask.reshape(1,2),x.square())
        return F.linear(c*self.w,self.proj,self.bias)+snapshot
def make():
    return Model().eval()
