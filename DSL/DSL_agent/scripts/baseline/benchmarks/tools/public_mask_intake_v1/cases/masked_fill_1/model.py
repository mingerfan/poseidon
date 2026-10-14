import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.04868919631965485, 0.09235767436870318], [0.05442078302928266, 0.260918409155131]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
        self.register_buffer("proj",torch.tensor([[-0.13712911358776753, 0.09420348441155729], [-0.06273225778971381, -0.09820761290807983]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([-0.030753083781615037, 0.020257997410789624],dtype=torch.float64))
    def forward(self,x):
        snapshot=x.clone()
        c=(x.square()+self.w).masked_fill(self.mask.reshape(1,2),x.new_full((),-.125))
        return F.linear(c*self.w,self.proj,self.bias)+snapshot
def make():
    return Model().eval()
