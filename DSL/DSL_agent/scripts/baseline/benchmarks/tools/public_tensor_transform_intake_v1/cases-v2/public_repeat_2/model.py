import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[0.3470601465158746, -0.10483191870659248]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[0.2118356374485567, -0.17478908084127448], [0.3129457596827243, 0.08430169336764382]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([0.10751815400463974, -0.14453468622848661],dtype=torch.float64))
    def forward(self,x):
        w=self.public.repeat(2,1)
        return F.linear(x.flip((-1,))*w,self.proj,self.bias)
def make():
    return Model().eval()
