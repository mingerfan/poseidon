import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[-0.23944668280250325, -0.43676594826748905]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[0.01000359842285492, 0.20284056378377957], [0.34163247855407997, 0.03645843965964113]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([0.14817729495376666, 0.11486803789019165],dtype=torch.float64))
    def forward(self,x):
        w=self.public.tile((2,1))
        return F.linear(x+w,self.proj,self.bias)+x
def make():
    return Model().eval()
