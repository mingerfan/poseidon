import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.09069768163863068, -0.22419472138811986], [-0.4199942014009556, -0.1690673380415879]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[0.06545262324523776, -0.052103110914072326], [-0.1502837490650755, -0.1904570045719013]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([-0.14027467990744893, -0.059822845413519704],dtype=torch.float64))
    def forward(self,x):
        c=x.new_full((1,2),.125000017,dtype=torch.float32).double()
        return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x
def make():
    return Model().eval()
