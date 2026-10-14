import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.09204369010103891, 0.4437267962566777], [0.43008086299305354, -0.263089785168925]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[-0.21066915989343932, -0.1285266738799956], [-0.18209508431965116, 0.2460494466072029]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([-0.09251519877727035, -0.09578450396822258],dtype=torch.float64))
    def forward(self,x):
        c=torch.full_like(x.roll(1,-1),-.375)
        return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x
def make():
    return Model().eval()
