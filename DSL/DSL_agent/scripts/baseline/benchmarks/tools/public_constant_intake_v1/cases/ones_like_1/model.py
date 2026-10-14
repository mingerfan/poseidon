import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.058702919008549126, -0.1303071993482371], [0.06716624997113657, -0.27709163663689523]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[0.12938820062955858, -0.2867571198763617], [-0.26828595490299034, 0.19521937207131707]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([0.11860734678522741, -0.06259584518571872],dtype=torch.float64))
    def forward(self,x):
        c=torch.ones_like(x,dtype=torch.int64).double()
        return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x
def make():
    return Model().eval()
