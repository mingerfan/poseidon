import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.027643255775300635, 0.14270409169939025], [-0.37771176526449896, -0.3010397772054633]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
        self.register_buffer("proj",torch.tensor([[-0.024522664120536275, -0.20954273908805499], [-0.0037581669855430433, 0.26259716605242084]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([0.11664474141735565, 0.01025040022542184],dtype=torch.float64))
    def forward(self,x):
        snapshot=x.clone()
        c=torch.where(condition=self.mask.reshape(1,2),input=x.square(),other=self.w)
        return F.linear(c*self.w,self.proj,self.bias)+snapshot
def make():
    return Model().eval()
