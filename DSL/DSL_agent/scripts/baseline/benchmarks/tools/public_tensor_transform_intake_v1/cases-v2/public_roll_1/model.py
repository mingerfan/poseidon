import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[-0.0845525621159875, -0.48598060089980144], [0.3528991309084225, 0.3193132342275963]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[0.1600208391327943, -0.1308281000223798], [0.009606237047294908, -0.28518746064738243]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([0.09063342550967726, -0.000889624183199933],dtype=torch.float64))
    def forward(self,x):
        w=self.public.roll(shifts=(1,-1),dims=(0,-1))
        return F.linear(x*w,self.proj,self.bias)+x
def make():
    return Model().eval()
