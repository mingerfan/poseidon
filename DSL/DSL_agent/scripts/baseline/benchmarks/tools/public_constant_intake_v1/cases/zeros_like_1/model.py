import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.4213532455370818, -0.3296468224305133], [0.3967831911487191, -0.011672137393164472]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[-0.11252629582931242, 0.11907534476192871], [0.029344316092316858, 0.2568357598345035]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([-0.027222603901495793, -0.03435802892542848],dtype=torch.float64))
    def forward(self,x):
        c=torch.zeros_like(x.square())
        return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x
def make():
    return Model().eval()
