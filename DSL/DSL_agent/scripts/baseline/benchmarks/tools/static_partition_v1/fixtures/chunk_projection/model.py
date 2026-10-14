import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(2,4,bias=False,dtype=torch.float64)
    def forward(self,x):
        a,b=self.linear(x).chunk(2,dim=-1)
        return a+b*0.5
def make(): return Model().double().eval()
