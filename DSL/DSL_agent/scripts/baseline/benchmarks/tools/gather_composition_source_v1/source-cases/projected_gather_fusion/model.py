import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indices",torch.tensor([[3,0,3],[1,2,0]],dtype=torch.int64))
        self.linear=torch.nn.Linear(3,4)
    def forward(self,x,y):
        projected=self.linear(x)
        return projected.gather(dim=1,index=self.indices)+y
def make():
    return Model().double().eval()
