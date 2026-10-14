import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indices",torch.tensor([4,0,4,2],dtype=torch.int64))
        self.linear=torch.nn.Linear(4,4)
    def forward(self,x):
        selected=torch.gather(x,0,self.indices)
        return self.linear(selected)+selected
def make():
    return Model().double().eval()
