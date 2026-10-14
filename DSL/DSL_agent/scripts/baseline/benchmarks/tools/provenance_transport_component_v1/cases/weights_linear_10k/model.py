import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(100,100,dtype=torch.float64)
    def forward(self,x):
        return self.fc(x)
def make():
    return Model().eval()
