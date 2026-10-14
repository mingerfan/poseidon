import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(2,3,dtype=torch.float64)
    def forward(self,x):
        h=self.fc(x.transpose(-2,-1))
        z=h.square()+h*.125
        return {'features':z,'summary':z.mean(dim=-1),'same':z}
def make(rank):
    return Model().eval()
