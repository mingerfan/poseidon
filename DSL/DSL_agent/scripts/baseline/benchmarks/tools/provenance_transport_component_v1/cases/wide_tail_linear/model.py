import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(1025,3,dtype=torch.float64)
    def forward(self,x,y):
        z=self.fc(x+y)
        pooled=z.mean(dim=-1,keepdim=True)
        return {'projection':z,'summary':pooled,'same':z}
def make():
    return Model().eval()
