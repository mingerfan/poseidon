import torch
class Model(torch.nn.Module):
    def __init__(self,rank):
        super().__init__()
        self.axes=tuple(range(rank-6))+tuple(reversed(range(rank-6,rank)))
        self.reductions=(rank-6,rank-4,rank-2)
        self.register_buffer('w',torch.zeros((1,)*(rank-6)+(2,)*6,dtype=torch.float64))
    def forward(self,x):
        z=x.permute(self.axes)*self.w+x.square()*.125
        pooled=z.mean(dim=self.reductions,keepdim=True)
        return {'features':z,'summary':pooled,'same':z}
def make(rank):
    return Model(rank).eval()
