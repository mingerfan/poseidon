import torch
class Model(torch.nn.Module):
    def __init__(self,index,axis,method):
        super().__init__()
        self.register_buffer("indices",torch.tensor(index,dtype=torch.int64))
        self.axis=axis
        self.method=method
    def forward(self,x):
        if self.method:
            return x.gather(dim=self.axis,index=self.indices)
        return torch.gather(x,self.axis,self.indices)
def make(index,axis,method):
    return Model(index,axis,method)
