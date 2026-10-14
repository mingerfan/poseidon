import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indices",torch.tensor([[[1,0]],[[0,1]],[[1,1]]],dtype=torch.int64))
    def forward(self,x):
        viewed=x.transpose(0,1)
        selected=torch.gather(viewed,-2,self.indices)
        return {"selected":selected,"row_mean":selected.mean(dim=-1)}
def make():
    return Model().double().eval()
