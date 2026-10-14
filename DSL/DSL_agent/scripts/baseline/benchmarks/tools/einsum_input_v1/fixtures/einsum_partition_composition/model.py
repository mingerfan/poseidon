import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight",torch.zeros(2,4,dtype=torch.float64))
    def forward(self,x):
        y=torch.einsum("bi,ij->bj",x,self.weight)
        a,b=y.chunk(2,-1)
        s,t=a.unbind(0)
        return {"sum":s+t,"right":b}
def make():
    return Model().eval()
