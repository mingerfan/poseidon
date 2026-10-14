import torch
class Model(torch.nn.Module):
    def forward(self,x):
        a,b,c=torch.chunk(input=x,chunks=3,dim=-1)
        return {"merge":a+b*2,"tail":c}
def make(): return Model().double().eval()
