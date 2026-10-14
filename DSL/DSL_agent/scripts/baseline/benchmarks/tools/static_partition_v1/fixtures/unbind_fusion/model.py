import torch
class Model(torch.nn.Module):
    def forward(self,x):
        a,b=x.unbind(0)
        return a+b*2
def make(): return Model().double().eval()
