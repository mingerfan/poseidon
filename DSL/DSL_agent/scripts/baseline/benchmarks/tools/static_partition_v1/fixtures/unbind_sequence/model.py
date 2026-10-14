import torch
class Model(torch.nn.Module):
    def forward(self,x):
        a,b=torch.unbind(input=x,dim=-2)
        return (a.transpose(0,1),b+0.5)
def make(): return Model().double().eval()
