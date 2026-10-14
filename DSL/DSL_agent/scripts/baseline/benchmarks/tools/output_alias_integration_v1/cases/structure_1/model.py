import torch
class Model(torch.nn.Module):
    def forward(self,x):
        y=x.square().sum()
        return {'scalar':y,'nested':[y,None],'tensor':(x,x)}
def make():
    return Model().eval()
