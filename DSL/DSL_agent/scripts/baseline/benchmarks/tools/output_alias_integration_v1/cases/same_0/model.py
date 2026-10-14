import torch
class Model(torch.nn.Module):
    def forward(self,x):
        y=x.square()*.5+.125
        return {'first':y,'second':y}
def make():
    return Model().eval()
