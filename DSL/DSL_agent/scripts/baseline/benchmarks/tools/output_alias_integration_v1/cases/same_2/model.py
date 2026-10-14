import torch
class Model(torch.nn.Module):
    def forward(self,x):
        y=x.square()*.5+.125
        return {'first':y,'second':y,'third':y,'fourth':y}
def make():
    return Model().eval()
