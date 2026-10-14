import torch
class Model(torch.nn.Module):
    def forward(self,x):
        y=x.square()*.25+x
        return {'main':y,'original':x,'again':y,'original_again':x}
def make():
    return Model().eval()
