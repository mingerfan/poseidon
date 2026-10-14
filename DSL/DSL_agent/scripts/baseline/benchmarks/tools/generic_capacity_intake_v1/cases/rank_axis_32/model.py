import torch
class Model(torch.nn.Module):
    def forward(self,x,y):
        z=(x+y).transpose(-1,-2)
        pooled=z.square().mean(dim=(-1,-3),keepdim=True)
        return {'pooled':pooled,'same':pooled,'flat':z.flatten()}
def make(rank):
    return Model().eval()
