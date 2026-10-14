import torch
class Model(torch.nn.Module):
    def forward(self,x):
        pooled=x.square().mean(dim=0)
        return {'pooled':pooled,'same':pooled,'head':pooled[:8]}
def make():
    return Model().eval()
