import torch
class Model(torch.nn.Module):
    def forward(self,x,y):
        z=torch.cat((x,y),dim=-1)
        pooled=z.mean(dim=-1)
        total=pooled.sum().reshape(1)
        return {'rows':pooled,'summary':total,'same':pooled}
def make():
    return Model().eval()
