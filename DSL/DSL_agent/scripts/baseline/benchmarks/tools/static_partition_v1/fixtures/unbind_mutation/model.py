import torch
class Model(torch.nn.Module):
    def forward(self,x):
        a,b=x.unbind(0)
        snapshot=a.clone()
        b.copy_(a*2)
        return {"snapshot":snapshot,"updated":x}
def make(): return Model().double().eval()
