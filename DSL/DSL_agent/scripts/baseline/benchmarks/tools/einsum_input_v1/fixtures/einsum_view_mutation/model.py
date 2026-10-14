import torch
class Model(torch.nn.Module):
    def forward(self,x):
        y=torch.einsum("ij->ji",x)
        saved=y.clone()
        y.add_(x*.25)
        return {"saved":saved,"updated":x}
def make():
    return Model().eval()
