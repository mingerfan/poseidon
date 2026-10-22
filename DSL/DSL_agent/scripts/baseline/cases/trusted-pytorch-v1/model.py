import torch
class SmallMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first=torch.nn.Linear(4,2)
        self.last=torch.nn.Linear(2,2)
        with torch.no_grad():
            self.first.weight.copy_(torch.tensor([[.1,.2,.3,.4],[-.2,.1,.2,.1]]))
            self.first.bias.copy_(torch.tensor([.01,-.02]))
            self.last.weight.copy_(torch.tensor([[.4,.2],[-.1,.3]]))
            self.last.bias.copy_(torch.tensor([.02,.03]))
    def forward(self,x):
        h=self.first(x)
        return self.last(h*h)
def make_model():
    return SmallMLP()
