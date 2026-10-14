import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[-0.1762781619379249], [-0.3979493417685598]],dtype=torch.float64))
    def forward(self,x):
        w=self.public.repeat(1,2)
        return x.square()*w+x
def make():
    return Model().eval()
