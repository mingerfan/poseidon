import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([-0.023250468312141614, -0.10401625150646487, -0.2840935404188901, 0.04150236929524098],dtype=torch.float64))
    def forward(self,x):
        w=self.public.flip((0,))
        return x*w
def make():
    return Model().eval()
