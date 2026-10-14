import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[-0.477907612498519, -0.3894306923704438], [0.10857869942017595, 0.32532837549323745]],dtype=torch.float32))
    def forward(self,x):
        w=self.public.float().repeat_interleave(2,dim=-1).double()
        return x*w+x
def make():
    return Model().eval()
