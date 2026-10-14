import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([0.24799775618668435, -0.35678594846066836, -0.0061172491341541235, -0.034611345606449684],dtype=torch.float64))
    def forward(self,x):
        c=x.new_zeros((1,))
        return x*(c+.125)+self.w
def make():
    return Model().eval()
