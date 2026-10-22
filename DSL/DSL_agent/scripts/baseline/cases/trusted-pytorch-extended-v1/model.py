import torch

class Elementwise4096(torch.nn.Module):
    def forward(self,x):
        return (x*0.5+0.125).square()

class CifarReduction(torch.nn.Module):
    def forward(self,x):
        shifted=x+0.125
        return torch.stack([shifted.sum(),shifted.mean()],dim=0)

class FourTensorFusion(torch.nn.Module):
    def forward(self,x,y,z,t):
        fusion=(x.mean()+y.mean())*(z.mean()-t.mean())
        return x+fusion,y+fusion,z+fusion,t+fusion

def make_elementwise():return Elementwise4096()
def make_cifar_reduction():return CifarReduction()
def make_four_tensor_fusion():return FourTensorFusion()
