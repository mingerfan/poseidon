import torch
def compute(x,scalar):
    return torch.stack((scalar,x.sum(),scalar*.5),dim=-1)
