import torch
def compute(x,w):
    return torch.cat((x,w),dim=0)
