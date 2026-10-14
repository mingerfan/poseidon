import torch
def reorder(w):
    return torch.roll(w,shifts=(5,-3),dims=(0,1))
