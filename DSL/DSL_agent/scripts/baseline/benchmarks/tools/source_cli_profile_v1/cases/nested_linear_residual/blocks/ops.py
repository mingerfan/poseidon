import torch
from torch.nn import functional as F
def compute(x,w,proj,bias,*,z):
    c=torch.stack((x,w.flip(0)),dim=-1).flatten(1)
    return F.linear(c,proj,bias)+z
