import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self,x):
        return F.dropout1d(x,training=False)
def make_model():
    return Model().double().eval()
