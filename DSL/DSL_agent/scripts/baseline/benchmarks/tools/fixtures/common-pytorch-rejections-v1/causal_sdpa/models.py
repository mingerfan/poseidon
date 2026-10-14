import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self,q,k,v):
        return F.scaled_dot_product_attention(q,k,v,is_causal=True)
def make_model():
    return Model().double().eval()
