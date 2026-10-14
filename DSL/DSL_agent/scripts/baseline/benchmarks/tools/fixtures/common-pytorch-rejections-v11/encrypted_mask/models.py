import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self,q,k,v,mask):
        return F.scaled_dot_product_attention(q,k,v,mask,dropout_p=0.)
def make_model():
    return Model().double().eval()
