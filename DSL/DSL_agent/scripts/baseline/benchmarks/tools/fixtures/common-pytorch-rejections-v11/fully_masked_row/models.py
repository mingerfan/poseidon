import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mask',torch.empty([2, 2],dtype=torch.bool))
    def forward(self,q,k,v):
        return F.scaled_dot_product_attention(q,k,v,self.mask,dropout_p=0.)
def make_model():
    return Model().double().eval()
