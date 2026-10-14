"""Public source-spelling probes; small, static, CPU float64 inference."""
import torch
from torch import nn
from torch.nn import functional as F

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=nn.MultiheadAttention(4,2,dropout=0.,batch_first=True,dtype=torch.float64)
    def forward(self,x):
        return self.attn(x,x,x,need_weights=False)[0]

class FunctionalMaxPool(nn.Module):
    def forward(self,x):
        return F.max_pool2d(x,kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)

class GeluTanh(nn.Module):
    def __init__(self):
        super().__init__();self.activation=nn.GELU(approximate="tanh")
    def forward(self,x):
        return self.activation(x)

class InplaceRelu(nn.Module):
    def __init__(self):
        super().__init__();self.activation=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.activation(x)

class ContiguousView(nn.Module):
    def forward(self,x):
        return x.transpose(1,2).contiguous().view(1,-1)

class PublicDivision(nn.Module):
    def forward(self,x):
        return torch.div(x,2.0)

def make_model(case):
    constructors={"multihead_attention":MultiHead,"functional_max_pool2d":FunctionalMaxPool,
        "gelu_tanh":GeluTanh,"relu_inplace":InplaceRelu,"contiguous_view":ContiguousView,
        "public_division":PublicDivision}
    return constructors[case]().cpu().double().eval()
