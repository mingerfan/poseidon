import torch.nn.functional as F
from helper import activation
def forward(x):
    h=F.linear(x, w1, b1)
    h=activation(h)
    return F.linear(h, w2, b2)
