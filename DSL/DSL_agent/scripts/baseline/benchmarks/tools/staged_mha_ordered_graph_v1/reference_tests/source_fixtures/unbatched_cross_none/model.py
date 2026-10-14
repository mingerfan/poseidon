"""Static eval native MHA composition; weights loaded by the trusted tool."""
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(4, 2, batch_first=True, kdim=None, vdim=None, add_bias_kv=False, add_zero_attn=False, dtype=torch.float64)
        self.register_buffer('attn_mask', None)
        self.register_buffer('padding_mask', None)

    def forward(self, q, k, v):
        projected = q
        output, weights = self.attention(projected, k, v, need_weights=False, average_attn_weights=True, attn_mask=self.attn_mask, key_padding_mask=self.padding_mask)
        return (output, None)

def make():
    return Model().eval()
