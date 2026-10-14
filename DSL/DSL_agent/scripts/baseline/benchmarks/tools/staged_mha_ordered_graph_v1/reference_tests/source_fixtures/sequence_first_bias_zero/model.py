"""Static eval native MHA composition; weights loaded by the trusted tool."""
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(2, 1, batch_first=False, kdim=None, vdim=None, add_bias_kv=True, add_zero_attn=True, dtype=torch.float64)
        self.post = nn.Linear(2, 2, dtype=torch.float64)
        self.register_buffer('attn_mask', None)
        self.register_buffer('padding_mask', None)

    def forward(self, q, k, v):
        projected = q
        output, weights = self.attention(projected, k, v, need_weights=True, average_attn_weights=True, attn_mask=self.attn_mask, key_padding_mask=self.padding_mask)
        output = self.post(output)
        output = output + projected
        return (output, [weights, None])

def make():
    return Model().eval()
