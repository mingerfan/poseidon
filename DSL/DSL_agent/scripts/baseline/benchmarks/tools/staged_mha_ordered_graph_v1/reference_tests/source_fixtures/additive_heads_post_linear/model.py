"""Static eval native MHA composition; weights loaded by the trusted tool."""
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(4, 2, batch_first=True, kdim=None, vdim=None, add_bias_kv=False, add_zero_attn=False, dtype=torch.float64)
        self.post = nn.Linear(4, 2, dtype=torch.float64)
        self.register_buffer('attn_mask', torch.tensor([[[-0.04, -0.0]], [[-0.08, -0.01]], [[-0.12, -0.02]], [[-0.16, -0.03]]], dtype=torch.float64))
        self.register_buffer('padding_mask', torch.tensor([[0.0, -0.025], [0.0, -0.05]], dtype=torch.float64))

    def forward(self, q, k, v):
        projected = q
        output, weights = self.attention(projected, k, v, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask, key_padding_mask=self.padding_mask)
        output = self.post(output)
        return {"result": output, "weights": weights}

def make():
    return Model().eval()
