"""Trusted developer fixtures for PyTorch spelling and input-boundary audits."""
import torch
from torch.nn import functional as F

class Forms(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind in ("module_linear", "functional_linear", "matrix_linear"):
            self.layer = torch.nn.Linear(2, 2, dtype=torch.float64)
            with torch.no_grad():
                self.layer.weight.copy_(torch.tensor([[.25, -.125], [.375, .5]], dtype=torch.float64))
                self.layer.bias.copy_(torch.tensor([.0625, -.03125], dtype=torch.float64))
        elif kind in ("module_conv", "functional_conv"):
            self.layer = torch.nn.Conv1d(1, 1, 2, dtype=torch.float64)
            with torch.no_grad():
                self.layer.weight.copy_(torch.tensor([[[.25, -.125]]], dtype=torch.float64))
                self.layer.bias.fill_(.0625)
        elif kind in ("module_pool", "functional_pool"):
            self.layer = torch.nn.AvgPool1d(2)
        elif kind in ("module_batch_norm", "functional_batch_norm"):
            self.layer = torch.nn.BatchNorm1d(2).double()
        elif kind == "module_layer_no_affine":
            self.layer = torch.nn.LayerNorm(2, eps=1., elementwise_affine=False)
        elif kind == "module_group_no_affine":
            self.layer = torch.nn.GroupNorm(1, 2, eps=1., affine=False)
        elif kind in ("float32_strict", "float32_convert"):
            self.register_buffer("gain", torch.tensor([.25, .5], dtype=torch.float32))
        self.eval()

    def forward(self, x):
        k = self.kind
        if k == "module_linear": return self.layer(x)
        if k == "functional_linear": return F.linear(x, self.layer.weight, self.layer.bias)
        if k == "matrix_linear": return x @ self.layer.weight.T + self.layer.bias
        if k == "module_conv": return self.layer(x)
        if k == "functional_conv": return F.conv1d(x, self.layer.weight, self.layer.bias)
        if k == "module_pool": return self.layer(x)
        if k == "functional_pool": return F.avg_pool1d(x, 2)
        if k == "module_batch_norm": return self.layer(x)
        if k == "functional_batch_norm":
            return F.batch_norm(x, self.layer.running_mean, self.layer.running_var,
                                self.layer.weight, self.layer.bias, training=False)
        if k == "module_layer_no_affine": return self.layer(x)
        if k == "functional_layer_no_affine": return F.layer_norm(x, (2,), eps=1.)
        if k == "module_group_no_affine": return self.layer(x)
        if k == "functional_group_no_affine": return F.group_norm(x, 1, eps=1.)
        if k == "shape_reshape": return x.reshape(x.size(0), -1)
        if k == "shape_loop":
            y = x
            for _ in range(x.size(1)): y = y + x * .125
            return y
        if k == "public_loop":
            y = x
            for _ in range(2): y = y + x * .125
            return y
        if k == "scalar_output": return x.mean()
        if k == "scalar_reshape": return x.mean().reshape(1)
        if k == "tuple_output": return x + x, x * x
        if k == "dict_output": return {"twice": x + x, "square": x * x}
        if k == "data_branch":
            if x.sum() > 0: return x + x
            return x * x
        if k in ("float32_strict", "float32_convert"): return x * self.gain
        if k == "sample_variance": return torch.var(x, dim=-1, keepdim=True)
        raise ValueError("Unknown audit fixture")

def make(kind): return Forms(kind)
