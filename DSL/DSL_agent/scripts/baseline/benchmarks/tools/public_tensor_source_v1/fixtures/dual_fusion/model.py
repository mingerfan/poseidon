import torch
class Model(torch.nn.Module):
    def __init__(self, mode, std_shape):
        super().__init__()
        self.mode = mode
        self.register_buffer('std', torch.zeros(std_shape, dtype=torch.float64))
        if mode == 'dense_projection':
            self.register_buffer('weight', torch.zeros((3, 2), dtype=torch.float64))
    def forward(self, x, y=None):
        if self.mode == 'vector_function':
            return torch.div(input=x, other=self.std, rounding_mode=None, out=None)
        if self.mode == 'channel_method':
            return x.divide(self.std)
        if self.mode == 'inplace_snapshot':
            saved = x.clone()
            x.div_(self.std)
            return {'snapshot': saved, 'out': x}
        if self.mode == 'mixed_scalar_power':
            return torch.pow(x / self.std, 2) / 2
        if self.mode == 'dual_fusion':
            return torch.div(x, self.std) + y.true_divide(self.std)
        if self.mode == 'dense_projection':
            return torch.mm(x / self.std, self.weight)
        return x / self.std
def make(mode, std_shape):
    return Model(mode, std_shape).eval()
