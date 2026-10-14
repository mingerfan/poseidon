from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = nn.Upsample(size=5, mode="linear", align_corners=False)

    def forward(self, x):
        return self.resize(x)

def make():
    return Model().eval()
