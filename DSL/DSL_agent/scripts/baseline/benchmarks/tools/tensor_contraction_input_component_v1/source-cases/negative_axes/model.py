import torch
class Model(torch.nn.Module):
    def __init__(self,dims,tensor_dimensions,mean_left):
        super().__init__()
        if tensor_dimensions:
            self.register_buffer("dimensions",torch.tensor(dims,dtype=torch.int64))
        else:
            self.dimensions=dims
        self.mean_left=mean_left
    def forward(self,left,right):
        if self.mean_left:
            left=left.mean()
        return torch.tensordot(left,right,dims=self.dimensions)
def make(dims,tensor_dimensions,mean_left):
    return Model(dims,tensor_dimensions,mean_left).double().eval()
