import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.11808492136525572, 0.19622026056634906], [-0.3489017463992625, -0.15869078757584826]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[0.28539070852487375, 0.10921151314324279], [0.06036529098539467, -0.012851591815547903]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([0.062038083164426105, 0.01614731050282142],dtype=torch.float64))
    def forward(self,x):
        c=x.new_ones(x.shape,dtype=torch.int64).double()
        return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x
def make():
    return Model().eval()
