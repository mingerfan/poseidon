import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[0.3653261780668356, -0.23727478642514038], [-0.33741376467416795, 0.23581564568353103]],dtype=torch.float64))
        self.register_buffer("proj",torch.tensor([[0.07968120057697392, 0.278173354449524], [0.02381753437733547, 0.006473501364626988]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([-0.05817557030710266, -0.035487581417211506],dtype=torch.float64))
    def forward(self,x):
        c=x.new_zeros(size=(1,2))
        return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x
def make():
    return Model().eval()
