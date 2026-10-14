import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.tensor([[-0.20910694762671067, -0.30643933545127044], [-0.18393458652610306, -0.3572528582331984]],dtype=torch.float64))
        self.register_buffer("mask",torch.tensor([True, False],dtype=torch.bool))
        self.register_buffer("proj",torch.tensor([[-0.22517670813304064, 0.1265182604008162], [-0.15725128262250126, 0.12239321361965083]],dtype=torch.float64))
        self.register_buffer("bias",torch.tensor([-0.05981405711180307, 0.07230489286247557],dtype=torch.float64))
    def forward(self,x):
        snapshot=x.clone()
        c=self.w.float().where(self.mask.reshape(1,2),-.125000017).double()
        return F.linear((x.square()+c)*self.w,self.proj,self.bias)+x
def make():
    return Model().eval()
