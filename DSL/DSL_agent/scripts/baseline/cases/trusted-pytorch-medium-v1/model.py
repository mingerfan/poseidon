import torch

class ReductionFusion(torch.nn.Module):
    def forward(self,x,y):
        joined=torch.cat([x,y],dim=0)
        total=torch.sum(joined,dim=-1,keepdim=True)
        average=joined.mean(dim=0)
        return torch.stack([total,total*0.5],dim=-1),average

class SpatialResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.Conv2d(2,2,3,padding=1,groups=2)
        self.pool=torch.nn.AvgPool2d(2)
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(18,dtype=torch.float64).reshape(2,1,3,3)*0.001-0.007)
            self.conv.bias.copy_(torch.tensor([0.01,-0.02],dtype=torch.float64))
    def forward(self,x):
        branch=self.conv(x)
        pooled=self.pool(branch.square()+x)
        average=pooled.mean(dim=(-2,-1))
        joined=torch.concat([average,average*0.5],dim=-1)
        totals=pooled.sum(dim=-1)
        return pooled,joined,torch.stack((totals,totals*0.25),dim=-1)

class LinearFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(65,4)
        with torch.no_grad():
            self.linear.weight.copy_((torch.arange(260,dtype=torch.float64).reshape(4,65)%17-8)*0.002)
            self.linear.bias.copy_(torch.tensor([0.01,-0.02,0.03,-0.04],dtype=torch.float64))
    def forward(self,x,y):
        left=self.linear(x)
        right=self.linear(y)
        stacked=torch.stack([left,right],dim=-1)
        total=torch.sum(stacked,dim=-1)
        average=torch.mean(total,dim=1,keepdim=True)
        return torch.concatenate([left,right],dim=1),stacked,average

def make_reduction_fusion():return ReductionFusion()
def make_spatial_residual():return SpatialResidual()
def make_linear_fusion():return LinearFusion()
