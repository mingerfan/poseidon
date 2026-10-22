import torch

class RankFourStats(torch.nn.Module):
    def forward(self,x):
        shifted=x+0.125
        return torch.stack([shifted.sum(),shifted.mean(),shifted.sum()-shifted.mean()],dim=-1)

class ScalarFusion(torch.nn.Module):
    def forward(self,x,y):
        a=x.mean()+y.sum()
        return torch.stack(tensors=(a,a.square(),a-0.125),dim=0)

def make_rank_four_stats():return RankFourStats()
def make_scalar_fusion():return ScalarFusion()
