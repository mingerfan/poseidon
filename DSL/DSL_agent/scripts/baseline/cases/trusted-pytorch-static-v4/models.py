"""Small trusted sources for declared-shape controls and rank-zero outputs."""
import torch

class Loop(torch.nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
    def forward(self,x,y=None):
        if self.kind==0:
            result=x
            for _ in range(x.shape[-1]):result=result+x*.125
            return result
        if self.kind==1:
            result=x.reshape(1,x.numel())
            for _ in range(result.size(1)):result=result+.125
            return result.reshape(x.shape)
        result=x
        if x.dim()==2 and len(x)==2:
            for n in x.shape:
                for _ in range(n):result=result+y*.125
        return result

class Scalar(torch.nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
    def forward(self,x,y=None):
        if self.kind==0:return torch.sum(x)
        if self.kind==1:return (x*x).mean()
        return {"score":torch.dot(x,y),"features":(x+y,[x.reshape(1,3)])}

# Separate explicit signatures avoid optional FX tensor placeholders.
class One(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,x):return self.model(x)
def make_loop(kind):return One(Loop(kind)) if kind<2 else Loop(kind)
def make_scalar(kind):return One(Scalar(kind)) if kind<2 else Scalar(kind)
