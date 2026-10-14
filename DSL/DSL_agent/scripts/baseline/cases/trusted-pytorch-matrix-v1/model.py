"""Small fixed matrix-product models for isolated FX capture."""
import torch

class Product(torch.nn.Module):
    def __init__(self, kind, spelling):
        super().__init__()
        self.kind=kind
        self.spelling=spelling
    def forward(self,x,y):
        if self.kind=="dot":
            if self.spelling==0:return torch.dot(x,y).reshape(1)
            if self.spelling==1:return x.dot(y).reshape(1)
            return torch.dot(input=x,tensor=y,out=None).reshape(1)
        if self.kind=="mm":
            if self.spelling==0:return torch.mm(x,y)
            if self.spelling==1:return x.mm(y)
            return torch.mm(input=x,mat2=y,out=None)
        if self.kind=="bmm":
            if self.spelling==0:return torch.bmm(x,y)
            if self.spelling==1:return x.bmm(y)
            return torch.bmm(input=x,mat2=y,out=None)
        if self.spelling==0:return x@y
        if self.spelling==1:return x.matmul(y)
        return torch.matmul(input=x,other=y,out=None)


def make_dot0():
    return Product("dot",0)

def make_dot1():
    return Product("dot",1)

def make_dot2():
    return Product("dot",2)

def make_mm0():
    return Product("mm",0)

def make_mm1():
    return Product("mm",1)

def make_mm2():
    return Product("mm",2)

def make_bmm0():
    return Product("bmm",0)

def make_bmm1():
    return Product("bmm",1)

def make_bmm2():
    return Product("bmm",2)
