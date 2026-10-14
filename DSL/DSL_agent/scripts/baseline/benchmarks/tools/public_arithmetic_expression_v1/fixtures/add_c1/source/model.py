"""Small independent reference fixtures for closed public arithmetic."""
import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self,operation,context):
        super().__init__();self.operation=operation;self.context=context
        def state(name,values,dtype=torch.float64):
            self.register_buffer(name,torch.tensor(values,dtype=dtype))
        if operation=="add"and context==0:
            state("std",[.5,.75,1.25]);state("mean",[.1,-.2,.05])
        elif operation=="add":
            state("a",[[.3,.7,1.1],[-.2,.6,.4]]);state("b",[[.2,-.1,.3],[.6,-.3,.1]])
        elif operation=="subtract"and context==0:
            state("mask",[True,False,True],torch.bool)
        elif operation=="subtract"and context==1:
            state("a",[[.7],[1.1]]);state("b",[[.1,.3,-.2]])
        elif operation=="subtract":
            state("a",[[.3,.7,1.1],[-.2,.6,.4]]);state("b",[[.2,-.1,.3],[.6,-.3,.1]])
        elif operation=="multiply"and context==0:
            state("a",[.3,.7,1.1]);state("b",[.6,.9,1.2])
        elif operation=="multiply"and context==1:
            state("a",[.3,.7,1.1])
        elif operation=="multiply":
            state("a",[[.3,.7,1.1],[-.2,.6,.4]]);state("b",[[.2,-.1,.3],[.6,-.3,.1]])
        elif operation=="divide":
            state("std",[.5,.75,1.25])
            if context:state("a",[.6,.9,1.2])
        elif operation=="negate"and context<2:
            state("a",[.3,.7,1.1])
        elif operation=="negate":
            state("a",[[.3,.7],[1.1,-.2],[.6,.4]])
        else:raise ValueError("fixture operation")
    def forward(self,x):
        if self.operation=="add":
            if self.context==0:return (x-self.mean)/(self.std+.01)
            if self.context==1:return x*(self.a+self.b)
            return x+torch.add(input=self.a,other=self.b,alpha=.25)
        if self.operation=="subtract":
            if self.context==0:return x*(1.0-self.mask.double())
            if self.context==1:return x*(self.a-self.b)
            return x*(self.a.sub(self.b,alpha=.5)).transpose(0,1)
        if self.operation=="multiply":
            if self.context==0:return x*(self.a.mul(self.b))
            if self.context==1:return x*(.5*self.a).unsqueeze(0)
            return F.linear(x,torch.multiply(self.a,self.b))
        if self.operation=="divide":
            if self.context==0:return x*(1.0/self.std)
            if self.context==1:return x/(self.std/self.a)
            return x*self.a.div(self.std+.01,rounding_mode=None)
        if self.operation=="negate":
            if self.context==0:return x*(-self.a)
            if self.context==1:return x+self.a.neg().reshape(1,3)
            return F.linear(x,torch.negative(input=self.a).transpose(0,1))
        raise ValueError("fixture operation")
def build(operation,context):
    return Model(operation,context).eval()
def input_shape(operation,context):
    if operation in("multiply","negate")and context==2:return [1,3]
    if operation=="subtract"and context==2:return [3,2]
    return [2,3]
