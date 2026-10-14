"""Trusted finite-ring embedding for actual upstream helpers.

R_P(v) repeats a P-vector into physical slots. R_P commutes with componentwise
arithmetic and cyclic rotations modulo P. Constants are exactly scalar or P
values; candidate code cannot instantiate this proxy or choose closure fields.
"""
import math
from benchmark_graph import require

class PeriodicExpr:
    __array_priority__=10000
    def __init__(self,value,period,record,limit=1024):
        self.value=value;self.period=period;self.record=record;self.limit=limit
    def spend(self,operation):
        self.record["operations"].append(operation)
        require(len(self.record["operations"])<=self.limit,"Upstream periodic operation budget")
    def other(self,value):
        if isinstance(value,PeriodicExpr):
            require(value.period==self.period and value.record is self.record,"Mixed periodic helper context")
            return value.value
        # Actual helpers produce public torch tensors. Preserve every scalar/element.
        if hasattr(value,"detach"):
            value=value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value,"tolist"):value=value.reshape(-1).tolist()
        if type(value) is list:
            require(len(value) in (1,self.period) and all(type(v) in (int,float) and math.isfinite(v)
                    and abs(v)<=1024 for v in value),"Upstream finite-ring constant shape/value")
            return value
        require(type(value) in (int,float) and math.isfinite(value) and abs(value)<=1024,"Upstream scalar")
        return value
    def binary(self,other,operation,reverse=False):
        value=self.other(other);self.spend(operation)
        left,right=(value,self.value) if reverse else (self.value,value)
        if operation=="add":result=left+right
        elif operation=="multiply":result=left*right
        else:result=left-right
        return type(self)(result,self.period,self.record,self.limit)
    def __add__(self,x):return self.binary(x,"add")
    def __radd__(self,x):return self.binary(x,"add",True)
    def __sub__(self,x):return self.binary(x,"subtract")
    def __rsub__(self,x):return self.binary(x,"subtract",True)
    def __mul__(self,x):return self.binary(x,"multiply")
    def __rmul__(self,x):return self.binary(x,"multiply",True)
    def rotate(self,step):
        require(type(step) is int,"Upstream rotation must be integral")
        normalized=step%self.period
        steps=[1<<j for j in range(self.period.bit_length()-1) if normalized&(1<<j)]
        self.record["rotations"].append(dict(requested=step,normalized=normalized,steps=steps))
        result=self.value
        for part in steps:self.spend("rotate");result=result.rotate(part)
        return type(self)(result,self.period,self.record,self.limit)

def proxy_type(expr_class):
    """A private subclass lets upstream Empty.resolveType retain the adapter.

    No existing Expr, Empty, rotation or helper function is monkeypatched.
    Hecate's metaclass installs arithmetic methods on new subclasses, so restore
    this adapter's own methods on the newly created type only.
    """
    cls=type("TrustedPeriodicExpr",(PeriodicExpr,expr_class),{})
    for name,value in PeriodicExpr.__dict__.items():
        if name not in ("__dict__","__weakref__","__module__","__doc__"):
            setattr(cls,name,value)
    return cls
