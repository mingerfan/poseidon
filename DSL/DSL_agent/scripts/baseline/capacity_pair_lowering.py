"""Shape-proven two-element centering for trusted capacity baselines only.

A sparse amplitude encodes [a, sign*a]. Materialize the second lane only when
an operation needs it. Other shapes and unproved expressions keep the ordinary
capacity lowering. No request, constants, reference or Provider source is changed.
"""
import math
import numpy as np
from benchmark_graph import require
from unified_graph_lowering import Scalar
from unified_capacity_lowering import Chunks, all_slot_sum, public_vector, vector_arithmetic

VERSION = "capacity-centered-pair-v1"

class Pair(Chunks):
    def __init__(self, em, amplitude, shape, sign=-1):
        require(math.prod(shape)==2 and sign in (-1,1), "Capacity pair shape")
        self.em=em;self.amplitude=amplitude;self.shape=tuple(shape)
        self.sign=sign;self.replicated=False;self._parts=None
        self.factor_base=amplitude;self.replicated_factors=()
    @property
    def parts(self):
        if self._parts is None:
            second=self.em.rotate(self.amplitude,-1)
            self._parts=(self.amplitude+second if self.sign==1 else self.amplitude-second,)
        return self._parts

def centered_pair(em,node,x,producers,shapes):
    if (node["op"]!="subtract" or not isinstance(x,Chunks)
            or math.prod(x.shape)!=2 or len(x.parts)!=1):
        return None
    mean=producers.get(node["inputs"][1])
    if (mean is None or mean["op"]!="mean" or mean["inputs"]!=[node["inputs"][0]]
            or math.prod(shapes[mean["outputs"][0]])!=1):
        return None
    axes={a%len(x.shape) for a in mean["attrs"]["axes"]}
    if math.prod(x.shape[a] for a in axes)!=2:
        return None
    mask=em.request["public_constants"].get("mask0")
    if mask!=[1.]+[0.]*(em.period-1):return None
    half=None
    if .5 not in em.constants:
        half=next((name for name,value in em.request["public_constants"].items()
                   if type(value) is list and len(value)==em.period and value[0]==.5),None)
        if half is None:return None
    delta=x.parts[0]-em.rotate(x.parts[0],1)
    scaled=delta*.5 if half is None else em.emit(delta.name+" * "+half)
    amplitude=em.emit(scaled.name+" * mask0")
    return Pair(em,amplitude,x.shape)

def arithmetic(em,op,xs,shape,*,node=None,packed=None):
    if math.prod(shape)!=2 or not any(isinstance(x,Pair) for x in xs):return None
    left,right=xs
    if isinstance(left,Pair) and isinstance(right,Pair):
        if op=="multiply":return Pair(em,left.amplitude*right.amplitude,shape,left.sign*right.sign)
        if left.sign==right.sign:
            a=left.amplitude+right.amplitude if op=="add" else left.amplitude-right.amplitude
            return Pair(em,a,shape,left.sign)
        return None
    index=0 if isinstance(left,Pair) else 1
    pair=xs[index];other=xs[1-index]
    if isinstance(other,np.ndarray) and other.size and np.all(other==other.reshape(-1)[0]):
        value=float(other.reshape(-1)[0])
        if op=="multiply":
            result=Pair(em,pair.amplitude*value,shape,pair.sign)
            result.factor_base=pair.factor_base
            result.replicated_factors=pair.replicated_factors+(value,)
            return result
        if value==0:
            return Pair(em,-pair.amplitude if op=="subtract" and index==1 else pair.amplitude,shape,pair.sign)
    if op=="multiply" and isinstance(other,Chunks) and math.prod(other.shape)==1:
        factor=other.broadcast_scalar()
        result=Pair(em,pair.amplitude*factor,shape,pair.sign)
        result.factor_base=pair.factor_base
        result.replicated_factors=pair.replicated_factors+(factor,)
        return result
    if (op=="multiply" and isinstance(other,np.ndarray) and other.size
            and pair.replicated_factors and node is not None and packed is not None):
        # The same bound public vector multiplies the two sparse lanes before
        # their common replicated factors. No plaintext rotation or new weights.
        coefficient=public_vector(em,packed,node["id"],"elementwise",operand=1-index,chunk=0)
        first=vector_arithmetic(em,"multiply",pair.factor_base,coefficient)
        second=vector_arithmetic(em,"multiply",em.rotate(pair.factor_base,-1),coefficient)
        value=first+second if pair.sign==1 else first-second
        for factor in pair.replicated_factors:value=value*factor
        return Chunks(em,[value],shape)
    return None

def unary(em,op,x,attrs,shape):
    if not isinstance(x,Pair):return None
    if op in ("reshape","flatten"):return Pair(em,x.amplitude,shape,x.sign)
    if op=="negate":return Pair(em,-x.amplitude,shape,x.sign)
    exponent=2 if op=="square" else attrs.get("exponent") if op=="power" else None
    if exponent in (2,4):return Pair(em,x.amplitude**exponent,shape,1)
    return None

def reduction(em,op,x,attrs,shape):
    if not isinstance(x,Pair) or math.prod(shape)!=1:return None
    axes={a%len(x.shape) for a in attrs["axes"]}
    if math.prod(x.shape[a] for a in axes)!=2:return None
    if x.sign==-1:return Chunks(em,[em.zero],shape,replicated=True)
    if op=="sum" and 2. not in em.constants:return None
    value=all_slot_sum(em,x.amplitude)
    if op=="sum":value=value*2.
    return Chunks(em,[value],shape,replicated=True)
