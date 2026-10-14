"""Manual-only exact Chebyshev basis conversion and Estrin feasibility probe.

No model/reference change, no coefficient truncation, and no provider rewriting.
Fraction identities precede the explicit float-literal rounding boundary.
"""
from fractions import Fraction
import math

def power_coefficients(coeff):
    coeff=list(coeff)
    if not 1<=len(coeff)<=129:raise ValueError("Polynomial coefficient budget")
    coeff=[Fraction(float(c)) for c in coeff]
    if not coeff:raise ValueError("Empty polynomial")
    result=[Fraction(0) for _ in coeff]
    previous=[Fraction(1)]
    current=[Fraction(0),Fraction(1)]
    for n,c in enumerate(coeff):
        if n==0:term=previous
        elif n==1:term=current
        else:
            term=[Fraction(0)]+[2*v for v in current]
            for i,v in enumerate(previous):term[i]-=v
            previous,current=current,term
        for i,v in enumerate(term):result[i]+=c*v
    return result

def estrin(x,coeff,zero,weight=None):
    values=power_coefficients(coeff)
    # Shared binary normalization respects the existing scalar literal bound.
    shift=0
    while max(abs(v) for v in values)>1024:
        values=[v/2 for v in values];shift+=1
    rounded=[float(v) for v in values]
    if any(not math.isfinite(v) or (exact and not v) for exact,v in zip(values,rounded)):
        raise ValueError("Unrepresentable coefficient")
    # All nonzero terms survive; the only approximation is float literals.
    if weight is not None:
        rounded=[weight*v for v in rounded]
    power=x
    while len(rounded)>1:
        rounded=[rounded[i]+power*rounded[i+1] if i+1<len(rounded) else rounded[i]
                 for i in range(0,len(rounded),2)]
        if len(rounded)>1:power=power*power
    result=rounded[0]
    for _ in range(shift):result=result+result
    return result if hasattr(result,"name") else zero+result
