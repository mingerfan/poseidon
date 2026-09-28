"""Manual public-coefficient probes, never an Agent answer transformer."""
from fractions import Fraction
import math
from chebyshev_lowering import balanced as basis_evaluate

def coefficients(values):
    values=list(values)
    if not values or len(values)>129:raise ValueError("Coefficient budget")
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in values):
        raise ValueError("Finite public coefficients required")
    return [Fraction(v) for v in values]

def multiply_argument(values,offset=0.):
    """Exact rational x*(sum(c_n*T_n(x))+offset); no discarded term."""
    c=coefficients(values);c[0]+=Fraction(offset)
    out=[Fraction(0)]*(len(c)+1);out[1]+=c[0]
    for n in range(1,len(c)):
        out[n-1]+=c[n]/2;out[n+1]+=c[n]/2
    return out

def parity_series(values):
    """P(x)=E(T2(x))+x*O(T2(x)), exact rational coefficient algebra."""
    c=coefficients(values);even=c[::2];odd=[Fraction(0)]*(len(c)//2)
    for n in range(1,len(c),2):
        k=(n-1)//2
        odd[0]+=((-1)**k)*c[n]
        for j in range(1,k+1):odd[j]+=2*((-1)**(k-j))*c[n]
    return even,odd

def finite_literals(values):
    out=[float(v) for v in values]
    if any(not math.isfinite(v) or abs(v)>1024 for v in out):
        raise ValueError("Existing literal budget")
    if any(v and f==0 for v,f in zip(values,out)):
        raise ValueError("Coefficient conversion underflow")
    return out

def parity_evaluate(x,values,zero):
    even,odd=parity_series([float(v) for v in values])
    square=x*x;y=square+square-1.
    answer=basis_evaluate(y,finite_literals(even),zero)
    if any(odd):answer=answer+x*basis_evaluate(y,finite_literals(odd),zero)
    return answer
