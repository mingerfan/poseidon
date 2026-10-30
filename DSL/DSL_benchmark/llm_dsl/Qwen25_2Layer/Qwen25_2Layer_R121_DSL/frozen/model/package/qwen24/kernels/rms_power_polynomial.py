"""Alternate evaluation of the SAME low-degree rsqrt polynomial.

Conversion is restricted to the normalized [-1,1] variable and bounded
coefficients; it is not a general replacement for high-degree Chebyshev.
"""
import numpy as np


def coefficients(profile):
    if profile.function != 'rsqrt' or len(profile.coefficients)>17:
        raise ValueError('Only degree <=16 inverse-sqrt profiles supported')
    values=np.polynomial.chebyshev.cheb2poly(profile.coefficients)
    if not np.isfinite(values).all() or np.max(np.abs(values))>4:
        raise ValueError('Power coefficients exceed bounded candidate contract')
    return values


def evaluate(x,profile,constant,block_size):
    values=coefficients(profile)
    lo,hi=profile.domain
    z=x*constant(2/(hi-lo))+constant(-(hi+lo)/(hi-lo))
    powers={1:z}

    def power(n):
        if n not in powers:
            powers[n]=power(n//2)*power(n-n//2)
        return powers[n]

    def block(start):
        end=min(start+block_size,len(values))
        result=constant(float(values[start]))
        for j in range(1,end-start):
            coefficient=float(values[start+j])
            if coefficient:
                result=result+power(j)*constant(coefficient)
        return result

    result=block(0)
    for start in range(block_size,len(values),block_size):
        result=result+power(start)*block(start)
    powers.clear()
    return result
