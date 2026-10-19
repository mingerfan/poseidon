"""Unvalidated segmented RMS prototype for concurrent tokens in one layer.

One-token decode uses the existing kernel; no future token is assumed known.
This file is not installed in the running full model or short CKKS test.
"""
import math
import numpy as np
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import ops,qwen_nonlinear as nonlinear,rms_power_polynomial as polynomial
from qwen24.kernels.expr import Plain
from qwen24.kernels import zero_tail_contract as tail


STRATEGIES=('replica16','tiled16','tiled32','tiled64','tiled128')
STRATEGY='tiled16'


def rms_norm_batch(vectors,weight,*,rsqrt,eps=1e-6,strategy=None):
    strategy=STRATEGY if strategy is None else strategy
    if strategy not in STRATEGIES:raise ValueError('Unknown RMS strategy')
    vectors=tuple(vectors)
    if not vectors:raise ValueError('At least one token required')
    size,slots=vectors[0].size,vectors[0].slots
    if any((v.size,v.slots)!=(size,slots) for v in vectors):
        raise ValueError('All concurrent tokens require identical shapes')
    weight=ops._real_array(weight,'RMSNorm weight')
    if weight.shape!=(size,) or not math.isfinite(eps) or eps<=0:
        raise ValueError('RMSNorm requires matching gamma and positive finite eps')
    if getattr(rsqrt,'function',None)!='rsqrt':raise ValueError('Expected rsqrt profile')
    if len(vectors)==1:return (ql.rms_norm(vectors[0],weight,rsqrt=rsqrt,eps=eps),)
    stride=1<<(size-1).bit_length()
    capacity=slots//stride
    if capacity<2:
        return tuple(ql.rms_norm(v,weight,rsqrt=rsqrt,eps=eps) for v in vectors)
    results=[]
    for start in range(0,len(vectors),capacity):
        group=vectors[start:start+capacity]
        if len(group)==1:
            results.append(ql.rms_norm(group[0],weight,rsqrt=rsqrt,eps=eps));continue
        packed=None
        for i,vector in enumerate(group):
            clean=ql.extract(vector,0,size).expr
            term=clean.rotate(-i*stride) if i else clean
            packed=term if packed is None else ops.add(packed,term)
        scaled=ops.mul(packed,Plain(np.full(slots,1/math.sqrt(size))))
        reduced=ops.mul(scaled,scaled)
        step=1
        while step<stride:
            reduced=ops.add(reduced,reduced.rotate(step));step*=2
        # Only group-start lanes contain the desired within-group sum.
        # Every other lane must be masked before broadcasting; summing
        # cyclic windows without this mask would leak adjacent groups.
        mean_mask=np.zeros(slots)
        mean_mask[np.arange(len(group))*stride]=1.0
        denominator=ops.mul(reduced,Plain(mean_mask))
        if strategy not in STRATEGIES:raise ValueError('Unknown evaluation strategy')
        sparse=True
        copies=min(stride,int(strategy[7:] if strategy.startswith('replica') else strategy[5:]))
        if copies>1:
            step=1
            while step<copies:
                denominator=ops.add(denominator,denominator.rotate(-step));step*=2
        if not sparse:
            step=1
            while step<stride:
                denominator=ops.add(denominator,denominator.rotate(-step));step*=2
        # Inactive lanes must use a legal inverse-sqrt input, never eps alone.
        if not rsqrt.domain[0]<=1<=rsqrt.domain[1]:
            raise ValueError('Inactive lanes require denominator 1 in the profile domain')
        offset=np.ones(slots)
        if sparse:
            for j in range(copies):offset[np.arange(len(group))*stride+j]=eps
        else:offset[:len(group)*stride]=eps
        denominator=ops.add(denominator,Plain(offset))
        constant=lambda value:Plain(np.full(slots,value))
        if isinstance(rsqrt,nonlinear.ChebyshevApproximation):
            inverse=polynomial.evaluate(denominator,rsqrt,constant,6)
        elif isinstance(rsqrt,ql.InverseSqrtApproximation):
            inverse=nonlinear.inverse_sqrt(denominator,rsqrt,constant)
        else:
            inverse=ql.approximate(ql.PackedVector(denominator,slots,slots),rsqrt,function='rsqrt').expr
        if sparse:
            # Only one inverse per token is mathematically needed. Mask all
            # other lanes before broadcasting, including padding and CKKS noise.
            if copies>1:
                average_mask=np.zeros(slots)
                for j in range(copies):average_mask[np.arange(len(group))*stride+j]=1/copies
                inverse=ops.mul(inverse,Plain(average_mask))
                tiled=strategy.startswith('tiled') and stride-size>=copies-1
                if tiled:
                    # Repeat the R inverse samples across this token's stride.
                    # A forward R-window remains inside the token for every
                    # valid lane iff size+R-1<=stride. Padding absorbs seams.
                    step=copies
                    while step<stride:
                        inverse=ops.add(inverse,inverse.rotate(-step));step*=2
                step=1
                while step<copies:
                    inverse=ops.add(inverse,inverse.rotate(step));step*=2
            else:tiled=False
            if not tiled:
                inverse=ops.mul(inverse,Plain(mean_mask))
                step=1
                while step<stride:
                    inverse=ops.add(inverse,inverse.rotate(-step));step*=2
        normalized=ops.mul(packed,inverse)
        for i in range(len(group)):
            shifted=normalized.rotate(i*stride) if i else normalized
            value=ql.PackedVector(ops.mul(shifted,Plain(weight)),size,slots)
            results.append(tail._proven_prefix(value))
    return tuple(results)
