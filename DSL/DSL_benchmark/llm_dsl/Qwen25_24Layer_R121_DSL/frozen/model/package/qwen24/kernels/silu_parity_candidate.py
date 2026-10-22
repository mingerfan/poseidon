"""Parity-aware SiLU with optional exact linear part and pre-square scaling."""
import math
from types import SimpleNamespace
from qwen24.kernels import qwen_nonlinear as nonlinear

_UNIT=nonlinear.chebyshev_unit
_CHEB=nonlinear.chebyshev
STRATEGIES=('none','zero_odd','fold2','fold4','fold8',
    'canonical_fold2','canonical_fold4','canonical_fold8','pre_fold2','pre_fold4','pre_fold8')
CURRENT='none'


def coefficients(profile,canonical):
    if tuple(profile.domain)!=(-2.,2.) or len(profile.coefficients)!=13:
        raise ValueError('SiLU experiment requires the audited symmetric degree12 profile')
    result=list(profile.coefficients)
    omitted=sum(abs(result[i]) for i in range(3,len(result),2))
    if canonical:omitted+=abs(result[1]-1.)
    if omitted>1e-13:raise ValueError('SiLU symmetry/linear coefficient outside audited tolerance')
    for i in range(3,len(result),2):result[i]=0.
    if canonical:result[1]=1.
    return result


def series(values,block):
    return SimpleNamespace(coefficients=values,evaluation='blocked',block_size=block)


def evaluate_unit(z,profile,constant):
    if CURRENT=='none' or profile.function!='silu':return _UNIT(z,profile,constant)
    canonical=CURRENT.startswith(('canonical_','pre_'))
    c=coefficients(profile,canonical)
    if CURRENT=='zero_odd':
        return _UNIT(z,SimpleNamespace(coefficients=c,evaluation=profile.evaluation,block_size=profile.block_size),constant)
    block=int(CURRENT.rsplit('fold',1)[1])
    y=z*z*constant(2)+constant(-1)
    linear=z if canonical else z*constant(c[1])
    return _UNIT(y,series(c[::2],block),constant)+linear


def evaluate(x,profile,constant):
    if not CURRENT.startswith('pre_fold') or profile.function!='silu':return _CHEB(x,profile,constant)
    c=coefficients(profile,True);block=int(CURRENT.rsplit('fold',1)[1])
    # T2(x/2)=x^2/2-1. Scale BEFORE squaring so both operands stay <=sqrt(2)
    # in the stated domain, with no post-square plaintext factor2/rescale.
    scaled=x*constant(1/math.sqrt(2))
    y=scaled*scaled+constant(-1)
    return _UNIT(y,series(c[::2],block),constant)+x*constant(.5)


def install(strategy):
    global CURRENT
    if strategy not in STRATEGIES:raise ValueError('Unknown SiLU parity strategy')
    CURRENT=strategy;nonlinear.chebyshev_unit=evaluate_unit;nonlinear.chebyshev=evaluate
