"""Fixed-statistics inference BatchNorm; channel axis 1 (PyTorch N,C,...).

Public preprocessing only. No training statistics, approximation, or encrypted
square root/division. The independent reference does not call this lowering.
"""
import math
from seal_artifact_gate import require


def coefficients(shape, mean, variance, weight, bias, eps, *, max_elements=8):
    require(type(max_elements) is int and max_elements in (8,256),'Invalid BatchNorm resource profile')
    require(type(shape) in (tuple,list) and 2 <= len(shape) <= 5 and
            all(type(n) is int and n > 0 for n in shape) and math.prod(shape) <= max_elements,
            'BatchNorm requires a bounded N,C,... tensor')
    channels=shape[1]
    require(type(eps) in (int,float) and math.isfinite(eps) and 0 <= eps <= 1,
            'BatchNorm epsilon must be finite and nonnegative')
    def vector(value,label,default=None):
        if value is None and default is not None:return [default]*channels
        require(type(value) is list and len(value)==channels and
                all(type(x) in (int,float) and math.isfinite(x) and abs(x)<=1024 for x in value),
                'BatchNorm '+label+' must be public per-channel finite values')
        return value
    mean=vector(mean,'running_mean')
    variance=vector(variance,'running_var')
    weight=vector(weight,'weight',1.)
    bias=vector(bias,'bias',0.)
    require(all(v>=0 and v+eps>0 for v in variance),'BatchNorm variance/epsilon invalid')
    gain=[g/math.sqrt(v+eps) for g,v in zip(weight,variance)]
    offset=[b-g*m for b,g,m in zip(bias,gain,mean)]
    require(all(math.isfinite(x) and abs(x)<=1024 for x in gain+offset),
            'BatchNorm folded constants exceed public range')
    return gain,offset


def expand_channels(shape, values):
    block=math.prod(shape[2:])
    return [values[(i//block)%shape[1]] for i in range(math.prod(shape))]
