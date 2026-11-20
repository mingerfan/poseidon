"""Independent exact square-normalize oracle for explicit k<=11 variant."""


def fixed_shift_squaring(values, keep, *, shift, steps):
    import math
    import numpy as np
    x,mask = np.asarray(values),np.asarray(keep)
    if (x.dtype != np.float64 or x.ndim < 1 or not np.isfinite(x).all()
            or mask.dtype != np.bool_ or type(steps) is not int or not 0<=steps<=11
            or not math.isfinite(shift)):
        raise ValueError('The explicit unit-width oracle requires float64, a boolean mask and k0..11')
    mask = np.broadcast_to(mask,x.shape)
    if not mask.any(axis=-1).all():
        raise ValueError('Every original attention row must retain at least one key')
    exponents = (x[mask]-shift)*2.**-steps
    probability = np.zeros_like(x)
    stages = []
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        probability[mask] = np.exp(exponents)
        for index in range(steps+1):
            if index:
                probability = probability*probability
            total = probability.sum(-1,keepdims=True)
            if not np.isfinite(total).all() or not (total>0).all():
                raise ValueError('The unit-width square normalization lost finite positive sums')
            row = dict(stage='initial' if index==0 else 'square',sum_min=float(total.min()),
                sum_max=float(total.max()),inv_min=float((1/total).min()),inv_max=float((1/total).max()))
            if index:
                row['index'] = index-1
            stages.append(row)
            probability = probability/total
    return probability,dict(shift=float(shift),steps=steps,stages=stages,
        initial_exponent_min=float(exponents.min()),initial_exponent_max=float(exponents.max()),
        variant='unit-width-public-square-v2',dynamic_max_used=False,coefficients_fitted=False,ckks_executions=0)
