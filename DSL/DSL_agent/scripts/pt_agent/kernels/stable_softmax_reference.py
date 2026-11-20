"""Independent floating-point oracle for the existing squaring softmax.

This is a numerical diagnostic of the existing public-shift algorithm.  It
does not select a model, fit coefficients, or stand in for CKKS execution.
"""


def fixed_shift_squaring(values, keep, *, shift, steps):
    import math
    import numpy as np
    x = np.asarray(values)
    mask = np.asarray(keep)
    if (x.dtype != np.float64 or x.ndim < 1 or not np.isfinite(x).all()
            or mask.dtype != np.bool_ or type(steps) is not int or steps not in range(9)
            or not math.isfinite(shift)):
        raise ValueError('Finite float64 scores, a boolean mask, public shift and steps0..8 are required')
    mask = np.broadcast_to(mask, x.shape)
    if not mask.any(axis=-1).all():
        raise ValueError('Every original attention row must retain at least one key')
    exponents = (x[mask]-shift)*(2.0**-steps)
    p = np.zeros_like(x)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        p[mask] = np.exp(exponents)
        total = p.sum(-1, keepdims=True)
        if not np.isfinite(total).all() or not (total > 0).all():
            raise ValueError('The fixed public configuration does not yield finite positive sums')
        stages = [dict(stage='initial', sum_min=float(total.min()), sum_max=float(total.max()),
                       inv_min=float((1/total).min()), inv_max=float((1/total).max()))]
        p = p/total
        for index in range(steps):
            p = p*p
            total = p.sum(-1, keepdims=True)
            if not np.isfinite(total).all() or not (total > 0).all():
                raise ValueError('A squaring normalization became nonfinite or empty')
            stages.append(dict(stage='square', index=index, sum_min=float(total.min()),
                               sum_max=float(total.max()), inv_min=float((1/total).min()),
                               inv_max=float((1/total).max())))
            p = p/total
    return p, dict(shift=float(shift), steps=steps, stages=stages,
                   initial_exponent_min=float(exponents.min()), initial_exponent_max=float(exponents.max()),
                   dynamic_max_used=False, coefficients_fitted=False, ckks_executions=0)
