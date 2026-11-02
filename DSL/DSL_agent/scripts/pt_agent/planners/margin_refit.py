"""A1 §6.23: refit a widened calibrated point so it keeps its original accuracy."""

import math

import numpy as np

from ..kernels import approx

GRID = 20001
LADDER = (7, 15, 31, 63, 127, 255)
BASE_LIMIT = 63
NEWTON_ITERS = range(0, 6)
SLACK = 1e-9


def grid(domain):
    lo, hi = map(float, domain)
    values = np.linspace(lo, hi, GRID)
    if lo > 0:
        values = np.concatenate((values, np.geomspace(lo, hi, GRID)))
    return np.unique(values)


def max_error(attrs):
    """Largest error on the point's own domain: relative for inv/rsqrt, absolute otherwise."""
    from ..transformer_tensor import polynomial
    x = grid(attrs['domain'])
    with np.errstate(all='ignore'):
        try:
            got = polynomial(x, attrs)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            return math.inf
        exact = approx.function(attrs['fn'], x)
        error = np.abs(got/exact-1) if attrs['fn'] in ('inv', 'rsqrt') else np.abs(got-exact)
    if not np.all(np.isfinite(error)):
        return math.inf
    return float(np.max(error))


def depth(attrs):
    from ..backends import CountBackend
    from ..kernels.poly import act_cheb
    backend = CountBackend(8)
    x = backend.input(0)
    if attrs['method'] == 'cheb':
        return act_cheb(backend, x, attrs['fn'], attrs['deg'], attrs['domain']).depth
    return approx.activation(backend, x, attrs).depth


CHEB_LIMIT = 255


def degrees(current, method='remez'):
    """cheb: the whole DSL range up to 255; remez/newton: the registered ladder up to 63, or the current degree."""
    limit = CHEB_LIMIT if method == 'cheb' else max(BASE_LIMIT, current)
    return [d for d in LADDER if d <= limit]


def settings(attrs):
    """Same method and fit; remez/cheb degrees and newton deg0 from the ladder, newton iters 0..5."""
    method = attrs['method']
    if method in ('remez', 'cheb'):
        return [dict(deg=d) for d in degrees(attrs['deg'], method)]
    if method == 'newton':
        return [dict(deg0=d, iters=i) for d in degrees(attrs['deg0'], method) for i in NEWTON_ITERS]
    return []


def refit(old_attrs, domain):
    """Choose settings on `domain` whose error stays within the error of `old_attrs` on its domain."""
    target = max_error(old_attrs)
    record = dict(old_domain=list(old_attrs['domain']), new_domain=list(domain), target=target)
    rows = []
    for setting in settings(old_attrs):
        attrs = dict(old_attrs, domain=tuple(domain), **setting)
        try:
            error = max_error(attrs)
            level = depth(attrs)
        except Exception as exc:  # numerical or kernel failure: candidate unusable
            rows.append(dict(setting, error=math.inf, failure=type(exc).__name__))
            continue
        rows.append(dict(setting, error=error, depth=level))
    usable = [r for r in rows if math.isfinite(r['error']) and 'depth' in r]
    if not usable:
        record.update(status='kept', setting=None)
        return {}, record
    order = lambda r: (r['depth'], r.get('deg', r.get('deg0')), r.get('iters', 0))
    within = [r for r in usable if r['error'] <= target*(1+SLACK)]
    choice = min(within, key=order) if within else min(usable, key=lambda r: (r['error'], *order(r)))
    setting = {k: choice[k] for k in ('deg', 'deg0', 'iters') if k in choice}
    record.update(status='within' if within else 'closest', setting=setting,
                  error=choice['error'], depth=choice['depth'])
    return setting, record
