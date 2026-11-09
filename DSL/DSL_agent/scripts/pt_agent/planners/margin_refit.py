"""A1 §6.23: refit a widened calibrated point so it keeps its original accuracy."""

import math
import time

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


def _evaluate(old_attrs, domain, setting):
    attrs = dict(old_attrs, domain=tuple(domain), **setting)
    if 'method' in setting:
        from ..dsl.ops_spec import new_approx_error
        problem = new_approx_error(dict(attrs, degrees=attrs.get('degrees'))) if setting['method'] != 'cheb' else (
            'cheb takes deg and domain only' if any(attrs.get(k) is not None for k in ('eps', 'deg0', 'iters')) else None)
        if problem:
            return dict(setting, error=math.inf, failure='dsl: '+problem)
    try:
        error = max_error(attrs)
        level = depth(attrs)
    except Exception as exc:  # numerical or kernel failure: candidate unusable
        return dict(setting, error=math.inf, failure=type(exc).__name__)
    return dict(setting, error=error, depth=level)


def _order(row):
    return (row['depth'], row.get('deg') or row.get('deg0'), row.get('iters') or 0, row.get('method', ''))


def extended_settings(attrs):
    """E8 §2.3 stage 1: the same method above the stage-0 limit, up to 255."""
    method = attrs['method']
    if method == 'cheb':
        return []
    limit = max(BASE_LIMIT, attrs['deg'] if method == 'remez' else attrs['deg0'])
    extra = [d for d in LADDER if limit < d <= CHEB_LIMIT]
    if method == 'remez':
        return [dict(deg=d) for d in extra]
    if method == 'newton':
        return [dict(deg0=d, iters=i) for d in extra for i in NEWTON_ITERS]
    return []


def alternative_settings(attrs):
    """E8 §2.3 stage 2: other methods of the same function, full ladder up to 255."""
    fn, method = attrs['fn'], attrs['method']
    methods = ('remez', 'newton') if fn in ('inv', 'rsqrt') else ('remez', 'cheb')
    if method not in methods:
        return []
    ladder = [d for d in LADDER if d <= CHEB_LIMIT]
    rows = []
    for other in sorted(m for m in methods if m != method):
        fit = attrs.get('fit') if other in ('remez', 'newton') and method in ('remez', 'newton') else None
        if other == 'newton':
            rows += [dict(method='newton', deg=None, deg0=d, iters=i, fit=fit) for d in ladder for i in NEWTON_ITERS]
        else:
            rows += [dict(method=other, deg=d, deg0=None, iters=None, fit=fit) for d in ladder]
    return rows


def refit(old_attrs, domain, *, extended=False):
    """Choose settings on `domain` whose error stays within the error of `old_attrs` on its domain.

    `extended` (tune rule v8-6.25) adds stage 1 and stage 2 when stage 0 has no setting within the target."""
    started = time.monotonic()
    target = max_error(old_attrs)
    record = dict(old_domain=list(old_attrs['domain']), new_domain=list(domain), target=target)
    stages = [settings(old_attrs)]
    if extended:
        stages += [extended_settings(old_attrs), alternative_settings(old_attrs)]
    usable, counts = [], []
    choice, stage_used = None, None
    for stage, candidates in enumerate(stages):
        rows = [_evaluate(old_attrs, domain, setting) for setting in candidates]
        counts.append(len(rows))
        found = [r for r in rows if math.isfinite(r['error']) and 'depth' in r]
        usable += [dict(r, stage=stage) for r in found]
        within = [r for r in found if r['error'] <= target*(1+SLACK)]
        if within:
            choice, stage_used = min(within, key=_order), stage
            break
    if extended:
        record.update(stage_counts=counts, seconds=time.monotonic()-started)
    if not usable:
        record.update(status='kept', setting=None)
        return {}, record
    status = 'within'
    if choice is None:
        choice = min(usable, key=lambda r: (r['error'], *_order(r)))
        stage_used, status = choice['stage'], 'closest'
    keys = ('method', 'deg', 'deg0', 'iters', 'fit') if 'method' in choice else ('deg', 'deg0', 'iters')
    setting = {k: choice[k] for k in keys if k in choice}
    record.update(status=status, setting=setting, error=choice['error'], depth=choice['depth'])
    if extended:
        record.update(stage=stage_used, method=setting.get('method', old_attrs['method']))
    return setting, record
