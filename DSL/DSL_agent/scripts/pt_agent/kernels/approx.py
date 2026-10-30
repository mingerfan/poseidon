"""U5 deterministic Remez fitting and arithmetic shared by trusted evaluators.

The exchange is performed in a Chebyshev basis. Failure is explicit; this
module never substitutes interpolation for a failed minimax fit.
"""

from functools import lru_cache

import numpy as np

from .layout import KernelError
from .poly import EXACT, cheb_eval


MAX_ITERATIONS = 50
RELATIVE_CONVERGENCE = 1e-6
NEW_METHODS = frozenset(('remez','composite','newton'))
REMEZ_FUNCTIONS = dict(EXACT, inv=lambda x: 1.0/x, rsqrt=lambda x: 1.0/np.sqrt(x),
                       hardswish=lambda x: x*np.minimum(np.maximum(x+3.0, 0.0), 6.0)/6.0,
                       square=lambda x: x*x)
NEWTON_FUNCTIONS = frozenset(('inv', 'rsqrt'))
COMPOSITE_FUNCTIONS = frozenset(('relu',))


def tolerance(amplitude, maximum_target):
    """U5 §7: compare the relative term with the float64 evaluation floor."""
    relative = RELATIVE_CONVERGENCE*abs(amplitude)
    tau = float(64*np.finfo(np.float64).eps*maximum_target)
    return max(relative, tau), tau, 'relative' if relative >= tau else 'float64-floor'


class ApproximationError(KernelError):
    """A finite exchange failed, with the last numerical evidence attached."""

    def __init__(self, message, *, iterations, max_error, error_amplitude, history,
                 stage=None, degree=None, domain=None):
        self.iterations = iterations
        self.max_error = float(max_error)
        self.error_amplitude = float(error_amplitude)
        self.history = history
        self.stage, self.degree, self.domain = stage, degree, domain
        super().__init__(f"{message}: iterations={iterations}, "
                         f"max|err|={max_error!r}, E={error_amplitude!r}")


def function(fn, value):
    """The declared mathematical targets; no calibration or random inputs."""
    if fn in REMEZ_FUNCTIONS:
        return REMEZ_FUNCTIONS[fn](value)
    raise KernelError(f"Unknown approximation function {fn!r}")


def _parameters(fn, deg, domain):
    if type(deg) is not int or deg < 1:
        raise KernelError('Remez requires a positive integer degree')
    if (len(domain) != 2 or not np.isfinite(domain).all()
            or not domain[0] < domain[1]):
        raise KernelError('Remez requires a finite domain with lo < hi')
    if fn in ('inv', 'rsqrt') and domain[0] <= 0:
        raise KernelError('inv/rsqrt requires a positive domain')
    function(fn, np.asarray([domain[0]], dtype=np.float64))


def _alternating(points, errors):
    """Keep the strongest extremum in each same-sign run, in input order."""
    chosen = []
    for index, error in enumerate(errors):
        if error == 0:
            continue
        if chosen and np.signbit(error) == np.signbit(errors[chosen[-1]]):
            if abs(error) > abs(errors[chosen[-1]]):
                chosen[-1] = index
        else:
            chosen.append(index)
    return points[chosen], errors[chosen]


def _extrema(error, grid):
    """Bracket on the dense grid, then refine all brackets deterministically."""
    values = error(grid)
    absolute = np.abs(values)
    indices = np.flatnonzero((absolute[1:-1] >= absolute[:-2]) &
                            (absolute[1:-1] >= absolute[2:]) &
                            ((absolute[1:-1] > absolute[:-2]) |
                             (absolute[1:-1] > absolute[2:])))+1
    left, right = grid[indices-1].copy(), grid[indices+1].copy()
    signs = np.where(values[indices] < 0, -1.0, 1.0)
    # Fixed ternary refinement uses only the function values and NumPy.
    for _ in range(64):
        a, b = (2*left+right)/3, (left+2*right)/3
        take_right = error(a)*signs < error(b)*signs
        left = np.where(take_right, a, left)
        right = np.where(take_right, right, b)
    refined = (left+right)/2
    # Retain the original grid point if roundoff made refinement worse.
    refined = np.where(np.abs(error(refined)) >= absolute[indices], refined, grid[indices])
    points = np.concatenate((grid[:1], refined, grid[-1:]))
    errors = error(points)
    return _alternating(points, errors), max(float(absolute.max()), float(np.abs(errors).max()))


def _references(points, errors, count):
    """Exchange extrema, removing weakest alternating pairs as necessary."""
    selected = list(range(len(points)))
    while len(selected) > count:
        if len(selected) == count+1:
            # Removing one interior point would break sign alternation.
            drop = 0 if abs(errors[selected[0]]) <= abs(errors[selected[-1]]) else -1
            selected.pop(drop)
            continue
        weakest = min(range(len(selected)), key=lambda i: (abs(errors[selected[i]]), i))
        if weakest == 0 or weakest == len(selected)-1:
            selected.pop(weakest)
        else:
            neighbor = (weakest-1 if abs(errors[selected[weakest-1]]) <=
                        abs(errors[selected[weakest+1]]) else weakest+1)
            for i in sorted((weakest, neighbor), reverse=True):
                selected.pop(i)
    return points[selected]


def _single_exchange(reference, point, error, amplitude):
    """A degenerate initial ripple can have fewer than degree+2 extrema.

    Exchange its largest-error point against the adjacent reference of the
    same ripple sign. This is the single-point Remez exchange; retaining the
    initial references here would simply solve the same system forever.
    """
    result = reference.copy()
    index = int(np.searchsorted(reference, point))
    ripple_sign = 1.0 if amplitude >= 0 else -1.0
    error_sign = 1.0 if error >= 0 else -1.0
    def sign(i):
        return ripple_sign*(-1.0 if i % 2 else 1.0)
    if index == 0:
        result = (np.concatenate(([point],reference[:-1])) if error_sign != sign(0)
                  else np.concatenate(([point],reference[1:])))
    elif index == len(reference):
        result = (np.concatenate((reference[1:],[point])) if error_sign != sign(len(reference)-1)
                  else np.concatenate((reference[:-1],[point])))
    else:
        result[index-1 if error_sign == sign(index-1) else index] = point
    return result


def _exchange(target, degree, *, odd=False, interval=(-1.0,1.0), w=None):
    count = (degree+1)//2+1 if odd else degree+2
    reference = np.cos(np.linspace(np.pi, 0.0, count))
    if odd:
        reference = interval[0]+(interval[1]-interval[0])*(reference+1)/2
    grid = np.linspace(*interval, max(8193, 64*(degree+1)+1))
    maximum_target = float(np.abs(target(grid)).max())
    signs = np.where(np.arange(count) % 2, -1.0, 1.0)
    history = []
    maximum, amplitude = np.inf, 0.0
    for iteration in range(1, MAX_ITERATIONS+1):
        if odd:
            basis = np.polynomial.chebyshev.chebvander(reference,degree)[:,1::2]
        else:
            basis = np.polynomial.chebyshev.chebvander(reference, degree)
        matrix = np.column_stack((basis, signs if w is None else signs/w(reference)))
        try:
            solution = np.linalg.solve(matrix, target(reference))
        except np.linalg.LinAlgError as exc:
            raise ApproximationError('Remez exchange has a singular reference system',
                iterations=iteration, max_error=maximum, error_amplitude=amplitude,
                history=history) from exc
        coefficients, amplitude = solution[:-1], float(solution[-1])
        if odd:
            expanded = np.zeros(degree+1)
            expanded[1::2] = coefficients
            coefficients = expanded
        def error(value):
            difference = target(value)-np.polynomial.chebyshev.chebval(value, coefficients)
            return difference if w is None else w(value)*difference
        (points, errors), maximum = _extrema(error, grid)
        ratio = ((maximum-abs(amplitude))/abs(amplitude) if amplitude else
                 (0.0 if maximum == 0.0 else np.inf))
        allowed, tau, criterion = tolerance(amplitude, maximum_target)
        history.append(dict(iteration=iteration, max_error=maximum, E=amplitude,
                            relative_gap=float(ratio), alternating_extrema=len(points),
                            tau=tau, gap=maximum-abs(amplitude),criterion=criterion,
                            minimum_extremal_error=float(np.abs(errors).min()) if len(errors) else 0.0))
        if maximum-abs(amplitude) <= allowed:
            coefficients.flags.writeable = False
            return coefficients, dict(iterations=iteration, max_error=maximum, E=amplitude,
                tau=tau,gap=maximum-abs(amplitude),criterion=criterion,
                reference=reference.tolist(), extrema=points.tolist(), errors=errors.tolist(), history=history)
        if len(points) >= count:
            reference = _references(points, errors, count)
        else:
            index = int(np.argmax(np.abs(errors)))
            reference = _single_exchange(reference,points[index],errors[index],amplitude)
            history[-1]['exchange'] = 'single-extremum'
    raise ApproximationError('Remez did not converge', iterations=MAX_ITERATIONS,
                             max_error=maximum, error_amplitude=amplitude, history=history)


@lru_cache(maxsize=None)
def _remez(fn, deg, domain):
    _parameters(fn, deg, domain)
    lo, hi = domain
    return _exchange(lambda u: function(fn, ((hi-lo)*u+(hi+lo))/2), deg)


@lru_cache(maxsize=None)
def _relative_remez(fn, deg, domain):
    _parameters(fn,deg,domain)
    if fn not in ('exp','inv','rsqrt'):
        raise KernelError('relative Remez requires exp, inv or rsqrt')
    lo,hi=domain
    def target(u):
        return function(fn,((hi-lo)*u+(hi+lo))/2)
    return _exchange(target,deg,w=lambda u:1/target(u))


def _fit_result(fn,deg,domain,fit):
    if fit=='absolute':return _remez(fn,deg,tuple(domain))
    if fit=='relative':return _relative_remez(fn,deg,tuple(domain))
    raise KernelError('fit must be absolute or relative')


def remez(fn, deg, domain, *, fit='absolute'):
    """Fit the target in Chebyshev basis with the specified strict exchange."""
    return _fit_result(fn,deg,domain,fit)[0]


def remez_report(fn, deg, domain, *, fit='absolute'):
    """Return the exchange's numerical evidence, including alternating extrema."""
    return _fit_result(fn,deg,domain,fit)[1]


def act_remez(b, x, fn, deg, domain, *, coefficients=None):
    c = remez(fn, deg, domain) if coefficients is None else coefficients
    return evaluate(b,x,fn,'remez',domain,c)


@lru_cache(maxsize=None)
def _composite(degrees, eps):
    if (not degrees or any(type(d) is not int or d < 1 or d % 2 == 0 for d in degrees)
            or not np.isfinite(eps) or not 0 < eps < 1):
        raise KernelError('composite requires positive odd degrees and 0 < eps < 1')
    lo,hi = float(eps),1.0
    coefficients,reports = [],[]
    for index,degree in enumerate(degrees):
        try:
            c, report = _exchange(lambda x: np.ones_like(x),degree,odd=True,interval=(lo,hi))
        except ApproximationError as exc:
            raise ApproximationError(f'composite stage {index+1}, degree={degree}, interval={(lo,hi)}: {exc}',
                iterations=exc.iterations,max_error=exc.max_error,error_amplitude=exc.error_amplitude,
                history=exc.history,stage=index+1,degree=degree,domain=(lo,hi)) from exc
        report = dict(report,degree=degree,domain=(lo,hi))
        coefficients.append(c)
        reports.append(report)
        lo,hi = 1-report['max_error'],1+report['max_error']
    coefficients[-1] = coefficients[-1]*0.5
    coefficients[-1].flags.writeable = False
    return tuple(coefficients),reports


def composite_coeffs(degrees, eps):
    """Odd stage fits to 1; the final stage includes the factor one half."""
    return _composite(tuple(degrees),float(eps))[0]


def composite_report(degrees, eps):
    return _composite(tuple(degrees),float(eps))[1]


def composite(b,x,degrees,domain,eps,*,coefficients=None):
    lo,hi = domain
    if not (hi > 0 and lo == -hi):
        raise KernelError('composite requires a symmetric domain (-B,B)')
    c = composite_coeffs(degrees,eps) if coefficients is None else coefficients
    return evaluate(b,x,'relu','composite',domain,c)


@lru_cache(maxsize=None)
def _stage_cost(degree):
    from ..backends.count import CountBackend
    backend = CountBackend(8)
    coefficients = np.zeros(degree+1)
    coefficients[1::2] = 1.0
    result = cheb_eval(backend,backend.input(0),coefficients)
    return result.depth,backend.stats()['ops_total']


class CompositeSearchError(KernelError):
    def __init__(self,message,*,excluded,attempted):
        self.exclusions = excluded
        self.attempted = attempted
        super().__init__(message+f'; excluded={len(excluded)}, attempted={attempted}')


@lru_cache(maxsize=None)
def search_composite(max_depth,eps,target_error):
    """Enumerate the prescribed finite odd tuples, ordered by depth and cost."""
    from itertools import product
    from ..backends.count import CountBackend
    if (type(max_depth) is not int or max_depth < 1 or not 0 < eps < 1
            or not np.isfinite(target_error) or target_error <= 0):
        raise KernelError('search requires a positive depth budget, error target and 0 < eps < 1')
    choices = tuple(range(1,64,2))
    candidates = []
    for length in range(1,5):
        for degrees in product(choices,repeat=length):
            depth = 2+sum(_stage_cost(d)[0] for d in degrees)
            if depth <= max_depth:
                operations = 3+sum(_stage_cost(d)[1] for d in degrees)
                candidates.append((depth,operations,degrees))
    grid = np.linspace(eps,1.0,2**16)
    attempted = 0
    excluded = []
    best = None
    for predicted_depth,_,degrees in sorted(candidates):
        if best is not None and predicted_depth > best['depth']:
            break
        attempted += 1
        try:
            coefficients = composite_coeffs(degrees,eps)
        except ApproximationError as exc:
            excluded.append(dict(degrees=degrees,stage=exc.stage,degree=exc.degree,
                domain=exc.domain,max_error=exc.max_error,E=abs(exc.error_amplitude),
                gap=exc.max_error-abs(exc.error_amplitude),iterations=exc.iterations,
                tau=exc.history[-1].get('tau') if exc.history else None))
            continue
        sign = grid
        for c in coefficients:
            sign = cheb_eval(None,sign,c)
        error = float(np.max(np.abs(sign-0.5)))
        if error <= target_error:
            b = CountBackend(8)
            result = composite(b,b.input(0),degrees,(-1.0,1.0),eps,coefficients=coefficients)
            candidate = dict(degrees=degrees,eps=eps,error=error,depth=result.depth,
                             operations=b.stats()['ops_total'])
            key = lambda row:(row['depth'],row['operations'],row['degrees'])
            if best is None or key(candidate) < key(best):
                best = candidate
    if best is not None:
        return dict(best,attempted=attempted,excluded=len(excluded),exclusions=excluded)
    raise CompositeSearchError(f'No composite tuple meets depth={max_depth}, eps={eps}, error={target_error}',
                               excluded=excluded,attempted=attempted)


def newton(b, x, fn, deg0, iters, domain, *, coefficients=None):
    """U5 §6: identical explicit order for NumPy, Mock and emitted arithmetic."""
    if fn not in NEWTON_FUNCTIONS or domain[0] <= 0:
        raise KernelError('newton requires inv/rsqrt and a positive domain')
    if type(iters) is not int or iters < 0:
        raise KernelError('newton iters must be a nonnegative integer')
    c = remez(fn,deg0,domain) if coefficients is None else coefficients
    return evaluate(b,x,fn,'newton',domain,c,iters=iters)


def evaluate(b,x,fn,method,domain,coefficients,*,iters=0):
    """Coefficient-only arithmetic, also copied verbatim into standalone P2."""
    lo,hi = domain
    if method == 'composite':
        sign = x*(1.0/hi)
        for stage in coefficients:
            sign = cheb_eval(b,sign,stage)
        return x*(sign+0.5)
    y = cheb_eval(b,x*(2/(hi-lo))-(hi+lo)/(hi-lo),coefficients)
    if method == 'remez':
        return y
    if fn == 'rsqrt':
        x_half = x*0.5
    for _ in range(iters):
        if fn == 'inv':
            t = x*y
            t = 2-t
            y = y*t
        else:
            t = y*y
            t = x_half*t
            t = 1.5-t
            y = y*t
    return y


def fitted_coefficients(attrs):
    method,domain,fn = attrs['method'],attrs['domain'],attrs.get('fn','relu')
    if method == 'remez':
        return remez(fn,attrs['deg'],domain,fit=attrs.get('fit') or 'absolute')
    if method == 'composite':
        return composite_coeffs(tuple(attrs['degrees']),attrs['eps'])
    if method == 'newton':
        return remez(fn,attrs['deg0'],domain,fit=attrs.get('fit') or 'absolute')
    raise KernelError(f'Unknown fitted method {method!r}')


def activation(b,x,attrs,*,coefficients=None):
    c = fitted_coefficients(attrs) if coefficients is None else coefficients
    return evaluate(b,x,attrs.get('fn','relu'),attrs['method'],attrs['domain'],c,
                    iters=attrs.get('iters') or 0)


def configuration(attrs):
    method = attrs['method']
    parameters = (attrs['deg'] if method == 'remez' else
                  (tuple(attrs['degrees']),attrs['eps']) if method == 'composite' else
                  (attrs['deg0'],attrs['iters']))
    result = (method,parameters,*attrs['domain'])
    return result+('relative',) if attrs.get('fit')=='relative' else result


def degree_bound(attrs):
    return (max(attrs['degrees']) if attrs['method'] == 'composite' else
            attrs['deg0'] if attrs['method'] == 'newton' else attrs['deg'])
