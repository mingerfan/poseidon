"""A1 r8-v8 domain provenance and the shared feasibility decision."""

import math
from contextlib import contextmanager
from contextvars import ContextVar


_violation_reports = ContextVar('a1_violation_reports', default=None)


def maximum_u(values, domain, *, fn=None, domain_source='calibrated', version='r8-v4'):
    """One coverage coordinate for validation, diagnostics and saved-row replay.

    Keep the historical linear expression's arithmetic order for older prompts.
    Endpoints suffice for replay because each coordinate is monotone before abs.
    """
    import numpy as np
    data = np.asarray(values)
    lo, hi = domain
    if version == 'r8-v10' and domain_source == 'calibrated' and fn in ('inv', 'rsqrt') and lo > 0:
        if not np.isfinite(data).all() or np.any(data <= 0):
            return float('inf')
        return float(np.max(np.abs(2*(np.log(data)-math.log(lo))/(math.log(hi)-math.log(lo))-1)))
    return float(np.max(np.abs(data*(2/(hi-lo))-(hi+lo)/(hi-lo))))


@contextmanager
def capture_violations():
    """Collect internal validation reports without adding Agent feedback fields."""
    reports = []
    token = _violation_reports.set(reports)
    try:
        yield reports
    finally:
        _violation_reports.reset(token)


def record_violations(report, callback=None):
    reports = _violation_reports.get()
    if reports is not None:
        reports.append(report)
    if callback is not None:
        callback(report)


class ViolationOrder:
    """First out-of-domain approximation input for each original sample.

    Events arrive before the public observer flattens or groups its values.
    Repeated token/head visits to one point update its range while preserving
    the position of its first violation. No input arrays are retained.
    """

    def __init__(self, descriptors, rows, total, *, version='r8-v8'):
        self.version = version
        self.rows = {(p.op.id, p.role or 'act'): row for p, row in zip(descriptors, rows)}
        self.points = {(p.op.id, p.role or 'act'): p for p in descriptors}
        self.inverse_outputs = {}
        self.relative_errors = {}
        self.chains = [{} for _ in range(total)]
        self.observed = [False] * total
        self.events = 0

    def inverse_result(self, op_id, role, values, inverse, *, start=0):
        """Keep only scalar ranges from this engine's actual inverse result."""
        import numpy as np
        point = self.points.get((op_id, role))
        if point is None or getattr(point.op, 'kind', None) != 'softmax':
            return
        data = np.asarray(values)
        sums = data*np.asarray(inverse)
        for local, sample in enumerate(sums):
            x = data[local]
            self.inverse_outputs[(op_id, role, start+local)] = dict(
                c_min=float(np.min(sample)), c_max=float(np.max(sample)),
                obs_min=float(np.min(x)), obs_max=float(np.max(x)))

    def _budget(self, op_id, role, index, event_index):
        import numpy as np
        point = self.points[(op_id, role)]
        if getattr(point.op, 'kind', None) != 'softmax':
            return None
        sequence = ['inv'] + [
            'final_inv' if point.op.attrs.get('final_inv_method') is not None and i == point.op.attrs['steps']-1
            else f'step_inv[{i}]' for i in range(point.op.attrs.get('steps', 0))]
        if role not in sequence[1:]:
            return None
        upstream = sequence[sequence.index(role)-1]
        values = self.inverse_outputs.get((op_id, upstream, index))
        if values is None:
            return None
        rho = point.op.attrs.get('inv_budget')
        interval = (1-rho, 1+rho) if rho is not None else (math.sqrt(2/3), math.sqrt(3/2))
        if interval[0] <= values['c_min'] and values['c_max'] <= interval[1]:
            return None
        parent = self.points[(op_id, upstream)]
        row = self.rows[(op_id, upstream)]
        key = (op_id, upstream)
        if key not in self.relative_errors:
            import numpy as np
            from .transformer_tensor import polynomial
            grid = np.geomspace(*parent.attrs['domain'], 100001)
            self.relative_errors[key] = float(np.max(np.abs(grid*polynomial(grid, parent.attrs)-1)))
        from .planners.approximation_failures import point_fields
        lo, hi = row['domain']
        # The historical budget arithmetic used Python floats, independently
        # of the enclosing engine's NumPy floating-point exception context.
        with np.errstate(over='ignore', invalid='ignore'):
            u = maximum_u((values['obs_min'], values['obs_max']), (lo, hi),
                          fn=parent.attrs.get('fn'), domain_source=row['domain_source'], version=self.version)
        return dict(event_index=event_index, op_id=op_id, role=upstream,
            program_line=row['program_line'], graph_tensor=row['graph_tensor'],
            domain_source=row['domain_source'], domain=row['domain'], kind='budget',
            **values, max_u=u, nonfinite=not all(math.isfinite(v) for v in values.values()),
            rho=rho, R=interval, max_relative_error=self.relative_errors[key],
            target=point_fields(parent)['target'])

    def observe(self, op_id, role, values, mask=None, *, start=0):
        import numpy as np
        row = self.rows[(op_id, role)]
        data = np.asarray(values)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        active = None if mask is None else np.broadcast_to(np.asarray(mask, dtype=bool), data.shape)
        lo, hi = row['domain']
        event_index = self.events
        self.events += 1
        for local, sample in enumerate(data):
            index = start + local
            self.observed[index] = True
            sample = sample.reshape(-1) if active is None else sample[active[local]]
            if not sample.size:
                continue
            nonfinite = not bool(np.isfinite(sample).all())
            with np.errstate(over='ignore', invalid='ignore'):
                u = maximum_u(sample, (lo, hi), fn=getattr(self.points[(op_id, role)], 'attrs', {}).get('fn'),
                              domain_source=row['domain_source'], version=self.version)
            if not nonfinite and u <= 1:
                continue
            if nonfinite:
                u = float('inf')
            minimum, maximum = float(np.min(sample)), float(np.max(sample))
            if not self.chains[index] and row['domain_source'] == 'analytic':
                budget = self._budget(op_id, role, index, event_index)
                if budget is not None:
                    self.chains[index][(op_id, budget['role'], 'budget')] = budget
            key = (op_id, role)
            previous = self.chains[index].get(key)
            if previous is None:
                self.chains[index][key] = dict(event_index=event_index, op_id=op_id, role=role,
                    program_line=row['program_line'], graph_tensor=row['graph_tensor'],
                    domain_source=row['domain_source'], domain=row['domain'],
                    obs_min=minimum, obs_max=maximum, max_u=u, nonfinite=nonfinite)
            else:
                previous['obs_min'] = min(previous['obs_min'], minimum) if not math.isnan(minimum) else minimum
                previous['obs_max'] = max(previous['obs_max'], maximum) if not math.isnan(maximum) else maximum
                previous['max_u'] = max(previous['max_u'], u)
                previous['nonfinite'] |= nonfinite

    @property
    def analytic_domain_violation(self):
        return any(chain and next(iter(chain.values()))['domain_source'] == 'analytic'
                   and next(iter(chain.values())).get('kind') != 'budget'
                   for chain in self.chains)

    def budget_feedback(self):
        return [dict(sample_index=index, **next(iter(chain.values())))
                for index, chain in enumerate(self.chains)
                if chain and next(iter(chain.values())).get('kind') == 'budget']

    def report(self, *, engine, split):
        samples = []
        for index, chain in enumerate(self.chains):
            entries = list(chain.values())
            samples.append(dict(sample_index=index, observed=self.observed[index],
                                first_violation=entries[0] if entries else None, chain=entries))
        return dict(engine=engine, split=split, total=len(samples),
                    analytic_domain_violation=self.analytic_domain_violation, samples=samples)


def enabled(ctx):
    return getattr(ctx, 'prompt_version', 'r8-v4') in ('r8-v8','r8-v9','r8-v10')


def analytic_domain(point, input_shape):
    op = point.op
    if op.kind != 'softmax' or point.prefix not in ('inv', 'step_inv', 'final_inv'):
        return None
    n = input_shape[-1]
    a = op.attrs
    if point.prefix == 'inv':
        # exp_domain already describes (scores-shift)*2**(-steps).
        lo, hi = a['exp_domain']
        try:
            return (math.exp(lo)/a['inv_scale'], n*math.exp(hi)/a['inv_scale'])
        except OverflowError:
            return None  # No finite declared domain can contain this bound.
    if a.get('final_inv_method') is not None:
        rho = a.get('inv_budget')
        if rho is None:
            return None  # An unprovided error budget proves no repeated bound.
        return ((1-rho)**2/(1.5*n), 1.5*(1+rho)**2)
    return (1/(1.5*n), 1.5)


def provenance(program, ctx, descriptors):
    from .dsl.typecheck import check
    checked = check(program, ctx.weight_shapes, prompt_version='r8-v8')
    result = []
    for point in descriptors:
        bound = analytic_domain(point, checked.shapes[point.op.inputs[0]])
        lo, hi = point.attrs['domain']
        analytic = bound is not None and contains_analytic_domain((lo, hi), bound)
        result.append(dict(domain_source='analytic' if analytic else 'calibrated',
                           analytic_domain=bound, u_limit=1.0 if analytic else 0.90))
        if getattr(ctx, 'prompt_version', None) == 'r8-v10':
            fn = point.attrs.get('fn')
            if not analytic and fn in ('inv', 'rsqrt') and lo > 0:
                result[-1].update(fn=fn, u_scale='log')
    return result


def contains_analytic_domain(declared, bound):
    return declared[0] <= bound[0]*(1+1e-9) and declared[1] >= bound[1]*(1-1e-9)


def summarize(rows):
    calibrated = [row['max_u'] for row in rows if row['domain_source'] == 'calibrated']
    analytic = [row['max_u'] for row in rows if row['domain_source'] == 'analytic']
    return dict(max_u_calibrated=max(calibrated, default=0.0),
                max_u_analytic=max(analytic, default=0.0),
                analytic_domain_violation=any(value > 1 for value in analytic))


def domain_ok(result):
    return result['max_u_calibrated'] <= 0.90 and result['max_u_analytic'] <= 1


def require_analytic_coverage(result):
    if result.get('analytic_domain_violation'):
        import json
        points = [r for r in result.get('rows', result.get('worst_acts', []))
                  if r.get('domain_source') == 'analytic' and r['max_u'] > 1]
        raise RuntimeError('A1 §6.7 analytical domain exceeded: '+json.dumps(points))
