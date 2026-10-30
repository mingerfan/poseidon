"""r8-v9 calibration-only minimum domains, checked before execution."""


def minimum_domain(lo, hi):
    return (min(1.5*lo, lo/1.5), max(1.5*hi, hi/1.5))


def contains(declared, required):
    lo, hi = required
    return declared[0] <= lo+1e-9*abs(lo) and declared[1] >= hi-1e-9*abs(hi)


def check_margin(program, ctx):
    from .transformer_validation import points, point_names
    errors = []
    if getattr(ctx, 'prompt_version', None) not in ('r8-v9','r8-v10'):
        return dict(ok=True, errors=errors)
    for point, name in zip(points(program), point_names(program, ctx)):
        # Inverse roles retain the registered analytical-domain rules.
        if point.op.kind == 'softmax' and point.role != 'exp':
            continue
        limits = ctx.calibration[name]
        required = minimum_domain(limits['min'], limits['max'])
        if point.op.kind == 'softmax':
            shift = point.op.attrs.get('shift', 0.)
            scale = 2.**(-point.op.attrs.get('steps', 0))
            required = tuple((value-shift)*scale for value in required)
        declared = point.attrs['domain']
        if not contains(declared, required):
            errors.append(dict(code='E_MARGIN', line=point.op.lineno,
                call_lines=list(point.op.call_lines), role=point.role or 'act',
                point=name, declared_domain=declared, min_domain=required,
                message='Declared domain must contain the calibration minimum domain M.'))
    return dict(ok=not errors, errors=errors)


def require_margin(program, ctx):
    result = check_margin(program, ctx)
    if not result['ok']:
        raise ValueError('A1 §6.15 minimum margin violated: '+repr(result['errors']))
    return result
