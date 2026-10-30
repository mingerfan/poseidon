"""Fixed U4 rule baseline for the normalized token/attention graph."""

import json
import math
import re
from functools import lru_cache


RELATIVE_TARGET = 1e-3
NEWTON_DEGREES = (7,15,31)
MAX_NEWTON_ITERS = 5
INV_BUDGETS = (0.5, 0.25, 0.1)


class ApproximationLimitError(ValueError):
    """The U4 §11.2 finite choices do not meet the requested error."""

    def __init__(self, key, report):
        self.key, self.report = key, report
        super().__init__(f'{key}: no permitted {report["fn"]} approximation meets '
                         f'relative error {report["target"]}; see candidate evidence')


def _relative_profile(fn, degree, domain, fit='absolute'):
    import numpy as np
    from ..kernels import approx
    from ..kernels.poly import cheb_eval
    from ..backends import CountBackend
    lo,hi=domain;half=(hi-lo)/2;mid=(hi+lo)/2
    c=approx.remez(fn,degree,domain,fit=fit)
    from .approximation_failures import require_finite
    require_finite(c,'coefficients')
    # Include the real stationary points of the relative-error function,
    # endpoints, and uniform/geometric grids. A failing point is a concrete
    # witness; these finite evaluations are recorded as numerical evidence.
    ch=np.polynomial.chebyshev
    coordinate=np.array([mid,half]);derivative=ch.chebder(c)
    if fn=='inv':
        stationary=ch.chebder(ch.chebmul(coordinate,c))
    elif fn=='rsqrt':
        stationary=ch.chebadd(half*c,2*ch.chebmul(coordinate,derivative))
    else:
        stationary=ch.chebsub(derivative,half*c)
    roots=ch.chebroots(stationary)
    roots=roots[np.isreal(roots)].real
    roots=roots[(roots>=-1)&(roots<=1)]
    grid=np.linspace(lo,hi,2**16+1)
    if lo>0:grid=np.concatenate((grid,np.geomspace(lo,hi,2**16+1)))
    grid=np.unique(np.concatenate((grid,mid+half*roots)))
    y0=cheb_eval(None,grid*(2/(hi-lo))-(hi+lo)/(hi-lo),c)
    z=y0/(np.exp(grid) if fn=='exp' else 1/grid if fn=='inv' else 1/np.sqrt(grid))
    error=np.abs(1-z);index=int(np.argmax(error))
    require_finite(z,'relative values')
    require_finite(error,'relative errors')
    backend=CountBackend(8)
    base=approx.evaluate(backend,backend.input(0),fn,'remez',domain,c).depth
    evidence=dict(degree=degree,fit=fit,relative_error=float(error[index]),depth=base,
                  minimum_initial_z=float(z.min()),maximum_initial_z=float(z.max()),
                  diagnostic_points=int(grid.size),
                  witness=dict(x=float(grid[index]),value=float(y0[index]),relative_error=float(error[index])),
                  fitting=approx.remez_report(fn,degree,domain,fit=fit))
    return evidence,z


def inverse_theory(domain,target=RELATIVE_TARGET):
    """A1 §1.8 stable diagnostics, independent of candidate selection."""
    lo,hi=map(float,domain)
    ratio=hi/lo if lo!=0 else None
    result=dict(domain=domain,ratio=ratio if ratio is not None and math.isfinite(ratio) else None,
        optimal_constant_newton_iters=None,relative_minimax_polynomial_degree=None,approximate=False,
        polynomial_formula='ceil(acosh(1/target)/log1p(e+sqrt(e*(2+e))))-1; e=2/(r-1)',
        newton_formula='0 if q<=target else ceil(log2(-log(target))-log2(-log1p(-2/(r+1))))',
        note='与原式数学等价；理论诊断只做记录，不参与选择')
    if lo<=0:result['reason']='nonpositive lower bound'
    elif ratio is None or not math.isfinite(ratio):result['reason']='ratio not finite'
    elif ratio<=1:result['reason']='ratio <= 1'
    else:
        log_q=math.log1p(-2/(ratio+1))
        iterations=0 if log_q<=math.log(target) else math.ceil(math.log2(-math.log(target))-math.log2(-log_q))
        e=2/(ratio-1)
        acosh_ratio=math.log1p(e+math.sqrt(e*(2+e)))
        degree=math.ceil(math.acosh(1/target)/acosh_ratio)-1
        result.update(optimal_constant_newton_iters=iterations if iterations<=2**53 else float(iterations),
                      relative_minimax_polynomial_degree=degree if degree<=2**53 else float(degree),
                      approximate=iterations>2**53 or degree>2**53)
    return result


@lru_cache(maxsize=128)
def _select_approximation(fn,domain,target,exp_ladder,reject_degenerate=False):
    import numpy as np
    from .tuner import LADDER
    from ..kernels import approx
    from .approximation_failures import NUMERICAL_ERRORS,failure as numerical_failure
    degrees=(15,31,63) if fn=='exp' and not exp_ladder else tuple(sorted(d for d in LADDER if d<=63))
    candidates=[];profiles={}
    for fit in ('absolute','relative'):
        for degree in degrees:
            try:profile,z=_relative_profile(fn,degree,domain,fit)
            except NUMERICAL_ERRORS as exc:
                if reject_degenerate:
                    failure=numerical_failure(exc,degree=degree,domain=domain)
                elif isinstance(exc,approx.ApproximationError):
                    failure=dict(reason='nonconvergent',iterations=exc.iterations,
                                 max_error=exc.max_error,abs_E=abs(exc.error_amplitude),
                                 history=exc.history,degree=degree,domain=domain)
                else:raise
                profiles[fit,degree]=(None,failure)
                candidates.append(dict(method='remez',fit=fit,deg=degree,feasible=False,**failure))
                continue
            profiles[fit,degree]=(profile,z)
            candidates.append(dict(method='remez',fit=fit,deg=degree,depth=profile['depth'],
                                   relative_error=profile['relative_error'],
                                   feasible=profile['relative_error']<=target,profile=profile))
        if fn in ('inv','rsqrt'):
            for degree in NEWTON_DEGREES:
                profile,z=profiles[fit,degree]
                if profile is None:
                    candidates.extend(dict(method='newton',fit=fit,deg0=degree,iters=iteration,
                                           feasible=False,**z) for iteration in range(1,MAX_NEWTON_ITERS+1))
                    continue
                z=z.copy();history=[float(np.max(np.abs(1-z)))]
                positive=bool(np.all(z>0))
                for iteration in range(1,65):
                    with np.errstate(over='ignore',invalid='ignore'):
                        z=1-(1-z)*(1-z) if fn=='inv' else z*(1.5-.5*z*z)
                        error=float(np.max(np.abs(1-z)))
                    history.append(error)
                    if iteration<=MAX_NEWTON_ITERS:
                        candidate=dict(method='newton',fit=fit,deg0=degree,iters=iteration,
                            depth=profile['depth']+(2 if fn=='inv' else 3)*iteration,
                            relative_error=error,feasible=positive and math.isfinite(error) and error<=target,
                            profile=profile)
                        if reject_degenerate and not math.isfinite(error):
                            candidate.update(numerical_failure(FloatingPointError('nonfinite Newton error'),
                                                               domain=domain))
                        candidates.append(candidate)
                    # Always record all 1..5 legal iterations. Beyond five,
                    # continue only to diagnose the required iteration count.
                    if iteration>=MAX_NEWTON_ITERS and (not math.isfinite(error) or error<=target):break
                required=next((i for i,e in enumerate(history) if math.isfinite(e) and e<=target and positive),None)
                profile['newton_model']=dict(history=history,required_iters=required,
                    required_depth=None if required is None else profile['depth']+(2 if fn=='inv' else 3)*required,
                    reason=('target reached' if required is not None else
                            'outside the convergence region' if not math.isfinite(history[-1]) or not positive
                            else 'target not reached within 64 diagnostic iterations'))
    accepted=[c for c in candidates if c['feasible']]
    choice=min(accepted,key=lambda c:(c['depth'],c['relative_error'],c['method'],c.get('deg',c.get('deg0',0)),c.get('iters',0),c['fit'])) if accepted else None
    settings=({key:choice[key] for key in ('method','deg','deg0','iters','fit') if key in choice}
              if choice is not None else None)
    required=inverse_theory(domain,target) if fn=='inv' else None
    return dict(fn=fn,domain=domain,domain_ratio=domain[1]/domain[0] if domain[0]>0 else None,
                target=target,selected=settings,selected_depth=None if choice is None else choice['depth'],
                status='selected' if choice is not None else 'infeasible',required=required,
                candidates=candidates,grid_note='endpoints, real stationary points and uniform/geometric diagnostic samples')


def select_approximation(fn,domain,*,target=RELATIVE_TARGET,exp_ladder=False,context=None):
    if context is None:
        return _select_approximation(fn,tuple(map(float,domain)),target,exp_ladder)
    import copy
    report=copy.deepcopy(_select_approximation(fn,tuple(map(float,domain)),target,exp_ladder,True))
    for candidate in report['candidates']:
        if candidate.get('reason') not in ('degenerate','nonfinite'):continue
        candidate.update(context,fn=fn,domain=domain,target=target)
        if candidate['method']=='remez':candidate['degree']=candidate['deg']
        else:candidate.pop('degree',None)
    return report


def softmax_adjustments(exp_calibration,inv_calibration):
    lo,hi=(float(exp_calibration[k]) for k in ('min','max'))
    smin,smax=(float(inv_calibration[k]) for k in ('min','max'))
    if smin<=0 or smax<=smin:raise ValueError('positive nondegenerate softmax sum calibration is required')
    shift=hi;scale=smax*math.exp(-shift)
    # Translate the existing 1.5-margin domain, and retain the same margin
    # after the public scale. Neither transform changes the domain ratio.
    return dict(shift=shift,inv_scale=scale,exp_domain=(lo*1.5-shift,hi*1.5-shift),
                inv_domain=(smin/1.5/smax,1.5))


def squared_softmax_adjustments(exp_calibration,key_count):
    lo,hi=(float(exp_calibration[k]) for k in ('min','max'))
    shift=hi;a0,b0=lo*1.5-shift,hi*1.5-shift
    steps=min(max(0,math.ceil(math.log2(b0-a0))),8)
    a,b=a0/(2**steps),b0/(2**steps)
    upper=key_count*math.exp(b)
    return dict(shift=shift,steps=steps,inv_scale=upper,exp_domain=(a,b),
                inv_domain=(math.exp(a)/upper,1.0),step_inv_domain=(1/(1.5*key_count),1.5))


def budget_softmax_adjustments(exp_calibration, key_count, steps, rho):
    """A1 analytical bounds; only exact calibration statistics enter here."""
    if steps not in range(9) or rho not in INV_BUDGETS or key_count < 1:
        raise ValueError('softmax budget requires k in 0..8, a registered rho and positive key count')
    lo,hi=(float(exp_calibration[key]) for key in ('min','max'))
    a,b=(lo*1.5-hi)/(2**steps),(hi*1.5-hi)/(2**steps)
    upper=key_count*math.exp(b)
    repeated=((1-rho)**2/(1.5*key_count),1.5*(1+rho)**2)
    return dict(shift=hi,steps=steps,inv_scale=upper,exp_domain=(a,b),
                inv_domain=(math.exp(a)/upper,1.0),step_inv_domain=repeated,
                final_inv_domain=repeated)


def softmax_budget_candidate(exp_calibration, key_count, steps, rho):
    """Return one feasible bounded configuration, with every fit attempt."""
    adjusted=budget_softmax_adjustments(exp_calibration,key_count,steps,rho)
    roles=[('exp','exp',RELATIVE_TARGET/(2**steps)),
           ('inv','inv',rho if steps else RELATIVE_TARGET)]
    if steps:roles.extend((('step_inv','inv',rho),('final_inv','inv',RELATIVE_TARGET)))
    attrs={key:adjusted[key] for key in ('shift','steps','inv_scale')}
    reports={}
    for role,fn,target in roles:
        domain=adjusted[role+'_domain']
        report=select_approximation(fn,domain,target=target,exp_ladder=fn=='exp',
                                    context=dict(role=role,k=steps,rho=rho))
        reports[role]=report
        if report['selected'] is not None:
            attrs[role+'_domain']=domain
            attrs.update({role+'_'+key:value for key,value in report['selected'].items()
                          if key!='fit' or value!='absolute'})
    if steps:attrs['inv_budget']=rho
    return dict(steps=steps,rho=rho if steps else None,attrs=attrs,reports=reports,
                feasible=all(report['selected'] is not None for report in reports.values()))


def _softmax_latency(attrs, input_shape, mask_shape, *, slots, placement, profile):
    import gc
    from ..cost import estimate
    from ..dsl.parser import parse
    params='x: Cipher['+', '.join(map(str,input_shape))+']'
    if mask_shape is not None:params+=', m: Mask['+', '.join(map(str,mask_shape))+']'
    lines=[f'@fhe.program(slots={slots}, refresh="compiler", placement={json.dumps(placement)})',
           f'def softmax_cost({params}):',
           ' a=pack(x,layout='+json.dumps('score' if len(input_shape)==4 else 'token')+')']
    if mask_shape is not None:lines.append(' am=pack(m,layout="flat")')
    fields=', '.join(key+'='+repr(value) for key,value in attrs.items())
    lines.extend([' y=softmax(a,mask='+('am' if mask_shape is not None else 'None')+', '+fields+')',' return y'])
    latency=estimate(parse('\n'.join(lines)+'\n'),{},profile=profile)['latency_pred_us']
    gc.collect()
    return latency


def select_softmax_budget(exp_calibration, input_shape, *, keep='all', mask_shape=None,
                          rho=0.25, slots=16384, placement='depth-dp', profile='qwen48'):
    """Compare k=0..8 with the normalized static cost model; no samples run."""
    candidates=[]
    for steps in range(9):
        candidate=softmax_budget_candidate(exp_calibration,input_shape[-1],steps,rho)
        if candidate['feasible']:
            candidate['attrs']['keep']=keep
            candidate['latency_pred_us']=_softmax_latency(candidate['attrs'],input_shape,mask_shape,
                                                         slots=slots,placement=placement,profile=profile)
        candidates.append(candidate)
    accepted=[candidate for candidate in candidates if candidate['feasible']]
    selected=min(accepted,key=lambda c:(c['latency_pred_us'],c['steps'])) if accepted else None
    return dict(selected=selected,candidates=candidates)


def plan(graph, calibration, *, slots=16384, placement='depth-dp', softmax_budget=False,
         profile='qwen48', softmax_reports=None):
    names={};used=set(graph.inputs);issues=[]
    def name(v):
        if v not in names:
            stem=re.sub(r'_+','_','v_'+re.sub(r'[^A-Za-z0-9]','_',v)).rstrip('_')
            result=stem;i=2
            while result in used:result=stem+'_'+str(i);i+=1
            names[v]=result;used.add(result)
        return names[v]
    def w(v):return 'W['+json.dumps(v)+']'
    def point(key,role,domain=None,*,fn=None,target=RELATIVE_TARGET,exp_ladder=False):
        if domain is None:
            lo,hi=(float(calibration[key][v]) for v in ('min','max'))
            if lo<=0 or hi<=lo:raise ValueError(f'{key}: positive nondegenerate calibration is required')
            domain=(lo/1.5,hi*1.5)
        report=select_approximation(fn or role,domain,target=target,exp_ladder=exp_ladder)
        if report['selected'] is None:
            issues.append(dict(key=key,report=report))
            return f'{role}_domain={domain!r}'
        fields=dict(report['selected'],domain=domain)
        if fields.get('fit')=='absolute':fields.pop('fit')
        return ', '.join(f'{role}_{field}={json.dumps(value) if isinstance(value,str) else repr(value)}'
                         for field,value in fields.items())
    params=[]
    for v in graph.inputs:
        t=graph.tensors[v];kind='Mask' if t.kind=='mask' else 'Cipher'
        params.append(f'{v}: {kind}[{", ".join(map(str,t.shape))}]')
    lines=[f'@fhe.program(slots={slots}, refresh="compiler", placement={json.dumps(placement)})',
           f'def {graph.name}({", ".join(params)}):']
    for v in graph.inputs:
        shape=graph.tensors[v].shape
        layout='token' if len(shape)==3 else 'flat'
        lines.append(f'    {name(v)} = pack({v}, layout="{layout}")')
    for node in graph.nodes:
        a=node.attrs;out=node.outputs[0];x=name(node.inputs[0]);kind=node.op
        if kind=='linear':
            expr=f'linear({x}, w={w(a["weight"])}, algo="bsgs"'
            if a.get('bias') is not None:expr+=', b='+w(a['bias'])
            expr+=')'
        elif kind in ('layernorm','rmsnorm'):
            args=[x,'eps='+repr(float(a['eps'])),point(out+'::rsqrt','rsqrt')]
            for src,dst in (('weight','w'),('bias','b')):
                if a.get(src) is not None:args.append(dst+'='+w(a[src]))
            expr=kind+'('+', '.join(args)+')'
        elif kind=='softmax':
            key_count=graph.tensors[node.inputs[0]].shape[-1]
            if softmax_budget:
                selection=select_softmax_budget(calibration[out+'::exp'],graph.tensors[node.inputs[0]].shape,
                    keep=a['keep'],mask_shape=graph.tensors[a['mask']].shape if a.get('mask') else None,
                    slots=slots,placement=placement,profile=profile)
                if softmax_reports is not None:softmax_reports[out]=selection
                if selection['selected'] is None:
                    failed=next(report for candidate in selection['candidates']
                                for report in candidate['reports'].values() if report['selected'] is None)
                    raise ApproximationLimitError(out+'::softmax',failed)
                fields=selection['selected']['attrs']
                args=[x,'axis=-1','mask='+(name(a['mask']) if a.get('mask') else 'None')]
                args.extend(key+'='+repr(value) for key,value in fields.items())
                lines.append(f'    {name(out)} = softmax('+', '.join(args)+')')
                continue
            adjusted=squared_softmax_adjustments(calibration[out+'::exp'],key_count)
            args=[x,'axis=-1','keep='+json.dumps(a['keep']),
                  'mask='+(name(a['mask']) if a.get('mask') else 'None'),
                  'shift='+repr(adjusted['shift']),'inv_scale='+repr(adjusted['inv_scale']),
                  'steps='+repr(adjusted['steps']),
                  point(out+'::exp','exp',adjusted['exp_domain'],
                        target=RELATIVE_TARGET/(2**adjusted['steps']),exp_ladder=True),
                  point(out+'::inv','inv',adjusted['inv_domain'])]
            if adjusted['steps']:
                args.append(point(out+'::step_inv','step_inv',adjusted['step_inv_domain'],fn='inv'))
            expr='softmax('+', '.join(args)+')'
        elif kind=='matmul':
            expr='matmul('+', '.join(name(v) for v in node.inputs)+', '+', '.join(
                key+'='+repr(a[key]) for key in ('transpose_b','heads','kv_heads','scale'))+')'
        elif kind in ('add','sub','mul'):
            rhs=name(node.inputs[1]) if len(node.inputs)==2 else w(a['const'])
            expr=f'{kind}({x}, {rhs})'
        elif kind=='act':
            lo,hi=(float(calibration[node.inputs[0]][v])*1.5 for v in ('min','max'))
            expr=f'act({x}, fn={json.dumps(a["fn"])}, method="cheb", deg=127, domain={(lo,hi)!r})'
        elif kind=='shuffle':
            expr=f'shuffle({x}, index={tuple(a["index"])!r}, sign={tuple(a["sign"])!r})'
        elif kind=='select':expr=f'select({x}, {name(node.inputs[1])})'
        elif kind=='flatten':expr=f'flatten({x}, start_dim=1)'
        elif kind=='slice':expr=f'slice({x}, axis={a["axis"]}, start={a["start"]}, end={a["end"]}, step={a["step"]})'
        else:raise ValueError(f'{node.id}: 暂不支持 {kind} in token rule planning')
        lines.append(f'    {name(out)} = {expr}')
    lines.append('    return '+', '.join(name(v) for v in graph.outputs))
    if issues:
        error=ApproximationLimitError(issues[0]['key'],issues[0]['report'])
        error.issues=issues
        raise error
    return '\n'.join(lines)+'\n'
