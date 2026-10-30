"""U4 applies the existing deterministic ladders to each prefixed point."""

import copy
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ..cost import estimate
from ..dsl.printer import print_program
from ..transformer_validation import points,point_names,validate,exact_check
from ..tensor_sim import cached_estimate
from ..kernels import approx
from .tuner import TuneResult


DEFAULT_TUNE_RULE = 'v8-6.11'
PRE_TUNE_RULE = 'v8-pre-6.11'
CONVERGENT_TUNE_RULE = 'v8-6.14'
ACCURACY_TUNE_RULE = 'v8-6.17'
_tune_rule = ContextVar('a1_tune_rule', default=DEFAULT_TUNE_RULE)


def convergent_rule(rule):
    return rule in (CONVERGENT_TUNE_RULE, ACCURACY_TUNE_RULE)


def move_accuracy_ok(result, ctx, rule):
    if rule != ACCURACY_TUNE_RULE:
        return True
    import math
    limit = min(1e-2, ctx.model_info.threshold.get('logit_rel_err', 1e-2))
    error = result.get('logit_rel_err')
    return error is not None and math.isfinite(error) and error <= limit


def validate_tune_rule(prompt_version, rule):
    if rule is None:
        rule = (ACCURACY_TUNE_RULE if prompt_version == 'r8-v10' else
                CONVERGENT_TUNE_RULE if prompt_version in ('r8-v8','r8-v9') else DEFAULT_TUNE_RULE)
    if rule not in (DEFAULT_TUNE_RULE, PRE_TUNE_RULE, CONVERGENT_TUNE_RULE, ACCURACY_TUNE_RULE):
        raise ValueError('Unknown tune_rule: '+str(rule))
    if prompt_version not in ('r8-v8','r8-v9','r8-v10') and rule != DEFAULT_TUNE_RULE:
        raise ValueError('Nondefault tune_rule requires r8-v8')
    return rule


@dataclass
class ConvergentTuneResult(TuneResult):
    """The new rule adds metadata without changing historical TuneResult text."""
    sweeps: int = 0
    converged: bool = False


def local_approximation_cost(attrs, role):
    """Count exactly the single approximation used by program_failures."""
    from ..backends import CountBackend
    from ..kernels import transformer as tr
    backend = CountBackend(8)
    value = tr.apply(backend, backend.input(0), tr.approximation(attrs, role,
                     'exp' if role == 'exp' else 'inv'))
    return dict(depth=value.depth, counts=backend.stats()['counts'])


def dominated_approximation(current, candidate):
    return (candidate['depth'] >= current['depth'] and
            all(candidate['counts'][kind] >= count for kind, count in current['counts'].items()))


@contextmanager
def tune_rule_scope(rule):
    """Select the internal rule without changing tuner dispatch or feedback."""
    validate_tune_rule('r8-v8', rule)
    token = _tune_rule.set(rule)
    try:
        yield
    finally:
        _tune_rule.reset(token)


def softmax_moves(program,ctx,op_index):
    """A1 moves in deterministic order, with calibration-only domain rebuilding."""
    from . import transformer_rule as rule
    op=program.ops[op_index]
    descriptors=points(program);names=point_names(program,ctx)
    key=next(name for p,name in zip(descriptors,names) if p.op is op and p.role=='exp')
    node=next(node for node in ctx.graph.nodes if node.outputs[0]+'::exp'==key)
    shape=ctx.graph.tensors[node.inputs[0]].shape
    calibration=ctx.calibration[key]
    steps=op.attrs.get('steps',0);rho=op.attrs.get('inv_budget') or 0.25
    choices=[(k,rho,'steps') for k in (steps-1,steps+1) if 0<=k<=8]
    position=rule.INV_BUDGETS.index(rho)
    if steps:
        choices.extend((steps,rule.INV_BUDGETS[i],'inv_budget')
                       for i in (position-1,position+1) if 0<=i<len(rule.INV_BUDGETS))
    def replacement(candidate):
        fields=dict(op.attrs)
        for prefix in ('exp','inv','step_inv','final_inv'):
            for name in ('method','domain','deg','degrees','eps','deg0','iters','fit'):
                fields[prefix+'_'+name]=None
        fields['inv_budget']=None
        fields.update(candidate['attrs'])
        return fields
    for k,budget,kind in choices:
        candidate=rule.softmax_budget_candidate(calibration,shape[-1],k,budget)
        if candidate['feasible']:
            yield dict(kind=kind,old=dict(steps=steps,rho=rho),new=dict(steps=k,rho=budget)),replacement(candidate)
        else:
            errors=[item for report in candidate['reports'].values() for item in report['candidates']
                    if item.get('reason') in ('degenerate','nonfinite')]
            yield dict(kind=kind,rejected='no_feasible_fit',steps=k,rho=budget,numerical_errors=errors),None
    domains=rule.budget_softmax_adjustments(calibration,shape[-1],steps,rho)
    bounded=steps and op.attrs.get('final_inv_method') is not None
    roles=[('exp','exp',rule.RELATIVE_TARGET/(2**steps)),
           ('inv','inv',rho if bounded else rule.RELATIVE_TARGET)]
    if steps:roles.append(('step_inv','inv',rho if bounded else rule.RELATIVE_TARGET))
    if bounded:roles.append(('final_inv','inv',rule.RELATIVE_TARGET))
    elif steps:domains['step_inv_domain']=(1/(1.5*shape[-1]),1.5)
    fields=dict(op.attrs)
    fields.update({key:domains[key] for key in ('shift','inv_scale')})
    for role,fn,target in roles:fields[role+'_domain']=domains[role+'_domain']
    for role,fn,target in roles:
        report=rule.select_approximation(fn,domains[role+'_domain'],target=target,exp_ladder=fn=='exp',
                                         context=dict(role=role,k=steps,rho=rho))
        for candidate in report['candidates']:
            if not candidate['feasible']:continue
            updated=dict(fields)
            for name in ('method','deg','degrees','eps','deg0','iters','fit'):
                updated[role+'_'+name]=None
            selected={name:candidate[name] for name in ('method','deg','deg0','iters','fit') if name in candidate}
            updated.update({role+'_'+name:value for name,value in selected.items()})
            if updated==op.attrs:continue
            yield dict(kind='approximation',role=role,attrs=selected),updated


def tune(program,ctx,*,budget_s=2400,max_sweeps=1,engine='tensor',cache=True,engine_profile='mock'):
    from .tuner import TuneResult,_new_settings,LADDER,TARGET_U,N_COARSE
    rule=validate_tune_rule(getattr(ctx,'prompt_version','r8-v4'),_tune_rule.get())
    started=time.monotonic();history=[];changes=[];coarse=full=0
    convergent=convergent_rule(rule)
    sweeps=0;converged=False
    from .. import domain_criteria as criteria
    v8=criteria.enabled(ctx)
    if v8:
        from ..dsl.interface import require_interface
        require_interface(program,ctx.model_info)
    current=copy.deepcopy(program)
    def cost(p):
        from ..margin import require_margin
        require_margin(p, ctx)
        return cached_estimate(p,ctx.weights,ctx.profile,estimate,cache=cache)
    def measure(p, *args, **kwargs):
        # Only the new classification tuning rule needs an extra statistic.
        # Validation's original feasibility and outward Agent feedback stay intact.
        if rule != ACCURACY_TUNE_RULE or ctx.model_info.task != 'classification':
            return validate(p, ctx, *args, **kwargs)
        import numpy as np
        from ..metrics import _relative
        outputs=[]
        result=validate(p, ctx, *args, output_callback=outputs.append, **kwargs)
        if outputs:
            actual=outputs[0]
            result['logit_rel_err']=_relative(ctx.val_exact_logits[:len(actual)],actual,
                                            np.isfinite(actual).reshape(len(actual),-1).all(axis=1))
        return result
    before=cost(current)
    initial=measure(current,engine=engine);full+=1
    if v8:criteria.require_analytic_coverage(initial)
    current_result=initial;current_cost=before
    a1=getattr(ctx,'prompt_version','r8-v4') in ('r8-v7','r8-v8','r8-v9','r8-v10')
    def accept(result):
        if v8:
            criteria.require_analytic_coverage(result)
            return (result['feasible'] and not result['diverged'] and criteria.domain_ok(result)
                    and move_accuracy_ok(result,ctx,rule))
        if a1:
            from ..metrics import passes
            return (result['feasible'] and not result['diverged'] and result['max_u']<=0.90
                    and passes(result,ctx.model_info.threshold))
        return result['feasible'] and result.get('acc_drop_pp',0)<=0.3
    def finish():
        result=measure(current,engine='mock' if v8 else 'tensor' if engine_profile=='tensor' else 'mock');
        if v8:criteria.require_analytic_coverage(result)
        if result['feasible']!=current_result['feasible'] or result['agree']!=current_result['agree']:
            raise RuntimeError('Final Mock verification differs from tensor')
        history.append(dict(phase='result',accepted=result['feasible'],feasible=result['feasible'],
                            latency_pred_us=current_cost['latency_pred_us'],agree=result.get('agree'),
                            total=result.get('total'),max_u=result['max_u']))
        if v8:
            history[-1].update({key:result[key] for key in ('max_u_calibrated','max_u_analytic','worst_acts')})
        if convergent:
            history[-1].update(sweeps=sweeps,converged=converged)
        result_type=ConvergentTuneResult if convergent else TuneResult
        return result_type(current,print_program(current),changes,before,current_cost,coarse,full,
                          time.monotonic()-started,history,result['feasible'],
                          **(dict(sweeps=sweeps,converged=converged) if convergent else {}))
    if initial['diverged'] or not exact_check(current,ctx,engine=engine)['ok']:return finish()
    repair_initial=v8 and rule!=PRE_TUNE_RULE and not initial['feasible']
    aligned=copy.deepcopy(current)
    names=point_names(aligned,ctx)
    alignment_errors=[]
    for index,point in enumerate(points(aligned)):
        # Softmax domains use the registered calibration/analytical rules;
        # held-out observations are used only to judge coverage.
        if point.op.kind=='softmax':continue
        measured=ctx.calibration[names[index]]
        lo,hi=measured['min'],measured['max']
        if lo is None or hi is None:continue
        if getattr(ctx, 'prompt_version', None) in ('r8-v9','r8-v10'):
            from ..margin import minimum_domain
            domain = minimum_domain(lo, hi)
        elif point.attrs['method'] in ('minimax_comp','composite'):
            radius=max(abs(lo),abs(hi))/TARGET_U;domain=(-radius,radius)
        else:
            center,half=(lo+hi)/2,(hi-lo)/2/TARGET_U;domain=(center-half,center+half)
            if point.attrs['fn'] in ('inv','rsqrt'):
                domain=(max(lo/1.5,domain[0]),domain[1])
        if getattr(ctx, 'prompt_version', None) not in ('r8-v9','r8-v10'):
            domain=tuple(round(v,3) for v in domain)
        if domain[0]>=domain[1] or point.attrs['fn'] in ('inv','rsqrt') and domain[0]<=0:
            if repair_initial:
                from .approximation_failures import failure,point_fields
                from ..kernels.layout import KernelError
                alignment_errors.append(failure(KernelError('Calibration alignment produced a degenerate domain'),
                    **dict(point_fields(point),domain=domain)))
            continue
        point.update(dict(domain=domain))
    from ..margin import require_margin
    require_margin(aligned, ctx)
    from .approximation_failures import program_failures
    errors=alignment_errors or (program_failures(aligned) if a1 else [])
    if errors:
        accepted=False
    else:
        aligned_result=measure(aligned,engine=engine);full+=1
        if v8:criteria.require_analytic_coverage(aligned_result)
        aligned_cost=cost(aligned)
        accepted=accept(aligned_result) and (repair_initial or
                  aligned_cost['latency_pred_us']<=current_cost['latency_pred_us'])
    row=dict(phase='domain',accepted=accepted,roles=[p.role or 'act' for p in points(aligned)])
    if errors:row['numerical_errors']=errors
    history.append(row)
    if accepted:current,current_result,current_cost=aligned,aligned_result,aligned_cost
    elif not initial['feasible'] and (not a1 or repair_initial):return finish()
    for sweep in range(min(max_sweeps,8) if convergent else max_sweeps):
        if convergent and time.monotonic()-started>=budget_s:break
        sweeps=sweep+1
        changes_before=len(changes)
        interrupted=False
        visited=set()
        descriptors=points(current)
        for index,point in enumerate(descriptors):
            identity=(point.op.id,'softmax' if a1 and point.op.kind=='softmax' else point.prefix)
            if identity in visited:continue
            visited.add(identity)
            if time.monotonic()-started>=budget_s:
                interrupted=True;break
            if a1:
                point=next(p for p in points(current) if p.op.id==point.op.id and p.prefix==point.prefix)
            if a1 and point.op.kind=='softmax':
                op=point.op;op_index=op.id
                for move,attrs in softmax_moves(current,ctx,op_index):
                    if time.monotonic()-started>=budget_s:
                        interrupted=True;break
                    row=dict(phase='softmax',sweep=sweep,line=op.lineno,move=move,accepted=False)
                    if attrs is None:
                        history.append(row);continue
                    candidate=copy.deepcopy(current)
                    candidate.ops[op_index].attrs.clear();candidate.ops[op_index].attrs.update(attrs)
                    require_margin(candidate, ctx)
                    errors=program_failures(candidate)
                    if errors:
                        row.update(rejected='numerical',numerical_errors=errors);history.append(row);continue
                    if convergent and move['kind']=='approximation':
                        local_current=local_approximation_cost(current.ops[op_index].attrs,move['role'])
                        local_candidate=local_approximation_cost(attrs,move['role'])
                        if dominated_approximation(local_current,local_candidate):
                            row.update(rejected='dominated',local_current=local_current,
                                       local_candidate=local_candidate)
                            history.append(row);continue
                    predicted=cost(candidate);row['latency_pred_us']=predicted['latency_pred_us']
                    if predicted['latency_pred_us']>=current_cost['latency_pred_us']:
                        row['rejected']='cost';history.append(row);continue
                    result=measure(candidate,N_COARSE,engine=engine);coarse+=1
                    row['coarse']=result;accepted=accept(result)
                    if accepted:
                        result=measure(candidate,engine=engine);full+=1;accepted=accept(result)
                        row['full']=result
                    row['accepted']=accepted;history.append(row)
                    if accepted:
                        changes.append(dict(line=op.lineno,call_lines=list(op.call_lines),
                                            role='softmax',old=dict(current.ops[op_index].attrs),new=dict(attrs),move=move))
                        current,current_result,current_cost=candidate,result,predicted
                continue
            if point.attrs['method'] in approx.NEW_METHODS:settings=_new_settings(point.attrs)
            elif point.attrs['method']=='cheb':
                degree=next((d for d in LADDER if d<point.attrs['deg']),None)
                settings=[] if degree is None else [dict(attrs=dict(deg=degree))]
            else:settings=[]
            for setting in settings:
                updates=setting['attrs']
                if updates is None:
                    history.append(dict(phase='skip_search',role=point.role,search=setting['search']));continue
                candidate=copy.deepcopy(current)
                selected=(next(p for p in points(candidate) if (p.op.id,p.prefix)==identity)
                          if a1 else points(candidate)[index])
                selected.update(updates)
                require_margin(candidate, ctx)
                errors=program_failures(candidate) if a1 else []
                if errors:
                    history.append(dict(phase='coarse',role=point.role or 'act',line=point.op.lineno,
                                        attrs=updates,accepted=False,rejected='numerical',numerical_errors=errors))
                    continue
                predicted=cost(candidate)
                if predicted['latency_pred_us']>=current_cost['latency_pred_us']:continue
                result=measure(candidate,N_COARSE,engine=engine);coarse+=1
                accepted=accept(result)
                history.append(dict(phase='coarse',role=point.role or 'act',line=point.op.lineno,
                                    attrs=updates,accepted=accepted,latency_pred_us=predicted['latency_pred_us']))
                if accepted:
                    result=measure(candidate,engine=engine);full+=1;accepted=accept(result)
                if accepted:
                    changes.append(dict(line=point.op.lineno,call_lines=list(point.op.call_lines),
                                        role=point.role or 'act',old=point.attrs,new=updates))
                    current,current_result,current_cost=candidate,result,predicted
        if convergent:
            interrupted=interrupted or time.monotonic()-started>=budget_s
            converged=not interrupted and len(changes)==changes_before
            history.append(dict(phase='sweep',sweeps=sweeps,converged=converged,
                                accepted_changes=len(changes)-changes_before,
                                budget_exhausted=interrupted))
            if converged or interrupted:break
    return finish()
