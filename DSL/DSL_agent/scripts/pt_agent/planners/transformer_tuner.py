"""U4 applies the existing deterministic ladders to each prefixed point."""

import copy
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ..cost import estimate, estimate_fast
from ..dsl.printer import print_program
from ..transformer_validation import points,point_names,validate,exact_check
from ..tensor_sim import cached_estimate, timed, estimate_calls
from ..kernels import approx
from .tuner import TuneResult

validate = timed('validation_s')(validate)


DEFAULT_TUNE_RULE = 'v8-6.11'
PRE_TUNE_RULE = 'v8-pre-6.11'
CONVERGENT_TUNE_RULE = 'v8-6.14'
ACCURACY_TUNE_RULE = 'v8-6.17'
MARGIN_TUNE_RULE = 'v8-6.19'
STRICT_MARGIN_TUNE_RULE = 'v8-6.21'
WIDE_MARGIN_TUNE_RULE = 'v8-6.22'
REFIT_MARGIN_TUNE_RULE = 'v8-6.23'
FLOOR_MARGIN_TUNE_RULE = 'v8-6.24'
ALIGN_FALLBACK_TUNE_RULE = 'v8-6.25'
MARGIN_TARGET_U = 0.80
MARGIN_ACCURACY_FACTOR = 0.5
# §6.19 keeps half of the task logit limit; §6.21 keeps a quarter, with the same alignment.
MARGIN_RULE_FACTORS = {MARGIN_TUNE_RULE: MARGIN_ACCURACY_FACTOR, STRICT_MARGIN_TUNE_RULE: 0.25,
                       WIDE_MARGIN_TUNE_RULE: MARGIN_ACCURACY_FACTOR, REFIT_MARGIN_TUNE_RULE: MARGIN_ACCURACY_FACTOR,
                       FLOOR_MARGIN_TUNE_RULE: MARGIN_ACCURACY_FACTOR, ALIGN_FALLBACK_TUNE_RULE: MARGIN_ACCURACY_FACTOR}
# §6.22 aligns to a 2x calibration margin; the others keep the 1.5x M.
MARGIN_DOMAIN_FACTORS = {WIDE_MARGIN_TUNE_RULE: 2.0, REFIT_MARGIN_TUNE_RULE: 2.0, FLOOR_MARGIN_TUNE_RULE: 2.0,
                         ALIGN_FALLBACK_TUNE_RULE: 2.0}
# §6.23 refits widened points to their original accuracy and judges alignment by accuracy only.
REFIT_RULES = (REFIT_MARGIN_TUNE_RULE, FLOOR_MARGIN_TUNE_RULE, ALIGN_FALLBACK_TUNE_RULE)
# §6.24 keeps each refit setting as the lower bound of later descent on that point.
FLOOR_RULES = (FLOOR_MARGIN_TUNE_RULE, ALIGN_FALLBACK_TUNE_RULE)
# E8 §2.3–§2.4: staged refit beyond the stage-0 ladder and per-point alignment fallback.
EXTENDED_REFIT_RULES = (ALIGN_FALLBACK_TUNE_RULE,)
FALLBACK_RULES = (ALIGN_FALLBACK_TUNE_RULE,)
_tune_rule = ContextVar('a1_tune_rule', default=DEFAULT_TUNE_RULE)


def convergent_rule(rule):
    return rule in (CONVERGENT_TUNE_RULE, ACCURACY_TUNE_RULE, *MARGIN_RULE_FACTORS)


def accuracy_rule(rule):
    return rule in (ACCURACY_TUNE_RULE, *MARGIN_RULE_FACTORS)


def margin_rule(rule):
    return rule in MARGIN_RULE_FACTORS


def move_logit_limit(ctx, rule):
    """Tuner acceptance limit on heldout logit_rel_err; margin rules keep a fraction of the task limit."""
    limit = min(1e-2, ctx.model_info.threshold.get('logit_rel_err', 1e-2))
    return MARGIN_RULE_FACTORS.get(rule, 1.0)*limit


def move_accuracy_ok(result, ctx, rule):
    if not accuracy_rule(rule):
        return True
    import math
    error = result.get('logit_rel_err')
    return error is not None and math.isfinite(error) and error <= move_logit_limit(ctx, rule)


def margin_domain(lo, hi, *, log_scale, target=MARGIN_TARGET_U, factor=1.5):
    """§6.19: hull of M and the interval that places calibration data at u <= target.

    §6.22 passes factor=2 to widen M.

    The scale is the one the u criterion uses for the point: logarithmic for r8-v10
    calibrated inv/rsqrt points with lo > 0, linear otherwise.
    """
    import math
    from ..margin import minimum_domain
    m = minimum_domain(lo, hi, factor)
    if log_scale and lo > 0:
        a, b = math.log(lo), math.log(hi)
        mid, half = (a+b)/2, (b-a)/2/target
        t = (math.exp(mid-half), math.exp(mid+half))
    else:
        center, half = (lo+hi)/2, (hi-lo)/2/target
        t = (center-half, center+half)
    return (min(m[0], t[0]), max(m[1], t[1]))


def refit_floor(record):
    """§6.24: a refit setting (within or closest) bounds later descent; kept points have no floor.

    Only the numeric components bound descent (E8 §2.3: a changed method never enters the floor)."""
    if record.get('status') in ('within', 'closest') and record.get('setting'):
        floor = {key: value for key, value in record['setting'].items()
                 if key in ('deg', 'deg0', 'iters') and value is not None}
        return floor or None
    return None


def below_floor(updates, floor):
    """True when any floored component of the move goes below its refit value."""
    return floor is not None and any(key in floor and value < floor[key] for key, value in updates.items())


def validate_tune_rule(prompt_version, rule):
    if rule is None:
        rule = (ACCURACY_TUNE_RULE if prompt_version in ('r8-v10','r8-v11') else
                CONVERGENT_TUNE_RULE if prompt_version in ('r8-v8','r8-v9') else DEFAULT_TUNE_RULE)
    if rule not in (DEFAULT_TUNE_RULE, PRE_TUNE_RULE, CONVERGENT_TUNE_RULE, ACCURACY_TUNE_RULE, *MARGIN_RULE_FACTORS):
        raise ValueError('Unknown tune_rule: '+str(rule))
    if prompt_version not in ('r8-v8','r8-v9','r8-v10','r8-v11') and rule != DEFAULT_TUNE_RULE:
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


from .transformer_session import validate_tune_search


def _point_cost(attrs):
    from ..backends import CountBackend
    from ..kernels import transformer as tr
    backend=CountBackend(8)
    value=tr.apply(backend,backend.input(0),attrs)
    return dict(depth=value.depth,counts=backend.stats()['counts'])


def _softmax_cost(attrs,width,slots):
    from ..backends import CountBackend
    from ..kernels import transformer as tr
    from ..kernels.layout import TokenLayout,layout_blocks
    backend=CountBackend(slots)
    layout=TokenLayout(slots,1,width)
    inputs=[backend.input(i) for i,_ in enumerate(layout_blocks(layout.token))]
    values,_=tr.softmax(backend,inputs,layout,tr.approximation(attrs,'exp','exp'),
        tr.approximation(attrs,'inv','inv'),shift=attrs.get('shift',0),inv_scale=attrs.get('inv_scale',1),
        steps=attrs.get('steps',0),step_inv=tr.approximation(attrs,'step_inv','inv') if attrs.get('steps') else None,
        final_inv=tr.approximation(attrs,'final_inv','inv') if attrs.get('final_inv_method') else None)
    return dict(depth=max(v.depth for v in values),counts=backend.stats()['counts'])


def build_tune_menu(program,ctx,floors):
    """The sweep's pre-estimation filters, evaluated against one round's program."""
    from .tuner import _new_settings,LADDER
    from .approximation_failures import program_failures
    from ..margin import require_margin
    result=[];visited=set()
    for point in points(program):
        identity=(point.op.id,'softmax' if point.op.kind=='softmax' else point.prefix)
        if identity in visited:continue
        visited.add(identity)
        if point.op.kind=='softmax':
            moves=softmax_moves(program,ctx,point.op.id)
        else:
            settings=(_new_settings(point.attrs) if point.attrs['method'] in approx.NEW_METHODS else
                [dict(attrs=dict(deg=d)) for d in [next((d for d in LADDER if d<point.attrs['deg']),None)] if d is not None]
                if point.attrs['method']=='cheb' else [])
            moves=((dict(kind='approximation',role=point.role or 'act'),setting['attrs'])
                   for setting in settings if setting['attrs'] is not None
                   and not below_floor(setting['attrs'],floors.get(identity)))
        for move,attrs in moves:
            if attrs is None:continue
            candidate=copy.deepcopy(program)
            if point.op.kind=='softmax':
                candidate.ops[point.op.id].attrs.clear();candidate.ops[point.op.id].attrs.update(attrs)
            else:
                next(p for p in points(candidate) if (p.op.id,p.prefix)==identity).update(attrs)
            require_margin(candidate,ctx)
            if program_failures(candidate):continue
            if point.op.kind=='softmax' and move['kind']=='approximation':
                before=local_approximation_cost(point.op.attrs,move['role'])
                after=local_approximation_cost(attrs,move['role'])
                if dominated_approximation(before,after):continue
            elif point.op.kind=='softmax':
                names=point_names(program,ctx)
                name=next(n for p,n in zip(points(program),names) if p.op.id==point.op.id and p.role=='exp')
                node=next(n for n in ctx.graph.nodes if n.outputs[0]+'::exp'==name)
                width=ctx.graph.tensors[node.inputs[0]].shape[-1]
                before=_softmax_cost(point.op.attrs,width,program.header.slots)
                after=_softmax_cost(attrs,width,program.header.slots)
            else:
                before=_point_cost(point.attrs)
                after=_point_cost(next(p for p in points(candidate) if (p.op.id,p.prefix)==identity).attrs)
            updates={k:dict(old=point.op.attrs.get(k),new=v) for k,v in candidate.ops[point.op.id].attrs.items()
                     if v!=point.op.attrs.get(k)}
            # A structural softmax move owns the whole softmax point; an
            # approximation move owns its role (all shared step_inv uses).
            prefix=move.get('role','softmax') if point.op.kind=='softmax' else point.prefix
            result.append(dict(id='m'+str(len(result)+1),line=point.op.lineno,
                call_lines=list(point.op.call_lines),point=dict(op_id=point.op.id,prefix=prefix,
                role=move.get('role') or point.role or 'softmax'),changes=updates,move=move,
                local_before=before,local_after=after,local_delta=dict(depth=after['depth']-before['depth'],
                counts={k:after['counts'][k]-v for k,v in before['counts'].items()})))
    return result


def _structural(item):
    return item['move']['kind'] in ('steps', 'inv_budget')


def _conflicting_points(items):
    for index, item in enumerate(items):
        for other in items[:index]:
            if item['point']['op_id'] != other['point']['op_id']:
                continue
            a, b = _structural(item), _structural(other)
            if a and b:
                return True
            # A structural move rebuilds the role domains; legality is checked
            # against that intermediate program in _compose_moves.
            if a != b:
                continue
            if (item['point']['prefix'] == other['point']['prefix'] or
                    set(item['changes']) & set(other['changes'])):
                return True
    return False


def _compose_moves(base, items, ctx):
    if _conflicting_points(items):
        return None, 'conflict'
    candidate = copy.deepcopy(base)
    structural_ops = {m['point']['op_id'] for m in items if _structural(m)}
    for item in sorted(items, key=lambda m: not _structural(m)):
        op_id = item['point']['op_id']
        if op_id in structural_ops and not _structural(item):
            matches = (attrs for move, attrs in softmax_moves(candidate, ctx, op_id)
                       if move.get('kind') == 'approximation' and
                       move.get('role') == item['move'].get('role') and
                       move.get('attrs') == item['move'].get('attrs') and attrs is not None)
            attrs = next(matches, None)
            if attrs is None:
                return None, 'not_legal_after_structural'
            candidate.ops[op_id].attrs.clear()
            candidate.ops[op_id].attrs.update(attrs)
        else:
            candidate.ops[op_id].attrs.update({k: v['new'] for k, v in item['changes'].items()})
    return candidate, None


def parse_tune_reply(reply,menu):
    """Read the first JSON object; malformed responses consume their one call."""
    if reply.get('finish_reason')=='length':return [],'length'
    text=reply.get('content','');start=text.find('{')
    try:
        obj,_=json.JSONDecoder().raw_decode(text[start:]) if start>=0 else (None,0)
    except (ValueError,TypeError):return [],'invalid_json'
    if not isinstance(obj,dict) or not isinstance(obj.get('candidates'),list):return [],'invalid_json'
    lookup={m['id']:m for m in menu};result=[]
    for candidate in obj['candidates'][:8]:
        ids=candidate.get('moves') if isinstance(candidate,dict) else None
        row=dict(moves=ids,note=candidate.get('note','') if isinstance(candidate,dict) else '',accepted=False)
        if not isinstance(ids,list) or not 1<=len(ids)<=4 or any(not isinstance(i,str) for i in ids):
            row['rejected']='invalid_moves'
        elif any(i not in lookup for i in ids):row['rejected']='unknown_id'
        elif _conflicting_points([lookup[i] for i in ids]):row['rejected']='conflict'
        result.append(row)
    return result,None


def _tune_context(program,ctx,cost,result,floors,rule):
    cost_fields={key:cost.get(key) for key in ('latency_pred_us','boots_pred')}
    plan=cost.get('refresh_plan') or {}
    cost_fields['placement_variant']=plan.get('selected_variant',plan.get('variant'))
    remaining={(r['line'],tuple(r['call_lines'])):r['mean'] for r in plan.get('levels_left_at_refresh',[])}
    cost_fields['hotspots']=[dict({k:r.get(k) for k in ('line','call_lines','latency_share')},
        levels_left_at_refresh=remaining.get((r['line'],tuple(r['call_lines'])))) for r in cost.get('hotspots',[])]
    rows=result.get('rows') or []
    precision={k:result.get(k) for k in ('agree','total','logit_rel_err','max_u_calibrated','max_u_analytic')}
    precision['points']=[dict(op_id=p.op.id,prefix=p.prefix,role=p.role or 'act',line=p.op.lineno,
        max_u=rows[i].get('max_u') if i<len(rows) else None,refit_floor=floors.get((p.op.id,p.prefix)))
        for i,p in enumerate(points(program))]
    return dict(program='\n'.join(f'{i}: {line}' for i,line in enumerate(print_program(program).splitlines(),1)),
        cost=cost_fields,precision=precision,acceptance=dict(max_u_calibrated=.90,max_u_analytic=1.,
        logit_rel_err=move_logit_limit(ctx,rule)))


def tune(program,ctx,*,budget_s=2400,max_sweeps=1,engine='tensor',cache=True,engine_profile='mock',
         client=None,iteration=0,align_points=None):
    """`align_points` (E8a §3.1 attribution only) restricts the alignment to these point indices and skips the fallback."""
    from .tuner import TuneResult,_new_settings,LADDER,TARGET_U,N_COARSE
    rule=validate_tune_rule(getattr(ctx,'prompt_version','r8-v4'),_tune_rule.get())
    search=validate_tune_search(ctx)
    if search=='llm-select' and client is None:raise ValueError('llm-select requires a tuning client')
    started=time.monotonic();history=[];changes=[];accepted_states=[];coarse=full=0
    estimates_before=estimate_calls();llm_calls=0;llm_s=0.
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
        estimator = estimate_fast if getattr(ctx, 'placement', None) else estimate
        return cached_estimate(p,ctx.weights,ctx.profile,estimator,cache=cache)
    def measure(p, *args, **kwargs):
        # Only the new classification tuning rule needs an extra statistic.
        # Validation's original feasibility and outward Agent feedback stay intact.
        if not accuracy_rule(rule) or ctx.model_info.task != 'classification':
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
    a1=getattr(ctx,'prompt_version','r8-v4') in ('r8-v7','r8-v8','r8-v9','r8-v10','r8-v11')
    def reject_reasons(result):
        """E8 §2.2: the failed acceptance terms; accept(result) holds exactly when this is empty."""
        reasons=[]
        if v8:
            criteria.require_analytic_coverage(result)
            if not result['feasible']:reasons.append('infeasible')
            if result['diverged']:reasons.append('diverged')
            if (result.get('max_u_calibrated') is None or result.get('max_u_analytic') is None
                    or not criteria.domain_ok(result)):reasons.append('domain')
            if not move_accuracy_ok(result,ctx,rule):reasons.append('accuracy')
            return reasons
        if a1:
            from ..metrics import passes
            if not result['feasible']:reasons.append('infeasible')
            if result['diverged']:reasons.append('diverged')
            if not result['max_u']<=0.90:reasons.append('domain')
            if not passes(result,ctx.model_info.threshold):reasons.append('accuracy')
            return reasons
        if not result['feasible']:reasons.append('infeasible')
        if not result.get('acc_drop_pp',0)<=0.3:reasons.append('accuracy')
        return reasons
    def accept(result):
        return not reject_reasons(result)
    def alignment_metrics(result):
        if result is None:return None
        metrics={key:result.get(key) for key in ('feasible','diverged','agree','total','agreement_pct','logit_rel_err',
                                                  'max_u_calibrated','max_u_analytic')}
        metrics['move_logit_limit']=move_logit_limit(ctx,rule) if v8 and accuracy_rule(rule) else None
        metrics['worst_acts']=list(result.get('worst_acts') or [])[:5]
        return metrics
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
                          search=search,llm_calls=llm_calls,llm_s=llm_s,estimates=estimate_calls()-estimates_before,
                          accepted_states=accepted_states,
                          **(dict(sweeps=sweeps,converged=converged) if convergent else {}))
    if initial['diverged'] or not exact_check(current,ctx,engine=engine)['ok']:return finish()
    repair_initial=v8 and rule!=PRE_TUNE_RULE and not initial['feasible']
    names=point_names(current,ctx)
    def align_point(program,index):
        """Align point `index` of `program` in place (the §6.19–§6.24 rule; E8 R1 under v8-6.25)."""
        point=points(program)[index]
        # Softmax domains use the registered calibration/analytical rules;
        # held-out observations are used only to judge coverage.
        if point.op.kind=='softmax':return None
        measured=ctx.calibration[names[index]]
        lo,hi=measured['min'],measured['max']
        if lo is None or hi is None:return None
        if margin_rule(rule):
            domain = margin_domain(lo, hi, log_scale=getattr(ctx, 'prompt_version', None) in ('r8-v10','r8-v11')
                                   and point.attrs['fn'] in ('inv', 'rsqrt'),
                                   factor=MARGIN_DOMAIN_FACTORS.get(rule, 1.5))
        elif getattr(ctx, 'prompt_version', None) in ('r8-v9','r8-v10','r8-v11'):
            from ..margin import minimum_domain
            domain = minimum_domain(lo, hi)
        elif point.attrs['method'] in ('minimax_comp','composite'):
            radius=max(abs(lo),abs(hi))/TARGET_U;domain=(-radius,radius)
        else:
            center,half=(lo+hi)/2,(hi-lo)/2/TARGET_U;domain=(center-half,center+half)
            if point.attrs['fn'] in ('inv','rsqrt'):
                domain=(max(lo/1.5,domain[0]),domain[1])
        if getattr(ctx, 'prompt_version', None) not in ('r8-v9','r8-v10','r8-v11'):
            domain=tuple(round(v,3) for v in domain)
        if domain[0]>=domain[1] or point.attrs['fn'] in ('inv','rsqrt') and domain[0]<=0:
            return dict(degenerate=True,point=point,domain=domain)
        old_attrs=dict(point.attrs)
        point.update(dict(domain=domain))
        info=dict(degenerate=False,point=point,domain=domain,changed=tuple(domain)!=tuple(old_attrs['domain']))
        if rule in REFIT_RULES and (domain[0]<old_attrs['domain'][0] or domain[1]>old_attrs['domain'][1]):
            from .margin_refit import refit
            setting,record=refit(old_attrs,domain,extended=rule in EXTENDED_REFIT_RULES)
            point.update(setting)
            info['refit']=dict(record,line=point.op.lineno,role=point.role or 'act',fn=point.attrs['fn'],
                               method=point.attrs['method'],old_setting={k:old_attrs.get(k) for k in setting})
            floor=refit_floor(record) if rule in FLOOR_RULES else None
            if floor is not None:
                info['floor']=((point.op.id,point.prefix),dict(line=point.op.lineno,role=point.role or 'act',setting=floor))
        return info
    from ..margin import require_margin
    from .approximation_failures import program_failures
    aligned=copy.deepcopy(current)
    alignment_errors=[]
    refits=[]
    candidate_floors={}
    for index in range(len(points(aligned))):
        if align_points is not None and index not in align_points:continue
        info=align_point(aligned,index)
        if info is None:continue
        if info['degenerate']:
            if repair_initial:
                from .approximation_failures import failure,point_fields
                from ..kernels.layout import KernelError
                alignment_errors.append(failure(KernelError('Calibration alignment produced a degenerate domain'),
                    **dict(point_fields(info['point']),domain=info['domain'])))
            continue
        if 'refit' in info:refits.append(info['refit'])
        if 'floor' in info:candidate_floors[info['floor'][0]]=info['floor'][1]
    require_margin(aligned, ctx)
    errors=alignment_errors or (program_failures(aligned) if a1 else [])
    aligned_result=None
    if errors:
        accepted=False
        reasons=['alignment_error' if alignment_errors else 'numerical_failure']
    else:
        aligned_result=measure(aligned,engine=engine);full+=1
        if v8:criteria.require_analytic_coverage(aligned_result)
        aligned_cost=cost(aligned)
        reasons=reject_reasons(aligned_result)
        if not (repair_initial or rule in REFIT_RULES or
                aligned_cost['latency_pred_us']<=current_cost['latency_pred_us']):reasons.append('cost')
        accepted=not reasons
    row=dict(phase='domain',accepted=accepted,roles=[p.role or 'act' for p in points(aligned)])
    if refits:row['refits']=refits
    # Floors bound descent only from the accepted aligned program.
    floors={key:value['setting'] for key,value in candidate_floors.items()} if accepted else {}
    if floors:row['floors']=list(candidate_floors.values())
    if errors:row['numerical_errors']=errors
    row['reasons']=reasons
    if aligned_result is not None:row['metrics']=alignment_metrics(aligned_result)
    fallback_program=None
    if not accepted and rule in FALLBACK_RULES and not repair_initial and initial['feasible'] and align_points is None:
        # E8 §2.4: widen one point at a time, in program order, keeping each accepted point.
        accumulated,accumulated_result,point_floors,entries=current,None,{},[]
        for index in range(len(points(current))):
            trial=copy.deepcopy(accumulated)
            point_started=time.monotonic()
            info=align_point(trial,index)
            if info is None or info['degenerate'] or not info['changed']:continue
            point=info['point']
            entry=dict(line=point.op.lineno,role=point.role or 'act',fn=point.attrs['fn'])
            require_margin(trial,ctx)
            result=None
            if a1 and program_failures(trial):
                entry_reasons=['numerical_failure']
            else:
                result=measure(trial,engine=engine);full+=1
                if v8:criteria.require_analytic_coverage(result)
                entry_reasons=reject_reasons(result)
            entry.update(accepted=not entry_reasons,reasons=entry_reasons,metrics=alignment_metrics(result),
                         refit=info.get('refit'),seconds=time.monotonic()-point_started)
            entries.append(entry)
            if not entry_reasons:
                accumulated,accumulated_result=trial,result
                if 'floor' in info:point_floors[info['floor'][0]]=info['floor'][1]
        row['fallback']=entries
        row['accepted_lines']=[e['line'] for e in entries if e['accepted']]
        row['fallback_accepted']=bool(row['accepted_lines'])
        if row['fallback_accepted']:
            fallback_program=(accumulated,accumulated_result)
            floors={key:value['setting'] for key,value in point_floors.items()}
            if floors:row['fallback_floors']=list(point_floors.values())
    history.append(row)
    if accepted:current,current_result,current_cost=aligned,aligned_result,aligned_cost
    elif fallback_program is not None:
        current,current_result=fallback_program;current_cost=cost(current)
    elif not initial['feasible'] and (not a1 or repair_initial):return finish()
    if search=='llm-select':
        from ..llm import LLMError
        from .prompts import build_tune_prompt
        empty_rounds=0
        for round_number in range(1,5):
            if time.monotonic()-started>=budget_s:break
            if hasattr(client,'calls') and hasattr(client,'limit') and client.calls>=client.limit:break
            menu=build_tune_menu(current,ctx,floors)
            if not menu or time.monotonic()-started>=budget_s:break
            base=current
            baseline_cost=current_cost['latency_pred_us']
            round_row=dict(phase='llm_round',round=round_number,menu_size=len(menu),menu=menu,
                           candidates=[],tokens={},llm_s=0.,finish_reason=None)
            messages=build_tune_prompt(_tune_context(base,ctx,current_cost,current_result,floors,rule),menu,
                                      [{k:r[k] for k in ('round','candidates','finish_reason')}
                                       for r in history if r['phase']=='llm_round'])
            call_started=time.monotonic();llm_calls+=1
            try:reply=client.chat(messages,purpose=f'tune-select-iter{iteration}-round{round_number}')
            except LLMError as exc:
                round_row.update(error=exc.code,llm_s=time.monotonic()-call_started)
                llm_s+=round_row['llm_s'];history.append(round_row)
                if exc.code=='budget_exhausted':break
                raise
            round_row.update(tokens=reply.get('usage') or {},llm_s=reply.get('seconds',time.monotonic()-call_started),
                             finish_reason=reply.get('finish_reason'))
            llm_s+=round_row['llm_s']
            candidates,parse_error=parse_tune_reply(reply,menu)
            round_row.update(candidates=candidates,parse_error=parse_error)
            lookup={m['id']:m for m in menu};alternatives=[];accepted_count=0;merge_attempts=0
            def expired():return time.monotonic()-started>=budget_s
            def not_reached(row, reason):row.update(rejected='not_reached',reason=reason)
            # Estimate every candidate against the same round baseline before
            # deciding which alternative receives the first numerical check.
            for index,row in enumerate(candidates):
                if row.get('rejected'):continue
                if expired():not_reached(row,'budget');continue
                items=[lookup[i] for i in row['moves']]
                candidate,error=_compose_moves(base,items,ctx)
                if error:row['rejected']=error;continue
                require_margin(candidate,ctx)
                errors=program_failures(candidate)
                if errors:row.update(rejected='numerical',numerical_errors=errors);continue
                row['evaluated']=True
                predicted=cost(candidate);row['latency_pred_us']=predicted['latency_pred_us']
                if predicted['latency_pred_us']>=baseline_cost:row['rejected']='cost';continue
                alternatives.append((predicted['latency_pred_us'],index,items,predicted))
            alternatives.sort(key=lambda item:(item[0],item[1]))
            accepted_items=[]
            def verify(candidate,row):
                nonlocal coarse,full
                if expired():not_reached(row,'budget');return None
                result=measure(candidate,N_COARSE,engine=engine);coarse+=1;row['coarse']=result
                if not accept(result):row['rejected']='coarse';return None
                if expired():not_reached(row,'budget');return None
                result=measure(candidate,engine=engine);full+=1;row['full']=result
                if not accept(result):row['rejected']='full';return None
                return result
            def record_changes(previous,candidate,items):
                for m in items:
                    op_id=m['point']['op_id']
                    updates={k:dict(old=previous.ops[op_id].attrs.get(k),new=v)
                             for k,v in candidate.ops[op_id].attrs.items()
                             if previous.ops[op_id].attrs.get(k)!=v}
                    changes.append(dict(line=m['line'],call_lines=m['call_lines'],role=m['point']['role'],
                        old={k:v['old'] for k,v in updates.items()},new={k:v['new'] for k,v in updates.items()},
                        move=m['move'],menu_id=m['id'],round=round_number))
            selected_index=None
            for _,index,items,predicted in alternatives:
                row=candidates[index]
                candidate,error=_compose_moves(base,items,ctx)
                result=verify(candidate,row)
                if result is None:continue
                row.update(accepted=True,merged=False)
                record_changes(current,candidate,items)
                accepted_items=items;selected_index=index
                current,current_result,current_cost=candidate,result,predicted
                accepted_count+=1
                break
            # Revisit every remaining alternative, including an alternative
            # whose standalone validation failed before the selected one.
            if accepted_items:
                for _,index,items,_ in alternatives:
                    if index==selected_index:continue
                    row=candidates[index]
                    if _conflicting_points(accepted_items+items):
                        if not row.get('rejected'):not_reached(row,'existing_better_candidate')
                        continue
                    if expired():
                        if not row.get('rejected'):not_reached(row,'budget')
                        continue
                    if merge_attempts>=3:
                        if not row.get('rejected'):not_reached(row,'merge_limit')
                        continue
                    merge_attempts+=1
                    target=dict(accepted=False);row['merge']=target
                    candidate,error=_compose_moves(base,accepted_items+items,ctx)
                    if error:target['rejected']=error;continue
                    require_margin(candidate,ctx)
                    errors=program_failures(candidate)
                    if errors:target.update(rejected='numerical',numerical_errors=errors);continue
                    predicted=cost(candidate);target.update(evaluated=True,latency_pred_us=predicted['latency_pred_us'])
                    if predicted['latency_pred_us']>=current_cost['latency_pred_us']:
                        target['rejected']='cost';continue
                    result=verify(candidate,target)
                    if result is None:continue
                    target['accepted']=True;row.update(accepted=True,merged=True)
                    record_changes(current,candidate,items)
                    accepted_items+=items
                    current,current_result,current_cost=candidate,result,predicted
                    accepted_count+=1
            round_row.update(accepted_candidates=accepted_count,merge_attempts=merge_attempts,
                             merge_accepted=sum(bool(r.get('merge',{}).get('accepted')) for r in candidates))
            history.append(round_row);sweeps=round_number
            if accepted_count:
                accepted_states.append(dict(source=print_program(current),cost=copy.deepcopy(current_cost),round=round_number))
            empty_rounds=0 if accepted_count else empty_rounds+1
            if empty_rounds>=2:converged=True;break
        return finish()
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
                if below_floor(updates,floors.get(identity)):
                    history.append(dict(phase='coarse',role=point.role or 'act',line=point.op.lineno,
                                        attrs=updates,accepted=False,rejected='refit_floor'))
                    continue
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
        if len(changes)>changes_before:
            accepted_states.append(dict(source=print_program(current),cost=copy.deepcopy(current_cost),round=sweep+1))
        if convergent:
            interrupted=interrupted or time.monotonic()-started>=budget_s
            converged=not interrupted and len(changes)==changes_before
            history.append(dict(phase='sweep',sweeps=sweeps,converged=converged,
                                accepted_changes=len(changes)-changes_before,
                                budget_exhausted=interrupted))
            if converged or interrupted:break
    return finish()
