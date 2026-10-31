"""Calibration/held-out feedback for explicit U4 approximation points."""

from dataclasses import dataclass
import gc
import numpy as np

from .analyzer import reference_forward,approximation_points,approximation_values
from .kernels import transformer as tr
from .backends import MockBackend,BackendError
from .lower import lower,pack_inputs,decode_outputs
from .tensor_sim import run_tensor
from .metrics import compare,passes


@dataclass
class Point:
    op: object
    role: str
    key: str
    attrs: dict

    @property
    def prefix(self):
        return self.role.split('[',1)[0]

    def update(self,fields):
        self.attrs.update(fields)
        self.op.attrs.update({self.prefix+'_'+key if self.role else key:value for key,value in fields.items()})


def points(program):
    result=[]
    for op in program.ops:
        if op.kind=='act':result.append(Point(op,'',op.inputs[0],dict(op.attrs)))
        elif op.kind in ('layernorm','rmsnorm'):
            result.append(Point(op,'rsqrt',op.outputs[0]+'::rsqrt',tr.approximation(op.attrs,'rsqrt','rsqrt')))
        elif op.kind=='softmax':
            for role in ('exp','inv'):
                result.append(Point(op,role,op.outputs[0]+'::'+role,tr.approximation(op.attrs,role,role)))
            for index in range(op.attrs.get('steps',0)):
                last=op.attrs.get('final_inv_method') is not None and index==op.attrs['steps']-1
                role='final_inv' if last else f'step_inv[{index}]'
                prefix='final_inv' if last else 'step_inv'
                result.append(Point(op,role,op.outputs[0]+'::'+role,tr.approximation(op.attrs,prefix,'inv')))
    return result


def point_names(program,ctx):
    original=iter(ctx.act_tensors);softmax_names={};result=[]
    for point in points(program):
        if point.prefix in ('step_inv','final_inv'):
            name=softmax_names[point.op.id]+'::'+point.role
        else:
            name=next(original)
            if point.op.kind=='softmax':softmax_names[point.op.id]=name.rsplit('::',1)[0]
        result.append(name)
    return result


def context(info,calibration,*,samples,n_val=1000,normalization=None):
    from .planners.agent_tools import AgentContext,VAL_BATCH
    graph=info.graph;inputs,labels=samples.get('heldout',n_val)
    names=[key for node,key in approximation_points(graph)]
    needed=dict.fromkeys(names,0.0);probes=[];logits=[]
    for start in range(0,len(inputs[graph.inputs[0]]),VAL_BATCH):
        arrays={k:v[start:start+VAL_BATCH] for k,v in inputs.items()}
        values=reference_forward(graph,arrays,keep=True)
        logits.append(values[graph.outputs[0]])
        for key,x in approximation_values(graph,values).items():
            lo,hi=calibration[key]['min'],calibration[key]['max']
            ratios=[float(np.max(x))/hi if hi else 1, float(np.min(x))/lo if lo else 1]
            needed[key]=max(needed[key],*ratios)
    probe={k:v[:4] for k,v in inputs.items()}
    probe_values=approximation_values(graph,reference_forward(graph,probe,keep=True))
    probes=[probe_values[key] for key in names]
    exact=np.concatenate(logits)
    return AgentContext(graph=graph,weights=graph.weights,calibration=calibration,
        weight_shapes={k:v.shape for k,v in graph.weights.items()},val_images=inputs[graph.inputs[0]],
        val_exact_logits=exact,val_labels=labels,val_exact_top1=None,probe_images=inputs[graph.inputs[0]][:4],
        probe_act_inputs=probes,needed_margin=needed,act_tensors=names,model_name=graph.name,
        normalization=normalization,model_info=info,samples=samples,val_inputs=inputs,val_split='heldout')


def execute(program,weights,inputs,*,engine='tensor',exact=False,observer=None):
    if engine=='tensor':
        return run_tensor(program,weights,inputs,observer=observer,act_mode='exact' if exact else 'program')
    from .planners.agent_tools import _unpack
    backend=MockBackend(program.header.slots,batch=len(next(iter(inputs.values()))))
    try:
        backend.bind(pack_inputs(program,[inputs[p.name] for p in program.params]))
        def observe(name,value,layout,cfg):
            if observer is not None:observer(name,_unpack(value,layout),cfg)
        if observer is not None and hasattr(observer, 'sample_event'):
            observe.sample_event = observer.sample_event
        if observer is not None and hasattr(observer, 'inverse_event'):
            observe.inverse_event = observer.inverse_event
        values,layouts=lower(program,weights,backend,act_mode='exact' if exact else 'program',observer=observe if observer else None)
        return decode_outputs([v.value for v in values],layouts)[0]
    finally:
        backend.release_batch()


def exact_check(program,ctx,*,engine='mock'):
    from .margin import require_margin
    require_margin(program, ctx)
    inputs={k:v[:4] for k,v in ctx.val_inputs.items()}
    observations=[]
    actual=execute(program,ctx.weights,inputs,engine=engine,exact=True,
                   observer=lambda name,value,cfg:observations.append(value))
    expected=ctx.val_exact_logits[:4]
    diff=float(np.max(np.abs(actual-expected))) if actual.shape==expected.shape else None
    first=None
    descriptors=points(program)
    reference=ctx.probe_act_inputs
    staged=any(p.prefix in ('step_inv','final_inv') for p in descriptors)
    if staged:
        reference=[]
        execute(program,ctx.weights,inputs,engine='tensor',exact=True,
                observer=lambda name,value,cfg:reference.append(value))
    names=point_names(program,ctx)
    for i,(a,b) in enumerate(zip(observations,reference)):
        point=descriptors[i]
        if point.op.kind=='softmax' and not staged:
            shift=point.op.attrs.get('shift',0.0);scale=point.op.attrs.get('inv_scale',1.0)
            if point.role=='exp' and shift!=0:b=np.asarray(b)-shift
            if point.role=='inv' and (shift!=0 or scale!=1):b=np.asarray(b)*np.exp(-shift)/scale
        a=np.sort(np.asarray(a).reshape(-1));b=np.sort(np.asarray(b).reshape(-1))
        difference=float(np.max(np.abs(a-b))) if a.shape==b.shape else None
        if difference is None or difference>1e-9:
            first=dict(act_index=i,graph_tensor=names[i],program_line=descriptors[i].op.lineno,max_diff=difference)
            break
    return dict(ok=diff is not None and diff<=1e-9 and first is None and len(observations)==len(reference),
                logit_max_diff=diff,first_mismatch=first,act_count=[len(observations),len(reference)])


def add_required_logit_error(measured,threshold,exact,approx):
    """A1 §6.20: a threshold that bounds logit_rel_err gets the value for every task.

    compare() reports it only for causal_lm; the formula is the same one the report uses.
    Thresholds without the key keep their original measured fields.
    """
    if 'logit_rel_err' not in threshold or 'logit_rel_err' in measured:
        return measured
    from .metrics import _relative
    exact,approx=np.asarray(exact),np.asarray(approx)
    valid=np.isfinite(approx).reshape(len(approx),-1).all(axis=1)
    measured['logit_rel_err']=_relative(exact,approx,valid)
    return measured


def validate(program,ctx,n=None,*,engine='tensor',stop_on_divergence=True,output_callback=None,
             diagnostic_callback=None,batch_callback=None,**kwargs):
    from .planners.agent_tools import VAL_BATCH,MOCK_VAL_BATCH,FEASIBLE_MAX_U
    total=len(ctx.val_images) if n is None else min(n,len(ctx.val_images))
    roles=points(program)
    from . import domain_criteria as criteria
    v8 = criteria.enabled(ctx)
    if v8:
        from .dsl.interface import require_interface
        require_interface(program, ctx.model_info)
    from .margin import require_margin
    require_margin(program, ctx)
    names=point_names(program,ctx)
    rows=[dict(program_line=p.op.lineno,graph_tensor=names[i],role=p.role or 'act',
               method=p.attrs['method'],deg=p.attrs.get('deg'),degrees=p.attrs.get('degrees'),
               domain=p.attrs['domain'],obs_min=None,obs_max=None,max_u=0.0,
               **({key:p.attrs[key] for key in ('deg0','iters')} if p.attrs['method']=='newton' else {}))
          for i,p in enumerate(roles)]
    if v8:
        for row, source in zip(rows, criteria.provenance(program, ctx, roles)):
            row.update(source)
    violations = criteria.ViolationOrder(roles, rows, total, version=ctx.prompt_version) if v8 else None
    outputs=np.full_like(ctx.val_exact_logits[:total],np.nan)
    diverged=[];done=0;error=None;numerical=[]
    a1=getattr(ctx,'prompt_version','r8-v4') in ('r8-v7', 'r8-v8')
    from .planners.approximation_failures import NUMERICAL_ERRORS,failure,point_fields
    batch_size=min(VAL_BATCH,MOCK_VAL_BATCH) if engine=='mock' else VAL_BATCH
    for start in range(0,total,batch_size):
        stop=min(start+batch_size,total)
        batch={k:v[start:stop] for k,v in ctx.val_inputs.items()}
        observed=0
        def observer(name,value,cfg):
            nonlocal observed
            row=rows[observed];observed+=1
            lo,hi=row['domain'];minimum=float(value.min());maximum=float(value.max())
            row['obs_min']=minimum if row['obs_min'] is None else min(row['obs_min'],minimum)
            row['obs_max']=maximum if row['obs_max'] is None else max(row['obs_max'],maximum)
            row['max_u']=max(row['max_u'],criteria.maximum_u(value, (lo,hi),
                fn=roles[observed-1].attrs.get('fn'), domain_source=row.get('domain_source','calibrated'),
                version=getattr(ctx,'prompt_version','r8-v4')))
        if violations is not None:
            def sample_event(op_id, role, value, mask=None):
                violations.observe(op_id, role, value, mask, start=start)
            observer.sample_event = sample_event
            def inverse_event(op_id, role, value, inverse):
                violations.inverse_result(op_id, role, value, inverse, start=start)
            observer.inverse_event = inverse_event
        try:outputs[start:stop]=execute(program,ctx.weights,batch,engine=engine,observer=observer)
        except (BackendError,*NUMERICAL_ERRORS) as exc:
            if not a1 and not isinstance(exc,(BackendError,FloatingPointError)):raise
            error=str(exc);diverged.extend(range(start,stop))
            if a1 and roles and (isinstance(exc,NUMERICAL_ERRORS) or 'nonfinite' in str(exc).lower()):
                fields=point_fields(roles[max(0,observed-1)])
                numerical.append(failure(exc,**fields))
            if stop_on_divergence:break
        finally:
            if engine=='mock':gc.collect()
        done=stop
        if batch_callback is not None:batch_callback(observer)
    info=ctx.model_info
    metric=dict(values=ctx.val_exact_logits[:total],task=info.task)
    if info.task=='causal_lm':
        metric.update(lengths=ctx.val_inputs['attention_mask'][:total].sum(1).astype(int),
                      selected_last=info.input_handling.get('single_lm_head',False))
    measured=compare(metric,outputs,None if ctx.val_labels is None else ctx.val_labels[:total])
    add_required_logit_error(measured,info.threshold,ctx.val_exact_logits[:total],outputs)
    max_u=max((r['max_u'] for r in rows),default=0.0)
    result=dict(measured,total=total,rows=rows,max_u=max_u,diverged=bool(diverged),n_diverged=len(diverged),
                diverged_indices=diverged,images_done=done,boot_input_max_abs=None,
                boot_input_bound=None,worst_acts=sorted(rows,key=lambda r:-r['max_u'])[:5],
                feasible=not diverged and max_u<=FEASIBLE_MAX_U and passes(measured,info.threshold))
    if v8:
        result.update(criteria.summarize(rows))
        result['analytic_domain_violation'] = violations.analytic_domain_violation
        result['feasible'] = not diverged and criteria.domain_ok(result) and passes(measured, info.threshold)
        result['worst_acts'] = sorted(rows, key=lambda r: (r['max_u'] <= r['u_limit'], -r['max_u']))[:5]
        budget = violations.budget_feedback()
        if budget:
            result['budget_violations'] = budget
    if error:result['error']=error
    if numerical:result['numerical_errors']=numerical
    if output_callback is not None:output_callback(outputs.copy())
    if violations is not None:
        criteria.record_violations(violations.report(engine=engine, split=ctx.val_split), diagnostic_callback)
    gc.collect()
    return result
