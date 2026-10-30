"""U4 §11 finite point diagnosis, using calibration and held-out samples only."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from ..paths import REPO, RESULTS_DIR


TARGET = 1e-3
MODEL_ITERATIONS = 64
GRID_POINTS = 2**16+1


def write(path, value):
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return 'nan' if math.isnan(item) else 'inf' if item > 0 else '-inf'
        if isinstance(item, dict): return {key: clean(v) for key, v in item.items()}
        if isinstance(item, (tuple, list)): return [clean(v) for v in item]
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def distribution(value, domain=None):
    import numpy as np
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result = dict(count=int(array.size), nonfinite=int(array.size-finite.size),
                  quantiles=None, minimum=None, maximum=None, ratio=None)
    if finite.size:
        q = np.percentile(finite, (0, 1, 50, 99, 100))
        result.update(quantiles=dict(zip(('p0', 'p1', 'p50', 'p99', 'p100'), map(float, q))),
                      minimum=float(q[0]), maximum=float(q[-1]),
                      ratio=float(q[-1]/q[0]) if q[0] > 0 else None)
    if domain is not None:
        lo, hi = domain
        result.update(domain=list(domain), outside=int(np.count_nonzero(~np.isfinite(array) | (array < lo) | (array > hi))),
                      max_u=float(np.max(np.abs(finite*(2/(hi-lo))-(hi+lo)/(hi-lo)))) if finite.size else float('inf'))
        if result['nonfinite']: result['max_u'] = float('inf')
    return result


def point_distribution(value,domain,mask=None):
    """Keep original sample/head/row coordinates for every coverage failure."""
    import numpy as np
    array=np.asarray(value,dtype=np.float64)
    selected=array if mask is None else array[mask]
    result=distribution(selected,domain)
    bad=~np.isfinite(array)|(array<domain[0])|(array>domain[1])
    if mask is not None:bad &= mask
    coordinates=np.argwhere(bad)
    result['outside_locations']=[dict(sample_index=int(c[0]),coordinates=[int(v) for v in c],
        row=int(c[-2]) if array.ndim>=3 else None,value=float(array[tuple(c)])) for c in coordinates]
    result['below_minimum']=float(array[bad & (array<domain[0])].min()) if np.any(bad & (array<domain[0])) else None
    result['above_maximum']=float(array[bad & (array>domain[1])].max()) if np.any(bad & (array>domain[1])) else None
    return result


def errors(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    difference = np.abs(actual-expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    result = dict(nonfinite=int(np.count_nonzero(~finite)), max_abs=float('inf'),
                  max_rel=float('inf'), relative_linf=float('inf'), zero_reference_nonzero=0)
    if finite.all():
        zero = expected == 0
        nonzero = ~zero
        zero_bad = int(np.count_nonzero(zero & (actual != 0)))
        relative = difference[nonzero]/np.abs(expected[nonzero])
        maximum = float(difference.max(initial=0))
        denominator = float(np.abs(expected).max(initial=0))
        result.update(max_abs=maximum, max_rel=float('inf') if zero_bad else float(relative.max(initial=0)),
                      relative_linf=maximum/denominator if denominator else (0.0 if maximum == 0 else float('inf')),
                      zero_reference_nonzero=zero_bad)
    return result


def newton_error_model(fn, z0, *, target=TARGET, maximum=MODEL_ITERATIONS):
    """U5 fixed-order residual model: inv e'=e²; rsqrt z'=z(3-z²)/2.

    z=x*y for inv, z=sqrt(x)*y for rsqrt. The reported relative error
    is |1-z|, so a negative rsqrt fixed point remains visibly wrong.
    """
    import numpy as np
    z = np.asarray(z0, dtype=np.float64).copy()
    history = []
    with np.errstate(over='ignore', invalid='ignore'):
        for iteration in range(maximum+1):
            error = float(np.max(np.abs(1-z)))
            history.append(error)
            if np.isfinite(error) and error <= target:
                return dict(required_iters=iteration, converges=True, initial_error=history[0],
                            final_error=error, history=history)
            if not np.isfinite(error): break
            if iteration == 0 and (fn == 'inv' and error >= 1 or fn == 'rsqrt' and np.any(z <= 0)):
                return dict(required_iters=None, converges=False, initial_error=history[0],
                            reason='initial value outside the positive Newton convergence region', history=history)
            z = 1-(1-z)*(1-z) if fn == 'inv' else z*(1.5-0.5*z*z)
    return dict(required_iters=None, converges=False, initial_error=history[0],
                reason='target not reached within diagnostic iteration budget', history=history)


def newton_profile(fn, degree, domain):
    import numpy as np
    from ..kernels import approx
    from ..kernels.poly import cheb_eval
    from ..backends import CountBackend
    lo, hi = map(float, domain)
    # Uniform and geometric grids include both endpoints. Grid evaluations
    # are diagnostic samples, not a claimed interval proof.
    grid = np.unique(np.concatenate((np.linspace(lo,hi,GRID_POINTS), np.geomspace(lo,hi,GRID_POINTS))))
    coefficients = approx.remez(fn, degree, (lo,hi))
    y0 = cheb_eval(None, grid*(2/(hi-lo))-(hi+lo)/(hi-lo), coefficients)
    multiplier = grid if fn == 'inv' else np.sqrt(grid)
    z = multiplier*y0
    model = newton_error_model(fn,z)
    worst = int(np.argmax(np.abs(1-z)))
    backend = CountBackend(8)
    value = approx.newton(backend,backend.input(0),fn,degree,0,(lo,hi),coefficients=coefficients)
    base_depth = value.depth
    model.update(fn=fn, deg0=degree, domain=[lo,hi], domain_ratio=hi/lo,
                 diagnostic_grid_points=int(grid.size), initial_depth=base_depth,
                 depth_per_iter=2 if fn=='inv' else 3,
                 required_depth=(base_depth+(2 if fn=='inv' else 3)*model['required_iters'])
                     if model['required_iters'] is not None else None,
                 witness=dict(x=float(grid[worst]), y0=float(y0[worst]),
                              relative_error=float(abs(1-z[worst]))),
                 endpoints=[dict(x=float(grid[i]),y0=float(y0[i]),relative_error=float(abs(1-z[i])))
                            for i in (0,-1)])
    return model


def point_id(op, role):
    return f'{op.id}:{role}'


def point_table(program, weights, arrays, directory, split):
    import numpy as np
    from ..transformer_tensor import run, polynomial
    from ..transformer_validation import points
    from ..backends import BackendError
    descriptors = points(program)
    reference_points, reference_steps, rows = {}, {}, []

    def exact_point(op, role, value, attrs, mask):
        reference_points[point_id(op,role)] = (op,role,np.array(value,copy=True),dict(attrs),
                                             None if mask is None else np.array(mask,copy=True))
        return polynomial(value,attrs,True)
    expected = run(program,weights,arrays,act_mode='exact',point_evaluator=exact_point,
                   step_observer=lambda op,value: reference_steps.update({op.id:np.array(value,copy=True)}))
    np.save(directory/(split+'-exact.npy'),expected)
    trajectory_points, trajectory_steps = {}, []

    def actual_point(op,role,value,attrs,mask):
        selected = value if mask is None else value[mask]
        target = polynomial(value,attrs,True)
        actual = polynomial(value,attrs,False)
        observed = actual if mask is None else actual[mask]
        truth = target if mask is None else target[mask]
        trajectory_points[point_id(op,role)] = dict(inputs=point_distribution(value,attrs['domain'],mask),
                                                  output_error=errors(observed,truth))
        return actual
    def step(op,value):
        trajectory_steps.append(dict(op_id=op.id,line=op.lineno,op=op.kind,output_error=errors(value,reference_steps[op.id])))
    failure = None
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        try:
            actual = run(program,weights,arrays,point_evaluator=actual_point,step_observer=step)
            trajectory_error = errors(actual,expected)
        except BackendError as exc:
            failure = str(exc);trajectory_error = None
    for index,point in enumerate(descriptors):
        role = point.role or 'act';key=point_id(point.op,role)
        op,role,value,attrs,mask = reference_points[key]
        selected = value if mask is None else value[mask]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            approximate = polynomial(selected,attrs,False)
            target = polynomial(selected,attrs,True)
        row = dict(index=index,point_id=key,line=op.lineno,op=op.kind,role=role,fn=attrs['fn'],
                   attrs=attrs,exact_prefix=dict(inputs=point_distribution(value,attrs['domain'],mask),
                                                output_error=errors(approximate,target)),
                   all_approx_prefix=trajectory_points.get(key),single_point=None)
        layers = []
        def isolated(operation,which,x,a,keep):
            return polynomial(x,a,point_id(operation,which)!=key)
        def single_step(operation,y):
            if operation.id >= op.id:
                layers.append(dict(op_id=operation.id,line=operation.lineno,op=operation.kind,
                                   relative_linf=errors(y,reference_steps[operation.id])['relative_linf']))
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            try:
                result=run(program,weights,arrays,point_evaluator=isolated,step_observer=single_step)
                row['single_point']=dict(output_error=errors(result,expected),layers=layers,
                                         maximum_downstream_relative=max((r['relative_linf'] for r in layers),default=0))
            except BackendError as exc:
                row['single_point']=dict(error=str(exc),layers=layers,output_error=None,
                                         maximum_downstream_relative=float('inf'))
        rows.append(row)
        write(directory/(split+'-points.partial.json'),rows)
        print(f'POINT {split} line={op.lineno} {role} range=({selected.min()},{selected.max()}) '
              f'rel={row["exact_prefix"]["output_error"]["max_rel"]}',flush=True)
    first = next((dict(point_id=r['point_id'],line=r['line'],role=r['role'],
                       error=r['all_approx_prefix']['output_error']) for r in rows
                  if r['all_approx_prefix'] is not None and
                  (r['all_approx_prefix']['output_error']['nonfinite'] or
                   r['all_approx_prefix']['output_error']['max_rel']>1)),None)
    return dict(split=split,samples=len(next(iter(arrays.values()))),points=rows,
                first_distorted_point=first,all_approx_failure=failure,
                all_approx_output_error=trajectory_error,step_errors=trajectory_steps),reference_points


def specialties(program, reference, calibration):
    import numpy as np
    from ..transformer_tensor import polynomial
    result = dict(softmax=[],norm=[])
    for op in program.ops:
        if op.kind=='softmax':
            _,_,scores,exp,mask = reference[point_id(op,'exp')]
            _,_,total,inv,_ = reference[point_id(op,'inv')]
            row_max = np.where(mask,scores,-np.inf).max(-1)
            row_min = np.where(mask,scores,np.inf).min(-1)
            approx_total=(polynomial(scores,exp,False)*mask).sum(-1)
            # shift and scale are derived exclusively from exact calibration.
            graph_key = next(k for k in calibration if k.endswith('::exp') and
                             k.removesuffix('::exp')==_graph_name(op.outputs[0]))
            shift=float(calibration[graph_key]['max'])
            total_key=graph_key.removesuffix('::exp')+'::inv'
            upper=float(calibration[total_key]['max'])
            factor=math.exp(-shift);scale=upper*factor
            domains=dict(original=inv['domain'],shift=tuple(float(v)*factor for v in inv['domain']),
                         shift_scale=tuple(float(v)/upper for v in inv['domain']))
            profiles={stage:newton_profile('inv',inv['deg0'],domain) for stage,domain in domains.items()}
            result['softmax'].append(dict(op_id=op.id,line=op.lineno,
                row_max_scores=row_max.tolist(),row_spans=(row_max-row_min).tolist(),
                row_max_distribution=distribution(row_max),row_span_distribution=distribution(row_max-row_min),
                exact_S=distribution(total),S_with_current_exp=distribution(approx_total),
                public_shift=shift,public_inv_scale=scale,exp_domain_shifted=[float(v)-shift for v in exp['domain']],
                transformed_S=dict(shift=distribution(total*factor),shift_scale=distribution(total/upper)),
                inv_profiles=profiles))
        elif op.kind in ('layernorm','rmsnorm'):
            _,_,value,attrs,_=reference[point_id(op,'rsqrt')]
            result['norm'].append(dict(op_id=op.id,line=op.lineno,variance=distribution(value),
                rsqrt_profile=newton_profile('rsqrt',attrs['deg0'],attrs['domain'])))
    return result


def _graph_name(value):
    # The rule planner's names are generally different from graph names. The
    # outer report uses point order, and this mapping is installed per program.
    return _point_graph_names[value]


_point_graph_names = {}


def diagnose(args):
    import numpy as np
    from ..models.info import ModelInfo
    from ..data.samples import StoredSplits
    from ..dsl.parser import parse
    from ..analyzer import approximation_points
    from ..transformer_validation import points
    info=ModelInfo.load(args.prepared/'model')
    source=args.prepared/'model/program.ptdsl'
    program=parse(source.read_text())
    calibration=json.loads((args.prepared/'model/calibration.json').read_text())
    for p,(node,key) in zip(points(program),approximation_points(info.graph)):
        if p.role:_point_graph_names[p.op.outputs[0]]=node.outputs[0]
    result=dict(round='U4',section='11.1',complete=False,prepared=str(args.prepared),program=str(source),
                target_relative_error=TARGET,model_iterations=MODEL_ITERATIONS,
                grid_base_points=GRID_POINTS,splits=[],adjustments_applied=False,
                llm_calls=0,report_images=0,native_compiles=0)
    samples=StoredSplits(args.prepared/'model/samples',info)
    for split in ('calibration','heldout'):
        arrays,_=samples.get(split)
        table,reference=point_table(program,info.graph.weights,arrays,args.out,split)
        table['specialties']=specialties(program,reference,calibration)
        result['splits'].append(table)
        write(args.out/'REPORT.json',result)
    result['complete']=True
    return result


def diagnose_choices(args):
    """Test public transforms and the finite calibration-only choice set."""
    import copy
    import numpy as np
    from ..models.info import ModelInfo
    from ..data.samples import StoredSplits
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..analyzer import approximation_points
    from ..transformer_tensor import run,polynomial
    from ..transformer_validation import points
    from ..planners.transformer_rule import (plan,select_approximation,softmax_adjustments,
                                            ApproximationLimitError)
    info=ModelInfo.load(args.prepared/'model')
    original=parse((args.prepared/'model/program.ptdsl').read_text())
    transformed=copy.deepcopy(original)
    calibration=json.loads((args.prepared/'model/calibration.json').read_text())
    before=json.loads((args.before/'REPORT.json').read_text())
    mapping={point_id(p.op,p.role or 'act'):key
             for p,(node,key) in zip(points(transformed),approximation_points(info.graph))}
    transforms=[]
    for op in transformed.ops:
        if op.kind!='softmax':continue
        exp_key=mapping[point_id(op,'exp')];inv_key=mapping[point_id(op,'inv')]
        change=softmax_adjustments(calibration[exp_key],calibration[inv_key])
        op.attrs.update(change)
        transforms.append(dict(op_id=op.id,line=op.lineno,exp_key=exp_key,inv_key=inv_key,**change))
    report=dict(round='U4',section='11.2-11.3',complete=False,prepared=str(args.prepared),
                before=str(args.before),transforms=transforms,splits=[],points=[],
                target_relative_error=TARGET,choices_from='calibration only',
                llm_calls=0,report_images=0,native_compiles=0,stop_required=False)
    samples=StoredSplits(args.prepared/'model/samples',info)
    for split in ('calibration','heldout'):
        arrays,_=samples.get(split);observed=[]
        def capture(op,role,value,attrs,mask):
            selected=value if mask is None else value[mask]
            observed.append(dict(point_id=point_id(op,role),line=op.lineno,role=role,
                                 inputs=point_distribution(value,attrs['domain'],mask)))
            return polynomial(value,attrs,True)
        exact=run(transformed,info.graph.weights,arrays,act_mode='exact',point_evaluator=capture)
        expected=np.load(args.before/(split+'-exact.npy'))
        delta=np.abs(exact-expected)
        math_ok=bool(np.all(delta<=1e-12+1e-12*np.abs(expected)))
        item=dict(split=split,samples=len(next(iter(arrays.values()))),points=observed,
                  exact_equivalence=errors(exact,expected),mathematical_check=math_ok)
        report['splits'].append(item);write(args.out/'REPORT.json',report)
        np.save(args.out/(split+'-transformed-exact.npy'),exact)
        if not math_ok:raise RuntimeError('Public softmax transforms changed exact outputs')
    original_points={row['point_id']:row for row in before['splits'][0]['points']}
    for descriptor in points(transformed):
        role=descriptor.role or 'act';key=point_id(descriptor.op,role)
        entry=dict(point_id=key,line=descriptor.op.lineno,role=role,fn=descriptor.attrs['fn'],
                   calibration_key=mapping[key],original_domain=original_points[key]['attrs']['domain'],
                   domain=descriptor.attrs['domain'])
        if role=='act':entry.update(status='unchanged',selected=descriptor.attrs)
        else:
            choice=select_approximation(descriptor.attrs['fn'],descriptor.attrs['domain'])
            entry.update(choice)
            if choice['selected'] is None:report['stop_required']=True
        report['points'].append(entry);write(args.out/'REPORT.json',report)
        print(f'CHOICE line={entry["line"]} {role} domain={entry["domain"]} '
              f'status={entry["status"]} selected={entry.get("selected")}',flush=True)
    # This is the §11.2 reference variant. The current rule entry uses
    # §11.6, so keep its status separate from this legacy choice table.
    report['rule_plan']=dict(complete=False,variant='11.2 reference only',
                            reason='No delivery program emitted by reference diagnosis')
    report.update(complete=True,eligible_for_mock=not report['stop_required'],
        stop_reason='U4 §11.3: no permitted finite inverse approximation meets the target'
                    if report['stop_required'] else None,
        heldout_approximate_run=False,mock_started=False)
    return report


def softmax_depths(program,weights):
    from ..dsl.typecheck import check
    from ..kernels import transformer as tr
    from ..kernels.layout import ScoreLayout,TokenLayout,multi_flat_layout,layout_blocks
    from ..backends import CountBackend
    checked=check(program,{k:v.shape for k,v in weights.items()});result=[]
    for op in program.ops:
        if op.kind!='softmax':continue
        a=op.attrs;shape=checked.shapes[op.inputs[0]];n=program.header.slots
        layout=ScoreLayout(n,shape[2],shape[1],shape[3]) if len(shape)==4 else TokenLayout(n,*shape[1:])
        backend=CountBackend(n);xs=[backend.input(i) for i in range(len(layout_blocks(layout)))]
        mask_layout=multi_flat_layout(n,shape[-1]) if a['mask'] is not None else None
        mask=[backend.input(len(xs)+i) for i in range(len(layout_blocks(mask_layout)))] if mask_layout else None
        values,_=tr.softmax(backend,xs,layout,tr.approximation(a,'exp','exp'),tr.approximation(a,'inv','inv'),
            keep=a['keep'],mask=mask,mask_layout=mask_layout,shift=a.get('shift',0),inv_scale=a.get('inv_scale',1),
            steps=a.get('steps',0),step_inv=tr.approximation(a,'step_inv','inv') if a.get('steps',0) else None)
        result.append(dict(line=op.lineno,op_id=op.id,steps=a.get('steps',0),
                           multiplication_depth=max(value.depth for value in values),counts=backend.stats()))
    return result


def diagnose_squared(args):
    import copy
    from types import SimpleNamespace
    import numpy as np
    from ..models.info import ModelInfo
    from ..data.samples import StoredSplits
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..dsl.printer import print_program
    from ..transformer_tensor import run
    from ..transformer_validation import points
    from ..planners.transformer_rule import (plan,ApproximationLimitError,select_approximation,
                                            squared_softmax_adjustments)
    info=ModelInfo.load(args.prepared/'model');weights=info.graph.weights
    original=parse((args.prepared/'model/program.ptdsl').read_text())
    calibration=json.loads((args.prepared/'model/calibration.json').read_text())
    before=json.loads((args.before/'REPORT.json').read_text())
    report=dict(round='U4',section='11.4-11.6',complete=False,prepared=str(args.prepared),
                before=str(args.before),exact_identity=[],selections=[],splits=[],
                llm_calls=0,report_images=0,native_compiles=0,stop_required=False)
    # Form the exact identity first. Approximation fields here are merely
    # placeholders; exact mode evaluates functions and performs no fitting.
    identity=copy.deepcopy(original)
    graph_softmax=[node for node in info.graph.nodes if node.op=='softmax']
    for op,node in zip((o for o in identity.ops if o.kind=='softmax'),graph_softmax):
        cfg=squared_softmax_adjustments(calibration[node.outputs[0]+'::exp'],
                                       info.graph.tensors[node.inputs[0]].shape[-1])
        op.attrs.update(cfg,step_inv_method='newton',step_inv_deg0=7,step_inv_iters=2)
    zero=copy.deepcopy(identity)
    for op in zero.ops:
        if op.kind=='softmax':op.attrs['steps']=0
    samples=StoredSplits(args.prepared/'model/samples',info)
    for split in ('calibration','heldout'):
        arrays,_=samples.get(split)
        exact=run(identity,weights,arrays,act_mode='exact');expected=run(zero,weights,arrays,act_mode='exact')
        difference=np.abs(exact-expected)
        passed=bool(np.all(difference<=1e-12+1e-12*np.abs(expected)))
        report['exact_identity'].append(dict(split=split,samples=len(next(iter(arrays.values()))),
            passed=passed,errors=errors(exact,expected)))
        write(args.out/'REPORT.json',report)
        if not passed:raise RuntimeError('steps=k versus steps=0 exact identity failed')
    # Finish every old finite choice table and record the previously
    # unreported nonconvergent candidate in a fresh evidence directory.
    legacy_dir=args.out/'legacy-choices';legacy_dir.mkdir()
    legacy=diagnose_choices(SimpleNamespace(prepared=args.prepared,before=args.before,out=legacy_dir))
    write(legacy_dir/'REPORT.json',legacy)
    report['legacy_choices']=dict(path=str(legacy_dir/'REPORT.json'),complete=legacy['complete'],
                                  infeasible_points=[p['point_id'] for p in legacy['points'] if p.get('selected') is None])
    old_stop=next((json.loads(p.read_text()) for p in [args.before.parent/'u4-points-choices-20261026T123231747385/STOP.json'] if p.exists()),None)
    if old_stop:
        first=next(p for p in legacy['points'] if p['point_id']=='8:inv')
        matching=[dict(method=c['method'],degree=c.get('deg',c.get('deg0')),iters=c.get('iters'))
            for c in first['candidates'] if c.get('reason')=='nonconvergent' and
            c['iterations']==old_stop['fitting']['iterations'] and
            c['max_error']==old_stop['fitting']['max_error'] and c['abs_E']==old_stop['fitting']['abs_E']]
        report['previous_stop_resolution']=dict(original=str(args.before.parent/'u4-points-choices-20261026T123231747385/STOP.json'),
                                                matching_candidates=matching)
    # Complete all current choices before the feasibility decision.
    for descriptor in points(identity):
        role=descriptor.role or 'act';attrs=descriptor.attrs
        item=dict(point_id=point_id(descriptor.op,role),line=descriptor.op.lineno,role=role,domain=attrs['domain'])
        if descriptor.prefix=='step_inv':item['shared_with']=point_id(descriptor.op,'step_inv[0]')
        if not descriptor.role:item.update(status='unchanged',selected=attrs)
        else:
            target=TARGET/(2**descriptor.op.attrs['steps']) if role=='exp' else TARGET
            item.update(select_approximation(attrs['fn'],attrs['domain'],target=target,exp_ladder=role=='exp'))
        report['selections'].append(item);write(args.out/'REPORT.json',report)
    infeasible=[p for p in report['selections'] if p.get('selected') is None]
    if infeasible:
        report.update(complete=True,stop_required=True,stop_reason='infeasible approximation choices',
                      eligible_for_mock=False,infeasible_points=[p['point_id'] for p in infeasible])
        return report
    text=plan(info.graph,calibration);(args.out/'program.ptdsl').write_text(text)
    program=parse(text)
    if not check(program,{k:v.shape for k,v in weights.items()}).ok:raise RuntimeError('squared softmax rule failed typecheck')
    report['program']=str(args.out/'program.ptdsl')
    old_depth=softmax_depths(original,weights);new_depth=softmax_depths(program,weights)
    report['softmax_depths']=[dict(line=n['line'],steps=n['steps'],old=o,new=n,
        added_depth=n['multiplication_depth']-o['multiplication_depth']) for o,n in zip(old_depth,new_depth)]
    for split in ('calibration','heldout'):
        arrays,_=samples.get(split)
        table,reference=point_table(program,weights,arrays,args.out,split)
        table['specialties']=dict(softmax=[],norm=[])
        for op in program.ops:
            if op.kind=='softmax':
                _,_,scores,exp,mask=reference[point_id(op,'exp')]
                _,_,s0,inv,_=reference[point_id(op,'inv')]
                table['specialties']['softmax'].append(dict(line=op.lineno,steps=op.attrs['steps'],
                    row_max=distribution(np.where(mask,scores,-np.inf).max(-1)),
                    row_span=distribution(np.where(mask,scores,-np.inf).max(-1)-np.where(mask,scores,np.inf).min(-1)),
                    scaled_S0=distribution(s0,inv['domain']),
                    stages=[dict(role=f'step_inv[{i}]',inputs=distribution(reference[point_id(op,f'step_inv[{i}]')][2],
                        op.attrs['step_inv_domain'])) for i in range(op.attrs['steps'])]))
            elif op.kind in ('layernorm','rmsnorm'):
                _,_,variance,attrs,_=reference[point_id(op,'rsqrt')]
                table['specialties']['norm'].append(dict(line=op.lineno,variance=distribution(variance,attrs['domain'])))
        table['comparison']=[]
        old={r['point_id']:r for r in next(s for s in before['splits'] if s['split']==split)['points']}
        for row in table['points']:
            previous=old.get(row['point_id'])
            table['comparison'].append(dict(point_id=row['point_id'],line=row['line'],role=row['role'],
                old=None if previous is None else previous['exact_prefix'],new=row['exact_prefix'],
                old_single=None if previous is None else previous['single_point']['output_error'],
                new_single=row['single_point']['output_error']))
        report['splits'].append(table);write(args.out/'REPORT.json',report)
    calibration_fail=[r['point_id'] for r in report['splits'][0]['points']
                      if r['exact_prefix']['inputs']['outside'] or
                         r.get('all_approx_prefix') is not None and r['all_approx_prefix']['inputs']['outside']]
    coverage_fail=[dict(split=s['split'],point_id=r['point_id'],line=r['line'],
        exact_prefix=r['exact_prefix']['inputs'],all_approx_prefix=None if r.get('all_approx_prefix') is None else r['all_approx_prefix']['inputs'])
        for s in report['splits'] for r in s['points'] if r['exact_prefix']['inputs']['outside'] or
        r.get('all_approx_prefix') is not None and r['all_approx_prefix']['inputs']['outside']]
    nonfinite=[s['split'] for s in report['splits'] if s['all_approx_failure']]
    report.update(complete=True,calibration_coverage_failures=calibration_fail,coverage_failures=coverage_fail,
                  nonfinite_splits=nonfinite,stop_required=bool(coverage_fail or nonfinite),
                  eligible_for_mock=not (coverage_fail or nonfinite),mock_started=False)
    if calibration_fail:report['stop_reason']='calibration inputs outside the registered domains'
    elif coverage_fail:report['stop_reason']='held-out domain coverage failed'
    elif nonfinite:report['stop_reason']='nonfinite approximate execution'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--stage',choices=('before','choices','squared'),default='before')
    parser.add_argument('--before',type=Path)
    args=parser.parse_args()
    if args.stage in ('choices','squared') and args.before is None:parser.error('--stage choices/squared requires --before')
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        result=dict(complete=False,round='U4',section='11.1')
        try:result={'before':diagnose,'choices':diagnose_choices,'squared':diagnose_squared}[args.stage](args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if (args.out/'REPORT.json').exists():result.update(json.loads((args.out/'REPORT.json').read_text()))
            result.update(complete=False,error=f'{type(exc).__name__}: {exc}')
        write(args.out/'REPORT.json',result)
        return (2 if result.get('stop_required') else 0) if result['complete'] else 1
    args.prepared=args.prepared or RESULTS_DIR/'u4-bert-prepare-20261026T115827230385'
    stamp=datetime.now(timezone.utc)
    run_id='u4-points-'+args.stage+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id
    args.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','1200','--',
             '-m','pt_agent.experiments.u4_point_diagnosis','--inside','--prepared',str(args.prepared),'--out',str(args.out),
             '--stage',args.stage]
    if args.before is not None:command.extend(['--before',str(args.before)])
    write(args.out/'argv.json',command)
    print('RUN='+str(args.out),flush=True)
    started=time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status=subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT,timeout=1230)
    path=args.out/'REPORT.json'
    result=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    result.update(run_id=run_id,git_rev=rev,exit=status.returncode,seconds=time.monotonic()-started)
    write(path,result)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model='bert',mode='point-diagnosis-'+args.stage,git_rev=rev,
                metrics=dict(complete=result['complete'],exit=status.returncode,llm_calls=0,
                             stop_required=result.get('stop_required',False)),
                artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=result['complete'],exit=status.returncode),indent=2),flush=True)
    return status.returncode


if __name__=='__main__':raise SystemExit(main())
