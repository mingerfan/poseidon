"""Closed retained13 component-plan provenance; fresh physical Count stays required.

Outer admission alone reads the registry/process table. Workers revalidate the
closed artifacts against its canonical proof; no new planner or verify is
claimed. Prior clocks are historical, and cache validation has its own span.
"""
import ast
import fcntl
import json
import math
from pathlib import Path
import shutil

MODULE = 'pt_agent.experiments.qwen_retained_component_replay'
POLICY = 'qwen-component-program-plan-retained13-v1'
PROOF = 'component-plan-cache-admission.json'
NAMES = tuple('layer'+str(i) for i in range(24))+('head',)
FIELDS = {'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}


def _read(path):
    return json.loads(Path(path).read_text())


def _same(value):
    return json.loads(json.dumps(value,allow_nan=False))


def _require(condition, message):
    if not condition:raise ValueError(message)


def eligible(source):
    return source.get('entry_module') == MODULE


def _inside(path, root):
    path=Path(path)
    _require(path.is_absolute() and not path.is_symlink(), 'Require canonical owned artifact path')
    resolved=path.resolve()
    _require(resolved==path and resolved.is_relative_to(root), 'Artifact escaped its original component-cost directory')
    return resolved


def _stat(path):
    info=Path(path).stat()
    return dict(size_bytes=info.st_size,mtime_ns=info.st_mtime_ns)


def validate_plan(plan, outputs):
    from ..plan import PLAN_FORMAT
    _require(isinstance(plan,dict) and plan.get('format')==PLAN_FORMAT
        and plan.get('source')=='pt_agent.lower'
        and plan.get('placement') in ('depth-dp','depth-dp+prune','stage-cut','stage-cut+prune')
        and plan.get('levels')==dict(input_level=5,boot_in_level=1,boot_out_level=5), 'Original plan format/profile changed')
    for key in ('calls','boots','barriers'):
        _require(type(plan.get(key)) is int and plan[key]>=0, 'Invalid original plan integer')
    _require(plan['calls']>=34, 'Original physical inputs absent from plan')
    for key in ('substitutions','outputs','dead','merged','cuts'):
        _require(type(plan.get(key)) is list, 'Incomplete original plan schema')
    _require(isinstance(plan.get('timing'),dict) and isinstance(plan.get('candidates'),dict), 'Original planner candidate/timing proof missing')
    _require(plan['placement'] in plan['candidates'] and 'error' not in plan['candidates'][plan['placement']],
        'Original selected planner candidate is unavailable')
    for key in ('lower_s','dp_s','candidates_s','map_s'):
        value=plan['timing'].get(key)
        _require(type(value) in (int,float) and math.isfinite(value) and value>=0,
            'Original planner timing schema is incomplete')
    _require(all(type(v) is int and v>=0 for v in plan['merged']), 'Invalid original planner index metadata')
    if plan['placement'] in ('stage-cut','stage-cut+prune'):
        # Stage cuts retain (statement, kernel stage, local multiplicative depth) labels.
        _require(all(type(row) is list and len(row)==3
            and all(type(v) is int and v>=0 for v in row) for row in plan['cuts']),
            'Invalid original planner index metadata')
    else:
        _require(all(type(v) is int and v>=0 for v in plan['cuts']), 'Invalid original planner index metadata')
    _require(isinstance(plan.get('latency_us'),(int,float)) and not isinstance(plan['latency_us'],bool)
        and math.isfinite(plan['latency_us']) and plan['latency_us']>=0, 'Invalid original plan latency')
    covered=set();positions=set()
    for row in plan['substitutions']:
        _require(type(row) is list and len(row)==3 and all(type(v) is int and v>=0 for v in row), 'Invalid original substitution')
        seq,pos,nid=row
        _require(seq<plan['calls'] and pos in (0,1) and (seq,pos) not in positions, 'Ambiguous original substitution')
        positions.add((seq,pos));covered.add(nid)
    output_positions=set()
    for row in plan['outputs']:
        _require(type(row) is list and len(row)==2 and all(type(v) is int and v>=0 for v in row), 'Invalid original output refresh')
        pos,nid=row
        _require(pos<outputs and pos not in output_positions, 'Ambiguous original output refresh')
        output_positions.add(pos);covered.add(nid)
    _require(len(covered)==plan['boots'], 'Original canonical bootstrap coverage differs')
    end=0
    for row in plan['dead']:
        _require(type(row) is list and len(row)==2 and all(type(v) is int for v in row), 'Invalid original dead range')
        start,stop=row
        _require(end<=start<stop<=plan['calls'], 'Unordered/out-of-range original dead calls')
        end=stop
    return plan


def _events(case):
    paths=list((case/'phases').glob('*/evaluation-stage-events.jsonl'))
    from .r10_paper_metrics import read_events,summarize_events
    streams=[(path,read_events(path)) for path in paths]
    streams=[(path,events) for path,events in streams if any(e.get('stage_id')=='cost' for e in events)]
    _require(len(streams)==1, 'Require the unique original retained cost event stream')
    path,events=streams[0];summary=summarize_events(events)
    _require(summary['status']=='ok' and summary['attribution_ready'] is True,
        'Original component evaluation events are incomplete')
    spans=summary['spans'];costs=[s for s in spans if s['stage_id']=='cost']
    _require(len(costs)==1 and costs[0]['status']=='ok', 'Original complete cost span missing')
    by_id={s['event_id']:s for s in spans}
    def below(span,ancestor):
        parent=span.get('parent_id')
        while parent is not None:
            if parent==ancestor:return True
            parent=by_id[parent].get('parent_id')
        return False
    proof={}
    for name in NAMES:
        matches=[s for s in spans if s['stage_id']=='component_cost_'+name]
        _require(len(matches)==1 and matches[0]['status']=='ok' and below(matches[0],costs[0]['event_id']), 'Original component cost span missing')
        component=matches[0];phases={}
        for stage in ('cost_estimate','record_lower','placement','plan_express','conditional_plan_replay'):
            found=[s for s in spans if s['stage_id']==stage and below(s,component['event_id'])]
            _require(len(found)==1 and found[0]['status']=='ok', 'Original component planning/verify span missing')
            phases[stage]=dict(event_id=found[0]['event_id'],inclusive_ns=found[0]['inclusive_ns'],exclusive_ns=found[0]['exclusive_ns'])
        proof[name]=dict(component_event_id=component['event_id'],historical_stages=phases)
    return dict(path=str(path.resolve()),components=proof)


def _typed_bindings(source, rows):
    from ..dsl.parser import parse
    from . import qwen_layered_structure as layered
    shapes={}
    for row in rows:
        for name,shape in row['weight_shapes']:
            shape=tuple(shape)
            _require(name not in shapes or shapes[name]==shape, 'Original component weight shapes conflict')
            shapes[name]=shape
    partition=layered.split_program(parse(Path(source['program']).read_text()),shapes)
    return [layered.source_binding(segment) for segment in (*partition.layers,partition.head)]


def _cost_authority(case,source,source_text):
    """New frozen candidate may cite an older complete resource-rejected cost."""
    original=source['original_source'];authority=original.get('completed_cost_authority')
    if authority is None:return case,None
    from . import qwen_completed_cost_cache as cached
    from .qwen_retained_component_replay import _completed_base_source,INVENTORY_SCOPE
    _require(original.get('logical_constant_inventory_gib')==32
        and original.get('logical_constant_inventory_scope')==INVENTORY_SCOPE,
        'Explicit logical inventory policy required for historical cost authority')
    old=Path(authority['authority_case']).resolve()
    proof_path=Path(original['completed_cost_admission'])
    _require(proof_path==case/cached.PROOF and not proof_path.is_symlink(),
        'New frozen source requires its canonical completed-cost proof')
    loaded=cached.load_worker(old,proof_path,_completed_base_source(original),source_text)
    _require(loaded['provenance']==authority and loaded['cost'].get('lowering_policy')==POLICY,
        'Frozen source historical cost authority differs')
    return old,loaded


def _artifacts(case, source):
    """Recompute all source/schema/selection/event facts; no registry here."""
    from ..planners.transformer_session import load_state,json_safe
    _require(eligible(source) and source.get('selected_iteration')==13
        and source.get('selected_origin')=='llm' and source.get('candidate_frozen') is True
        and source.get('original_source',{}).get('component_cost_policy')==POLICY,
        'Only the exact retained13 component policy can reuse a plan')
    record=_read(case/'result.json');done=_read(case/'completion.json')
    _require(record.get('status')=='ok' and record.get('candidate_frozen') is True
        and record.get('source')==source['original_source']
        and record.get('api_calls')==0 and record.get('termination') is None
        and record.get('terminal_budget_error') is None
        and done.get('status')=='ok' and done.get('round')==record.get('round')
        and done.get('required_artifacts_persisted') is True and done.get('registry_persisted') is True,
        'Require successful closed same-source retained component C2')
    phases=record.get('phases',[])
    _require([row.get('phase') for row in phases]==['calibration','prompt','step','step'],
        'Require the original retained evaluation and confirmation phase chain')
    for index,row in enumerate(phases,1):
        process=row.get('process',{})
        _require(type(process.get('exit_code')) is int and process['exit_code']==0
            and process.get('timed_out') is False and process.get('signal') is None
            and process.get('remaining_pids')==[] and process.get('signals_sent')==[]
            and row.get('worker',{}).get('status')=='ok'
            and _read(case/'phases'/(str(index).zfill(3)+'-'+row['phase'])/'worker-result.json')==row['worker'],
            'Original retained phase supervision is unclean or differs from saved worker')
    state=load_state(case/'session/SESSION.json');agent=_read(case/'session/AGENT.json');frozen=_read(case/'session/FROZEN.json')
    selected=agent.get('confirm',{}).get('selected')
    _require(isinstance(selected,dict) and selected.get('iteration')==13 and selected.get('origin')=='llm'
        and state.get('phase')=='done' and state.get('failed') is None and state.get('calls')==13
        and state.get('pending') is None and state.get('component_cost_policy')==POLICY
        and state.get('confirmation',{}).get('selected')==selected and frozen.get('iteration')==13
        and frozen.get('origin')=='llm' and frozen.get('confirm')==agent.get('confirm')
        and Path(frozen.get('program','')).resolve()==case/'session/program.ptdsl', 'Original selected C2 transaction differs')
    actual_source=Path(source['program']).read_text()
    _require(ast.dump(ast.parse(selected['source']),include_attributes=False)==ast.dump(ast.parse(actual_source),include_attributes=False), 'Selected candidate source changed')
    saved_record=state['records'][(13,'llm',selected['source'])][0]
    _require(json_safe(saved_record)==agent.get('best') and json_safe(state.get('best'))==agent.get('best')
        and state.get('best_program')==selected['source'] and agent.get('best_program')==selected['source'], 'Original selected cost record differs')
    components=saved_record.get('refresh_plan',{}).get('components')
    _require(isinstance(components,list) and [r.get('name') for r in components]==list(NAMES), 'All25 selected component plans required')
    paths=[Path(r['plan']) for r in components]
    authority_case,authority=_cost_authority(case,source,actual_source)
    root=paths[0].parent.parent.resolve()
    _require(root.parent==authority_case/'session' and root.name.startswith('component-cost-') and root.name[len('component-cost-'):].isdigit()
        and not root.is_symlink(), 'Use the selected original component cost directory')
    if authority is not None:
        _require(root==Path(authority['provenance']['authority_cost_directory']).resolve()
            and all(_same(saved_record.get(k))==_same(v) for k,v in authority['cost'].items()),
            'New selected cost differs from the completed historical aggregate')
    cost=_read(root/'result.json');partial=_read(root/'partial.json')
    _require(cost.get('status')=='ok' and cost.get('placement_model')=='component-program-plan'
        and cost.get('lowering_policy')==POLICY and partial==dict(status='running',completed=25,components=list(NAMES))
        and all(_same(saved_record.get(k))==v for k,v in cost.items() if k!='status'), 'Selected complete component result differs')
    for key in ('counts','ops_total','complete_count_replay_counts','complete_count_replay_ops_total',
                'boots_pred','component_internal_boots','explicit_boundary_boots','max_depth',
                'rot_keys','distinct_consts','const_bytes','latency_pred_us'):
        _require(key in cost, 'Original aggregate cost schema is incomplete')
    for counts,total in (('counts','ops_total'),('complete_count_replay_counts','complete_count_replay_ops_total')):
        _require(isinstance(cost[counts],dict) and all(type(v) is int and v>=0 for v in cost[counts].values())
            and sum(cost[counts].values())==cost[total] and cost[counts].get('boot')==cost['boots_pred'],
            'Original aggregate inventory differs')
    _require(cost['explicit_boundary_boots']==24*32+19
        and cost['boots_pred']==cost['component_internal_boots']+cost['explicit_boundary_boots']
        and cost['component_internal_boots']==sum(row['internal_boots'] for row in components)
        and cost['const_bytes']==cost['distinct_consts']*16384*8, 'Original aggregate boundaries/constants differ')
    c2=_read(case/'session/component-c2-access.json')
    _require(c2.get('status')=='completed' and c2.get('calls')==13, 'Original full C2 is incomplete')
    artifacts={};bindings=[]
    for name,row,path in zip(NAMES,components,paths):
        path=_inside(path,root)
        _require(path==root/name/'refresh-plan.json', 'Original component plan path differs')
        source_path=_inside(path.parent/'source-binding.json',root)
        count_path=_inside(path.parent/'complete-count.json',root)
        binding=_read(source_path);count=_read(count_path)
        outputs=19 if name=='head' else 34;boundary=19 if name=='head' else 32
        plan=validate_plan(_read(path),outputs)
        interface=count.get('interface',{});stats=count.get('stats',{})
        _require(interface.get('status')=='ok' and interface.get('source_binding')==binding
            and interface.get('explicit_boundary_bootstrap_calls')==boundary
            and interface.get('internal_plan_bootstrap_calls')==plan['boots']
            and interface.get('symbolic_output_levels')==[5]*boundary
            and stats.get('counts',{}).get('boot')==plan['boots']+boundary
            and row.get('internal_boots')==plan['boots'] and row.get('boundary_boots')==boundary
            and row.get('output_levels')==[5]*boundary, 'Original successful physical Count/interface differs')
        bindings.append(binding);artifacts[name]=dict(plan_path=str(path),source=binding,count=count,
            plan_summary={k:plan[k] for k in ('format','placement','calls','boots','levels','timing')},
            stat_provenance={p.name:_stat(p) for p in (path,source_path,count_path)})
    _require(_same(_typed_bindings(source,bindings))==bindings, 'Typed original frozen program component bindings differ')
    events=_events(authority_case)
    return dict(source_case=str(case),source=source,cost_directory=str(root),
        cost_result=cost,components=artifacts,events=events,
        **(dict(completed_cost_authority=authority['provenance']) if authority is not None else {}),
        clock_scope='Historical component DAG/planning/verify; never part of this fresh native placement clock')


def prepare_outer(case, source):
    """Called only in the original outer source admission before any worker."""
    if not eligible(source):return None
    case=Path(case).resolve();facts=_artifacts(case,source)
    if 'completed_cost_authority' in facts:
        # The new C2 source and the old cost authority each need outer closure.
        from . import qwen_completed_cost_cache as cached
        from .qwen_retained_component_replay import _completed_base_source
        cached.prepare_outer(Path(facts['completed_cost_authority']['authority_case']),
            _completed_base_source(source['original_source']))
    record=_read(case/'result.json');rows=[]
    with (case.parent/'registry.jsonl').open() as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row=json.loads(line)
                if row.get('run_id')==case.name:rows.append(row)
    _require(len(rows)==1 and set(rows[0])==FIELDS and rows[0]['artifacts_dir']==str(case)
        and all(rows[0].get(k)==record.get(k) for k in ('round','model','mode'))
        and rows[0]['metrics'].get('status')=='ok', 'Unique successful eight-field original registry identity required')
    births={}
    for path in case.glob('*resources.jsonl'):
        with path.open() as stream:
            for line in stream:
                sample=json.loads(line)
                for item in [*sample.get('processes',[]),sample.get('outer_process')]:
                    if item is not None:births[int(item['pid'])]=int(item['starttime'])
    _require(births, 'Original actual process-birth evidence required')
    from .r10_paper_metrics import _process
    for pid,birth in births.items():
        current=_process(pid)
        _require(not current or current['starttime']!=birth or current['state']=='Z', 'Original source process birth remains active')
    return dict(format='qwen-component-plan-cache-admission-v1',facts=facts,registry=rows[0],
        process_births={str(k):v for k,v in births.items()},registry_checked_by_outer=True,all_process_births_closed=True)


def persist(path, proof):
    if proof is None:return
    from .r10_paper_metrics import write_json
    path=Path(path)/PROOF
    _require(not path.exists() and not path.is_symlink(), 'Preserve an existing cache admission proof')
    write_json(path,proof)
    shutil.copyfile(__file__,path.parent/Path(__file__).name)


def worker_plan(args, source, segment, recorder):
    """No registry/process/config reads in this worker revalidation path."""
    if not eligible(source):return None
    from . import qwen_layered_structure as layered
    case=Path(args.source_case).resolve();native_case=Path(args.codegen).resolve().parent
    with recorder.span('validated_component_plan_cache_hit',cache_hit=True):
        proof_path=native_case/PROOF
        _require(not proof_path.is_symlink() and _read(native_case/'source-record.json')==source,
            'Canonical original outer source record required')
        proof=_read(proof_path)
        _require(proof.get('format')=='qwen-component-plan-cache-admission-v1'
            and proof.get('registry_checked_by_outer') is True and proof.get('all_process_births_closed') is True,
            'Explicit original outer cache admission proof required')
        row=proof.get('registry',{});record=_read(case/'result.json')
        _require(set(row)==FIELDS and row.get('run_id')==case.name and row.get('artifacts_dir')==str(case)
            and all(row.get(k)==record.get(k) for k in ('round','model','mode'))
            and row.get('metrics',{}).get('status')=='ok'
            and isinstance(proof.get('process_births'),dict) and proof['process_births']
            and all(str(pid).isdigit() and type(birth) is int and birth>=0 for pid,birth in proof['process_births'].items()),
            'Original outer registry/process proof identity differs')
        facts=_artifacts(case,source)
        _require(proof.get('facts')==facts, 'Current component artifacts differ from the exact admitted source')
        name=segment.role+('' if segment.index is None else str(segment.index))
        artifact=facts['components'][name]
        _require(_same(layered.source_binding(segment))==artifact['source'], 'Loaded callback source differs from accepted cached plan')
        plan=_read(artifact['plan_path']);historical=facts['events']['components'][name]
        verified=dict(boots=plan['boots'],calls=plan['calls'],inner_boots=plan['boots'],
            verification_origin='prior closed retained13 component cost conditional_plan_replay',
            fresh_verify_executed=False,historical_stages=historical['historical_stages'])
        return dict(plan=plan,verification=verified,count=artifact['count'],
            provenance=dict(cache_hit=True,fresh_planner_executed=False,fresh_verify_executed=False,
                fresh_complete_physical_count_required=True,source_case=str(case),plan_path=artifact['plan_path'],
                historical_stages=historical['historical_stages'],clock_scope=facts['clock_scope']))


def compare_count(cached, fresh):
    _require(_same(cached['stats'])==_same(fresh['stats']) and _same(cached['interface'])==_same(fresh['interface']),
        'Fresh complete physical Count differs from the accepted cached Count')
