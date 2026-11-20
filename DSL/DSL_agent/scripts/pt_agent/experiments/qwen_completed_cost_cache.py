"""Reuse a complete retained13 cost after its sole logical-constant rejection.

The caller first performs its original source admission. Only outer admission
reads registry/process state. Workers recheck canonical saved metadata and the
original candidate AST, without rerunning Count or inheriting quality/C2.
"""
import ast
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import shutil

from . import qwen_component_plan_cache as plans

GIB=1024**3
OLD_CONST_BYTES=4*GIB
NEW_CONST_BYTES=32*GIB
MODULE='pt_agent.experiments.qwen_retained_component_replay'
SOURCE_ROUND='autonomous-qwen-retained13-component-program-plan'
SOURCE_MODE='warm-retained13-full-model-component-program-plan-no-new-provider'
POLICY=plans.POLICY
PROOF='completed-cost-cache-admission.json'
FORMAT='qwen-completed-resource-rejected-cost-v1'


def _need(value,message):
    if not value:raise ValueError(message)


def _json(value):return json.loads(json.dumps(value,allow_nan=False))


def _canonical(path):
    path=Path(path)
    _need(path.is_absolute()and not path.is_symlink()and path.resolve()==path,'Require an explicit canonical owned path')
    return path


def _read(path):
    path=_canonical(path)
    return json.loads(path.read_text())


def _ast(source):
    _need(isinstance(source,str),'Original candidate source must be text')
    return ast.dump(ast.parse(source),include_attributes=False)


def _argument(argv,name):
    _need(argv.count(name)==1 and argv.index(name)+1<len(argv),'One saved source argument required: '+name)
    return argv[argv.index(name)+1]


def _source(case,expected):
    record=_read(case/'result.json');complete=_read(case/'completion.json')
    _need(record.get('round')==SOURCE_ROUND and record.get('mode')==SOURCE_MODE and record.get('model')=='Qwen2-0.5B'
        and record.get('status')=='quality_rejected'and record.get('candidate_frozen')is False
        and record.get('source')==_json(expected)and record.get('api_calls')==0
        and record.get('report_samples')==0 and record.get('termination')is None
        and record.get('terminal_budget_error')is None and record.get('error')is None
        and record.get('monitor_error')is None
        and complete.get('status')==record['status']and complete.get('round')==record.get('round')
        and complete.get('required_artifacts_persisted')is True and complete.get('registry_persisted')is True,
        'Require the closed normal retained resource rejection, no freeze/API/report/guard failure')
    _need(expected.get('component_cost_policy')==POLICY
        and expected.get('retained_candidate')==13 and expected.get('new_provider_calls_allowed')==0
        and expected.get('original_layers')==24,
        'Retain original24-layer candidate13 component policy and no Provider')
    argv=_read(case/'command-1.json')
    _need(isinstance(argv,list)and all(isinstance(v,str)for v in argv)and argv.count(MODULE)==1
        and argv[argv.index(MODULE)-1]=='-m'
        and _argument(argv,'--worker-phase')in ('calibration','prompt')
        and _canonical(_argument(argv,'--session'))==case/'session',
        'Original retained worker command must bind this exact session')
    partial=_canonical(_argument(argv,'--partial-from'))
    local=_canonical(case/'session/sources/iteration-13.ptdsl')
    original=_canonical(partial/'session/sources/iteration-13.ptdsl')
    source_ast=_ast(local.read_text())
    _need(source_ast==_ast(original.read_text()),'Original paid reply13 AST changed')
    phases=record.get('phases',[])
    _need([r.get('phase')for r in phases]==['calibration','prompt','step','step'],
        'Retain the complete original evaluate/empty-confirm phase chain')
    for i,row in enumerate(phases,1):
        observed=row.get('process',{});worker=row.get('worker',{})
        _need(type(observed.get('exit_code'))is int and observed['exit_code']==0
            and observed.get('timed_out')is False and observed.get('signal')is None
            and observed.get('remaining_pids')==[] and observed.get('signals_sent')==[]
            and worker.get('status')in ('ok','running','quality_rejected')
            and _read(case/'phases'/(str(i).zfill(3)+'-'+row['phase'])/'worker-result.json')==worker,
            'Original source phase must be clean and equal its saved worker')
    return record,source_ast,local,original


def _selection(case):
    from ..planners.transformer_session import load_state
    state=load_state(_canonical(case/'session/SESSION.json'))
    _need(state.get('phase')=='done'and state.get('iteration')==14 and state.get('calls')==13
        and state.get('failed')is None and state.get('pending')is None
        and state.get('max_iters')==13 and state.get('tuner')is False
        and state.get('prompt_version')=='r8-v11'and state.get('placement')=='program-plan'
        and state.get('confirm_rule')=='c2'and state.get('component_cost_policy')==POLICY
        and state.get('new_provider_calls_allowed')==0 and state.get('pool')==[]
        and state.get('records')=={} and state.get('best')is None
        and state.get('best_program')is None,'No accepted/foreign cost candidate may enter this cache')
    confirmation=state.get('confirmation')or{}
    _need(confirmation.get('selected')is None and confirmation.get('evals')==[],
        'Only original empty-pool C2 with zero candidate evaluations is eligible')
    agent=_read(case/'session/AGENT.json')
    _need(agent.get('status')=='confirm_failed'and agent.get('confirm')==_json(confirmation)
        and agent.get('best')is None and agent.get('best_program')is None,
        'Final original Agent must preserve the empty unaccepted pool')
    _need(not any((case/'session'/name).exists()for name in ('FROZEN.json','program.ptdsl')),
        'Frozen/program artifacts are forbidden for the rejected source')
    marker=_read(case/'session/component-c2-access.json')
    _need(marker.get('status')=='completed'and marker.get('calls')==13
        and marker.get('policy')==POLICY and marker.get('report_samples')==0,
        'Original empty-pool terminal transaction must be durably closed')
    history=state.get('history',[])
    _need([r.get('iteration')for r in history]==list(range(1,14))
        and history[-1].get('stage')=='resources'and history[-1].get('feasible')is False,
        'Original retained13 resource history is incomplete')
    replay=_read(case/'session/guarded-replay.json')
    _need(replay.get('status')=='committed'and replay.get('HTTP_added')==0
        and replay.get('committed_calls')==13,'Original reply13 must commit without another HTTP call')
    return history[-1]


def _cost(case,source_ast,local):
    roots=list((case/'session').glob('component-cost-*'))
    _need(roots==[case/'session/component-cost-1'],'Require the unique original candidate13 cost directory')
    root=_canonical(roots[0]);cost=_read(root/'result.json')
    _need(cost.get('status')=='ok'and cost.get('placement_model')=='component-program-plan'
        and cost.get('lowering_policy')==POLICY
        and _read(root/'partial.json')==dict(status='running',completed=25,components=list(plans.NAMES)),
        'All25 original component measurements must have completed normally')
    rows=cost.get('refresh_plan',{}).get('components')
    _need(isinstance(rows,list)and [r.get('name')for r in rows]==list(plans.NAMES),
        'Retain all24 original layers followed by the vocabulary head')
    raw=Counter();rotations=set();bindings=[];artifacts={};sum_consts=0;max_consts=0
    for name,row in zip(plans.NAMES,rows):
        directory=root/name;path=plans._inside(row['plan'],root)
        _need(path==directory/'refresh-plan.json','Original component plan path differs')
        binding_path=plans._inside(directory/'source-binding.json',root)
        count_path=plans._inside(directory/'complete-count.json',root)
        binding=_read(binding_path);counted=_read(count_path)
        boundary=19 if name=='head'else 32
        plan=plans.validate_plan(_read(path),19 if name=='head'else 34)
        stats=counted.get('stats',{});interface=counted.get('interface',{})
        counts=stats.get('counts',{})
        _need(isinstance(counts,dict)and set(counts)=={'add','sub','neg','mul_cc','mul_cp','rot','boot'}
            and all(type(v)is int and v>=0 for v in counts.values())
            and stats.get('ops_total')==sum(counts.values())
            and type(stats.get('distinct_consts'))is int and stats['distinct_consts']>=0
            and type(stats.get('max_depth'))is int and stats['max_depth']>=0
            and type(stats.get('max_total_depth'))is int and stats['max_total_depth']>=stats['max_depth']
            and type(stats.get('min_level'))is int and 0<=stats['min_level']<=5
            and stats.get('outputs')==boundary and type(stats.get('rot_steps'))is list
            and all(type(v)is int and 0<v<16384 for v in stats['rot_steps']),
            'Original full raw Count inventory/schema is incomplete')
        _need(interface.get('status')=='ok'and interface.get('source_binding')==binding
            and interface.get('explicit_boundary_bootstrap_calls')==boundary
            and interface.get('internal_plan_bootstrap_calls')==plan['boots']
            and interface.get('symbolic_output_levels')==[5]*boundary
            and counts['boot']==plan['boots']+boundary
            and row.get('internal_boots')==plan['boots']and row.get('boundary_boots')==boundary
            and row.get('output_levels')==[5]*boundary,'Original component Count/plan/I/O contract differs')
        raw.update(counts);rotations.update(stats['rot_steps'])
        sum_consts+=stats['distinct_consts'];max_consts=max(max_consts,stats['distinct_consts'])
        bindings.append(binding);artifacts[name]=dict(plan=str(path),binding=binding,count=counted,
            stat_provenance={p.name:plans._stat(p)for p in (path,binding_path,count_path)})
    _need(_json(plans._typed_bindings(dict(program=str(local)),bindings))==bindings,
        'Complete original reply13 typed component source differs')
    _need(cost.get('complete_count_replay_counts')==dict(raw)
        and cost.get('complete_count_replay_ops_total')==sum(raw.values())
        and cost.get('explicit_boundary_boots')==787
        and cost.get('component_internal_boots')==sum(r['internal_boots']for r in rows)
        and cost.get('boots_pred')==raw['boot']==cost['component_internal_boots']+787,
        'Complete25 aggregate physical inventory differs')
    counts=cost.get('counts',{})
    _need(isinstance(counts,dict)and all(type(v)is int and v>=0 for v in counts.values())
        and sum(counts.values())==cost.get('ops_total')and counts.get('boot')==cost['boots_pred']
        and cost.get('rot_keys')==len(rotations)
        and type(cost.get('max_depth'))is int and cost['max_depth']>=0
        and type(cost.get('latency_pred_us'))in (int,float)and math.isfinite(cost['latency_pred_us'])and cost['latency_pred_us']>=0
        and cost.get('placement_exact')is True and cost.get('global_optimality')is False
        and type(cost.get('distinct_consts'))is int and max_consts<=cost['distinct_consts']<=sum_consts
        and cost.get('const_bytes')==cost['distinct_consts']*16384*8
        and OLD_CONST_BYTES<cost['const_bytes']<=NEW_CONST_BYTES,
        'Only a complete logical constant inventory within the declared new32GiB budget is reusable')
    return root,cost,artifacts


def _facts(case,expected):
    record,source_ast,local,original=_source(case,expected);history=_selection(case)
    root,cost,artifacts=_cost(case,source_ast,local)
    feedback_path=case/'session/full-feedback/iteration-0013.full.json';full=_read(feedback_path)
    feedback=full.get('feedback',{});exact=feedback.get('exact')or{}
    _need(full.get('iteration')==13 and full.get('best_feasible_so_far')is None and full.get('tuned')is None
        and feedback.get('stage')=='resources'and feedback.get('feasible')is False
        and feedback.get('validation')is None and feedback.get('error')is None and feedback.get('stop')is None
        and exact.get('ok')is True and exact.get('first_mismatch')is None
        and feedback.get('errors')==[dict(code='E_RESOURCE',limit='const_bytes',value=cost['const_bytes'],max=OLD_CONST_BYTES)],
        'Require exact math success followed solely by E_RESOURCE const_bytes above4GiB, before heldout acceptance')
    for key in ('latency_pred_us','boots_pred','max_depth','ops_total','rot_keys','placement_model','placement_exact','barriers','boot_share'):
        _need(feedback.get('cost',{}).get(key)==cost.get(key),'Original cost feedback differs: '+key)
    _need(history.get('latency_pred_us')==cost.get('latency_pred_us')and history.get('boots_pred')==cost.get('boots_pred'),
        'Original rejected history and full aggregate differ')
    events=plans._events(case)
    from .r10_paper_metrics import read_events,summarize_events
    spans=summarize_events(read_events(events['path']))['spans']
    exacts=[s for s in spans if s['stage_id']=='independent_exact_math']
    costs=[s for s in spans if s['stage_id']=='cost']
    _need(len(exacts)==1 and exacts[0]['status']=='ok'and exacts[0]['end_ns']<=costs[0]['start_ns']
        and not any(s['stage_id']=='heldout_validation'for s in spans),
        'Successful original exact math must precede the complete cost, with no heldout evaluation')
    clocks=dict(cost={k:costs[0][k]for k in ('event_id','inclusive_ns','exclusive_ns')},
        independent_exact_math={k:exacts[0][k]for k in ('event_id','inclusive_ns','exclusive_ns')},
        components=events['components'])
    metadata=(case/'result.json',case/'completion.json',case/'session/SESSION.json',feedback_path,
        root/'result.json',root/'partial.json',local,original,Path(events['path']))
    return dict(source_case=str(case),source=_json(expected),source_ast=source_ast,
        original_reply13=str(original),local_reply13=str(local),cost_directory=str(root),cost=cost,
        components=artifacts,events=events['path'],historical_events=clocks,
        sole_rejection=dict(code='E_RESOURCE',limit='const_bytes',value=cost['const_bytes'],max=OLD_CONST_BYTES),
        original_C2_candidate_evaluations=0,heldout_or_C2_acceptance_inherited=False,
        stat_provenance={str(p):plans._stat(p)for p in metadata})


def prepare_outer(case,expected_source):
    """Call after original source admission; only this function probes processes."""
    case=_canonical(case);facts=_facts(case,expected_source);record=_read(case/'result.json')
    rows=[]
    with (case.parent/'registry.jsonl').open()as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row=json.loads(line)
                if row.get('run_id')==case.name:rows.append(row)
    _need(len(rows)==1 and set(rows[0])==plans.FIELDS and rows[0].get('artifacts_dir')==str(case)
        and all(rows[0].get(k)==record.get(k)for k in ('round','model','mode'))
        and rows[0].get('metrics',{}).get('status')==record['status'],
        'Unique original eight-field closed resource-rejected producer required')
    births=set()
    for path in case.rglob('*resources.jsonl'):
        _canonical(path)
        with path.open()as stream:
            for line in stream:
                if not line.strip():continue
                sample=json.loads(line)
                for item in [*sample.get('processes',[]),sample.get('outer_process')]:
                    if item is not None:births.add((int(item['pid']),int(item['starttime'])))
    _need(births,'Actual closed source birth metadata required')
    from .r10_paper_metrics import _process
    for pid,birth in births:
        now=_process(pid)
        _need(not now or now['starttime']!=birth or now['state']in ('Z','X'),'Original producer birth remains active')
    return dict(format=FORMAT,facts=facts,registry=rows[0],registry_checked_by_outer=True,
        process_births=[dict(pid=pid,starttime=birth)for pid,birth in sorted(births)],all_process_births_closed=True)


def persist(directory,proof):
    directory=_canonical(directory);case=Path(proof['facts']['source_case'])
    _need(directory.is_dir()and not directory.is_relative_to(case)and not case.is_relative_to(directory),
        'Save proof outside the immutable original producer case')
    path=directory/PROOF
    with path.open('x')as stream:
        json.dump(proof,stream,allow_nan=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    shutil.copyfile(__file__,directory/Path(__file__).name)
    return path


def load_worker(case,proof_path,expected_source,source_text):
    """Pure saved-metadata revalidation: no registry/proc/config or arithmetic."""
    case=_canonical(case);proof_path=_canonical(proof_path)
    _need(proof_path.name==PROOF and not proof_path.is_relative_to(case),'Explicit canonical independent cache proof required')
    proof=_read(proof_path)
    births=proof.get('process_births')
    _need(proof.get('format')==FORMAT and proof.get('registry_checked_by_outer')is True
        and proof.get('all_process_births_closed')is True and type(births)is list and births
        and all(isinstance(r,dict)and set(r)=={'pid','starttime'}
            and type(r['pid'])is int and r['pid']>0 and type(r['starttime'])is int and r['starttime']>=0 for r in births)
        and len({(r['pid'],r['starttime'])for r in births})==len(births)
        and set(proof.get('registry',{}))==plans.FIELDS,'Original outer closed producer proof required')
    facts=_facts(case,expected_source);row=proof['registry'];record=_read(case/'result.json')
    _need(row.get('run_id')==case.name and row.get('artifacts_dir')==str(case)
        and all(row.get(k)==record.get(k)for k in ('round','model','mode'))
        and row.get('metrics',{}).get('status')==record['status']
        and proof.get('facts')==facts and _ast(source_text)==facts['source_ast'],
        'Current original candidate/cost artifacts differ from the admitted closed source')
    return dict(cost={k:v for k,v in _json(facts['cost']).items()if k!='status'},provenance=dict(
        authority_case=str(case),authority_cost_directory=facts['cost_directory'],
        cost_case=str(case),cost_directory=facts['cost_directory'],events=facts['events'],
        historical_events=facts['historical_events'],policy=POLICY,original_candidate=13,
        cached_cost_reused=True,fresh_cost_estimate_executed=False,fresh_Count_executed=False,
        original_clock_scope='Historical completed25 original cost/math/plan/Count; fresh cache validation is separate',
        heldout_or_C2_acceptance_inherited=False,old_const_bytes_limit=OLD_CONST_BYTES,
        new_const_bytes_limit=NEW_CONST_BYTES,physical_RAM_or_disk_limits_changed=False))
