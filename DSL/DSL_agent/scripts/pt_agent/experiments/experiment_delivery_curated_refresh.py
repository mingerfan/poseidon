"""Finite metadata curation for the predeclared cold VGG and two frozen reports.

Reads explicit saved records with a streaming field selector. Conversations,
program text, sample arrays, weights and keys are never materialized. Historical
delivery rows stay in place; a closed accepted cold case can become primary by
the predeclared source identity, independently of final report accuracy.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path
import resource
import time

from .. import paths
from .experiment_delivery_refresh import atomic_write, DEADLINE
from .live_execution_observer import _Selector, projection, Unavailable

MIB=1024**2
MODEL='vgg16_silu_avgpool'
FACTORY='pt_agent.experiments.r13_common_models:vgg16_silu_avgpool'
PROFILE=Path(__file__).resolve().parents[1]/'compiler_profiles/qwen48/compiler-config.json'
BASE='THREE_CNN_FOUR_SYSTEMS_EXPERIMENTS'
PHASES=('intake','calibration','agent','placement','emit','trace','compile')
PROCESS={k:True for k in ('exit_code','timed_out','signal','started_ns','ended_ns','outer_peak_rss_kib')}
PROCESS.update(signals_sent={'*':{'pid':True,'signal':True}},remaining_pids={'*':True},peak_rss_kib_by_pid={'*':True},limits={'memory_gib':True,'file_gib':True,'wall_seconds':True})
SPAN={k:True for k in ('event_id','stage_id','parent_id','inclusive_ns','exclusive_ns','status')}
METRICS={k:True for k in ('iteration','origin','latency_pred_us','boots_pred','passed','feasible','top1','exact_top1','acc_drop_pp','max_u','max_u_calibrated','max_u_analytic','agree','total','diverged')}
CHECK={k:True for k in ('passed','finite','shape_ok','dtype_ok','max_absolute_error','max_error_ratio','criterion')}
COST={k:True for k in ('boots_pred','latency_pred_us','placement_variant')}
SELECTED={k:True for k in ('iteration','origin','boots_pred','latency_pred_us')}
TUNE={k:True for k in ('feasible','seconds','coarse_evals','full_evals')}
TUNE.update(cost_after=COST,history={'*':dict(METRICS,phase=True)})
TRACE=dict(iteration=True,seconds=True,model_call=True,tuned_reused_from=True,
           fb={'stage':True,'feasible':True,'cost':COST,'validation':METRICS},tuned=TUNE)
WORKER=dict(status=True,error={},frozen=True,tuner_enabled=True,actual_api_calls=True,
            selected_program=True,boots_pred=True,latency_pred_us=True,
            strict_math_probes={'*':{'sample_id':True,'exact':CHECK,'approximate':CHECK}},
            agent={'best_iteration':True,'best_source':True,'llm_calls':True,'prompt_tokens':True,'completion_tokens':True,
                   'confirm':{'selected':SELECTED,'evals':{'*':METRICS}}})
QUALITY=dict(status=True,formal_frozen=True,quality_acceptance_passed=True,strict_math_passed=True,
             model=True,selected_program=True,actual_api_calls=True,session_phase=True,source_result=True,confirmation=METRICS)
COMPLETION={k:True for k in ('status','complete_case_elapsed_seconds','required_artifacts_persisted','registry_persisted','timing_status','elapsed_scope')}
COMPLETION['phase_spans']={'*':SPAN}
PREDICTION=dict(table_known_sum_us=True,scope=True,untimed={'*':True})
PHASE=dict(status=True,process=PROCESS,agent_result=True,boots_pred=True,latency_pred_us=True,
           refresh_plan=True,compiled_bootstraps=True,runtime_plan=True,prediction=PREDICTION,
           execution_ops={'*':True},steps=True)
RESULT={k:True for k in ('round','model','mode','status','variant','api_delta','compiled_bootstraps','monitor_error','resource_monitor_error','historical_elapsed_reused','original_intake_reused','training_calibration_reused','failed_stage')}
RESULT.update(attempted={'*':True},completed={'*':True},phases={'*':PHASE},processes={'*':{'phase':True,'stage':True,'observation':PROCESS}},source={'factory':True,'compiler_profile':True,'original_sources':{'source':True,'checkpoint':True,'inference_preprocessing_source':True}},quality_audit=True)
REPORT={k:True for k in ('round','mode','model','source_case','status','report_completed','report_samples_read','report_samples_completed','access_marker','llm_calls','compiler_runs','ckks_runs','feedback_to_agent','automatic_retry','batch_size','reference_correct','approximate_correct','finite_samples','report_samples','ids_exactly_test_0_through_7999','reference_top1_percent','approximate_top1_percent','accuracy_drop_pp','all_finite','report_policy_passed','independent_report_observation','used_for_selection')}
REPORT['processes']={'*':{'phase':True,'stage':True,'observation':PROCESS}}
SESSION={k:True for k in ('phase','iteration','calls','max_iters','tuner')}
COMPONENT_KEYS=('covered_operator_core_seconds','bootstrap_galois_generation_seconds',
    'residual_galois_generation_seconds','fixed_key_generation_seconds',
    'constructor_dft_seconds','constructor_other_seconds',
    'plaintext_read_encode_transfer_seconds','cold_parameters_input_encryption_final_seconds')
PROJECTION_NULLS=('formal_complete_prediction_seconds','recording_and_checkpoint_overhead_seconds',
    'native_mathematical_acceptance','security_acceptance','per_cell_repeated_variance')
COMPONENT_PROJECTION=dict(status=True,created_utc=True,measurements_per_cell=True,
    per_cell_repeated_variance=True,full_native_model_clock_inputs=True,
    formal_complete_prediction_seconds=True,task_accuracy_acceptance=True,security_acceptance=True,
    new_actions={'*':True},rows={'*':dict(model=True,source_case=True,source_result=True,source_plan=True,
        source_check={'*':True},source_operations=True,bootstrap_count=True,source_op_counts={'*':True},
        unique_cells=True,covered_cells=True,component_point_seconds={'*':True},
        engineering_component_point_seconds=True,projected_native_variant=True,parameter_group=True,
        symbolic_metadata_mapping=True,profile_prediction_source=True,profile_audit_source=True,
        **dict.fromkeys(PROJECTION_NULLS,True))})


def finite(value):
    return type(value)in(int,float)and math.isfinite(value)


def selected(path,fields,maximum=64*MIB):
    return projection(path,fields,maximum=maximum)


def clean(obs):
    return (isinstance(obs,dict)and type(obs.get('exit_code'))is int and obs['exit_code']==0
            and obs.get('timed_out')is False and obs.get('signal')is None
            and obs.get('signals_sent')==[]and obs.get('remaining_pids')==[])


def elapsed(obs):
    a,b=obs.get('started_ns'),obs.get('ended_ns')
    return(b-a)/1e9 if type(a)is int and type(b)is int and b>=a else None


def same(left,right):
    return isinstance(left,(str,Path))and Path(left).resolve()==Path(right).resolve()


def read_csv(path):
    with Path(path).open()as f:
        reader=csv.DictReader(f);return list(reader),reader.fieldnames


def csv_text(rows,fields):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(rows);return stream.getvalue()


def manifest_value(value,root):
    required={'schema','cold_case','retained_compile_case','retained_session_case','reports'}
    if(not isinstance(value,dict)or not required.issubset(value)
            or set(value)-required-{'cold_component_projection'}
            or value['schema']!='predeclared-vgg-curated-refresh-v1'):
        raise ValueError('Explicit predeclared VGG metadata manifest required')
    root=Path(root).resolve();out=copy.deepcopy(value)
    def case(path):
        if not isinstance(path,str):raise ValueError('Explicit saved case path required')
        p=Path(path).resolve()
        if p==root or not p.is_relative_to(root):raise ValueError('Case must be below configured result root')
        return str(p)
    for key in ('cold_case','retained_compile_case','retained_session_case'):out[key]=case(out[key])
    if len({out[k]for k in ('cold_case','retained_compile_case','retained_session_case')})!=3:
        raise ValueError('Cold, retained compile and retained session identities must differ')
    if not isinstance(out['reports'],list)or len(out['reports'])!=2:raise ValueError('Two explicit independent reports required')
    for row in out['reports']:
        if not isinstance(row,dict)or set(row)!={'case','frozen_case','compiled_case'}:raise ValueError('Explicit report/freeze/compile identities required')
        for key in row:row[key]=case(row[key])
    pairs={(x['frozen_case'],x['compiled_case'])for x in out['reports']}
    if pairs!={(out['cold_case'],out['cold_case']),(out['retained_session_case'],out['retained_compile_case'])}:
        raise ValueError('Reports must bind the predeclared cold and retained frozen identities')
    if len({x['case']for x in out['reports']})!=2:raise ValueError('Report cases must differ')
    if 'cold_component_projection'in out:
        path=out['cold_component_projection']
        if not isinstance(path,str)or not Path(path).is_absolute():
            raise ValueError('Optional cold component projection requires an explicit absolute path')
        out['cold_component_projection']=str(Path(path).resolve())
    return out


def validate_cold_component_projection(artifact,*,case,record):
    """Check a saved component export against an already admitted cold compile.

    This reads no plan, profile, model or measured whole-model clock. The source
    export's original component sum is checked directly; no estimate is fitted.
    """
    if not isinstance(artifact,dict):raise ValueError('Component projection must be an object')
    rows=artifact.get('rows')
    if not isinstance(rows,list)or len(rows)!=1 or not isinstance(rows[0],dict):
        raise ValueError('Exactly one explicitly bound component projection row required')
    row=rows[0];case=Path(case).resolve();phase=record.get('phases',{}).get('compile',{})
    for key,expected in (('source_case',case),('source_result',case/'result.json'),('source_plan',phase.get('runtime_plan'))):
        actual=row.get(key)
        if(not isinstance(actual,str)or not Path(actual).is_absolute()
                or not isinstance(expected,(str,Path))or not same(actual,expected)):
            raise ValueError('Component projection belongs to a different cold source or plan')
    if row.get('model')!='VGG16':raise ValueError('Cold component projection model differs')
    checks=row.get('source_check')
    if not isinstance(checks,dict)or any(checks.get(k)is not True for k in
            ('compiled_path_matches','clean_native_compile','same_model_quality_program_source')):
        raise ValueError('Component export source checks must pass')
    ops=phase.get('execution_ops');source_ops=row.get('source_op_counts')
    if(not isinstance(ops,dict)or not ops or not isinstance(source_ops,dict)
            or any(type(v)is not int or v<0 for v in ops.values())
            or any(type(v)is not int or v<0 for v in source_ops.values())or source_ops!=ops):
        raise ValueError('Component operation counts differ from the actual compile')
    total=sum(ops.values());boots=record.get('compiled_bootstraps')
    if(type(row.get('source_operations'))is not int or row['source_operations']!=total
            or type(boots)is not int or boots<0 or type(phase.get('compiled_bootstraps'))is not int
            or phase['compiled_bootstraps']!=boots or type(row.get('bootstrap_count'))is not int
            or row['bootstrap_count']!=boots or ops.get('boot')!=boots):
        raise ValueError('Component total or bootstrap count differs from the actual compile')
    if phase.get('steps')is not None and(type(phase['steps'])is not int or phase['steps']!=total):
        raise ValueError('Saved compile step count differs from its operation inventory')
    if(any(type(row.get(k))is not int or row[k]!=38 for k in ('unique_cells','covered_cells'))
            or type(artifact.get('measurements_per_cell'))is not int or artifact['measurements_per_cell']!=1):
        raise ValueError('Require complete saved 38-cell coverage and its one measurement per cell')
    for key in ('formal_complete_prediction_seconds','task_accuracy_acceptance','security_acceptance','per_cell_repeated_variance'):
        if key not in artifact or artifact[key]is not None:
            raise ValueError('Component projection formal, security and variance fields must remain null')
    if any(k not in row or row[k]is not None for k in PROJECTION_NULLS):
        raise ValueError('Component row must retain unknown formal, recording, security and variance fields')
    actions=artifact.get('new_actions')
    if(type(artifact.get('full_native_model_clock_inputs'))is not int or artifact['full_native_model_clock_inputs']!=0
            or not isinstance(actions,dict)or not {'model_runs','crypto_runs','API_calls','explicit_integrity_checks'}.issubset(actions)
            or any(type(v)is not int or v!=0 for v in actions.values())):
        raise ValueError('Component export must use no full native model clock or new experimental actions')
    pieces=row.get('component_point_seconds');point=row.get('engineering_component_point_seconds')
    if(not isinstance(pieces,dict)or set(pieces)!=set(COMPONENT_KEYS)
            or any(not finite(v)or v<0 for v in pieces.values())or not finite(point)or point<0
            or point!=sum(pieces.values())):
        raise ValueError('Engineering point must equal the original finite component sum')
    labels=('projected_native_variant','parameter_group','symbolic_metadata_mapping',
            'profile_prediction_source','profile_audit_source')
    if any(not isinstance(row.get(k),str)or not row[k] for k in labels):
        raise ValueError('Native projection variant, mapping and independent profile provenance required')
    return dict(seconds=point,source_result=row['source_result'],source_plan=row['source_plan'],
        source_case=row['source_case'],component_point_seconds=copy.deepcopy(pieces),
        unique_cells=38,covered_cells=38,measurements_per_cell=1,per_cell_repeated_variance=None,
        formal_complete_prediction_seconds=None,security_acceptance=None,full_native_model_clock_inputs=0,
        **{k:row[k]for k in labels},
        scope='Engineering component point for this exact admitted cold RuntimePlan, using its saved independent cell measurements and static request inventory. '
              'The explicit native variant and symbolic scale mapping are recorded separately from the compiler parameters. '
              'Whole-model native time is not an input; recording overhead, repeated variance and formal latency/security acceptance remain unknown.')


def curate_cold_projection(delivery,manifest,record):
    """Admit only the optional same-cold export; preserve previously known values."""
    case=Path(manifest['cold_case']);cell=next((c for c in delivery['cells']
        if same(c.get('terminal',{}).get('result_path'),case/'result.json')),None)
    if cell is None:return dict(status='cold_not_admitted')
    info=dict(status='not_configured',reason='No explicit cold component projection path configured')
    if cell.get('metadata_admission',{}).get('status')!='accepted' or not cell.get('primary_matrix_cell'):
        info=dict(status='cold_not_admitted',reason='Cold closure, source and quality admission must pass first')
    elif manifest.get('cold_component_projection'):
        path=Path(manifest['cold_component_projection']);info.update(source_projection=str(path))
        try:
            if path.is_symlink()or not path.is_file()or path.stat().st_size>8*MIB:
                raise ValueError('Bounded regular saved component projection required')
            artifact=selected(path,COMPONENT_PROJECTION,maximum=8*MIB)
            admitted=validate_cold_component_projection(artifact,case=case,record=record)
        except(Unavailable,ValueError,KeyError,TypeError,OSError)as exc:
            info.update(status='not_admitted',reason=str(exc),error_class=type(exc).__name__)
        else:
            admitted['source_projection']=str(path)
            cell['engineering_component_projection']=admitted
            cell['predicted_latency_seconds']=admitted['seconds']
            cell['predicted_latency_scope']=admitted['scope']+' Source: '+str(path)
            if str(path)not in cell['sources']:cell['sources'].append(str(path))
            cell['unmeasured']=[v for v in cell.get('unmeasured',[])if not v.startswith('Engineering prediction')]
            info.update(status='accepted',reason='Exact declared cold source/plan and original component sum verified')
    info['previous_nonempty_point_preserved']=info['status']!='accepted'and cell.get('predicted_latency_seconds')is not None
    cell['component_projection_admission']=info
    return info


def observe(case,fields):
    case=Path(case);info=dict(case=str(case),closed=False,status='unobserved')
    try:
        completion=selected(case/'completion.json',COMPLETION)
    except Unavailable:
        # Only bounded status/session projection while active; no quality/count claims.
        try:info['status']=selected(case/'result.json',{'status':True}).get('status','unobserved')
        except Unavailable:pass
        session=case/'work/agent/SESSION.json'
        try:info['session_progress']=selected(session,SESSION)
        except Unavailable:pass
        return info,None,None
    if completion.get('status')in(None,'prepared','running'):
        info['status']=completion.get('status','unobserved');return info,None,None
    if(completion.get('registry_persisted')is not True or completion.get('required_artifacts_persisted')is not True
            or not finite(completion.get('complete_case_elapsed_seconds'))):
        info['status']='closure_not_verified';return info,None,None
    record=selected(case/'result.json',fields)
    info.update(closed=True,status=record.get('status'),completion_status=completion['status'])
    return info,record,completion


def accept_cold(case,record,completion,worker,quality,session):
    if(record.get('round')!='R30'or record.get('model')!=MODEL or record.get('status')!='ok'
            or completion.get('status')!='ok'or record.get('source',{}).get('factory')!=FACTORY
            or record.get('historical_elapsed_reused')is not False
            or record.get('original_intake_reused')is not False or record.get('training_calibration_reused')is not False
            or not same(record.get('source',{}).get('compiler_profile'),PROFILE)):
        raise ValueError('Require an actual fresh original-model R30 cold completion')
    for phase in PHASES:
        if(record.get('attempted',{}).get(phase)!=1 or record.get('completed',{}).get(phase)!=1
                or record.get('phases',{}).get(phase,{}).get('status')!='ok'):
            raise ValueError('Every original cold phase must complete once')
    for key in ('monitor_error','resource_monitor_error'):
        if record.get(key)is not None:raise ValueError('Cold monitor error retained')
    if not record.get('processes')or any(not clean(x.get('observation'))for x in record['processes']):
        raise ValueError('All saved cold children must complete cleanly')
    compile_phase=record['phases']['compile'];boots=record.get('compiled_bootstraps')
    if type(boots)is not int or boots<0 or compile_phase.get('compiled_bootstraps')!=boots or not clean(compile_phase.get('process')):
        raise ValueError('Compiled final count and actual native process must agree')
    plan=Path(compile_phase.get('runtime_plan','')).resolve()
    if not plan.is_relative_to(Path(case)/'post-freeze/emit')or not plan.name.endswith('.runtime-plan.json')or not plan.is_file():
        raise ValueError('The compiled RuntimePlan pointer must belong to this new cold artifact')
    program=Path(case)/'work/agent/program.ptdsl';worker_path=Path(case)/'work/agent/worker-result.json'
    if(not same(record['phases']['agent'].get('agent_result'),worker_path)
            or not same(record.get('quality_audit'),Path(case)/'post-run-quality-audit.json')
            or not same(worker.get('selected_program'),program)or not same(quality.get('selected_program'),program)
            or not same(quality.get('source_result'),worker_path)):
        raise ValueError('Bind saved frozen source and quality to this cold case')
    calls=record.get('api_delta');agent=worker.get('agent')or{};chosen=agent.get('confirm',{}).get('selected')or{}
    if(type(calls)is not int or not 0<=calls<=6 or worker.get('actual_api_calls')!=calls
            or agent.get('llm_calls')!=calls or quality.get('actual_api_calls')!=calls or session.get('calls')!=calls
            or worker.get('status')!='ok'or worker.get('frozen')is not True or worker.get('tuner_enabled')is not True
            or quality.get('session_phase')!='done'or session.get('phase')!='done'
            or any(quality.get(k)is not True for k in ('formal_frozen','quality_acceptance_passed','strict_math_passed'))):
        raise ValueError('Original hybrid policy, successful calls and frozen acceptance must agree')
    confirmation=quality.get('confirmation')or{}
    if(confirmation.get('passed')is not True or confirmation.get('total')!=1000
            or any(not finite(confirmation.get(k))for k in('top1','exact_top1','acc_drop_pp','max_u'))
            or confirmation['acc_drop_pp']>.5 or confirmation['max_u']>.9
            or any(confirmation.get(k)!=chosen.get(k)for k in ('iteration','origin'))):
        raise ValueError('The one saved confirmation must select the frozen candidate')
    checks=[v for p in worker.get('strict_math_probes',[])for k,v in p.items()if k in('exact','approximate')]
    if len(checks)!=6 or any(x.get('passed')is not True or x.get('finite')is not True
            or x.get('criterion')!='1e-12+1e-12*abs(reference)'or not finite(x.get('max_error_ratio'))or x['max_error_ratio']>1 for x in checks):
        raise ValueError('Retain six original strict mathematical checks')
    return checks


def resource_summary(case,record):
    traces=[]
    for path in sorted(Path(case).rglob('*resources.jsonl')):
        if path.is_symlink()or path.stat().st_size>128*MIB:continue
        d=dict(path=str(path),samples=0,single_pid_HWM_KiB=0,max_sampled_VmSize_KiB=0,simultaneous_subtree_RSS_KiB=0,subtree_plus_sampled_supervisor_RSS_KiB=0)
        with path.open()as f:
            for line in f:
                if not line.strip():continue
                try:x=json.loads(line)
                except ValueError:
                    d['malformed_lines']=d.get('malformed_lines',0)+1;continue
                procs=x.get('processes',[]);d['samples']+=1
                if not isinstance(procs,list):continue
                d['single_pid_HWM_KiB']=max([d['single_pid_HWM_KiB']]+[v.get('vmhwm_kib',0)or 0 for v in procs])
                d['max_sampled_VmSize_KiB']=max([d['max_sampled_VmSize_KiB']]+[v.get('vmsize_kib',0)or 0 for v in procs])
                rss=sum(v.get('vmrss_kib',0)or 0 for v in procs)
                if procs:
                    d['simultaneous_subtree_RSS_KiB']=max(d['simultaneous_subtree_RSS_KiB'],rss)
                    d['subtree_plus_sampled_supervisor_RSS_KiB']=max(d['subtree_plus_sampled_supervisor_RSS_KiB'],rss+(x.get('outer_process')or{}).get('vmrss_kib',0))
        if d['samples']:traces.append(d)
    obs=[x.get('observation')or{}for x in record.get('processes',[])]
    hwm=max([x['single_pid_HWM_KiB']for x in traces]+[max(x.get('peak_rss_kib_by_pid',{}).values(),default=0)for x in obs]+[0])
    return dict(single_pid_HWM_KiB=hwm or None,simultaneous_subtree_RSS_KiB=max([x['simultaneous_subtree_RSS_KiB']for x in traces]+[0])or None,
                subtree_plus_sampled_supervisor_RSS_KiB=max([x['subtree_plus_sampled_supervisor_RSS_KiB']for x in traces]+[0])or None,
                max_sampled_VmSize_KiB=max([x['max_sampled_VmSize_KiB']for x in traces]+[0])or None,
                sampling_scope='Saved individual traces only; cross-phase peaks are not added.',limits={'AS_gib':12,'scope':'Original cold child limits remain recorded per process; resource curation changes none.'},traces=traces)


def add_stages(delivery,stage_rows,cell,case,completion):
    for s in completion.get('phase_spans',[]):
        if any(x['source_case']==str(case)and x['event_id']==s['event_id']for x in stage_rows):continue
        row=dict(cell=cell['cell_id'],run=Path(case).name,source_case=str(case),event_id=s['event_id'],stage_id=s['stage_id'],parent_id=s.get('parent_id'),status=s.get('status'),inclusive_ns=s.get('inclusive_ns'),exclusive_ns=s.get('exclusive_ns'),scope='Saved process stage span. Parent and nested spans overlap; do not sum all rows.',source_completion=str(Path(case)/'completion.json'),source_result=str(Path(case)/'result.json'))
        stage_rows.append(row)
        delivery['stage_clock_rows'].append(dict(cell_id=cell['cell_id'],source_case=str(case),event_id=s['event_id'],stage=s['stage_id'],parent_id=s.get('parent_id'),status=s.get('status'),inclusive_seconds=s['inclusive_ns']/1e9 if finite(s.get('inclusive_ns'))else None,exclusive_seconds=s['exclusive_ns']/1e9 if finite(s.get('exclusive_ns'))else None,scope=row['scope']))


def trace_rounds(case,worker):
    path=Path(case)/'work/agent/agent_trace.jsonl'
    if path.is_symlink()or not path.is_file()or path.stat().st_size>64*MIB:raise ValueError('Bounded saved Agent trace required')
    with path.open()as f:
        selector=_Selector(f,64*MIB)
        while True:
            selector.space()
            if not selector.peek():break
            item=selector.value(TRACE)
            if not isinstance(item,dict):raise ValueError('Agent trace metadata object required')
            yield item


def round_rows(case,worker):
    chosen=worker['agent']['confirm']['selected'];evals=worker['agent']['confirm']['evals'];rows=[]
    for x in trace_rounds(case,worker):
        n=x['iteration'];fb=x.get('fb')or{};t=x.get('tuned');candidates=[('raw','llm',fb.get('cost')or{},fb.get('validation')or{},fb.get('feasible'),fb.get('stage'))]
        if isinstance(t,dict):
            final=[h for h in t.get('history',[])if h.get('phase')=='result']
            candidates.append(('tuned','tuned',t.get('cost_after')or{},final[-1]if final else{},t.get('feasible'),'tuner-result'))
        for kind,origin,cost,val,feasible,stage in candidates:
            confirmation=next((e for e in evals if e.get('iteration')==n and e.get('origin')==origin),{})
            rejected=None;codes=[]
            if feasible is False:
                rejected=stage if stage!='complete'else'heldout-validation-policy';codes.append('saved-feasible-false')
            if confirmation.get('passed')is False:rejected='one-time-confirmation';codes.append('confirmation-policy-rejected')
            rows.append(dict(model='VGG16',run=Path(case).name,variant='predeclared-primary-cold-hybrid',iteration=n,candidate_kind=kind,origin=origin,predicted_bootstraps=cost.get('boots_pred'),predicted_latency_us=cost.get('latency_pred_us'),predicted_scope='Saved pre-freeze Count/plan; distinct from post-freeze planner and compiled RuntimePlan.',placement_variant=cost.get('placement_variant'),feasible=feasible,max_u=val.get('max_u'),max_u_calibrated=val.get('max_u_calibrated'),max_u_analytic=val.get('max_u_analytic'),heldout_top1_percent=val.get('top1'),heldout_reference_top1_percent=val.get('exact_top1'),heldout_accuracy_drop_pp=val.get('acc_drop_pp'),heldout_argmax_agree=val.get('agree'),heldout_samples=val.get('total'),quality_split='CIFAR10 test[9000:10000]',saved_evaluation_stage=stage,rejection_stage=rejected,rejection_reason_codes=';'.join(codes),selected_frozen=n==chosen['iteration']and origin==chosen['origin'],selected_iteration=chosen['iteration'],selected_origin=chosen['origin'],confirmation_evaluated=bool(confirmation),confirmation_passed=confirmation.get('passed'),confirmation_top1_percent=confirmation.get('top1'),confirmation_reference_percent=confirmation.get('exact_top1'),confirmation_max_u=confirmation.get('max_u'),confirmation_split='CIFAR10 test[8000:9000]'if confirmation else None,actual_API_response_for_this_round=kind=='raw',LLM_response_seconds=x.get('seconds')if kind=='raw'else None,tuner_reported_seconds=t.get('seconds')if kind=='tuned'else None,tuner_reused_from=x.get('tuned_reused_from')if kind=='tuned'else None,new_tune_executed=kind=='tuned'and x.get('tuned_reused_from')is None,tuner_time_scope='Cached tuner duration is retained metadata, not a new elapsed span.'if kind=='tuned'else None,missing_quality_reason='Saved tuner result omits heldout top1; do not substitute raw or confirmation accuracy.'if kind=='tuned'and val.get('top1')is None else None,source_agent_trace=str(Path(case)/'work/agent/agent_trace.jsonl'),source_worker=str(Path(case)/'work/agent/worker-result.json')))
    return rows


def count_row(case,worker,record):
    path=record['phases']['placement']['refresh_plan']
    if not same(path,Path(case)/'post-freeze/refresh-plan.json'):raise ValueError('Current case refresh plan path required')
    plan=selected(path,{'boots':True,'placement':True,'candidates':{'*':{'boots':True}}})
    choices=plan['candidates'];chosen=worker['agent']['confirm']['selected'];confirmed=next(e for e in worker['agent']['confirm']['evals']if e['iteration']==chosen['iteration']and e['origin']==chosen['origin'])
    final=record['compiled_bootstraps']
    if plan.get('boots')!=final:raise ValueError('Actual compiled count differs from selected post-freeze plan')
    return dict(model='VGG16',run=Path(case).name,source_candidate_run=Path(case).name,variant='predeclared-primary-cold-hybrid',cold_full_E2E=True,selected_iteration=chosen['iteration'],selected_origin=chosen['origin'],selected_candidate_pre_freeze_boots=confirmed['boots_pred'],one_time_confirmation_estimated_boots=confirmed['boots_pred'],postfreeze_normalized_depth_dp_boots=choices['depth-dp']['boots'],postfreeze_normalized_depth_dp_prune_boots=choices['depth-dp+prune']['boots'],postfreeze_normalized_stage_cut_boots=choices['stage-cut']['boots'],postfreeze_selected_stage_cut_prune_boots=plan['boots'],actual_compiled_final_boots=final,compiled_minus_pre_freeze_candidate=final-confirmed['boots_pred'],compiled_minus_postfreeze_normalized_depth_dp=final-choices['depth-dp']['boots'],compiled_minus_postfreeze_normalized_depth_dp_prune=final-choices['depth-dp+prune']['boots'],compiled_minus_postfreeze_selected_plan=final-plan['boots'],scope='Separate candidate estimates, normalized planner candidates and actual compiler instructions; no model execution.',source_refresh_plan=path,source_result=str(Path(case)/'result.json'))


def pending_cell(case,exemplar):
    return dict(cell_id='VGG16:Agent:predeclared-primary-cold-hybrid:'+Path(case).name,
                primary_matrix_cell=False,model='VGG16',system='Agent',approximation_variant='predeclared-primary-cold-hybrid',
                method_variant_role='predeclared-primary-pending',bootstrap_count=None,bootstrap_scope=None,
                native_compile_seconds=None,native_compile_success=False,cold_full_E2E_seconds=None,
                full_E2E_scope='Predeclared new cold VGG; no metrics admitted until complete saved closure and source checks.',
                API_calls=None,API_scope='This new cold invocation only; retained session and no-tuner ablation remain separate.',
                quality_observations=[],stage_clocks={},CKKS_parameters=copy.deepcopy(exemplar.get('CKKS_parameters',{})),
                predicted_latency_seconds=None,predicted_latency_scope=None,real_latency_seconds=None,real_latency_scope=None,
                complete_model_CKKS_observed=False,formal_matrix_acceptance=False,error_stage=None,failure_class=None,
                sources=[str(Path(case)/'result.json'),str(Path(case)/'completion.json')],
                unmeasured=['New cold closure','New frozen confirmation/strict mathematics','Independent8000report','Engineering prediction bound to this exact new plan','Complete-model CKKS deferred'],
                terminal=dict(result_status='unobserved',error=None,completion_present=False,complete_case_seconds=None,
                    resource=dict(single_pid_HWM_KiB=None,simultaneous_subtree_RSS_KiB=None,subtree_plus_sampled_supervisor_RSS_KiB=None,limits=None),result_path=str(Path(case)/'result.json')))


def terminal_record(case,record,completion):
    return dict(result_status=record.get('status'),error=None,completion_present=True,
        complete_case_seconds=completion['complete_case_elapsed_seconds'],complete_case_clock_scope=completion.get('elapsed_scope'),
        required_artifacts_persisted=True,registry_persisted=True,timing_status=completion.get('timing_status'),
        remaining_pids=[p for x in record.get('processes',[])for p in(x.get('observation')or{}).get('remaining_pids',[])],
        resource=resource_summary(case,record),result_path=str(Path(case)/'result.json'))


def registry_metadata(root,case):
    records=[];path=Path(root)/'registry.jsonl'
    if not path.is_file():return records
    # Stream only field names and identity; nested metrics/payloads are skipped.
    fields={k:True for k in ('run_id','round','model','mode','artifacts_dir','ts')}
    with path.open()as f:
        selector=_Selector(f,64*MIB)
        while True:
            selector.space()
            if not selector.peek():break
            row=selector.value(fields)
            if isinstance(row,dict)and same(row.get('artifacts_dir'),case):records.append(row)
    return records


def unique_terminal(rows):
    return len([r for r in rows if r.get('mode')!='llm-call'])==1


def upsert(rows,row,keys):
    matches=[i for i,x in enumerate(rows)if all(str(x.get(k))==str(row.get(k))for k in keys)]
    if len(matches)>1:raise ValueError('Duplicate curated metadata identity')
    if matches:rows[matches[0]]=dict(rows[matches[0]],**row)
    else:rows.append(row)


def curate_cold(delivery,tables,manifest,root):
    case=Path(manifest['cold_case']);exemplar=next(c for c in delivery['cells']if c['model']=='VGG16'and c['system']=='Agent')
    cell=next((c for c in delivery['cells']if same(Path(c['terminal']['result_path']).parent,case)),None)
    if cell is None:
        cell=pending_cell(case,exemplar);delivery['cells'].append(cell)
    info,record,completion=observe(case,RESULT)
    if not info['closed']:
        cell['terminal']['result_status']=info['status'];return info
    cell['terminal']=terminal_record(case,record,completion);add_stages(delivery,tables['stages'],cell,case,completion)
    rows=registry_metadata(root,case);cell['registry']={str(case):rows}
    cell['registry_summary']=dict(terminal_matches=sum(x.get('mode')!='llm-call'for x in rows),terminal_unique=unique_terminal(rows),API_usage_rows=sum(x.get('mode')=='llm-call'for x in rows))
    delivery.setdefault('registry_checks',{}).setdefault('cases',{})[str(case)]=dict(rows=rows,**cell['registry_summary'])
    cell['API_calls']=record.get('api_delta')
    if record.get('status')!='ok'or completion.get('status')!='ok':
        cell.update(method_variant_role='predeclared-primary-closed-error',failure_class='Closed original case error; cold success not admitted.',error_stage=record.get('failed_stage'))
        info['admission']='closed_error';return info
    try:
        worker=selected(case/'work/agent/worker-result.json',WORKER)
        quality=selected(case/'post-run-quality-audit.json',QUALITY)
        session=selected(case/'work/agent/SESSION.json',SESSION)
        checks=accept_cold(case,record,completion,worker,quality,session)
        if not unique_terminal(rows):raise ValueError('Unique persisted terminal registry required')
        candidates=round_rows(case,worker);count=count_row(case,worker,record)
        if sum(x['candidate_kind']=='raw'for x in candidates)!=record['api_delta']or sum(x['selected_frozen']for x in candidates)!=1:
            raise ValueError('Round metadata must retain all calls and one selected frozen identity')
    except (ValueError,KeyError,TypeError,OSError)as exc:
        info.update(admission='metadata_not_admitted',metadata_error_class=type(exc).__name__)
        cell['metadata_admission']=dict(status='not_admitted',error_class=type(exc).__name__)
        return info
    chosen=worker['agent']['confirm']['selected'];confirmation=quality['confirmation']
    for old in delivery['cells']:
        if old is not cell and old['model']=='VGG16'and old['system']=='Agent'and old.get('primary_matrix_cell'):
            old['primary_matrix_cell']=False
            old['method_variant_role']='retained-hybrid-comparison'if old.get('postfreeze_compilation_only')else old.get('method_variant_role','historical-comparison')
    cell.update(primary_matrix_cell=True,method_variant_role='predeclared-primary-cold-hybrid',bootstrap_count=record['compiled_bootstraps'],
        bootstrap_scope='Actual compiled RuntimePlan ciphertext instructions from this new cold hybrid source.',native_compile_success=True,
        native_compile_seconds=elapsed(record['phases']['compile']['process']),cold_full_E2E_seconds=completion['complete_case_elapsed_seconds'],
        full_E2E_scope='This new cold intake/calibration/Agent+confirmation/placement/Emit/trace/archive/compile through durable closure; separate frozen report excluded.',
        candidate_attribution=dict(selected_iteration=chosen['iteration'],selected_origin=chosen['origin'],tuner_enabled=True,prompt_tokens=worker['agent'].get('prompt_tokens'),completion_tokens=worker['agent'].get('completion_tokens')),
        original_model_sources=record['source'].get('original_sources'),
        total_session_API_calls=record['api_delta'],strict_math=dict(split='test:9000..9002',checks=6,passed=True,criterion='1e-12+1e-12*abs(reference)',maximum_absolute_error=max(x['max_absolute_error']for x in checks),maximum_error_ratio=max(x['max_error_ratio']for x in checks)),
        error_stage=None,failure_class=None,metadata_admission=dict(status='accepted',selection_basis='Predeclared cold case identity and existing frozen confirmation/strict mathematics; final report is not used.'),
        unmeasured=['Independent8000report until its own closure','Engineering prediction tied to this exact new compiled plan','Complete-model CKKS deferred','Formal latency/security acceptance'])
    upsert(cell['quality_observations'],dict(kind='one-time frozen confirmation',split='CIFAR10 test[8000:9000]',samples=1000,**confirmation),('kind','split'))
    spans=completion.get('phase_spans',[]);rootid=next((s['event_id']for s in spans if s.get('stage_id')=='fresh_end_to_end'),None)
    cell['stage_clocks']={s['stage_id']:s['inclusive_ns']/1e9 for s in spans if s.get('parent_id')==rootid and finite(s.get('inclusive_ns'))}
    cell['Agent_substage_clocks']={name:sum(s['inclusive_ns']for s in spans if s.get('stage_id')==name and finite(s.get('inclusive_ns')))/1e9 for name in('llm_call','agent_tune','agent_evaluate','agent_confirm_select','strict_packed_arithmetic')}
    pred=record['phases']['compile'].get('prediction')or{}
    cell['external_compiler_profile_partial_sum']=dict(seconds=pred['table_known_sum_us']/1e6 if finite(pred.get('table_known_sum_us'))else None,untimed=pred.get('untimed'),scope=pred.get('scope'),measured_local_CKKS=False,formal_prediction=False)
    cell['planner_profile_latency_seconds']=record['phases']['placement']['latency_pred_us']/1e6
    for row in candidates:upsert(tables['rounds'],row,('run','iteration','candidate_kind'))
    upsert(tables['counts'],count,('run',));info['admission']='accepted'
    for p in(case/'work/agent/worker-result.json',case/'post-run-quality-audit.json',case/'post-freeze/refresh-plan.json'):
        if str(p)not in cell['sources']:cell['sources'].append(str(p))
    info['component_projection']=curate_cold_projection(delivery,manifest,record)
    if cell.get('predicted_latency_seconds')is not None:
        cell['unmeasured']=[v for v in cell['unmeasured']if not v.startswith('Engineering prediction')]
    return info


def accept_report(case,record,completion,binding):
    if(record.get('status')!='ok'or completion.get('status')!='ok'or record.get('model')!=MODEL
            or not same(record.get('source_case'),binding['frozen_case'])or record.get('report_completed')is not True
            or any(record.get(k)!=8000 for k in('report_samples_read','report_samples_completed','report_samples'))
            or type(record.get('finite_samples'))is not int or not 0<=record['finite_samples']<=8000
            or record.get('all_finite')is not(record['finite_samples']==8000)
            or record.get('ids_exactly_test_0_through_7999')is not True
            or record.get('batch_size')!=1 or any(record.get(k)!=0 for k in('llm_calls','compiler_runs','ckks_runs'))
            or any(record.get(k)is not False for k in('feedback_to_agent','automatic_retry','used_for_selection'))
            or record.get('independent_report_observation')is not True):
        raise ValueError('Complete independent finite8000report aggregate required')
    if not record.get('processes')or any(not clean(x.get('observation'))for x in record['processes']):
        raise ValueError('Report process must complete cleanly')
    for correct,percent in(('reference_correct','reference_top1_percent'),('approximate_correct','approximate_top1_percent')):
        n=record.get(correct);p=record.get(percent)
        if type(n)is not int or not 0<=n<=8000 or not finite(p)or abs(p-n/80)>1e-10:
            raise ValueError('Report aggregate counts and accuracy disagree')
    if not finite(record.get('accuracy_drop_pp'))or abs(record['accuracy_drop_pp']-(record['reference_correct']-record['approximate_correct'])/80)>1e-10:
        raise ValueError('Preserve observed report accuracy difference')
    marker=Path(case).parent/'frozen-cnn-report-access'/(Path(binding['frozen_case']).name+'.json')
    if not same(record.get('access_marker'),marker):raise ValueError('One frozen-identity access marker required')
    claim=selected(marker,{'source_case':True,'out':True})
    if not same(claim.get('source_case'),binding['frozen_case'])or not same(claim.get('out'),case):
        raise ValueError('Report access claim belongs to a different source or output')


def curate_report(delivery,tables,binding,root):
    case=Path(binding['case']);info,record,completion=observe(case,REPORT)
    if not info['closed']:return info
    cell=next((c for c in delivery['cells']if same(Path(c['terminal']['result_path']).parent,binding['compiled_case'])),None)
    if cell is None:info['admission']='compiled_source_not_admitted';return info
    if not cell.get('native_compile_success'):
        info['admission']='compiled_source_not_admitted';return info
    rows=registry_metadata(root,case)
    delivery.setdefault('registry_checks',{}).setdefault('cases',{})[str(case)]=dict(rows=rows,terminal_unique=unique_terminal(rows),terminal_matches=sum(x.get('mode')!='llm-call'for x in rows))
    try:
        if not unique_terminal(rows):raise ValueError('Unique report registry terminal required')
        accept_report(case,record,completion,binding)
    except(ValueError,KeyError,TypeError,OSError)as exc:
        info.update(admission='closed_report_not_admitted',metadata_error_class=type(exc).__name__)
        upsert(cell.setdefault('report_terminal_observations',[]),dict(case=str(case),status=record.get('status'),completion_status=completion.get('status'),samples_completed=record.get('report_samples_completed'),complete_case_seconds=completion.get('complete_case_elapsed_seconds'),aggregate_accuracy_admitted=False,source_result=str(case/'result.json')),('case',))
        return info
    # Quality rejection is a report observation; it never chooses/demotes a candidate.
    report=dict(kind='independent frozen report',split='CIFAR10 test[0:8000]',**record,
                complete_case_seconds=completion['complete_case_elapsed_seconds'],source_result=str(case/'result.json'))
    report.pop('processes',None)
    upsert(cell['quality_observations'],report,('kind','source_result'))
    if str(case/'result.json')not in cell['sources']:cell['sources'].append(str(case/'result.json'))
    add_stages(delivery,tables['stages'],cell,case,completion)
    cell['unmeasured']=[s for s in cell.get('unmeasured',[])if not s.startswith('Independent8000')]
    info['admission']='accepted_independent_observation';return info


def coverage(delivery):
    rows=[x for x in delivery['cells']if x.get('primary_matrix_cell')];c=delivery['coverage']
    main_rows=[x for x in rows if x['system']=='Agent']
    c.update(primary_cells=len(rows),primary_cells_with_saved_terminal_results=sum(x['terminal']['result_status']not in('prepared','running','unobserved')for x in rows),supplemental_variant_rows=len(delivery['cells'])-len(rows),successful_native_compiler_cells=sum(x.get('native_compile_success')is True for x in rows),main_method_successful_native_compiler_cells=sum(x.get('native_compile_success')is True for x in main_rows))
    c['main_method_compiler_cells_total']=len(main_rows)
    c['main_method_Agent_total_cells']=len(main_rows)
    c['main_method_Agent_compiled_cells']=c['main_method_successful_native_compiler_cells']
    c['main_method_Agent_cold_success_cells']=sum(finite(x.get('cold_full_E2E_seconds'))for x in main_rows)
    c['main_method_Agent_cold_scope']='Current admitted main-method cold successes; retained post-freeze clocks and no-tuner ablation remain separate historical rows.'
    c['successful_native_compiler_cells_scope']='Primary model/system artifact inventory; retained, cold and ablation identities are explicit on each row.'
    def reported(x):return any(q.get('kind')=='independent frozen report'for q in x.get('quality_observations',[]))
    c['independent_report8000_cells']=sum(reported(x)for x in rows);c['main_method_independent_report8000_cells']=sum(reported(x)for x in main_rows)
    c['experimental_inventory_report8000_cells_including_ablation']=sum(reported(x)for x in delivery['cells'])
    c['experimental_inventory_report8000_scope']='All admitted cells with their own independent frozen report, including historical continuation and no-tuner ablation rows; each cell counted once.'
    c['missing_independent_report8000_cells']=[x['cell_id']for x in rows if not reported(x)]
    c['main_method_report_scope']='Independent test0..7999 aggregates admitted only from the explicit frozen source/report access identity. Outcomes never select a candidate.'
    c['missing_successful_native_compiler_cells']=[x['cell_id']for x in rows if not x.get('native_compile_success')]
    c['missing_complete_model_CKKS_output_cells']=[x['cell_id']for x in rows if not x.get('complete_model_CKKS_observed')]
    c['cells_with_any_bootstrap_inventory']=sum(x.get('bootstrap_count')is not None for x in rows)
    c['cells_without_bootstrap_inventory']=[x['cell_id']for x in rows if x.get('bootstrap_count')is None]
    c['interpretation']='Primary selection follows the predeclared cold source after saved closure/confirmation/math checks. Report outcomes remain independent. Cold, retained compilation and no-tuner ablation clocks stay separate.'


def metadata_rules(delivery,tables):
    """Describe the current inventory without retaining historical count literals."""
    cells=delivery['cells'];primary=[c for c in cells if c.get('primary_matrix_cell')]
    delivery['comparison_rules']=[
        f"{len(primary)} primary cells and {len(cells)-len(primary)} supplemental rows are identified by primary_matrix_cell and method_variant_role. Current admitted primary cold sources, retained continuations and no-tuner ablations retain separate provenance.",
        'Same-backend fixed-polynomial controls retain their control role; official DaCapo, Orbit and HEIR rows retain their own program, parameter and artifact scopes.',
        'Bootstrap inventories retain their units: RuntimePlan ciphertext instructions, Orbit graph nodes, HEIR static IR sites and Count estimates are reported separately.',
        'Compiler process clocks, durable wrapper clocks, cold E2E clocks, continuation clocks, independent report clocks and native client clocks retain their recorded boundaries. Failed resource or layout attempts describe that attempt.',
        'Independent frozen test0..7999 reports never select a candidate or feed Agent feedback. Heldout9000..9999 and one-time confirmation8000..8999 retain their separate roles.',
        'Historical variants retain their original values and source identities. Current primary sources are selected only by the declared cold policy and saved source/confirmation/mathematical checks.',
        'Engineering component points are labelled with their exact plan, independent profile, native variant and symbolic metadata mapping; whole-model clocks, security acceptance and formal complete predictions remain separate fields.',
        'Coverage reports the current artifact inventory. Missing formal matrix evidence remains explicit.'
    ]
    stages=tables['stages'];rounds=tables['rounds'];counts=tables['counts']
    raw=sum(x.get('candidate_kind')=='raw'for x in rounds)
    tuned=sum(x.get('candidate_kind')=='tuned'for x in rounds)
    missing=sum(x.get('candidate_kind')=='tuned'and x.get('heldout_top1_percent')in(None,'')for x in rounds)
    delivery['attribution_exports']['rules']=[
        f'{len(stages)} saved stage rows retain original integer nanoseconds and source clocks. Parent and nested spans overlap.',
        'AGENT_ROUNDS retains raw and tuned origins and each saved frozen selected identity; conversations, program text and error text are excluded.',
        f'{raw} raw rows and {tuned} tuned rows are present in the current candidate table. Cached tuner duration is labelled with its original execution source.',
        f'{missing} tuned rows have no saved heldout top1. Missing heldout values remain empty; independent confirmation values have their own columns.',
        f'{len(counts)} count-scope rows retain candidate estimates, normalized post-freeze planner alternatives and actual compiled final counts separately, each bound to its own source.',
        'Primary cold, retained hybrid continuation and no-tuner ablation rows keep their distinct source identities, placement counts and elapsed clocks. Report accuracy never changes these roles.'
    ]


def main_row(cell,old):
    row=dict(old);terminal=cell['terminal'];rr=terminal.get('resource')or{};at=cell.get('candidate_attribution')or{};sub=cell.get('Agent_substage_clocks')or{}
    reports=[q for q in cell['quality_observations']if q.get('kind')=='independent frozen report'];conf=next((q for q in cell['quality_observations']if q.get('kind')in('one-time frozen confirmation','one-time frozen continuation confirmation')),{});q=reports[-1]if reports else conf
    row.update(cell_id=cell['cell_id'],primary_matrix_cell=cell['primary_matrix_cell'],model=cell['model'],system=cell['system'],method_variant_role=cell['method_variant_role'],approximation_variant=cell['approximation_variant'],status=terminal['result_status'],bootstrap_count=cell['bootstrap_count'],bootstrap_scope=cell['bootstrap_scope'],native_compile_success=cell['native_compile_success'],native_compile_seconds=cell['native_compile_seconds'],cold_full_E2E_seconds=cell['cold_full_E2E_seconds'],wrapper_complete_seconds=terminal.get('complete_case_seconds'),full_E2E_scope=cell['full_E2E_scope'],API_calls=cell['API_calls'],API_scope=cell['API_scope'],selected_iteration=at.get('selected_iteration'),selected_origin=at.get('selected_origin'),tuner_enabled=at.get('tuner_enabled'),LLM_seconds=sub.get('llm_call'),tuner_seconds=sub.get('agent_tune'),quality_kind=q.get('kind'),quality_split=q.get('split'),quality_samples=q.get('report_samples_completed',q.get('samples',q.get('total'))),reference_accuracy_percent=q.get('reference_top1_percent',q.get('exact_top1')),approximate_accuracy_percent=q.get('approximate_top1_percent',q.get('top1')),accuracy_drop_pp=q.get('accuracy_drop_pp',q.get('acc_drop_pp')),max_u=q.get('max_u'),confirmation_reference_percent=conf.get('exact_top1'),confirmation_approximate_percent=conf.get('top1'),confirmation_max_u=conf.get('max_u'),report_complete_seconds=reports[-1].get('complete_case_seconds')if reports else None,predicted_latency_seconds=cell['predicted_latency_seconds'],predicted_latency_scope=cell['predicted_latency_scope'],real_latency_seconds=cell['real_latency_seconds'],real_latency_scope=cell['real_latency_scope'],complete_model_CKKS_observed=cell['complete_model_CKKS_observed'],formal_matrix_acceptance=cell['formal_matrix_acceptance'],error_stage=cell['error_stage'],failure_class=cell['failure_class'],peak_single_PID_HWM_KiB=rr.get('single_pid_HWM_KiB'),peak_simultaneous_subtree_RSS_KiB=rr.get('simultaneous_subtree_RSS_KiB'),peak_subtree_plus_sampled_supervisor_RSS_KiB=rr.get('subtree_plus_sampled_supervisor_RSS_KiB'),result_path=terminal['result_path'],completion_present=terminal.get('completion_present'),registry_matches=cell.get('registry_summary',{}).get('terminal_matches'),unmeasured='; '.join(cell.get('unmeasured',[])),CKKS_parameters_json=json.dumps(cell['CKKS_parameters'],separators=(',',':')),stage_clocks_json=json.dumps(cell['stage_clocks'],separators=(',',':')),resource_limits_json=json.dumps(rr.get('limits'),separators=(',',':')),source_paths_json=json.dumps(cell['sources'],separators=(',',':')),total_session_API_calls=cell.get('total_session_API_calls',cell['API_calls'])if cell['system']=='Agent'else None)
    return row


def curate(delivery,tables,manifest,root,now):
    observations=[]
    try:observations.append(curate_cold(delivery,tables,manifest,root))
    except(ValueError,KeyError,TypeError,OSError)as exc:observations.append(dict(case=manifest['cold_case'],admission='metadata_read_error',metadata_error_class=type(exc).__name__))
    for binding in manifest['reports']:
        try:observations.append(curate_report(delivery,tables,binding,root))
        except(ValueError,KeyError,TypeError,OSError)as exc:observations.append(dict(case=binding['case'],admission='metadata_read_error',metadata_error_class=type(exc).__name__))
    coverage(delivery);delivery['curated_refresh']=dict(observed_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),observations=observations,source_policy='Explicit predeclared source identities only; report outcomes never select or tune.',whole_model_CKKS='deferred per latest user direction',new_model_API_or_numeric_runs=0)
    cold=next((x for x in observations if x.get('case')==manifest['cold_case']),{})
    delivery['current_primary_VGG_cold_trial']=dict(case=manifest['cold_case'],status_observed=cold.get('status','unobserved'),admission=cold.get('admission'),predeclared_primary=True,candidate_not_selected_from_report=True)
    for p in[manifest['cold_case']]+[x['case']for x in manifest['reports']]:
        if p not in delivery['source_metadata_paths']:delivery['source_metadata_paths'].append(p)
    return observations


def run_once(args,manifest,now):
    directory=Path(args.delivery_dir);target=directory/(BASE+'.json')
    if target.is_symlink()or not target.is_file()or target.stat().st_size>8*MIB:raise ValueError('Existing curated experimental JSON required')
    lock=directory/'.curated-refresh.lock'
    fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        delivery=json.loads(target.read_text());tables={};headers={}
        filenames=dict(main=BASE+'.csv',stages='STAGE_ATTRIBUTION.csv',rounds='AGENT_ROUNDS.csv',counts='COMPILED_COUNT_SCOPES.csv')
        for key,name in filenames.items():
            p=directory/name
            if p.is_symlink():raise ValueError('Curated CSV symlinks refused')
            tables[key],headers[key]=read_csv(p)
        old={r['cell_id']:r for r in tables['main']};observations=curate(delivery,tables,manifest,args.results_root,now)
        # Every historical cell stays present. Only the explicitly predeclared row can be added/promoted.
        if not set(old).issubset({c['cell_id']for c in delivery['cells']}):raise ValueError('Historical delivery rows must remain')
        owned_results=[Path(manifest[k])/'result.json'for k in ('cold_case','retained_compile_case')]
        main=[]
        for cell in delivery['cells']:
            previous=old.get(cell['cell_id'])
            if previous is None or any(same(cell['terminal']['result_path'],p)for p in owned_results):
                main.append(main_row(cell,previous or{}))
            else:
                # Other systems and controls retain their original quality schema and values.
                row=dict(previous);row['primary_matrix_cell']=cell['primary_matrix_cell'];main.append(row)
        tables['main']=main
        delivery['attribution_exports'].update(stage_rows=len(tables['stages']),candidate_rows=len(tables['rounds']),compiled_count_rows=len(tables['counts']),raw_rounds=sum(x.get('candidate_kind')=='raw'for x in tables['rounds']),tuned_candidate_rows=sum(x.get('candidate_kind')=='tuned'for x in tables['rounds']),selected_rows=sum(x.get('selected_frozen')in(True,'True')for x in tables['rounds']),tuned_rows_without_heldout_top1=sum(x.get('candidate_kind')=='tuned'and x.get('heldout_top1_percent')in(None,'')for x in tables['rounds']),curated_updated_utc=datetime.fromtimestamp(now,timezone.utc).isoformat())
        metadata_rules(delivery,tables)
        for key,name in filenames.items():atomic_write(directory/name,csv_text(tables[key],headers[key]))
        atomic_write(target,json.dumps(delivery,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        atomic_write(directory/'ATTRIBUTION_EXPORT.json',json.dumps(delivery['attribution_exports'],ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        status=dict(schema='finite-curated-experiment-refresh-v1',observed_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),observations=observations,
            source_manifest=str(args.manifest),phase_rows=len(tables['stages']),candidate_rows=len(tables['rounds']),cell_rows=len(tables['main']),
            historical_rows_preserved=True,new_model_API_CKKS_or_numeric_runs=0,until=args.until,
            preserved_external_data='Existing neighborhood and component-prediction source files remain intact. Only an explicit component export bound to the admitted cold source and exact compiled plan can update its engineering point.')
        atomic_write(directory/'CURATED_REFRESH_STATUS.json',json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        return status
    finally:os.close(fd)


def execute(args,*,now=time.time,sleep=time.sleep):
    if type(args.poll_seconds)is not int or args.poll_seconds not in(30,60):raise ValueError('Use30or60second polling')
    stop=datetime.fromisoformat(args.until.replace('Z','+00:00'));maximum=datetime.fromisoformat(DEADLINE.replace('Z','+00:00'))
    if stop.tzinfo is None or stop>maximum:raise ValueError('Finite timezone-aware cutoff no later than2026-11-17T10:00Z required')
    start=now()
    if stop.timestamp()>start+7*86400:raise ValueError('Finite maximum7day observer duration')
    directory=Path(args.delivery_dir).resolve();root=Path(args.results_root).resolve()
    if directory==root or directory.is_relative_to(root)or root.is_relative_to(directory):raise ValueError('Delivery publication must be separate from raw results')
    manifest=manifest_value(json.loads(Path(args.manifest).read_text()),root);polls=0
    while now()<stop.timestamp():
        run_once(args,manifest,now());polls+=1
        if args.once:break
        sleep(min(args.poll_seconds,max(0,stop.timestamp()-now())))
    return polls


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--delivery-dir',type=Path,required=True)
    parser.add_argument('--results-root',type=Path,default=paths.RESULTS_DIR);parser.add_argument('--poll-seconds',type=int,default=60)
    parser.add_argument('--until',default=DEADLINE);parser.add_argument('--once',action='store_true');args=parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_AS,(256*MIB,256*MIB));execute(args);return 0


if __name__=='__main__':raise SystemExit(main())
