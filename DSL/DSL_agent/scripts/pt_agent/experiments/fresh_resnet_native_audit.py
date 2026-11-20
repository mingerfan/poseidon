"""Read-only terminal audit of the fresh123 C2S52/P61 native experiment.

Only saved JSON, source, events, resource metadata and ten public logits are
read. Pending executions never produce a terminal success page.
"""
import argparse
from collections import Counter
from datetime import datetime
import html
import json
import math
from pathlib import Path
import shutil

from .completion_metrics import artifact_bytes
from .r10_paper_metrics import read_events, summarize_events
from .r31_lattigo_binding import NativeBinding,map_precision_plan
from . import r31_fresh_plan_math as source_math

EXECUTIONS=17863
BOOTSTRAPS=123
POINT_SECONDS=4974.14904172775
BINDING=NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,
    bootstrap_normalization_log2=3,auxiliary_prime_bits=61,c2s_bits=52)
VARIANT=BINDING.variant(32)
REGISTRY_FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
FIXED_KEY_STAGES={'residual.secret_key','residual.public_key','residual.relinearization_key',
    'bootstrap.ephemeral_secret','bootstrap.dense_to_sparse','bootstrap.sparse_to_dense','bootstrap.relinearization_key'}


def read(path):return json.loads(Path(path).read_text())


def same(a,b):return isinstance(a,(str,Path)) and isinstance(b,(str,Path)) and Path(a).resolve()==Path(b).resolve()


def recompute_ckks(values,expected):
    if (not isinstance(values,list) or not isinstance(expected,list) or len(values)!=len(expected) or not values
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in values+expected)):
        raise ValueError('Finite equally sized selected output/reference lists are required')
    rows=[]
    for index,(got,want) in enumerate(zip(values,expected)):
        delta=abs(got-want);allowance=1e-5+1e-4*abs(want)
        rows.append(dict(index=index,observed=got,reference=want,absolute_error=delta,
            original_allowance=allowance,error_ratio=delta/allowance,passed=delta<=allowance))
    return dict(passed=all(r['passed'] for r in rows),elements=len(rows),absolute_tolerance=1e-5,
        relative_tolerance=1e-4,max_absolute_error=max(r['absolute_error'] for r in rows),
        max_error_ratio=max(r['error_ratio'] for r in rows),rows=rows,
        scope='Independent recomputation from saved selected logits, in original frozen-program output units; no model or cryptographic rerun')


def native_events(path):
    """Use the native stage stack; DFT observer events omit an explicit parent."""
    opened={};stack=[];rows=[];issues=[];requests=Counter();maximum=0
    with Path(path).open() as stream:
        for line in stream:
            try:event=json.loads(line)
            except ValueError:issues.append('Incomplete/invalid native event JSON');continue
            maximum=max(maximum,event.get('elapsed_ns',0))
            if event.get('type')=='galois_request':requests[event.get('cache')]+=1
            if event.get('type')=='begin':
                key=event.get('event_id')
                if key in opened:issues.append('Duplicate native stage begin')
                opened[key]=dict(event,inferred_parent=stack[-1] if stack else None);stack.append(key)
            elif event.get('type')=='end':
                key=event.get('event_id');begin=opened.pop(key,None)
                if begin is None:issues.append('Native end without begin');continue
                if not stack or stack[-1]!=key:issues.append('Native stage stack order differs')
                else:stack.pop()
                duration=event.get('duration_ns')
                if type(duration) is not int or duration<0:issues.append('Invalid native duration');continue
                rows.append(dict(event_id=key,parent_id=begin['inferred_parent'],stage=begin['stage'],
                    metadata=begin.get('metadata') or {},inclusive_ns=duration,status=event.get('status'),
                    start_elapsed_ns=begin.get('elapsed_ns'),end_elapsed_ns=event.get('elapsed_ns')))
    if opened or stack:issues.append('Unclosed native stages')
    by_id={r['event_id']:r for r in rows};children={}
    for row in rows:children.setdefault(row['parent_id'],[]).append(row)
    for row in rows:
        row['exclusive_ns']=row['inclusive_ns']-sum(c['inclusive_ns'] for c in children.get(row['event_id'],[]))
        if row['exclusive_ns']<0:issues.append('Negative native exclusive duration')
    key_rows=[r for r in rows if r['stage'] in FIXED_KEY_STAGES or r['stage'].endswith('_lazy.galois')]
    computes=[r for r in rows if r['stage']=='runtime.compute']
    constructors=[r for r in rows if r['stage']=='bootstrap.evaluator_constructor']
    total=lambda values:sum(r['inclusive_ns'] for r in values)/1e9
    # Accumulate small nested categories once per ancestor. The full experiment
    # has thousands of key stages and instructions, so a Cartesian scan would
    # add avoidable audit work without improving the saved evidence.
    nested_keys=Counter();nested_dft=Counter()
    for row in rows:
        is_key=row['stage'] in FIXED_KEY_STAGES or row['stage'].endswith('_lazy.galois')
        is_dft=row['stage']=='bootstrap.dft_matrix'
        if not is_key and not is_dft:continue
        parent=row['parent_id'];visited=set()
        while parent is not None and parent not in visited:
            visited.add(parent)
            if is_key:nested_keys[parent]+=row['inclusive_ns']
            if is_dft:nested_dft[parent]+=row['inclusive_ns']
            parent=by_id.get(parent,{}).get('parent_id')
    constructor_other=sum(r['inclusive_ns']-nested_keys[r['event_id']]-nested_dft[r['event_id']]
        for r in constructors)/1e9
    components=dict(covered_operator_core_seconds=sum(r['inclusive_ns']-nested_keys[r['event_id']]
        for r in computes)/1e9,
        bootstrap_galois_generation_seconds=total([r for r in rows if r['stage']=='bootstrap_lazy.galois']),
        residual_galois_generation_seconds=total([r for r in rows if r['stage']=='residual_lazy.galois']),
        fixed_key_generation_seconds=total([r for r in rows if r['stage'] in FIXED_KEY_STAGES]),
        constructor_dft_seconds=total([r for r in rows if r['stage']=='bootstrap.dft_matrix']),
        constructor_other_seconds=constructor_other,
        plaintext_read_encode_transfer_seconds=total([r for r in rows if r['stage']=='plaintext.read_encode']),
        cold_parameters_input_encryption_final_seconds=total([r for r in rows if r['stage'] in
            {'parameters.construct','input.read','input.encode_encrypt','final.decrypt_decode_compare'}]))
    return dict(issues=issues,stage_counts=dict(Counter(r['stage'] for r in rows)),
        error_stages=[dict(stage=r['stage'],event_id=r['event_id']) for r in rows if r['status'] not in (None,'ok')],
        key_generation_stages=len(key_rows),galois_requests=dict(requests),
        native_event_elapsed_endpoint_seconds=maximum/1e9,components=components,
        same_component_sum_seconds=sum(components.values()),
        same_component_scope='runtime.compute inclusive minus nested key-generation stage timers; all galois/fixed keys, DFT, constructor remainder, plaintext.read_encode, parameters/input/final timers counted once. Parent operator/constructor remainder retains nested instrumentation and cache-eviction GC performed within its timer. Bootstrap secret-basis extension, root interstage intervals and local instrumentation outside these timers are excluded.',
        bootstrap_secret_basis_extension_seconds=total([r for r in rows if r['stage']=='bootstrap.extend_same_secret']),
        compute_rows=[dict(ordinal=r['metadata'].get('ordinal'),op=r['metadata'].get('op'),index=r['metadata'].get('index'))
            for r in sorted(computes,key=lambda r:r['event_id'])])


def resource_summary(path):
    result=dict(samples=0,first_ts_ns=None,last_ts_ns=None,subtree_sampled_peak_rss_bytes=0,
        single_pid_sampled_hwm_kib=0,single_pid_sampled_vmsize_kib=0,outer_sampled_hwm_kib=0)
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line);result['samples']+=1
            if result['first_ts_ns'] is None:result['first_ts_ns']=row['ts_ns']
            result['last_ts_ns']=row['ts_ns'];family=row.get('processes') or []
            if not family and row.get('observed') is not None:family=[row['observed']]
            result['subtree_sampled_peak_rss_bytes']=max(result['subtree_sampled_peak_rss_bytes'],sum(p.get('vmrss_kib',0)*1024 for p in family))
            for p in family:
                result['single_pid_sampled_hwm_kib']=max(result['single_pid_sampled_hwm_kib'],p.get('vmhwm_kib',0))
                result['single_pid_sampled_vmsize_kib']=max(result['single_pid_sampled_vmsize_kib'],p.get('vmsize_kib',0))
            result['outer_sampled_hwm_kib']=max(result['outer_sampled_hwm_kib'],(row.get('outer_process') or {}).get('vmhwm_kib',0))
    result['scope']='Sampled /proc observations. Per-PID HWM, sampled simultaneous child-tree RSS and outer HWM are separate.'
    return result


def validate_sources(result,math_result,input_prepare,*,observed_command=None):
    mathematical=read(math_result);source=mathematical['source'];worker=mathematical.get('worker') or {}
    if mathematical.get('status')!='ok' or worker.get('mathematical_equivalence_verified') is not True:
        raise ValueError('Accepted fresh original43 one-training-input mathematical proof is required')
    current=source_math.source_gate(input_prepare,source['source_case'],source['archive_manifest'])
    for key in ('input_prepare','source_case','plan','bundle','input','expected','program','model_info'):
        if not same(source.get(key),current[key]):raise ValueError('Current frozen mathematical source differs: '+key)
    proof_path=Path(worker['source_math_proof']);proof=read(proof_path)
    if (proof_path.resolve()!=Path(math_result).resolve().parent/'worker/source-math-proof.json'
            or proof.get('status')!='strict_math_passed' or proof.get('sample_ids')!=['train:0']
            or proof.get('operation_count')!=EXECUTIONS or proof.get('ideal_bootstrap_count')!=BOOTSTRAPS
            or proof.get('criterion')!='1e-12+1e-12*abs(reference)'):
        raise ValueError('Current original43 mathematical proof is incomplete')
    for key,expected in [('source_plan',current['plan']),('current_program',current['program']),
        ('input_prepare',current['input_prepare']),('input',current['input']),('reference',current['expected']),('source_case',current['source_case'])]:
        if not same(proof.get(key),expected):raise ValueError('Current program/input proof path differs: '+key)
    if observed_command is None:
        processes=[r for r in result.get('processes',[]) if r.get('stage')=='execute']
        if len(processes)!=1:raise ValueError('Exactly one native execution command is required')
        argv=processes[0]['argv'];process=processes[0]['observation']
    else:
        argv=observed_command['argv'];process=None
    def argument(name):
        if not isinstance(argv,list) or argv.count(name)!=1:raise ValueError('Missing/duplicate actual native argument '+name)
        index=argv.index(name)+1
        if index>=len(argv):raise ValueError('Missing native argument value '+name)
        return argv[index]
    mapped_path=Path(argument('-plan') if observed_command is not None else result['source_plan'])
    original=read(current['plan']);mapped=read(mapped_path)
    reconstructed,_=map_precision_plan(original,BINDING,bootstrap_cache=32)
    if mapped!=reconstructed:raise ValueError('Actual native plan differs from the explicit original43→50/C2S52 mapping')
    if observed_command is None:
        if not same(result.get('source_bundle'),current['bundle']):raise ValueError('Use the same canonical original runtime bundle')
        if not same((result.get('source_model_result') or {}).get('path'),Path(current['source_case'])/'result.json'):
            raise ValueError('Actual runtime source must identify the independently frozen fresh R30 case')
    for flag,path in [('-plan',mapped_path),('-bundle',current['bundle']),('-input',current['input']),('-expected',current['expected'])]:
        if not same(argument(flag),path):raise ValueError('Actual native argument/source differs: '+flag)
    for flag,value in [('-precision-bits','50'),('-bootstrap-cache','32'),('-mod1-inv-degree','15'),
        ('-log-message-ratio','2'),('-bootstrap-normalization-log2','3'),('-auxiliary-prime-bits','61'),
        ('-c2s-bits','52'),('-checkpoint-interval','1000')]:
        if argument(flag)!=value:raise ValueError('Actual native parameter argument differs: '+flag)
    expected=read(current['expected'])
    if expected.get('sample_id')!='train:0' or expected.get('indices')!=list(range(10)) or expected.get('values')!=current['reference_values']:
        raise ValueError('Require current independent original frozen-program Tensor expected values')
    for name in ('-allow-bootstrap','-trace-key-requests','-check-input-ssa'):
        if argv.count(name)!=1:raise ValueError('Required native observation/operation flag differs: '+name)
    if argument('-mode')!='execute':raise ValueError('Actual native command must be execute mode')
    return current,mapped,expected,process


def audit_records(result,client,plan,expected,completion,registry,go,resources,*,case_bytes):
    checks=[]
    def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=detail))
    execution=plan['execution'];counts=dict(Counter(r['op'] for r in execution));ssa=client.get('input_ssa_observations') or []
    check('complete_original_instruction_inventory',len(execution)==EXECUTIONS and counts.get('boot')==BOOTSTRAPS)
    check('actual_count_and_by_kind',client.get('compute_attempts')==EXECUTIONS and client.get('compute_completed')==EXECUTIONS
        and client.get('bootstrap_attempts')==BOOTSTRAPS and client.get('bootstrap_completed')==BOOTSTRAPS
        and client.get('op_attempts_by_kind')==counts and client.get('op_completed_by_kind')==counts)
    check('every_original_ssa_metadata_preserved_in_order',client.get('input_ssa_check_enabled') is True and len(ssa)==len(execution)
        and all(r.get('metadata_preserved') is True and (r.get('ordinal'),r.get('op'),r.get('inputs'))==
            (p['ordinal'],p['op'],p['inputs']) for r,p in zip(ssa,execution)))
    check('native_event_instruction_order',go['compute_rows']==[dict(ordinal=p['ordinal'],op=p['op'],index=i) for i,p in enumerate(execution)])
    check('one_encryption_one_final_no_intermediate',all(client.get(k)==1 for k in
        ('encryption_attempts','encryption_completed','final_decryption_attempts','final_decryption_completed'))
        and client.get('intermediate_decryptions')==0 and client.get('get_log_enabled') is False)
    boots=client.get('bootstrap_observations') or [];policy=client.get('bootstrap_precision_policy') or {}
    check('actual_c2s52_p61_inv15_mr2_div8_group',client.get('variant')==VARIANT and client.get('precision_bits')==50
        and client.get('c2s_bits')==52 and client.get('auxiliary_prime_bits')==61 and client.get('bootstrap_cache_capacity')==32
        and client.get('bootstrap_normalization_log2')==3 and client.get('bootstrap_input_divisor')==8
        and policy.get('coeffs_to_slots_log_scales')==[[52]]*4 and policy.get('slots_to_coeffs_log_scales')==[[39]]*3
        and policy.get('mod1_inverse_degree')==15 and policy.get('log_message_ratio')==2)
    check('all_native_bootstraps_preserve_input_and_restore_value',len(boots)==BOOTSTRAPS
        and client.get('native_bootstrap_api_attempts')==BOOTSTRAPS and client.get('native_bootstrap_api_completed')==BOOTSTRAPS
        and client.get('bootstrap_input_value_divisions')==BOOTSTRAPS and client.get('bootstrap_output_integer_restores')==BOOTSTRAPS
        and all(r.get('native_completed') is True and r.get('input_ssa_preserved') is True and r.get('input_value_divisor')==8
            and r.get('output_integer_multiplier')==8 and r.get('restored_output_level')==5 for r in boots))
    residual=client.get('residual_parameters') or {};bootstrap=client.get('bootstrap_parameter_summary') or {}
    check('actual_public_constructor_dimensions_and_prefix',residual.get('n')==65536 and residual.get('logical_log_slots')==14
        and residual.get('log_default_scale')==50 and len(residual.get('q',[]))==9 and len(residual.get('p',[]))==3
        and bootstrap.get('n')==65536 and len(bootstrap.get('q',[]))==28 and len(bootstrap.get('p',[]))==4
        and bootstrap.get('q',[])[:9]==residual.get('q')
        and residual.get('p',[None])[0]==bootstrap.get('p',[None])[0]
        and set(residual.get('p',[])).issubset(bootstrap.get('p',[])))
    independent=recompute_ckks(client['final_selected_values'],expected['values'])
    allowed_errors=[] if independent['passed'] else ['final.decrypt_decode_compare']
    check('go_event_complete_and_key_counts_match',not go['issues']
        and [r['stage'] for r in go['error_stages']]==allowed_errors
        and client.get('keygen_api_attempts')==client.get('keygen_api_completed')==go['key_generation_stages'])
    check('saved_ckks_comparison_matches_independent_original_criterion',len(expected['values'])==10
        and client.get('comparison',{}).get('passed')==independent['passed']
        and client.get('comparison',{}).get('absolute_tolerance')==1e-5 and client.get('comparison',{}).get('relative_tolerance')==1e-4)
    check('successful_runtime_claim_matches_math',client.get('encrypted_plan_completed')==independent['passed']
        and (client.get('status')=='ok')==independent['passed'])
    if result is not None:
        spans=result.get('timing',{}).get('spans') or [];roots=[r for r in spans if r.get('parent_id') is None]
        check('python_exclusive_attribution',result.get('timing',{}).get('attribution_ready') is True
            and sum(r.get('exclusive_ns',0) for r in spans)==sum(r.get('inclusive_ns',0) for r in roots))
        check('complete_persistence_endpoint',completion.get('required_artifacts_persisted') is True and completion.get('registry_persisted') is True
            and completion.get('status')==result.get('status') and completion.get('case_artifact_bytes')==case_bytes
            and completion.get('complete_case_elapsed_ns',0)>=result.get('recorded_elapsed_ns',0))
        check('registry_unique_eight_fields',len(registry)==1 and set(registry[0])==REGISTRY_FIELDS
            and registry[0].get('metrics')==result.get('metrics'))
    check('resource_sampled_rss_within_allowance',resources['subtree_sampled_peak_rss_bytes']<=14*1024**3)
    check('formal_and_prime_certificate_unverified',(result is None or result.get('security_acceptance') is None) and client.get('security_acceptance') is None
        and client.get('accuracy_acceptance') is None and client.get('concrete_prime_binding_verified') is False
        and (result is None or (result.get('CKKS_latency') is None and result.get('compiled_bootstraps') is None)))
    if result is not None:
        check('no_new_generation_install_or_integrity_action',all(result.get(k)==0 for k in ('llm_calls','downloads','installations','extra_integrity_checks'))
            and client.get('extra_integrity_checks')==0)
    else:check('native_saved_integrity_counter',client.get('extra_integrity_checks')==0)
    return dict(checks=checks,evidence_consistent=all(r['passed'] for r in checks),independent_ckks=independent,
        original_math_criterion_unchanged=True,scope='One accepted current saved train:0; numerical native correctness and engineering clocks. Task accuracy and formal security remain separate.')


def registry_rows(path,case):
    rows=[]
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('run_id')==case.name or same(row.get('artifacts_dir'),case):rows.append(row)
    return rows


def check_recovery_observation(recovery,command,completion,registry,case,recovery_case,*,case_bytes):
    """Validate saved observation facts without manufacturing a child exit status."""
    checks=[]
    def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=detail))
    initial=recovery.get('initial_identity') or {}
    check('recovery_identifies_original_process_and_source',same(recovery.get('source_case'),case)
        and recovery.get('round')=='R31-SURVIVOR-OBSERVATION'
        and recovery.get('mode')=='existing-native-process-after-supervisor-interruption'
        and type(recovery.get('native_pid')) is int and recovery.get('native_pid')>0
        and type(recovery.get('native_starttime_ticks')) is int and recovery.get('native_starttime_ticks')>0
        and initial.get('pid')==recovery.get('native_pid')
        and initial.get('starttime_ticks')==recovery.get('native_starttime_ticks') and initial.get('ppid')==1)
    check('recovery_command_is_same_process_later_observation',command.get('native_pid')==recovery.get('native_pid')
        and command.get('native_starttime_ticks')==recovery.get('native_starttime_ticks')
        and same(command.get('executable'),recovery.get('native_executable'))
        and command.get('capture_perf_counter_ns',0)>=recovery.get('observation_start_ns',1)
        and command.get('environment_read') is False and command.get('credentials_read') is False
        and command.get('extra_integrity_checks')==0
        and isinstance(command.get('argv'),list) and bool(command['argv'])
        and same(command['argv'][0],command.get('executable')))
    argv=command.get('argv') or []
    def arg_path(flag,target):
        return argv.count(flag)==1 and argv.index(flag)+1<len(argv) and same(argv[argv.index(flag)+1],target)
    check('recovery_keeps_original_client_and_event_paths',same(recovery.get('native_client'),case/'client.json')
        and same(recovery.get('native_events'),case/'go-events.jsonl')
        and same(recovery.get('original_resources'),case/'execute-resources.jsonl')
        and same(recovery.get('resources'),recovery_case/'resources.jsonl')
        and arg_path('-output',case/'client.json') and arg_path('-events',case/'go-events.jsonl'))
    check('original_supervision_interruption_and_unknown_native_exit_preserved',
        recovery.get('original_supervisor_exit_code')==143 and recovery.get('original_supervisor_signal')==15
        and recovery.get('original_supervisor_cause')=='unknown' and recovery.get('original_supervisor_clean') is False
        and recovery.get('original_continuous_outer_completion') is None
        and recovery.get('native_exit_code') is None and recovery.get('waitpid_available') is False)
    check('recovery_launches_no_native_or_crypto',all(recovery.get(k)==0 for k in
        ('native_spawned','native_restarted','additional_encryption','additional_keygen','additional_CKKS_runs','llm_calls','extra_integrity_checks')))
    start=recovery.get('observation_start_ns');end=recovery.get('observation_end_ns')
    original=recovery.get('original_execute_start_ns');last=recovery.get('last_original_resource_ns')
    check('recovery_clock_and_observation_gap_are_explicit',all(type(x) is int for x in (start,end,original,last))
        and original<=last<=start<=end
        and recovery.get('resource_observation_gap_seconds')==(start-last)/1e9
        and recovery.get('observed_total_since_original_execute_seconds')==(end-original)/1e9)
    timing=recovery.get('timing') or {};spans=timing.get('spans') or []
    check('recovery_exclusive_clock_attribution',bool(spans) and timing.get('attribution_ready') is True
        and sum(r.get('exclusive_ns',0) for r in spans)==sum(r.get('inclusive_ns',0) for r in spans if r.get('parent_id') is None))
    check('recovery_completion_and_unique_registry',completion.get('required_artifacts_persisted') is True
        and completion.get('registry_persisted') is True and completion.get('status')==recovery.get('status')
        and completion.get('case_artifact_bytes')==case_bytes and len(registry)==1
        and set(registry[0])==REGISTRY_FIELDS and same(registry[0].get('artifacts_dir'),recovery_case)
        and registry[0].get('mode')==recovery.get('mode') and registry[0].get('metrics',{}).get('status')==recovery.get('status'))
    check('recovery_observed_terminal_with_no_guard_stop',recovery.get('native_alive_after_observation') is False
        and recovery.get('guard_stop') is None and recovery.get('termination_signals_sent')==[]
        and recovery.get('error') is None
        and recovery.get('status') in ('native_completed_observed','native_error_observed'))
    return checks


def pending_record(case,client,recovery=None,recovery_case=None):
    return dict(status='pending',terminal=False,case=str(case),active_stage=client.get('active_stage'),
        compute_completed=client.get('compute_completed'),bootstrap_completed=client.get('bootstrap_completed'),
        terminal_success_claim=False,recovery_case=None if recovery_case is None else str(recovery_case),
        recovery_observation_status=None if recovery is None else recovery.get('status'),
        original_continuous_outer_completion=None if recovery is None else recovery.get('original_continuous_outer_completion'),
        original_supervisor_clean=None if recovery is None else recovery.get('original_supervisor_clean'),
        scope='Public counter snapshot only; terminal audit waits for saved native and observation endpoints')


def audit_recovered_case(args,case,recovery_case):
    client=read(case/'client.json');recovery=read(recovery_case/'result.json')
    if (client.get('status')=='running' or recovery.get('status')=='observing'
            or not (recovery_case/'completion.json').is_file()):
        return pending_record(case,client,recovery,recovery_case)
    # All provenance below comes from saved records. The absent original result
    # and completion stay absent; no synthetic result object stands in for them.
    command=read(recovery_case/'native-command-observed.json');completion=read(recovery_case/'completion.json')
    current,plan,expected,_=validate_sources(None,args.math_result,args.input_prepare,observed_command=command)
    go=native_events(case/'go-events.jsonl');original_resources=resource_summary(case/'execute-resources.jsonl')
    recovered_resources=resource_summary(recovery_case/'resources.jsonl')
    combined=dict(subtree_sampled_peak_rss_bytes=max(original_resources['subtree_sampled_peak_rss_bytes'],
        recovered_resources['subtree_sampled_peak_rss_bytes']))
    rows=registry_rows(args.registry,recovery_case)
    observed=audit_records(None,client,plan,expected,None,[],go,combined,case_bytes=artifact_bytes(case))
    observed['checks']+=check_recovery_observation(recovery,command,completion,rows,case,recovery_case,
        case_bytes=artifact_bytes(recovery_case))
    observed['checks'].append(dict(name='raw_original_terminal_record_absence_preserved',
        passed=not (case/'result.json').exists() and not (case/'completion.json').exists(),detail=None))
    observed['checks'].append(dict(name='native_terminal_matches_observer',
        passed=recovery.get('native_client_terminal_status')==client.get('status')
            and recovery.get('native_client_counts')=={k:client.get(k) for k in
                ('compute_completed','bootstrap_completed','encryption_completed','final_decryption_completed','intermediate_decryptions')}
            and recovery.get('native_client_comparison')==client.get('comparison'),detail=None))
    observed['evidence_consistent']=all(row['passed'] for row in observed['checks'])
    prediction=prediction_comparison(args.prediction,go)
    observed.update(status='audited_terminal_with_supervision_interruption' if observed['evidence_consistent'] else 'terminal_evidence_inconsistent',
        terminal=True,case=str(case),recovery_case=str(recovery_case),raw_runtime_status=None,
        saved_native_status=client.get('status'),recovery_observation_status=recovery.get('status'),
        ckks_passed=observed['independent_ckks']['passed'],source=current,variant=client.get('variant'),
        operation_counts=client.get('op_completed_by_kind'),native_counts={k:client.get(k) for k in
            ('compute_completed','bootstrap_completed','keygen_api_completed','encryption_completed','final_decryption_completed','intermediate_decryptions')},
        original_supervisor=dict(exit_code=143,signal=15,cause='unknown',clean=False,
            result_present=False,completion_present=False,native_exit_code=None,waitpid_available=False),
        timing=dict(native_process_seconds=None,recorded_pre_persistence_seconds=None,complete_persistence_seconds=None,
            original_continuous_outer_completion=None,
            recovery_observation_seconds=(recovery['observation_end_ns']-recovery['observation_start_ns'])/1e9,
            recovery_complete_persistence_seconds=completion.get('complete_case_elapsed_seconds'),
            observed_since_original_execute_seconds=recovery['observed_total_since_original_execute_seconds'],
            resource_observation_gap_seconds=recovery['resource_observation_gap_seconds'],
            native_event_elapsed_endpoint_seconds=go['native_event_elapsed_endpoint_seconds'],
            measured_same_component_sum_seconds=go['same_component_sum_seconds'],go_and_python_clocks_merged=False,
            unprojected_process_remainder_seconds=None),
        resources=dict(original=original_resources,recovery=recovered_resources,
            sampled_peak_rss_bytes_across_observed_segments=combined['subtree_sampled_peak_rss_bytes'],
            gap_seconds=recovery['resource_observation_gap_seconds'],gap_continuous_rss_unknown=True,
            scope='Original and recovery observations are separate. The final process HWM covers its lifetime; simultaneous RSS within the observation gap is unrecorded.'),
        parameter_group=parameter_group(client),prediction_comparison=prediction,
        original_registry_unique_rows=len(registry_rows(args.registry,case)),recovery_registry_unique_rows=len(rows),
        formal_security=None,formal_task_accuracy=None,
        artifact_source_paths=dict(original_case=str(case),original_client=str(case/'client.json'),
            original_events=str(case/'go-events.jsonl'),original_resources=str(case/'execute-resources.jsonl'),
            recovery_result=str(recovery_case/'result.json'),recovery_completion=str(recovery_case/'completion.json'),
            recovery_command=str(recovery_case/'native-command-observed.json'),recovery_resources=str(recovery_case/'resources.jsonl')),
        audits_executed_new_models=0,audits_executed_new_crypto=0,audits_executed_API=0,integrity_checks=0)
    return observed


def prediction_comparison(path,go):
    prediction=read(path)
    if prediction.get('engineering_total_point_seconds')!=POINT_SECONDS or prediction.get('variant')!=VARIANT:
        raise ValueError('Use the independently frozen same-group4974.14904172775 component point')
    projection=[]
    for name,point in prediction['component_point_seconds'].items():
        actual=go['components'].get(name);projection.append(dict(component=name,predicted_seconds=point,
            actual_same_scope_seconds=actual,difference_seconds=None if actual is None else actual-point))
    return dict(source=str(Path(path).resolve()),frozen_point_seconds=POINT_SECONDS,component_rows=projection,
        actual_same_components_seconds=go['same_component_sum_seconds'],point_difference_seconds=go['same_component_sum_seconds']-POINT_SECONDS,
        old90_outcomes_used=False,refitting_performed=False,formal_complete_prediction_seconds=None,
        per_cell_variance=None,recording_and_checkpoint_prediction_seconds=None,scope=prediction['scope'],
        actual_component_scope=go['same_component_scope'])


def parameter_group(client):
    bootstrap=client.get('bootstrap_parameter_summary') or {}
    return dict(residual=client.get('residual_parameters') or {},bootstrap=bootstrap,
        policy=client.get('bootstrap_precision_policy'),actual_secret=client.get('actual_residual_secret_generation'),
        ephemeral=client.get('bootstrap_ephemeral_sampler'),
        conservative_full_logqp_screen=dict(observed_logqp=bootstrap.get('log_qp'),reference_bound=1747,
            within_screen=type(bootstrap.get('log_qp')) in (int,float) and bootstrap['log_qp']<=1747,
            guideline='https://cic.iacr.org/p/1/4/26/pdf',
            scope='Public full-modulus planning screen; original concrete-prime binding, sigma/distribution and coupled auxiliary-key certification remain unverified'),
        formal_security_certificate=None,original_concrete_prime_mapping_certificate=None)


def audit_case(args):
    case=Path(args.case).resolve()
    if not (case/'result.json').is_file() or not (case/'completion.json').is_file():
        recovery=getattr(args,'recovery_case',None)
        if recovery is not None:return audit_recovered_case(args,case,Path(recovery).resolve())
        try:client=read(case/'client.json')
        except (OSError,ValueError):client={}
        return pending_record(case,client)
    result=read(case/'result.json');client=read(case/'client.json');completion=read(case/'completion.json')
    if client.get('status')=='running':return dict(status='pending',terminal=False,case=str(case),terminal_success_claim=False)
    current,plan,expected,process=validate_sources(result,args.math_result,args.input_prepare)
    go=native_events(case/'go-events.jsonl');resources=resource_summary(case/'execute-resources.jsonl')
    registry=[]
    with Path(args.registry).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('run_id')==case.name or same(row.get('artifacts_dir'),case):registry.append(row)
    observed=audit_records(result,client,plan,expected,completion,registry,go,resources,case_bytes=artifact_bytes(case))
    clean=(process.get('exit_code')==0 and process.get('timed_out') is False and process.get('signal') is None
        and process.get('signals_sent')==[] and process.get('remaining_pids')==[])
    observed['checks'].append(dict(name='native_process_clean_and_no_descendants',passed=clean,detail=process))
    observed['evidence_consistent']=all(r['passed'] for r in observed['checks'])
    prediction=read(args.prediction)
    if prediction.get('engineering_total_point_seconds')!=POINT_SECONDS or prediction.get('variant')!=VARIANT:
        raise ValueError('Use the independently frozen same-group4974.14904172775 component point')
    projection=[]
    for name,point in prediction['component_point_seconds'].items():
        actual=go['components'].get(name);projection.append(dict(component=name,predicted_seconds=point,
            actual_same_scope_seconds=actual,difference_seconds=None if actual is None else actual-point))
    process_seconds=(process['ended_ns']-process['started_ns'])/1e9
    observed.update(status='audited_terminal' if observed['evidence_consistent'] else 'terminal_evidence_inconsistent',terminal=True,
        case=str(case),raw_runtime_status=result['status'],ckks_passed=observed['independent_ckks']['passed'],
        source=current,variant=client['variant'],operation_counts=client.get('op_completed_by_kind'),
        native_counts={k:client.get(k) for k in ('compute_completed','bootstrap_completed','keygen_api_completed',
            'encryption_completed','final_decryption_completed','intermediate_decryptions')},
        timing=dict(native_process_seconds=process_seconds,recorded_pre_persistence_seconds=result['recorded_elapsed_ns']/1e9,
            complete_persistence_seconds=completion['complete_case_elapsed_seconds'],
            native_event_elapsed_endpoint_seconds=go['native_event_elapsed_endpoint_seconds'],
            measured_same_component_sum_seconds=go['same_component_sum_seconds'],
            unprojected_process_remainder_seconds=process_seconds-go['same_component_sum_seconds'],
            go_and_python_clocks_merged=False,python_exclusive_root_sum_ns=sum(r['exclusive_ns'] for r in result['timing']['spans'])),
        resources=dict(resources,per_pid_observed_hwm_kib=process.get('peak_rss_kib_by_pid'),
            supervisor_hwm_kib=process.get('outer_peak_rss_kib'),policy=result.get('memory_policy'),limits=process.get('limits')),
        parameter_group=dict(residual=residual if (residual:=client.get('residual_parameters')) else {},
            bootstrap=client.get('bootstrap_parameter_summary'),policy=client.get('bootstrap_precision_policy'),
            actual_secret=client.get('actual_residual_secret_generation'),ephemeral=client.get('bootstrap_ephemeral_sampler'),
            conservative_full_logqp_screen=dict(observed_logqp=client['bootstrap_parameter_summary']['log_qp'],
                reference_bound=1747,within_screen=client['bootstrap_parameter_summary']['log_qp']<=1747,
                guideline='https://cic.iacr.org/p/1/4/26/pdf',scope='Public full-modulus planning screen; original concrete-prime binding, sigma/distribution and coupled auxiliary-key certification remain unverified'),
            formal_security_certificate=None,original_concrete_prime_mapping_certificate=None),
        prediction_comparison=dict(source=str(Path(args.prediction).resolve()),frozen_point_seconds=POINT_SECONDS,
            component_rows=projection,actual_same_components_seconds=go['same_component_sum_seconds'],
            point_difference_seconds=go['same_component_sum_seconds']-POINT_SECONDS,
            old90_outcomes_used=False,refitting_performed=False,formal_complete_prediction_seconds=None,
            per_cell_variance=None,recording_and_checkpoint_prediction_seconds=None,
            scope=prediction['scope'],actual_component_scope=go['same_component_scope']),
        registry_unique_rows=len(registry),formal_security=None,formal_task_accuracy=None,
        artifact_source_paths={name:str(case/name) for name in ('result.json','client.json','completion.json','events.jsonl','go-events.jsonl','execute-resources.jsonl')},
        audits_executed_new_models=0,audits_executed_new_crypto=0,audits_executed_API=0,integrity_checks=0)
    return observed


def write_report(paper,record):
    if record.get('terminal') is not True:raise ValueError('Pending execution cannot produce a terminal result page')
    paper=Path(paper);evidence=paper/'fresh-resnet20-full-native-evidence';evidence.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__),evidence/'audit-source.py')
    test=Path(__file__).parents[1]/'tests/test_fresh_resnet_native_audit.py'
    if test.is_file():shutil.copyfile(test,evidence/test.name)
    target=paper/'FRESH_RESNET20_FULL_NATIVE_RESULTS.json';target.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    body='<html><meta charset="utf-8"><title>Fresh ResNet20 原生整网</title><style>body{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.5}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><body><h1>Fresh ResNet20 原生整网终态审计</h1>'
    body+='<p>17863条原生指令、123次自举，范围为当前冻结程序的训练样本0。十个保存输出按原CKKS容差独立重算。任务精度和正式安全证明分列。</p>'
    body+='<p>证据一致：'+str(record['evidence_consistent'])+'；原CKKS数值判据通过：'+str(record['ckks_passed'])+'。</p><table><tr><th>计时范围</th><th>秒</th></tr>'
    for k,v in record['timing'].items():
        if k.endswith('_seconds'):body+='<tr><td>'+html.escape(k)+'</td><td>'+html.escape(str(v))+'</td></tr>'
    body+='</table><p>冻结点4974.14904172775秒属于独立组件预测。差异按同范围组件列出；完整原生进程、记录开销和持久化终点另列。预测未按整网实测重新拟合。</p>'
    if record.get('original_supervisor') is not None:
        body+='<p>原监督进程以143退出，信号来源未知。原result/completion缺失，连续外层终点与原生进程退出码保留空值。恢复观察沿用同一PID和启动标识，未新增计算；资源观察缺口为'+html.escape(str(record['timing']['resource_observation_gap_seconds']))+'秒。原Go事件时钟、恢复观察时钟与恢复持久化终点分别列出。</p>'
    body+='<p><a href="'+str(target)+'">完整审计JSON、逐项检查与十个误差</a></p></body></html>'
    (paper/'FRESH_RESNET20_FULL_NATIVE_RESULTS.html').write_text(body)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','math-result','input-prepare','registry','prediction','paper'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--recovery-case',type=Path)
    args=parser.parse_args(argv);record=audit_case(args)
    if record['terminal']:write_report(args.paper,record)
    print(json.dumps({k:record.get(k) for k in ('status','terminal','evidence_consistent','ckks_passed','bootstrap_completed')}))
    return 2 if not record['terminal'] else 0 if record['evidence_consistent'] else 1


if __name__=='__main__':raise SystemExit(main())
