"""Read-only terminal audit of the bounded original HEIR initializer probe.

Read JSON, bounded log excerpts and /proc JSONL observations only. The result
case remains unchanged; an optional audit output must live outside that case.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re

from .completion_metrics import artifact_bytes
from .heir_initializer_probe import read,log_excerpt,startup_observation,classify,interruption_status,durable_json

FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
ZERO=('keygen_runs','encryptions','decryptions','bootstraps','model_forwards')


def resource_summary(path,observation=None):
    """Stream samples, preserving simultaneous RSS and per-process HWM/VSZ."""
    path=Path(path);count=malformed=0;by_pid={};tree=with_outer=outer_hwm=None
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError:malformed+=1;continue
                if not isinstance(row,dict) or not isinstance(row.get('processes'),list):
                    malformed+=1;continue
                count+=1;processes=row['processes'];rss=sum(x.get('vmrss_kib',0) for x in processes)
                outer=row.get('outer_process') or {}
                if processes:tree=max(tree or 0,rss)
                if processes or outer:with_outer=max(with_outer or 0,rss+outer.get('vmrss_kib',0))
                if outer:outer_hwm=max(outer_hwm or 0,outer.get('vmhwm_kib',0))
                for item in processes:
                    key=(item['pid'],item.get('starttime'))
                    saved=by_pid.setdefault(key,dict(pid=item['pid'],starttime=item.get('starttime'),
                        max_VmHWM_KiB=0,max_VmSize_KiB=0,max_VmRSS_KiB=0))
                    for dest,source in (('max_VmHWM_KiB','vmhwm_kib'),('max_VmSize_KiB','vmsize_kib'),('max_VmRSS_KiB','vmrss_kib')):
                        saved[dest]=max(saved[dest],item.get(source,0))
    observed_peak=(observation or {}).get('peak_rss_kib_by_pid') or {}
    return dict(path=str(path),present=path.is_file(),samples=count,malformed_or_partial_rows=malformed,
        sampled_processes=list(by_pid.values()),single_PID_sampled_HWM_KiB=max((x['max_VmHWM_KiB'] for x in by_pid.values()),default=None),
        single_PID_observation_HWM_KiB=max(observed_peak.values(),default=None),
        single_PID_sampled_maxVmSize_KiB=max((x['max_VmSize_KiB'] for x in by_pid.values()),default=None),
        simultaneous_child_subtree_RSS_KiB=tree,simultaneous_subtree_plus_outer_supervisor_RSS_KiB=with_outer,
        sampled_outer_supervisor_HWM_KiB=outer_hwm,
        observation_outer_supervisor_HWM_KiB=(observation or {}).get('outer_peak_rss_kib'),
        source='Saved R10 /proc samples; HWM/VSZ are per-process maxima, tree RSS is a simultaneous sum. Missing observations stay null.')


def audit(case,registry,readiness):
    case=Path(case).resolve();result=read(case/'result.json');completion=read(case/'completion.json')
    if (result.get('round')!='HEIR-INIT-PROBE' or result.get('status')=='running'
            or completion.get('status')!=result.get('status')
            or completion.get('registry_persisted') is not True or completion.get('required_artifacts_persisted') is not True
            or completion.get('ended_ns',0)<completion.get('started_ns',0)):
        raise ValueError('A closed original initializer probe with matching completion is required')
    matched=[]
    with Path(registry).open() as stream:
        fcntl.flock(stream,fcntl.LOCK_SH)
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            if row.get('run_id')==case.name:matched.append(row)
    if (len(matched)!=1 or set(matched[0])!=FIELDS or Path(matched[0]['artifacts_dir']).resolve()!=case
            or matched[0].get('metrics',{}).get('status')!=result['status']):
        raise ValueError('One matching original eight-field registry row is required')
    resources={};processes={}
    for name in ('build','initialize'):
        rows=[x for x in result.get('processes',[]) if x.get('stage')==name]
        if len(rows)>1:raise ValueError('The bounded probe must retain at most one attempt per phase')
        process=rows[0] if rows else {};observation=process.get('observation') or None
        resources[name]=resource_summary(case/(name+'-resources.jsonl'),observation)
        seconds=None
        if observation and all(type(observation.get(k)) is int for k in ('started_ns','ended_ns')):
            seconds=(observation['ended_ns']-observation['started_ns'])/1e9
        processes[name]=dict(attempts=result.get('attempts',{}).get(name),completed=result.get('completed',{}).get(name),
            observation=observation,supervision_exception=process.get('supervision_exception'),
            last_resource_sample=process.get('last_resource_sample'),observed_process_elapsed_seconds=seconds,
            limits=None if not observation else observation.get('limits'))
    worker=read(case/'worker-result.json') if (case/'worker-result.json').is_file() else result.get('worker') or {}
    logs={name:log_excerpt(case/(name+'.log')) if (case/(name+'.log')).is_file() else '' for name in processes}
    progress=startup_observation(worker,logs['initialize'])
    observed_stop=interruption_status(result,None)
    remaining=[pid for row in processes.values() for pid in ((row['observation'] or {}).get('remaining_pids') or [])]
    initialized=processes['initialize'];build=processes['build']
    if remaining:classification='supervision_error'
    elif observed_stop:classification=observed_stop
    elif initialized['observation']:
        obs=initialized['observation']
        classification=('supervision_error' if obs.get('exit_code')==0 and obs.get('signals_sent')
            and not obs.get('timed_out') and obs.get('signal') is None else classify(obs,logs['initialize'],worker))
    elif initialized['attempts']:classification='supervision_error'
    elif build['observation'] and (build['observation'].get('exit_code')!=0 or build['observation'].get('timed_out')
            or build['observation'].get('signal') is not None or build['observation'].get('signals_sent')):
        obs=build['observation']
        classification=('timeout' if obs.get('timed_out') else 'interrupted' if obs.get('signal') is not None
            else 'supervision_error' if obs.get('exit_code')==0 and obs.get('signals_sent')
            else 'memory_failure' if re.search(r'out of memory|cannot allocate memory|failed to reserve|address space',logs['build'],re.I)
            else 'build_error')
    elif result['status']=='prepared':classification='prepared'
    else:classification='supervision_error' if build['attempts'] else 'preparation_error'
    if initialized['attempts']:
        stage=('original-generated-package-initialization' if progress['actual_failure_before_own_main'] is True
            else 'original-preprocessing-encoding' if progress['preprocessing_entered'] is True
            else 'own-main-before-encoding' if progress['main_entered'] is True
            else 'initialization-entry-location-unobserved')
    else:stage='build' if build['attempts'] else 'preparation'
    static=read(readiness).get('source_static_estimates') or {}
    evidence=[]
    for name,log in logs.items():
        for line in log.splitlines():
            if re.search(r'out of memory|cannot allocate|failed to reserve|address space|heircnn.*\.init|preprocessing',line,re.I):
                evidence.append(dict(stage=name,text=line[:1000]))
                if len(evidence)>=12:break
    compatible=all(result.get(k)==0 for k in ZERO) and all(worker.get(k,0)==0 for k in ZERO)
    current_bytes=artifact_bytes(case)
    return dict(schema='heir-original-initializer-terminal-audit-v1',case=str(case),reported_status=result['status'],
        audited_classification=classification,failure_or_progress_stage=stage,startup_markers=progress,
        original_limits=result.get('limits'),complete_case_elapsed_seconds=completion.get('complete_case_elapsed_seconds'),
        processes=processes,resource_observations=resources,
        actual_encoding_entry_observed=progress['preprocessing_entered'],
        actual_encoding_completed=(worker['preprocessing_completed']==1) if type(worker.get('preprocessing_completed')) is int else None,
        completed_PT_count=worker.get('plaintext_count'),completed_LT_count=worker.get('linear_transform_count'),
        source_call_scope='Original generated package init, public CKKS parameter construction, encoder and original preprocessing only',
        driver_declared_operation_counts={k:result.get(k) for k in ZERO},
        worker_reported_operation_counts={k:worker.get(k) for k in ZERO},zero_crypto_contract_compatible=compatible,
        formal_security_acceptance=None,formal_accuracy_acceptance=None,CKKS_latency=None,
        static_source_estimates=dict(readiness=str(Path(readiness).resolve()),
            full_encoded_coefficient_GiB=static.get('full_encoded_coefficients_GiB'),
            preprocessing_encoded_coefficient_GiB=static.get('probe_preprocessing_coefficient_GiB'),
            largest_single_LT_GiB=static.get('largest_single_LT_GiB'),
            generated_package_global_inventories=static.get('current_package_global_inventory'),
            scope='Source-derived payload estimates only. Per-package global declarations do not establish actual linker retention, RSS or address-space usage.'),
        log_evidence=evidence,registry=dict(path=str(Path(registry).resolve()),matching_rows=1,exact_eight_fields=True),
        artifact_accounting=dict(completion_bytes=completion.get('case_artifact_bytes'),current_bytes=current_bytes,
            fixed_point_matches=completion.get('case_artifact_bytes')==current_bytes),
        interpretation='A bounded initializer/build/encoding observation. Per-process VmSize and HWM, simultaneous tree RSS and enforced AS limits remain distinct. This diagnostic supplies no encrypted inference time or executed key generation.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True);parser.add_argument('--readiness',type=Path,required=True)
    parser.add_argument('--registry',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if args.out and (args.out.resolve()==args.case.resolve() or args.out.resolve().is_relative_to(args.case.resolve())):
        parser.error('Keep the terminal audit output outside the original case')
    record=audit(args.case,args.registry or args.case.parent/'registry.jsonl',args.readiness)
    if args.out:
        if args.out.exists():raise ValueError('Preserve the previous audit output')
        durable_json(args.out,record)
    print(json.dumps(record,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
