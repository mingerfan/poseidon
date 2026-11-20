"""Durable, serial continuation of explicit trusted experiment commands.

The manifest contains local module argv and dependencies. Each experiment owns
its model policy and resource checks; this wrapper adds tree cleanup and clocks.
Report outputs are never used to construct another experiment's Agent inputs.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from .. import paths
from .r10_paper_metrics import _process, run_process, write_json

PREFIX = 'pt_agent.experiments.'
MODULES = {'orbit_common_model_frontend', 'orbit_common_model_optimizer',
           'dacapo_original_compile', 'dacapo_compile_continuation', 'dacapo_fresh_cold', 'failed_trace_archive',
           'fig2_uniform_degree',
           'heir_common_model_pipeline', 'r30_frozen_cnn_report',
           'bert_serial_continuation', 'dacapo_native_execute',
           'r30_continue_common_session', 'r30_continue_alex_session',
           'r30_alex_postfreeze', 'r30_alex_continuation_report', 'qwen_bridge_agent',
           'qwen_bounded_feedback', 'qwen_guarded_replay', 'r30_vgg_postfreeze', 'r31_common_cnn_native',
           'r30_common_model_e2e', 'orbit_common_resource_retry', 'qwen_prefill_compile', 'qwen_layer_native_probe',
           'orbit_backend_compile_only', 'orbit_backend_parse_only', 'orbit_common_backend_parse',
           'dacapo_vgg_math_restore', 'dacapo_vgg16_plan_math', 'heir_vgg_fused_pipeline',
           'heir_alexnet_wide_pool', 'heir_common_go_compile', 'qwen_remaining_prefill_compile',
           'bert_compile_first_deferred_report', 'qwen_generation_kernel_compile', 'qwen_component_link_validation',
           'bert_retained_budget_confirm', 'bert_budget_fork_compile',
           'heir_alexnet_host_serial_json', 'heir_host_go_compile', 'bert_frozen_fork_report',
           'dacapo_vgg_fresh_cold_rss16', 'orbit_fresh_compile_e2e', 'qwen_agent_cost_lifetime', 'qwen_agent_compact_record', 'closed_dacapo_trace_archive', 'heir_saved_math_no_linear_fusion', 'heir_no_fusion_go_compile', 'qwen_agent_grouped_record', 'heir_saved_math_layout_16384', 'heir_layout_16384_go_compile', 'qwen_retained_component_replay', 'heir_saved_math_explicit_linear_transform', 'a1_bert_fixed_current', 'a2_native_profile_driver', 'a3_native_bert_admission', 'a4_prepared_disk_run'}
MODULES.update({'a4_calibration_run', 'a5_a6_rule63_search', 'a7_qwen_layer0_sweep',
                'a_group_closed_raw_archive', 'a2_native_bootstrap_pricing',
                'a_group_future_closed_archive', 'cold_rule63_whole_model',
                'a_group_cold_closed_archive', 'a_group_same_config_feasibility',
                'cold_rule63_paper_policy', 'a5_a6_shared_rule63_compile',
                'qwen_rule63_block_compile', 'qwen_rule63_shared_optimized_compile',
                'bert_rule63_shared_optimized_compile',
                'bert_rule63_restored_input_archive',
                'a5r_resnet_rule63_shared_optimized_compile',
                'orbit_rule63_compile', 's1_orion_compile', 'bert_fresh_cold_2h'})
GIB = 1024**3
FIELDS = {'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
CONDITIONAL_PHASE_FIELDS = {'trace_attempts','trace_completed','compile_attempts','compile_completed'}


def deadline_timestamp(value):
    """An optional absolute experiment cutoff; retain time for terminal records."""
    if value is None:return None
    if not isinstance(value,str):raise ValueError('Deadline must be a timezone-aware ISO timestamp')
    try:parsed=datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
    except ValueError:raise ValueError('Deadline must be a timezone-aware ISO timestamp') from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('Deadline must name its timezone')
    return parsed.timestamp()


def parent_limits(job):
    """Explicit guarded compilers may inherit their declared larger hard limits."""
    memory=job.get('outer_address_space_gib',24)
    file_size=job.get('outer_file_gib',16)
    if type(memory)is not int or type(file_size)is not int:
        raise ValueError('Parent process limits must use explicit whole GiB values')
    if (memory,file_size)!=(24,16):
        argv=job.get('argv') or []
        qwen=(len(argv)>=4 and argv[3]==PREFIX+'qwen_prefill_compile' and (memory,file_size)==(32,64))
        def option(name):
            return argv[argv.index(name)+1]if argv.count(name)==1 and argv.index(name)+1<len(argv)else None
        layer=(len(argv)>=4 and argv[3]==PREFIX+'qwen_layer_native_probe'
               and (memory,file_size)==(24,64)
               and all(option(k)==v for k,v in (('--memory-gib','24'),('--case-gib','128'),
                   ('--seconds','10800'),('--through','compile')))
               and all(not item.startswith('--') or item in
                   ('--out','--source-case','--intake-python','--memory-gib','--case-gib',
                    '--seconds','--through','--diagnostic-elide-final-ir') for item in argv[4:])
               and argv.count('--diagnostic-elide-final-ir')==1)
        remaining=(len(argv)>=4 and argv[3]==PREFIX+'qwen_remaining_prefill_compile'
               and (memory,file_size)==(24,64)
               and all(option(k)==v for k,v in (('--memory-gib','24'),('--case-gib','128'),
                   ('--seconds','43200'),('--through','compile')))
               and all(not item.startswith('--') or item in
                   ('--out','--source-case','--probe-case','--intake-python','--memory-gib',
                    '--case-gib','--seconds','--through','--diagnostic-elide-final-ir') for item in argv[4:])
               and argv.count('--diagnostic-elide-final-ir')==1)
        generation_kernel=(len(argv)>=4 and argv[3]==PREFIX+'qwen_generation_kernel_compile'
               and (memory,file_size)==(24,64)
               and all(option(k)==v for k,v in (('--memory-gib','24'),('--case-gib','128'),
                   ('--seconds','21600'),('--through','compile')))
               and all(not item.startswith('--') or item in
                   ('--out','--source-case','--intake-python','--memory-gib',
                    '--case-gib','--seconds','--through','--diagnostic-elide-final-ir') for item in argv[4:])
               and argv.count('--diagnostic-elide-final-ir')==1)
        dacapo=(len(argv)>=4 and argv[3]==PREFIX+'dacapo_compile_continuation'
                and (memory,file_size)==(32,16)
                and all(option(k)==v for k,v in (('--memory-gib','32'),('--rss-gib','16'),('--file-gib','16'))))
        cold=(len(argv)>=4 and argv[3]==PREFIX+'dacapo_fresh_cold'
              and (memory,file_size)==(32,16)
              and all(option(k)==v for k,v in (('--model','alexnet'),('--memory-gib','12'),
                  ('--file-gib','16'),('--case-gib','32'),('--planning-gib','24'),
                  ('--waterline','40'),('--trace-seconds','1800'),('--compile-seconds','7200')))
              and not any(item.startswith(flag+'=') for item in argv for flag in
                  ('--model','--memory-gib','--file-gib','--case-gib','--planning-gib',
                   '--waterline','--trace-seconds','--compile-seconds'))
              and all(not item.startswith('--') or item in
                  ('--out','--model','--memory-gib','--file-gib','--case-gib','--planning-gib',
                   '--waterline','--trace-seconds','--compile-seconds','--diagnostic-elide-final-ir')
                  for item in argv[4:])
              and argv.count('--diagnostic-elide-final-ir')==1)
        if not(qwen or layer or remaining or generation_kernel or dacapo or cold):
            raise ValueError('Expanded parent limits require an explicit guarded Qwen or DaCapo compiler')
    return memory,file_size


def read(path):
    return json.loads(Path(path).read_text())


def check_command(argv, out):
    # These four metadata outers parse the typed DSL, whose typechecker imports
    # NumPy. Their pure Nix workers still receive the original intake Python.
    numeric_outer=(isinstance(argv,list)and len(argv)>3
        and argv[3] in (PREFIX+'qwen_layer_native_probe',PREFIX+'qwen_remaining_prefill_compile',
                       PREFIX+'qwen_generation_kernel_compile',PREFIX+'qwen_component_link_validation')
        and paths.PAPER_DATA_PYTHON is not None and isinstance(argv[0],str)
        and Path(argv[0]).absolute()==paths.PAPER_DATA_PYTHON.absolute())
    if (not isinstance(argv, list) or not all(isinstance(x,str) and x for x in argv)
            or len(argv)<6 or (argv[0] not in ('python3',sys.executable)and not numeric_outer)
            or argv[1:3]!=['-B','-m'] or not argv[3].startswith(PREFIX)
            or argv[3][len(PREFIX):] not in MODULES or '--detach' in argv
            or argv.count('--out')!=1):
        raise ValueError('Only explicit synchronous trusted experiment modules are supported')
    if Path(argv[argv.index('--out')+1]).absolute()!=Path(out).absolute():
        raise ValueError('Command output differs from its declared original case')
    return argv


def accepted_status(argv, status):
    if status=='ok':return True
    return (len(argv)>3 and argv[3]==PREFIX+'qwen_bridge_agent' and '--through' in argv
            and argv.index('--through')+1<len(argv) and argv[argv.index('--through')+1]=='prompt'
            and status=='prepared')


def validate(manifest, results_root):
    root=Path(results_root).resolve()
    deadline_timestamp(manifest.get('deadline_utc'))
    watch=manifest.get('watch') or {}
    if manifest.get('schema_version')!=1 or not isinstance(manifest.get('jobs'),list):
        raise ValueError('Expected finite version1 experiment batch')
    Path(watch['case']).absolute().relative_to(root)
    if type(watch.get('pid')) is not int or type(watch.get('starttime')) is not int:
        raise ValueError('Original external process identity is required')
    if not 1<=watch.get('seconds',0)<=604800:raise ValueError('External wait must be finite')
    ids=set();outputs=set()
    for job in manifest['jobs']:
        parent_limits(job)
        ident=job.get('id')
        if not isinstance(ident,str) or not ident or ident in ids or '/' in ident:
            raise ValueError('Each finite job has one unique simple ID')
        if any(x not in ids for x in job.get('requires',[])):
            raise ValueError('Dependencies must refer to earlier jobs')
        out=Path(job['out']).absolute();out.relative_to(root)
        if out in outputs or out==root or out==Path(watch['case']).absolute():
            raise ValueError('Keep independent output cases')
        if not 1<=job.get('seconds',0)<=90000:raise ValueError('Job time must be finite')
        reserve=job.get('minimum_free_gib',4 if job.get('kind')=='archive' else 10)
        if type(reserve) is not int or not 4<=reserve<=10:
            raise ValueError('Use an explicit whole-GiB disk reserve from4 through10GiB')
        for source,status in job.get('requires_case_statuses',{}).items():
            Path(source).absolute().relative_to(root)
            if status not in ('ok','error','prepared','quality_rejected'):raise ValueError('Require an explicit original case status')
        for source,fields in job.get('requires_case_fields',{}).items():
            Path(source).absolute().relative_to(root)
            if (not isinstance(fields,dict) or not fields or set(fields)-CONDITIONAL_PHASE_FIELDS
                    or any(type(v) is not int or v not in (0,1) for v in fields.values())):
                raise ValueError('Conditional phase counts must be explicit zero or one')
        if job.get('kind','experiment')=='archive':
            src=Path(job['source_case']).absolute();src.relative_to(root)
        elif job.get('kind','experiment')=='experiment':check_command(job['argv'],out)
        else:raise ValueError('Unknown job kind')
        ids.add(ident);outputs.add(out)
    return manifest


def wait_closed(watch, *, probe=_process, sleep=time.sleep, clock=time.monotonic):
    began=clock();case=Path(watch['case'])
    while True:
        proc=probe(watch['pid'])
        alive=bool(proc and proc['starttime']==watch['starttime'] and proc['state']!='Z')
        if not alive:
            if not (case/'completion.json').is_file() or not (case/'result.json').is_file():
                raise RuntimeError('Original process ended without durable terminal artifacts')
            result=read(case/'result.json');completion=read(case/'completion.json')
            if (result.get('status') not in ('ok','error','completed_with_observations') or completion.get('status')!=result['status']
                    or completion.get('required_artifacts_persisted') is not True
                    or completion.get('registry_persisted') is not True):
                raise RuntimeError('Original terminal status remains unresolved')
            observations=[x.get('observation') or {} for x in result.get('processes',[])]
            observations.extend((x.get('process') or {}) for x in result.get('jobs',[]))
            if (any(x.get('remaining_pids') for x in observations)
                    or any(x.get('cleanup_unknown') for x in result.get('jobs',[]))
                    or any(x.get('signal') is not None or (x.get('process') or {}).get('signal') is not None
                           for x in result.get('jobs',[]))
                    or result.get('signal') is not None):
                raise RuntimeError('Original descendant cleanup remains unresolved')
            if result.get('mode')=='finite-trusted-serial-experiments' and (case/'active.json').is_file():
                active=read(case/'active.json');rows=[x for x in result.get('jobs',[]) if x.get('id')==active.get('id')]
                observed=(rows[0].get('process') or {}) if len(rows)==1 else {}
                if (len(rows)!=1 or type(observed.get('exit_code')) is not int
                        or type(observed.get('timed_out')) is not bool
                        or not isinstance(observed.get('remaining_pids'),list)
                        or 'signal' not in observed or 'signals_sent' not in observed):
                    raise RuntimeError('Last active experiment has no closed process observation')
            return {'status':result.get('status'),'complete_case_seconds':completion.get('complete_case_elapsed_seconds'),
                    'elapsed_wait_seconds':clock()-began,'same_original_process_not_restarted':True}
        if clock()-began>watch['seconds']:raise TimeoutError('External wait allowance exhausted')
        sleep(10)


def archive_argv(job, manifest_file, results_root):
    from .completed_compile_archive import validate_completed_case, tree_inventory, file_metadata
    source=validate_completed_case(Path(job['source_case'])/'result.json',results_root=results_root)
    rows=[]
    for kind,path in source['eligible_files'].items():
        if path.is_file():rows.append({'kind':kind,'path':str(path),'inventory':[file_metadata(path,path.name)]})
    bundle=source['eligible_bundle']
    if bundle.is_dir():rows.append({'kind':'runtime_bundle','path':str(bundle),'inventory':tree_inventory(bundle)})
    if not rows:raise ValueError('No remaining eligible derived artifacts to archive')
    write_json(manifest_file,{'case_result':str(Path(job['source_case'])/'result.json'),
                            'scope':'Explicit closed successful case; derived payload only; retain all original results/plans',
                            'files':rows})
    return [sys.executable,'-B','-m',PREFIX+'completed_compile_archive','--manifest',str(manifest_file),
            '--out',job['out'],'--minimum-free-gib',str(job.get('minimum_free_gib',4))]


def case_phase_matches(case, expected):
    path=Path(case)/'result.json'
    if not path.is_file():return False
    observed=read(path)
    return (isinstance(observed,dict)
            and all(type(observed.get(key)) is int and observed[key]==wanted for key,wanted in expected.items()))


def execute_jobs(jobs, *, out, results_root, runner=run_process, free=lambda p:shutil.disk_usage(p).free,
                 deadline=None, wall_clock=time.time):
    records=[];by_id={}
    for job in jobs:
        row={'id':job['id'],'out':job['out'],'status':'pending'}
        if any(by_id[x]['status']!='ok' for x in job.get('requires',[])):
            row.update(status='dependency_skipped',reason='Required earlier experiment did not complete successfully')
        elif any(not (Path(x)/'result.json').is_file() or read(Path(x)/'result.json').get('status')!='ok'
                 for x in job.get('requires_cases',[])):
            row.update(status='dependency_skipped',reason='Required original case is not a closed success')
        elif any(not (Path(x)/'result.json').is_file() or read(Path(x)/'result.json').get('status')!=wanted
                 for x,wanted in job.get('requires_case_statuses',{}).items()):
            row.update(status='dependency_skipped',reason='Original case does not require this conditional continuation')
        elif any(not case_phase_matches(x,fields) for x,fields in job.get('requires_case_fields',{}).items()):
            row.update(status='dependency_skipped',reason='Original phase counts do not require this conditional continuation')
        else:
            reserve=job.get('minimum_free_gib',4 if job.get('kind')=='archive' else 10)
            remaining=None if deadline is None else deadline-wall_clock()-30
            if remaining is not None and remaining<=0:
                row.update(status='resource_deferred',reason='Absolute experiment cutoff reached; reserve terminal persistence; no child launched')
            elif free(results_root)<reserve*GIB:
                row.update(status='resource_deferred',reason='Physical disk reserve unavailable; no process launched')
            elif Path(job['out']).exists():
                row.update(status='preserved_existing_case',reason='Output already exists; automatic retry disabled')
            else:
                try:
                    argv=(archive_argv(job,out/(job['id']+'-archive-manifest.json'),results_root)
                          if job.get('kind')=='archive' else check_command(job['argv'],job['out']))
                    write_json(out/(job['id']+'-command.json'),argv)
                    row.update(status='running',argv=argv,started_utc=datetime.now(timezone.utc).isoformat())
                    row['absolute_deadline_applied']=deadline is not None
                    write_json(out/'active.json',row)
                    remaining=None if deadline is None else deadline-wall_clock()-30
                    if remaining is not None and remaining<=0:
                        row.update(status='resource_deferred',reason='Absolute cutoff reached during command preparation; no child launched')
                        (out/'active.json').rename(out/(job['id']+'-unstarted-active.json'))
                        write_json(out/(job['id']+'-deadline-deferral.json'),row)
                    else:
                        allowed=job['seconds'] if remaining is None else min(job['seconds'],remaining)
                        row['allowed_wall_seconds']=allowed
                        memory,file_size=parent_limits(job)
                        row['parent_address_space_gib']=memory
                        row['parent_file_gib']=file_size
                        row['cleanup_unknown']=True
                        obs=runner(argv,cwd=str(Path(__file__).parents[2]),log=out/(job['id']+'.log'),
                                   resources=out/(job['id']+'-resources.jsonl'),timeout=allowed,
                                   mem_gib=memory,fsize_gib=file_size)
                        row['cleanup_unknown']=False
                        row['process']=obs
                        clean=(obs.get('exit_code')==0 and not obs.get('timed_out') and obs.get('signal') is None
                               and not obs.get('remaining_pids') and not obs.get('signals_sent'))
                        result=Path(job['out'])/'result.json'
                        value=read(result) if result.is_file() else {}
                        row.update(status='ok' if clean and accepted_status(argv,value.get('status')) else 'failed',
                                   experiment_status=value.get('status'),ended_utc=datetime.now(timezone.utc).isoformat())
                except Exception as exc:
                    row.update(status='failed',error=type(exc).__name__+': '+str(exc))
                    if isinstance(exc,BatchInterrupted):row['signal']=exc.signum
        records.append(row);by_id[job['id']]=row
        write_json(out/'progress.json',{'status':'running','jobs':records,'single_heavy_job':True})
        observed=row.get('process') or {}
        if (observed.get('signal') is not None or observed.get('remaining_pids')
                or row.get('cleanup_unknown') or row.get('signal') is not None):break
    return records


class BatchInterrupted(Exception):
    def __init__(self, signum):
        self.signum=signum
        super().__init__('Batch interrupted with signal '+str(signum))


def _interrupt(signum, frame):
    raise BatchInterrupted(signum)


def acquire_lease(lease,seconds,*,clock=time.monotonic,sleep=time.sleep):
    began=clock()
    while True:
        try:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);return
        except BlockingIOError:
            if clock()-began>=seconds:raise TimeoutError('Serial execution lease wait exhausted') from None
            sleep(1)


def run(args):
    manifest=validate(read(args.manifest),paths.RESULTS_DIR)
    if args.out.exists():raise ValueError('Preserve previous batch evidence')
    args.out.mkdir(parents=True);started=time.perf_counter_ns();result={'status':'running','jobs':[]}
    handlers={s:signal.signal(s,_interrupt) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        write_json(args.out/'manifest.json',manifest)
        # A successor may itself still be waiting for this lease. Waiting for
        # its durable closure while holding the lease would block that source.
        result['watch']=wait_closed(manifest['watch'])
        with (paths.RESULTS_DIR/'autonomous-heavy-experiments.lock').open('a') as lease:
            acquire_lease(lease,getattr(args,'lease_wait_seconds',0))
            write_json(args.out/'supervisor.json',{'pid':os.getpid(),'process':_process(os.getpid())})
            result['jobs']=execute_jobs(manifest['jobs'],out=args.out,results_root=paths.RESULTS_DIR,
                deadline=deadline_timestamp(manifest.get('deadline_utc')))
            result['deadline_utc']=manifest.get('deadline_utc')
            result['status']=('error' if len(result['jobs'])!=len(manifest['jobs'])
                              or any((x.get('process') or {}).get('signal') is not None
                                     or (x.get('process') or {}).get('remaining_pids')
                                     or x.get('cleanup_unknown') or x.get('signal') is not None for x in result['jobs'])
                              else 'ok' if all(x['status']=='ok' for x in result['jobs'])
                              else 'completed_with_observations')
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
        if isinstance(exc,BatchInterrupted):result['signal']=exc.signum
    finally:
        # Preserve terminal evidence after cancellation; R10 restores these
        # handlers when its own descendant cleanup has finished.
        for s in handlers:signal.signal(s,signal.SIG_IGN)
        result.update(round='AUTONOMOUS-SERIAL',mode='finite-trusted-serial-experiments',
                      work_elapsed_before_persistence_seconds=(time.perf_counter_ns()-started)/1e9,
                      single_heavy_job=True,report_feedback_to_Agent=False,extra_integrity_checks=0)
        write_json(args.out/'result.json',result)
        from ..registry import git_rev
        row={'run_id':args.out.name,'round':result['round'],'model':'original-model-experiment-queue',
             'mode':result['mode'],'git_rev':git_rev(),'metrics':{'status':result['status'],'jobs':result['jobs']},
             'artifacts_dir':str(args.out),'ts':datetime.now(timezone.utc).isoformat()}
        assert set(row)==FIELDS
        with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        from .completion_metrics import artifact_bytes, persist_case_completion
        persist_case_completion(args.out,round_label=result['round'],started_ns=started,status=result['status'],
            required_artifact_bytes=artifact_bytes(args.out),
            timing={'status':'ok','attribution_ready':False,'diagnostics':[],'spans':[]})
        for s,handler in handlers.items():signal.signal(s,handler)
    return 0 if result['status'] in ('ok','completed_with_observations') else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--lease-wait-seconds',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.lease_wait_seconds<=604800:parser.error('Lease wait must be finite and at most7days')
    return run(args)

if __name__=='__main__':raise SystemExit(main())
