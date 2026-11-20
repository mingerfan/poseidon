"""R25: one full saved packed-graph check against fixed physical-unit references."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import sys
import threading
import time

from .. import paths
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-saved-packed-graph-three-probe-check'
OUT_NAME = 'orbit-resnet20-packed-r25-20261115'
MODULE = 'pt_agent.experiments.r25_orbit_packed_worker'
CATEGORIES = ('tests', 'checks')
OVERALL_SECONDS = 1800
CASE_BYTES, FREE_BYTES = 4 * 1024**3, 10 * 1024**3
FORMAL = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')
STAGE_FLAGS = ('source_checked','native_packed_equivalence_verified')
SCOPE = 'R22 original unoptimized packed graph on train:0,train:1,train:2; finite-input numerical agreement only'
COUNTERS = ('graph_load_attempts','graph_load_completed','constant_load_attempts','constant_load_completed',
            'probe_load_attempts','probe_load_completed','packed_forward_attempts','packed_forward_completed',
            'math_check_attempts','math_check_completed','op_attempts','op_completed')
OP_COUNTS = dict(constant=2712,rotate=1382,mul=2920,add=3022,negate=76)
ZERO_FIELDS = ('llm_calls','downloads','installations','new_weights','new_datasets','fit_calls',
               'agent_calls','tuner_calls','count_runs','emit_runs','trace_runs','compiler_runs',
               'optimizer_runs','solver_runs','keygen_runs','real_ckks_attempts','real_ckks_completed',
               'heir_calls','heldout_samples','confirm_samples','report_samples','freeze_attempts','raw_cifar_reads',
               'torch_model_builds','torch_model_forwards','checkpoint_reads','poly_text_reads')
FALSE_FIELDS = ('backend_ready','security_validated','accuracy_validated','formal_matrix_complete')


def _locations():
    root, results = paths.ORBIT_ROOT.resolve(), paths.RESULTS_DIR.resolve()
    r13=results/'r13-common-resnet20-silu-20261115'
    r22=results/'orbit-resnet20-trace-tdag-r22-20261115'
    return dict(results=results,repo=paths.REPO,scripts=paths.REPO/'DSL/DSL_agent/scripts',
                orbit=root,numpy_python=root.parent/'orbit-venv/bin/python',
                probes=r13/'worker/training_probes.npz',probe_source=r13/'worker/training_probe_source.json',
                r13_result=r13/'result.json',r22_result=r22/'result.json',
                r24_result=results/'orbit-resnet20-polynomial-r24-20261115/result.json',
                graph=r22/'work/parse-worker/tdag.json',
                constants=r22/'work/trace-worker/constants/_hecate_ResNetSiLU64k.cst')


def _source_preflight(locations):
    names=('graph','constants','probes','probe_source','r13_result','r22_result','r24_result')
    required=[locations[k] for k in names]
    if any(not p.is_file() or not p.stat().st_size for p in required):
        raise ValueError('R25 requires the existing saved graph, compiled constants and three training probes')
    if not locations['numpy_python'].is_file():raise ValueError('R25 requires the existing NumPy interpreter')
    records={k:json.loads(locations[k+'_result'].read_text()) for k in ('r13','r22','r24')}
    for key,record in records.items():
        if record.get('round')!=key.upper() or record.get('model')!=MODEL or record.get('status')!='ok':
            raise ValueError('R25 requires successful original saved source: '+key)
    a,c,d=(records[k] for k in ('r13','r22','r24'))
    artifacts=a.get('worker',{}).get('artifacts',{})
    for key,saved in (('probes','training_probes'),('probe_source','training_probe_source')):
        if Path(artifacts.get(saved,'')).resolve()!=locations[key].resolve():
            raise ValueError('R25 only uses the original saved training probes')
    if (c.get('mode')!='orbit-resnet20-trace-tdag-preflight'
            or c.get('orbit_model_trace_passed') is not True or c.get('tdag_parse_passed') is not True
            or any(c.get(k,{}).get(p)!=1 for k in ('attempted','completed') for p in ('trace','parse'))):
        raise ValueError('R25 requires the original completed R22 trace/parse')
    trace,parse=(c.get('phase_results',{}).get(k,{}) for k in ('trace','parse'))
    files=trace.get('constant_files',[])
    if (trace.get('status')!='ok' or parse.get('status')!='ok'
            or parse.get('parser_metadata',{}).get('max_slot')!=65536
            or Path(parse.get('tdag','')).resolve()!=locations['graph'].resolve()
            or len(files)!=1 or Path(files[0].get('path','')).resolve()!=locations['constants'].resolve()
            or files[0].get('size_bytes')!=784347640
            or locations['constants'].stat().st_size!=784347640):
        raise ValueError('R25 preserves the exact saved graph and compiled-constant source paths and metadata')
    if (locations['graph'].resolve()!=(locations['r22_result'].parent/'work/parse-worker/tdag.json').resolve()
            or locations['constants'].resolve()!=(locations['r22_result'].parent/'work/trace-worker/constants/_hecate_ResNetSiLU64k.cst').resolve()):
        raise ValueError('R25 requires the original R22 result layout')
    w=d.get('worker') or {}
    if (d.get('mode')!='orbit-native-polynomial-physical-scale-check'
            or any(w.get(k) is not True for k in ('original_reference_passed','native_polynomial_math_passed','physical_scaled_forward_completed'))
            or any(w.get(k)!=v for k,v in dict(original_forward_attempts=3,original_forward_completed=3,
                   approx_forward_attempts=3,approx_forward_completed=3,math_check_attempts=57,math_check_completed=57).items())
            or Path(d.get('source_records',{}).get('records',{}).get('r22','')).resolve()!=locations['r22_result'].resolve()
            or w.get('probe_source')!=a.get('worker',{}).get('source')):
        raise ValueError('R25 requires the saved R24 native physical-unit references and source identity')
    probes=w.get('approximate_probes',[])
    if (len(probes)!=3 or [q.get('probe_index') for q in probes]!=[0,1,2]
            or [q.get('sample_id') for q in probes]!=['train:0','train:1','train:2']
            or any(not isinstance(q.get('approximate'),list) or len(q['approximate'])!=1
                   or not isinstance(q['approximate'][0],list) or len(q['approximate'][0])!=10 for q in probes)):
        raise ValueError('R25 requires three original-shaped R24 approximate logits')
    return dict(model=MODEL,records={k:str(locations[k+'_result']) for k in records},
                saved_sample_ids=['train:0','train:1','train:2'],sample_contents_read=False,compiled_constants_read=False,
                original_mlir_metadata=trace.get('mlir'),
                training_preprocessing_verified=False,coefficient_source_equivalence_verified=False,
                inputs=[dict(path=str(p),bytes=p.stat().st_size) for p in required])


def phase_command(locations,work):
    command=[str(locations['numpy_python']),'-B','-m',MODULE,'--work-dir',str(work/'checks-worker')]
    for key in ('graph','constants','probes','probe_source','r13_result','r22_result','r24_result'):
        command += ['--'+key.replace('_','-'),str(locations[key])]
    return command


def child_environment(work, locations):
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','NIX_','__NIX_','HECATE_'))
         and k not in ('HECATE','IN_NIX_SHELL','PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(work/'tmp'),
               R25_TEST_WORK=str(work/'test-worker'),R25_NUMPY_PYTHON=str(locations['numpy_python']))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    return env


def case_size(out):
    total=0
    for parent,_,names in os.walk(out):
        for name in names:
            try:total+=(Path(parent)/name).stat().st_size
            except FileNotFoundError:pass
    return total


def check_budget(out):
    size=case_size(out)
    if size>CASE_BYTES or shutil.disk_usage(out).free<FREE_BYTES:
        raise RuntimeError('R25 artifact or disk reserve allowance exceeded')
    return size


def write_terminal_result(out,result):
    for _ in range(20):
        result['metrics'].update(status=result['status'],case_artifact_bytes=result.get('case_artifact_bytes',0))
        write_json(out/'result.json',result)
        size=case_size(out)
        if size==result.get('case_artifact_bytes'):return
        result['case_artifact_bytes']=result['artifact_bytes']=size
    raise RuntimeError('R25 terminal artifact fixed point did not converge')


def append_registry(out,result,registry):
    row=dict(run_id=out.name,round='R25',model=MODEL,mode=MODE,git_rev=result['git_rev'],metrics=result['metrics'],
             artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()


def execute(args):
    started=time.perf_counter_ns();deadline=time.monotonic()+OVERALL_SECONDS;locations=_locations()
    out=Path(args.out).absolute()
    if out.exists() or out.is_symlink() or out.resolve()!=(locations['results']/OUT_NAME).resolve():
        raise ValueError('R25 requires the single new approved result directory')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    saved_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    saved_timer=signal.getitimer(signal.ITIMER_REAL);received_signals=[]
    def expired(signum,frame):
        received_signals.append(signum);raise RuntimeError('R25 outer signal '+str(signum))
    for sig in saved_handlers:signal.signal(sig,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.monotonic()))
    out.mkdir();work=out/'work';work.mkdir()
    for name in ('tmp','test-worker'):(work/name).mkdir()
    result=dict(run_id=out.name,round='R25',model=MODEL,mode=MODE,status='running',error=None,failed_stage=None,
                git_rev=None,git_branch=None,attempted=dict.fromkeys(CATEGORIES,0),completed=dict.fromkeys(CATEGORIES,0),
                processes=[],worker=None,worker_record_errors=[],retry_count=0,git_metadata_attempted=0,git_metadata_completed=0,
                artifact_bytes=0,case_artifact_bytes=0,source_records=None,outer_received_signals=received_signals,
                resource_monitor_error=None,worker_budget_records=[],native_packed_equivalence_verified=False,
                backend_ready=False,security_validated=False,accuracy_validated=False,formal_matrix_complete=False,
                new_llm_calls=0,new_downloads=0,new_installations=0,trace_runs=0,compiler_runs=0,optimizer_runs=0,
                solver_runs=0,keygens=0,ckks_runs=0,heldout_samples=0,confirm_samples=0,report_samples=0,candidates_frozen=0,
                native_packed_equivalence_scope=SCOPE,launcher_runs=0,extra_integrity_checks=0)
    result.update(dict.fromkeys(STAGE_FLAGS,False));recorder=StageRecorder(out/'events.jsonl');env=child_environment(work,locations)
    stop=threading.Event();child_events=[]
    def checkpoint():
        if result['resource_monitor_error']:raise RuntimeError(result['resource_monitor_error'])
        check_budget(out)
        if time.monotonic()>=deadline:raise TimeoutError('R25 entire-batch allowance exhausted')
    def monitor():
        while not stop.wait(1):
            try:
                size,free=case_size(out),shutil.disk_usage(out).free
                with (out/'artifact-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(),artifact_bytes=size,disk_free_bytes=free))+'\n')
                if size>CASE_BYTES or free<FREE_BYTES:raise RuntimeError('R25 periodic artifact/disk allowance exceeded')
            except BaseException as exc:
                result['resource_monitor_error']=f'{type(exc).__name__}: {exc}';os.kill(os.getpid(),signal.SIGTERM);return
    watcher=threading.Thread(target=monitor,daemon=True)
    def launch(name,argv,seconds,category=None):
        checkpoint();result['failed_stage']=name
        if category:result['attempted'][category]+=1
        else:result['git_metadata_attempted']+=1
        command=dict(stage=name,argv=[str(x) for x in argv],cwd=str(locations['scripts']),observation=None,native_termination_signal=None)
        result['processes'].append(command);print(json.dumps(dict(stage=name,state='started')),flush=True)
        with recorder.span(name,command=command['argv']) as parent:
            try:
                obs=run_process(command['argv'],cwd=locations['scripts'],log=out/(name+'.log'),resources=out/(name+'-resources.jsonl'),
                                timeout=min(seconds,max(.01,deadline-time.monotonic())),mem_gib=4,fsize_gib=8,env=env)
                command['observation']=obs;code=obs['exit_code']
                command['native_termination_signal']=-code if isinstance(code,int) and code<0 else None
                if obs.get('signal') is not None:received_signals.append(obs['signal'])
            finally:
                if category=='checks':
                    wd=work/'checks-worker'
                    try:
                        for event in read_events(wd/'events.jsonl'):
                            if event.get('type') in ('begin','end') and event.get('parent_id') is None:event['parent_id']=parent['event_id']
                            child_events.append(event)
                        if (wd/'phase_result.json').is_file():
                            result['worker']=json.loads((wd/'phase_result.json').read_text())
                            for flag in STAGE_FLAGS:result[flag]=result['worker'].get(flag)
                        for filename in ('budget-fatal.json','budget-monitor-error.json'):
                            if (wd/filename).is_file():result['worker_budget_records'].append(json.loads((wd/filename).read_text()))
                    except BaseException as exc:result['worker_record_errors'].append(f'{type(exc).__name__}: {exc}')
            if code!=0 or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']:
                raise RuntimeError('R25 supervised phase failed: '+name)
            if category=='checks':
                w=result['worker'] or {}
                if (result['worker_record_errors'] or w.get('status')!='ok'
                        or any(w.get(k)!=v for k,v in dict(round='R25',model=MODEL,mode=MODE).items())
                        or any(w.get(k) is not True for k in STAGE_FLAGS)
                        or w.get('native_packed_equivalence_scope')!=SCOPE
                        or any(k not in w or type(w[k]) is not int or w[k]!=0 for k in ZERO_FIELDS)
                        or any(w.get(k) is not False for k in FALSE_FIELDS)
                        or any(k not in w or w[k] is not None for k in FORMAL)
                        or any(w.get(k)!=v for k,v in dict(graph_load_attempts=1,graph_load_completed=1,
                            constant_load_attempts=1,constant_load_completed=1,probe_load_attempts=1,probe_load_completed=1,
                            packed_forward_attempts=3,packed_forward_completed=3,math_check_attempts=3,math_check_completed=3,
                            op_attempts=30336,op_completed=30336).items())
                        or any(w.get(k)!={op:count*3 for op,count in OP_COUNTS.items()}
                               for k in ('op_attempted_by_kind','op_completed_by_kind'))):
                    raise RuntimeError('R25 fixed mathematical checks did not complete')
            if category:result['completed'][category]+=1
            else:result['git_metadata_completed']+=1
        checkpoint();print(json.dumps(dict(stage=name,state='completed')),flush=True);return obs
    try:
        watcher.start()
        with recorder.span('r25-preflight'):
            result['failed_stage']='source-preflight'
            with recorder.span('source-preflight'):
                checkpoint();result['source_records']=_source_preflight(locations);write_json(out/'source-metadata.json',result['source_records'])
                base=Path(__file__).parent.parent
                for rel in ('experiments/r25_orbit_packed_worker.py','experiments/r25_orbit_packed_preflight.py',
                            'tests/test_r25_orbit_packed.py','tests/_r25_orbit_packed_cases.py'):
                    target=work/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/rel,target)
            launch('git-revision',['git','rev-parse','--short','HEAD'],30);result['git_rev']=(out/'git-revision.log').read_text().strip()
            launch('git-branch',['git','branch','--show-current'],30);result['git_branch']=(out/'git-branch.log').read_text().strip()
            launch('r25-tests',[sys.executable,'-B','-m','unittest','discover','-s','pt_agent/tests','-p','test_r25_*.py','-v'],600,'tests')
            log=(out/'r25-tests.log').read_text();counts=re.findall(r'Ran (\d+) tests? in ',log)
            if not counts or int(counts[-1])<=0 or 'skipped=' in log or '... skipped ' in log:raise RuntimeError('R25 requires positive tests and zero skips')
            summary=json.loads((work/'test-worker/inner-wrapper-result.json').read_text())
            if summary.get('status')!='ok' or summary.get('wrapper_runs')!=1 or not isinstance(summary.get('inner_tests'),int) or summary['inner_tests']<=0 or summary.get('inner_skips')!=0:
                raise RuntimeError('R25 unique inner fake suite did not pass')
            result['test_summary']=dict(system_tests=int(counts[-1]),inner=summary)
            launch('model-checks',phase_command(locations,work),960,'checks');result.update(status='ok',failed_stage=None)
    except BaseException as exc:result.update(status='stopped_on_error',error=f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        if watcher.is_alive():watcher.join()
        ended=time.perf_counter_ns();events=read_events(out/'events.jsonl')+child_events
        with (out/'combined-events.jsonl').open('w') as stream:
            for event in events:stream.write(json.dumps(event)+'\n')
        timing=summarize_events(events,termination_observed_ns=ended)
        timing['outer_interval']=dict(start_ns=started,end_ns=ended,elapsed_ns=ended-started,boundary='Before terminal timing/result fixed-point writes and registry append')
        write_json(out/'timing.json',timing)
        result['timing']=dict(attribution_ready=timing['attribution_ready'],diagnostics=timing['diagnostics'],span_count=len(timing['spans']),
                              recorded_outer_seconds=(ended-started)/1e9,complete_batch_elapsed_seconds=None,boundary=timing['outer_interval']['boundary'])
        if timing['diagnostics'] or (result['status']=='ok' and not timing['attribution_ready']):result.update(status='stopped_on_error',timing_record_error=True)
        if result['resource_monitor_error']:result.update(status='stopped_on_error',error=result['resource_monitor_error'])
        result['root_raw_getrusage_hwm_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        w=result['worker'] or {}
        result['metrics']=dict(status=result['status'],mathematical_check_only=True,attempted=dict(result['attempted']),completed=dict(result['completed']),
                               counts={k:w.get(k,None if result['attempted']['checks'] else 0) for k in COUNTERS},
                               op_attempted_by_kind=w.get('op_attempted_by_kind'),op_completed_by_kind=w.get('op_completed_by_kind'),
                               **{k:result[k] for k in STAGE_FLAGS},native_packed_equivalence_scope=SCOPE,
                               formal_matrix_complete=False,new_llm_calls=0,ckks_runs=0,heldout_samples=0,confirm_samples=0,report_samples=0,
                               **dict.fromkeys(FORMAL,None))
        write_terminal_result(out,result)
        try:check_budget(out)
        except BaseException as exc:
            result.update(status='stopped_on_error',terminal_budget_error=f'{type(exc).__name__}: {exc}')
            if result['error'] is None:result['error']=result['terminal_budget_error']
            write_terminal_result(out,result)
        append_registry(out,result,locations['results']/'registry.jsonl')
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in saved_handlers.items():signal.signal(sig,handler)
        if saved_timer[0] or saved_timer[1]:signal.setitimer(signal.ITIMER_REAL,*saved_timer)
        print(json.dumps(dict(status=result['status'],error=result['error'],failed_stage=result['failed_stage'],out=str(out),
                              attempted=result['attempted'],completed=result['completed'],artifact_bytes=result['artifact_bytes'])),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    return 0 if execute(parser.parse_args(argv))['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
