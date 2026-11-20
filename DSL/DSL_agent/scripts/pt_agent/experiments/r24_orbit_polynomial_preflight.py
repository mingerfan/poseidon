"""R24: one original-polynomial and physical-unit check on saved train probes."""
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
MODE = 'orbit-native-polynomial-physical-scale-check'
OUT_NAME = 'orbit-resnet20-polynomial-r24-20261115'
MODULE = 'pt_agent.experiments.r24_orbit_polynomial_worker'
CATEGORIES = ('tests', 'checks')
OVERALL_SECONDS = 1500
CASE_BYTES, FREE_BYTES = 4 * 1024**3, 10 * 1024**3
FORMAL = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')
STAGE_FLAGS = ('original_reference_passed', 'native_polynomial_math_passed', 'physical_scaled_forward_completed')
COUNTERS = ('native_load_attempts', 'native_load_completed', 'checkpoint_load_attempts', 'checkpoint_load_completed', 'model_build_attempts', 'model_build_completed',
            'deepcopy_attempts', 'deepcopy_completed', 'probe_load_attempts', 'probe_load_completed',
            'original_forward_attempts', 'original_forward_completed', 'approx_forward_attempts', 'approx_forward_completed',
            'activation_attempts', 'activation_completed', 'math_check_attempts', 'math_check_completed',
            'original_reference_checks', 'silu_modules_replaced')
ZERO_FIELDS = ('llm_calls', 'downloads', 'installations', 'new_weights', 'new_datasets',
               'fit_calls', 'agent_calls', 'tuner_calls', 'count_runs', 'emit_runs', 'trace_runs',
               'compiler_runs', 'optimizer_runs', 'solver_runs', 'keygen_runs',
               'real_ckks_attempts', 'real_ckks_completed', 'heir_calls',
               'heldout_samples', 'confirm_samples', 'report_samples', 'freeze_attempts', 'raw_cifar_reads')
FALSE_FIELDS = ('native_packed_equivalence_verified', 'backend_ready', 'security_validated',
                'accuracy_validated', 'formal_matrix_complete')
POLY_FILES = ('MPCB.py','Poly.py','Func.py','data/treeStr.txt','data/coeffStr.txt')


def _locations():
    root, results = paths.ORBIT_ROOT.resolve(), paths.RESULTS_DIR.resolve()
    r13 = results / 'r13-common-resnet20-silu-20261115'
    r22 = results / 'orbit-resnet20-trace-tdag-r22-20261115'
    return dict(results=results, repo=paths.REPO, scripts=paths.REPO/'DSL/DSL_agent/scripts',
                intake_python=paths.INTAKE_PYTHON.absolute(), orbit=root,
                model_source=root/'frontend/dacapo/python/poly/poly/models/ResNetSiLU.py',
                checkpoint=paths.DACAPO_UPSTREAM/'examples/data/resnet20.silu.model',
                probes=r13/'worker/training_probes.npz', probe_source=r13/'worker/training_probe_source.json',
                r13_result=r13/'result.json',
                r16_result=results/'r16-baseline-compatibility-20261115/orbit-resnet20_silu/result.json',
                r22_result=r22/'result.json',
                r23_result=results/'orbit-resnet20-optimizer-r23-20261115/result.json',
                poly_dir=r22/'work/local-poly/poly')


def _source_preflight(locations):
    names = ('model_source','checkpoint','probes','probe_source','r13_result','r16_result','r22_result','r23_result')
    required = [locations[n] for n in names] + [locations['poly_dir']/n for n in POLY_FILES]
    if any(not p.is_file() or not p.stat().st_size for p in required):
        raise ValueError('R24 requires the original saved model/probe/polynomial sources')
    interpreter = locations['intake_python']
    if not (interpreter.is_file() or interpreter.is_symlink()):
        raise ValueError('R24 requires the existing intake interpreter')
    records = {k: json.loads(locations[k+'_result'].read_text()) for k in ('r13','r16','r22','r23')}
    for key, r in records.items():
        if r.get('round') != key.upper() or r.get('status') != 'ok' or r.get('model') != MODEL:
            raise ValueError('R24 requires saved original-model success: '+key)
    a,b,c,d = (records[k] for k in ('r13','r16','r22','r23'))
    if b.get('baseline') != 'orbit' or b.get('worker',{}).get('compatibility_passed') is not True:
        raise ValueError('R24 requires the original Orbit compatibility record')
    r13_worker=a.get('worker',{});artifacts=r13_worker.get('artifacts',{})
    for key, saved in (('probes','training_probes'),('probe_source','training_probe_source')):
        if Path(artifacts.get(saved,'')).resolve()!=locations[key].resolve():
            raise ValueError('R24 preserves the original three saved training probes')
    summary=b.get('worker',{}).get('source_summary',{})
    for key in ('checkpoint','model_source'):
        if Path(summary.get(key,'')).resolve()!=locations[key].resolve():
            raise ValueError('R24 model/checkpoint path differs from saved Orbit reference')
    references=b.get('worker',{}).get('math_probes',[])
    if (len(references)!=3 or [q.get('sample_id') for q in references]!=['train:0','train:1','train:2']
            or any(q.get('passed') is not True or q.get('criterion')!='1e-12+1e-12*abs(reference)'
                   or q.get('reference_shape')!=[1,10] or q.get('reference_dtype')!='float64' for q in references)):
        raise ValueError('R24 requires all three fixed float64 R16 references')
    if (c.get('mode')!='orbit-resnet20-trace-tdag-preflight' or c.get('orbit_model_trace_passed') is not True
            or c.get('tdag_parse_passed') is not True or d.get('mode')!='orbit-resnet20-native-optimizer-preflight'
            or d.get('native_optimizer_completed') is not True or d.get('placement_artifact_checked') is not True
            or d.get('placement_bootstrap_nodes')!=19):
        raise ValueError('R24 requires the saved complete R22/R23 native preparation')
    if locations['poly_dir'].resolve()!=(locations['r22_result'].parent/'work/local-poly/poly').resolve():
        raise ValueError('R24 only uses the saved R22 polynomial source combination')
    return dict(model=MODEL,records={k:str(locations[k+'_result']) for k in records},
                saved_sample_ids=['train:0','train:1','train:2'],sample_contents_read=False,weights_read=False,
                training_preprocessing_verified=False,coefficient_source_equivalence_verified=False,
                native_packed_equivalence_verified=False,
                inputs=[dict(path=str(p),bytes=p.stat().st_size) for p in required])


def phase_command(locations, work):
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','360',
             '--python',str(locations['intake_python']),'--','-B','-m',MODULE,
             '--work-dir',str(work/'checks-worker')]
    for key in ('model_source','checkpoint','probes','probe_source','r13_result','r16_result','r22_result','r23_result','poly_dir'):
        command += ['--'+key.replace('_','-'),str(locations[key])]
    return command


def child_environment(work, locations):
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','NIX_','__NIX_','HECATE_'))
         and k not in ('HECATE','IN_NIX_SHELL','PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(work/'tmp'),
               R24_TEST_WORK=str(work/'test-worker'),R24_INTAKE_PYTHON=str(locations['intake_python']))
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
        raise RuntimeError('R24 artifact or disk reserve allowance exceeded')
    return size


def write_terminal_result(out,result):
    for _ in range(20):
        result['metrics'].update(status=result['status'],case_artifact_bytes=result.get('case_artifact_bytes',0))
        write_json(out/'result.json',result)
        size=case_size(out)
        if size==result.get('case_artifact_bytes'):return
        result['case_artifact_bytes']=result['artifact_bytes']=size
    raise RuntimeError('R24 terminal artifact fixed point did not converge')


def append_registry(out,result,registry):
    row=dict(run_id=out.name,round='R24',model=MODEL,mode=MODE,git_rev=result['git_rev'],metrics=result['metrics'],
             artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()


def _base_environment_snapshot():
    observed=[]
    for relative in ('deps/nix-portable-v012','deps/nix-portable-v012/launcher.lock','deps/nix-portable-v012/xdg-config',
                     'deps/nix-portable-v012/xdg-state','cache/nix-portable-v012','cache/nix-portable-tmp'):
        p=paths.PLATFORM_DIR/relative;q=dict(path=str(p),exists=p.exists())
        if p.exists():
            stat=p.stat();q.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,directory=p.is_dir())
        observed.append(q)
    return observed


def execute(args):
    started=time.perf_counter_ns();deadline=time.monotonic()+OVERALL_SECONDS;locations=_locations()
    out=Path(args.out).absolute()
    if out.exists() or out.is_symlink() or out.resolve()!=(locations['results']/OUT_NAME).resolve():
        raise ValueError('R24 requires the single new approved result directory')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    saved_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    saved_timer=signal.getitimer(signal.ITIMER_REAL);received_signals=[]
    def expired(signum,frame):
        received_signals.append(signum);raise RuntimeError('R24 outer signal '+str(signum))
    for sig in saved_handlers:signal.signal(sig,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.monotonic()))
    out.mkdir();work=out/'work';work.mkdir()
    for name in ('tmp','test-worker'):(work/name).mkdir()
    result=dict(run_id=out.name,round='R24',model=MODEL,mode=MODE,status='running',error=None,failed_stage=None,
                git_rev=None,git_branch=None,attempted=dict.fromkeys(CATEGORIES,0),completed=dict.fromkeys(CATEGORIES,0),
                processes=[],worker=None,worker_record_errors=[],retry_count=0,git_metadata_attempted=0,git_metadata_completed=0,
                artifact_bytes=0,case_artifact_bytes=0,source_records=None,outer_received_signals=received_signals,
                resource_monitor_error=None,worker_budget_records=[],native_packed_equivalence_verified=False,
                backend_ready=False,security_validated=False,accuracy_validated=False,formal_matrix_complete=False,
                new_llm_calls=0,new_downloads=0,new_installations=0,trace_runs=0,compiler_runs=0,optimizer_runs=0,
                solver_runs=0,keygens=0,ckks_runs=0,heldout_samples=0,confirm_samples=0,report_samples=0,candidates_frozen=0,
                limited_original_launcher_check_approved=True,extra_integrity_checks=0)
    result.update(dict.fromkeys(STAGE_FLAGS,False));recorder=StageRecorder(out/'events.jsonl');env=child_environment(work,locations)
    stop=threading.Event();child_events=[]
    def checkpoint():
        if result['resource_monitor_error']:raise RuntimeError(result['resource_monitor_error'])
        check_budget(out)
        if time.monotonic()>=deadline:raise TimeoutError('R24 entire-batch allowance exhausted')
    def monitor():
        while not stop.wait(1):
            try:
                size,free=case_size(out),shutil.disk_usage(out).free
                with (out/'artifact-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(),artifact_bytes=size,disk_free_bytes=free))+'\n')
                if size>CASE_BYTES or free<FREE_BYTES:raise RuntimeError('R24 periodic artifact/disk allowance exceeded')
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
                raise RuntimeError('R24 supervised phase failed: '+name)
            if category=='checks':
                w=result['worker'] or {}
                if (result['worker_record_errors'] or w.get('status')!='ok'
                        or any(w.get(k)!=v for k,v in dict(round='R24',model=MODEL,mode=MODE).items())
                        or any(w.get(k) is not True for k in STAGE_FLAGS)
                        or any(k not in w or type(w[k]) is not int or w[k]!=0 for k in ZERO_FIELDS)
                        or any(w.get(k) is not False for k in FALSE_FIELDS)
                        or any(k not in w or w[k] is not None for k in FORMAL)
                        or any(w.get(k)!=v for k,v in dict(native_load_attempts=1,native_load_completed=1,
                            checkpoint_load_attempts=1,checkpoint_load_completed=1,model_build_attempts=1,model_build_completed=1,
                            deepcopy_attempts=1,deepcopy_completed=1,probe_load_attempts=1,probe_load_completed=1,
                            original_forward_attempts=3,original_forward_completed=3,approx_forward_attempts=3,approx_forward_completed=3,
                            activation_attempts=57,activation_completed=57,math_check_attempts=57,math_check_completed=57,
                            original_reference_checks=3,silu_modules_replaced=10).items())):
                    raise RuntimeError('R24 fixed mathematical checks did not complete')
            if category:result['completed'][category]+=1
            else:result['git_metadata_completed']+=1
        checkpoint();print(json.dumps(dict(stage=name,state='completed')),flush=True);return obs
    try:
        watcher.start()
        with recorder.span('r24-preflight'):
            result['failed_stage']='source-preflight'
            with recorder.span('source-preflight'):
                checkpoint();result['source_records']=_source_preflight(locations);write_json(out/'source-metadata.json',result['source_records'])
                result['base_environment_before']=_base_environment_snapshot()
                result['base_environment_scope']='Known original launcher directory-entry metadata only; no whole-filesystem audit'
                base=Path(__file__).parent.parent
                for rel in ('experiments/r24_orbit_polynomial_worker.py','experiments/r24_orbit_polynomial_preflight.py',
                            'tests/test_r24_orbit_polynomial.py','tests/_r24_orbit_polynomial_cases.py'):
                    target=work/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/rel,target)
            launch('git-revision',['git','rev-parse','--short','HEAD'],30);result['git_rev']=(out/'git-revision.log').read_text().strip()
            launch('git-branch',['git','branch','--show-current'],30);result['git_branch']=(out/'git-branch.log').read_text().strip()
            launch('r24-tests',[sys.executable,'-B','-m','unittest','discover','-s','pt_agent/tests','-p','test_r24_*.py','-v'],600,'tests')
            log=(out/'r24-tests.log').read_text();counts=re.findall(r'Ran (\d+) tests? in ',log)
            if not counts or int(counts[-1])<=0 or 'skipped=' in log or '... skipped ' in log:raise RuntimeError('R24 requires positive tests and zero skips')
            summary=json.loads((work/'test-worker/inner-wrapper-result.json').read_text())
            if summary.get('status')!='ok' or summary.get('wrapper_runs')!=1 or not isinstance(summary.get('inner_tests'),int) or summary['inner_tests']<=0 or summary.get('inner_skips')!=0:
                raise RuntimeError('R24 unique inner fake suite did not pass')
            result['test_summary']=dict(system_tests=int(counts[-1]),inner=summary)
            launch('model-checks',phase_command(locations,work),480,'checks');result.update(status='ok',failed_stage=None)
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
        result['base_environment_after']=_base_environment_snapshot()
        result['root_raw_getrusage_hwm_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        w=result['worker'] or {}
        result['metrics']=dict(status=result['status'],mathematical_check_only=True,attempted=dict(result['attempted']),completed=dict(result['completed']),
                               counts={k:w.get(k,None if result['attempted']['checks'] else 0) for k in COUNTERS},
                               **{k:result[k] for k in STAGE_FLAGS},native_packed_equivalence_verified=False,
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
