"""Finite original HEIR package initialization and preprocessing diagnostic.

Preparation writes an own client. --run additionally builds offline and attempts
the unchanged generated package initializers and real plaintext/LT encoding.
No key constructor or ciphertext operation is called by this client.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample

GIB=1024**3
ROUND='HEIR-INIT-PROBE'
MODE='original-generated-heir-initialization-and-encoding-only'
MODEL='resnet20_silu'
FORBIDDEN=re.compile(r'\b(?:NewKeyGenerator|Gen\w*Key\w*|NewEncryptor|NewDecryptor|Encrypt\w*|Decrypt\w*|Bootstrap\w*)\s*\(')


def read(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):raise ValueError('Object metadata required')
    return value


def durable_json(path,value):
    write_json(path,value)
    with Path(path).open('r') as stream:os.fsync(stream.fileno())
    directory=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


@contextmanager
def serial_lock(root):
    with (Path(root)/'.heir-initializer-probe.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An initializer probe already owns this serial slot') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def _function(source,name):
    match=re.search(r'^func '+re.escape(name)+r'\([^\n]*\{',source,re.M)
    if not match:raise ValueError('Original generated function missing: '+name)
    begin=source.index('{',match.start());depth=1;end=begin+1
    while end<len(source) and depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    if depth:raise ValueError('Unclosed generated function')
    return source[begin:end]


def source_gate(go_result,clear_result):
    go_result,clear_result=Path(go_result).resolve(),Path(clear_result).resolve()
    generated,clear=read(go_result),read(clear_result)
    artifact=generated.get('native_artifact') or {}
    if (generated.get('status')!='ok' or generated.get('model')!=MODEL
            or generated.get('library_build_passed') is not True
            or artifact.get('static_bootstrap_operations')!=102
            or clear.get('status')!='ok' or clear.get('model',MODEL)!=MODEL):
        raise ValueError('Accepted current102 generated library and three-input clear case required')
    worker=read(clear_result.parent/'worker-result.json')
    if (worker.get('status')!='ok' or worker.get('math_probes_completed')!=3
            or worker.get('native_math_completed')!=3 or worker.get('keygen_runs')!=0
            or any(row.get('passed') is not True for row in worker.get('samples',[]))
            or len(worker.get('samples',[]))!=3):raise ValueError('Three original clear mathematics probes required')
    parameters=artifact.get('parameters') or []
    if len(parameters)!=1:raise ValueError('One original native residual literal required')
    p=parameters[0]
    if p.get('logN')!=16 or len(p.get('Q',[]))!=11 or len(p.get('P',[]))!=3 or p.get('log_default_scale')!=45:
        raise ValueError('Original N16/Q11/P3/scale45 literal required')
    module=go_result.parent/'module'
    if not (module/'go.mod').read_text().startswith('module heircnn\n'):
        raise ValueError('Original generated module identity differs')
    sources={name:(module/name).read_text() for name in ('heircnn.go','heircnn_utils/heircnn_utils.go')}
    configure=_function(sources['heircnn.go'],'Resnet20_silu__configure')
    literal=re.search(r'ckks.NewParametersFromLiteral\(ckks.ParametersLiteral\{(.*?)\}\)',configure,re.S)
    if not literal:raise ValueError('Original public parameter literal missing')
    body=literal[1]
    actual=dict(logN=int(re.search(r'LogN:\s*(\d+)',body)[1]),
        log_default_scale=int(re.search(r'LogDefaultScale:\s*(\d+)',body)[1]))
    for key in ('Q','P'):
        actual[key]=[int(value.strip()) for value in re.search(key+r':\s*\[\]uint64\{([^}]*)\}',body)[1].split(',') if value.strip()]
    if any(actual[key]!=p[key] for key in actual):raise ValueError('Generated configure and accepted public literals differ')
    for name,source in sources.items():
        if FORBIDDEN.search(_function(source,'init')):raise ValueError('Generated package init invokes cryptography')
        for line in source.splitlines():
            if line.startswith('var ') and '=' in line and FORBIDDEN.search(line):
                raise ValueError('Generated global initialization invokes cryptography')
    preprocessing=_function(sources['heircnn_utils/heircnn_utils.go'],'Resnet20_silu__preprocessing')
    if FORBIDDEN.search(preprocessing):raise ValueError('Generated preprocessing invokes cryptography')
    if 'encoder.Encode(' not in preprocessing or 'lintrans.NewTransformation(' not in preprocessing:
        raise ValueError('Original real encoder/LT preparation calls required')
    return dict(go_result=str(go_result),clear_result=str(clear_result),module=str(module),parameters=p,
        static_bootstrap_operations=102,preprocessing_function='heircnn_utils.Resnet20_silu__preprocessing',
        original_configure_called=False,original_generated_sources_modified=False,
        generated_package_initialization='Automatic, before own main; both heircnn and heircnn_utils imported',
        external_resources='Original absolute paths in the unchanged generated Go source',
        source_files=[dict(path=str(module/name),size_bytes=(module/name).stat().st_size) for name in sources])


def client_source(parameters):
    vector=lambda values:','.join(map(str,values))
    source=r'''package main
import (
 "encoding/json"
 "fmt"
 "os"
 "runtime"
 "runtime/debug"
 "syscall"
 "time"
 _ "heircnn"
 "heircnn/heircnn_utils"
 "github.com/tuneinsight/lattigo/v6/schemes/ckks"
)
func main(){
 runtime.GOMAXPROCS(1);debug.SetMemoryLimit(8<<30)
 if len(os.Args)!=2{fmt.Fprintln(os.Stderr,"worker result path required");os.Exit(2)}
 started:=time.Now()
 result:=map[string]any{"status":"running","main_entered":true,"preprocessing_attempts":0,"preprocessing_completed":0,
 "keygen_runs":0,"encryptions":0,"decryptions":0,"bootstraps":0,"model_forwards":0,
 "security_acceptance":nil,"accuracy_acceptance":nil,"scope":"Original generated package init and real PT/LT encoding only"}
 write:=func(){var usage syscall.Rusage;syscall.Getrusage(syscall.RUSAGE_SELF,&usage);result["ru_maxrss_kib"]=usage.Maxrss;result["elapsed_seconds"]=time.Since(started).Seconds()
 raw,e:=json.MarshalIndent(result,"","  ");if e!=nil{panic(e)};f,e:=os.OpenFile(os.Args[1]+".tmp",os.O_CREATE|os.O_WRONLY|os.O_TRUNC,0600);if e!=nil{panic(e)};if _,e=f.Write(append(raw,'\n'));e!=nil{panic(e)};if e=f.Sync();e!=nil{panic(e)};if e=f.Close();e!=nil{panic(e)};if e=os.Rename(os.Args[1]+".tmp",os.Args[1]);e!=nil{panic(e)}}
 defer func(){if value:=recover();value!=nil{result["status"]="error";result["error"]=fmt.Sprint(value);write();os.Exit(1)};write()}()
 write()
 p,e:=ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN:16,Q:[]uint64{Q_LITERAL},P:[]uint64{P_LITERAL},LogDefaultScale:45})
 if e!=nil{panic(e)}
 encoder:=ckks.NewEncoder(p)
 result["public_parameters_completed"]=true;result["preprocessing_attempts"]=1;write()
 pts,lts:=heircnn_utils.Resnet20_silu__preprocessing(p,encoder)
 result["plaintext_count"]=len(pts);result["linear_transform_count"]=len(lts)
 result["preprocessing_completed"]=1;result["status"]="ok";write()
 runtime.KeepAlive(pts);runtime.KeepAlive(lts)
}
'''
    return source.replace('Q_LITERAL',vector(parameters['Q'])).replace('P_LITERAL',vector(parameters['P']))


def physical_observation(sample,available,free):
    rss=None if sample is None else sample['subtree_rss_bytes']
    reasons=[]
    if rss is not None and rss>14*GIB:reasons.append('child subtree RSS exceeds14GiB')
    if available<1.25*GIB:reasons.append('host available RAM below1.25GiB')
    if free<10*GIB:reasons.append('free disk below10GiB')
    return dict(stop=bool(reasons),reasons=reasons,subtree_rss_bytes=rss,
        available_ram_bytes=available,free_disk_bytes=free,
        sample_ts_ns=None if sample is None else sample.get('sample_ts_ns'))


def interruption_status(result,default='error'):
    if result.get('guard_trigger') is not None:return 'resource_stopped'
    if result.get('timeout_signal') is not None or result.get('wall_timeout_observed'):return 'timeout'
    if result.get('monitor_stop_reason') is not None:return 'supervision_error'
    if result.get('incoming_signal') is not None:return 'interrupted'
    return default


def receive_signal(result,signum):
    if signum==signal.SIGALRM:result['timeout_signal']=signum
    elif result.get('guard_trigger') is not None:result['guard_stop_signal']=signum
    elif result.get('monitor_stop_reason') is not None:result['monitor_stop_signal']=signum
    else:result['incoming_signal']=signum
    raise RuntimeError('HEIR supervisor received signal '+str(signum))


def observe_termination(result,observation):
    if observation.get('timed_out'):result['wall_timeout_observed']=True
    signum=observation.get('signal')
    if signum is not None:
        if result.get('guard_trigger') is not None:result['guard_stop_signal']=signum
        elif result.get('monitor_stop_reason') is not None:result['monitor_stop_signal']=signum
        else:result['incoming_signal']=signum


def classify(observation,log,worker,guard=None):
    if observation.get('remaining_pids'):return 'supervision_error'
    if guard:return 'resource_stopped'
    if observation.get('timed_out'):return 'timeout'
    if observation.get('signal') is not None:return 'interrupted'
    if observation.get('exit_code')!=0:
        if re.search(r'out of memory|cannot allocate memory|failed to reserve|address space',log,re.I):return 'memory_failure'
        return 'initialization_error'
    if observation.get('signals_sent'):return 'interrupted'
    if (worker.get('status')!='ok' or worker.get('main_entered') is not True
            or worker.get('preprocessing_attempts')!=1 or worker.get('preprocessing_completed')!=1
            or worker.get('plaintext_count')!=1743 or worker.get('linear_transform_count')!=23
            or any(worker.get(key)!=0 for key in ('keygen_runs','encryptions','decryptions','bootstraps','model_forwards'))):
        return 'protocol_error'
    return 'ok'


def log_excerpt(path,limit=65536):
    """Keep startup failure and final stack context without loading a large log."""
    with Path(path).open('rb') as stream:
        head=stream.read(limit);stream.seek(0,os.SEEK_END);size=stream.tell()
        if size<=limit:return head.decode(errors='replace')
        stream.seek(max(limit,size-limit));tail=stream.read(limit)
    return (head+b'\n'+tail).decode(errors='replace')


def startup_observation(worker,log):
    """A missing durable main marker leaves entry unobserved without an init stack."""
    init_stack=bool(re.search(r'\bheircnn(?:/heircnn_utils)?\.init(?:\.\d+)?\(',log))
    main=True if worker.get('main_entered') is True else (False if init_stack else None)
    attempts=worker.get('preprocessing_attempts')
    entered=(attempts==1) if type(attempts) is int and attempts in (0,1) else None
    return dict(main_entered=main,preprocessing_entered=entered,
        public_parameters_completed=True if worker.get('public_parameters_completed') is True else None,
        worker_marker_present=bool(worker),generated_package_init_failure_stack=init_stack,
        actual_failure_before_own_main=True if init_stack and main is not True else (False if main is True else None),
        failure_location_scope='A durable main marker proves entry. An original generated-package init stack supports failure before main; missing markers alone leave entry and failure location unobserved.')


def environment(out):
    env={key:value for key,value in os.environ.items() if key not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')
         and not key.startswith(('LD_','NIX_','__NIX_','GO'))}
    env.update(GOTOOLCHAIN='local',GOPROXY='off',GOSUMDB='off',GOMAXPROCS='1',GOMEMLIMIT='8GiB',
        GODEBUG='inittrace=1',CGO_ENABLED='0',GOPATH=str(out/'gopath'),GOMODCACHE=str(out/'module-cache'),
        GOCACHE=str(out/'build-cache'),GOTMPDIR=str(out/'tmp'),TMPDIR=str(out/'tmp'),
        TEST_TELEMETRY_DIR=str(out/'telemetry'),PYTHONDONTWRITEBYTECODE='1')
    return env


def _execute(args):
    started=time.perf_counter_ns();out=Path(args.out).resolve()
    if out.exists() or not out.is_relative_to(paths.RESULTS_DIR.resolve()) or out==paths.RESULTS_DIR.resolve():
        raise ValueError('A new isolated result directory is required')
    out.mkdir(parents=True)
    for name in ('module','tmp','telemetry'):(out/name).mkdir()
    (out/'telemetry/mode').write_text('off\n')
    result=dict(round=ROUND,mode=MODE,model=MODEL,status='running',error=None,attempts=dict(build=0,initialize=0),
        completed=dict(build=0,initialize=0),processes=[],worker=None,main_entered=None,preprocessing_entered=None,
        keygen_runs=0,encryptions=0,decryptions=0,bootstraps=0,model_forwards=0,llm_calls=0,downloads=0,installations=0,
        original_generated_sources_modified=False,original_libraries_modified=False,
        security_acceptance=None,accuracy_acceptance=None,CKKS_latency=None,compiled_bootstraps=None,
        formal_matrix_complete=False,diagnostic_completed=False,incoming_signal=None,timeout_signal=None,
        wall_timeout_observed=False,guard_trigger=None,guard_stop_signal=None,
        limits=dict(native_AS_gib=12,subtree_RSS_gib=14,minimum_available_RAM_gib=1.25,minimum_free_disk_gib=10,
            native_wall_seconds=300,build_AS_gib=4,build_wall_seconds=300,total_wall_seconds=660,
            artifact_case_gib=4,concurrency=1,GOMEMLIMIT='8GiB'))
    recorder=StageRecorder(out/'events.jsonl');stop=threading.Event();failures=[];active=[None]
    deadline=time.monotonic()+660
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupted(signum,frame):receive_signal(result,signum)
    def check():
        if failures:raise RuntimeError(failures[0])
        if time.monotonic()>deadline:
            result['wall_timeout_observed']=True;raise TimeoutError('Initializer probe case wall allowance exhausted')
        if artifact_bytes(out)>4*GIB:
            result['guard_trigger']=dict(kind='artifact',allowance_bytes=4*GIB)
            raise RuntimeError('Initializer probe artifact allowance exhausted')
        observed=physical_observation(_latest_rss_sample(active[0]) if active[0] else None,
            _available_ram_bytes(),shutil.disk_usage(out).free)
        if observed['stop']:
            result['resource_guard']=observed;result['guard_trigger']=observed
            durable_json(out/'resource-guard.json',observed)
            raise RuntimeError('; '.join(observed['reasons']))
    def monitor():
        while not stop.wait(.1):
            try:check()
            except BaseException as exc:
                failures.append(type(exc).__name__+': '+str(exc))
                try:
                    result['monitor_stop_reason']=failures[-1]
                finally:
                    try:os.kill(os.getpid(),signal.SIGTERM)
                    except OSError as signal_error:result['monitor_signal_error']=str(signal_error)
                return
    watcher=threading.Thread(target=monitor,daemon=True)
    for sig in old:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,660)
    try:
        check();watcher.start()
        with recorder.span('source-and-client-preparation'):
            source=source_gate(args.go_result,args.clear_result);result['source']=source
            original=Path(source['module'])
            text=(original/'go.mod').read_text()
            (out/'module/go.mod').write_text(text.replace('module heircnn\n','module heir-initializer-probe\n',1)
                +'\nrequire heircnn v0.0.0\nreplace heircnn => '+json.dumps(str(original))+'\n')
            shutil.copy2(original/'go.sum',out/'module/go.sum')
            (out/'module/main.go').write_text(client_source(source['parameters']))
            shutil.copy2(Path(__file__),out/'driver.py');write_json(out/'source-inventory.json',source)
        result['status']='prepared';write_json(out/'result.json',result)
        if args.run:
            with recorder.span('private-offline-cache-copy'):
                for name in ('module-cache','build-cache'):shutil.copytree(Path(args.go_result).parent/name,out/name)
            env=environment(out)
            def phase(name,command,memory):
                check();result['attempts'][name]=1;result['active_stage']=name
                row=dict(stage=name,argv=command,observation=None,supervision_exception=None)
                result['processes'].append(row)
                durable_json(out/'result.json',result) # Durable launch record precedes package init.
                with recorder.span(name,argv=command):
                    active[0]=out/(name+'-resources.jsonl')
                    try:
                        observation=run_process(command,cwd=out/'module',log=out/(name+'.log'),resources=active[0],
                            timeout=min(300,max(.01,deadline-time.monotonic())),mem_gib=memory,fsize_gib=1,env=env)
                        row['observation']=observation;observe_termination(result,observation)
                    except BaseException as exc:
                        row['supervision_exception']=type(exc).__name__+': '+str(exc)
                        row['last_resource_sample']=_latest_rss_sample(active[0]);raise
                    finally:active[0]=None
                return observation
            binary=out/'initializer-probe'
            build=phase('build',[str(args.go),'build','-p=1','-mod=readonly','-buildvcs=false','-o',str(binary),'.'],4)
            if (build.get('exit_code')!=0 or build.get('timed_out') or build.get('signal') is not None
                    or build.get('remaining_pids') or build.get('signals_sent')):
                result['status']=interruption_status(result,'build_error')
            else:
                result['completed']['build']=1
                observation=phase('initialize',[str(binary),str(out/'worker-result.json')],12)
                worker=read(out/'worker-result.json') if (out/'worker-result.json').is_file() else {}
                result['worker']=worker or None
                log=log_excerpt(out/'initialize.log')
                result.update(startup_observation(worker,log))
                result['status']=classify(observation,log,worker,result.get('guard_trigger'))
                result['diagnostic_completed']=not observation.get('remaining_pids')
                if result['status']=='ok':result['completed']['initialize']=1
    except BaseException as exc:
        result.update(status=interruption_status(result),
            error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in old.items():signal.signal(sig,handler)
        result['resource_monitor_error']=failures[0] if failures else None
        if failures:result['status']=interruption_status(result)
        worker_path=out/'worker-result.json'
        if result['worker'] is None and worker_path.is_file():
            try:result['worker']=read(worker_path)
            except (OSError,ValueError) as exc:result['partial_worker_read_error']=str(exc)[:1000]
        if result['worker'] or result['attempts']['initialize']:
            log=log_excerpt(out/'initialize.log') if (out/'initialize.log').is_file() else ''
            result.update(startup_observation(result['worker'] or {},log))
        timing=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns())
        result['timing']=timing;write_json(out/'result.json',result)
        registry=Path(args.registry or paths.RESULTS_DIR/'registry.jsonl')
        with registry.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
            if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):raise ValueError('Existing probe registry identity')
            stream.seek(0,os.SEEK_END);stream.write(json.dumps(dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,
                git_rev=None,metrics={key:result[key] for key in ('status','attempts','completed','main_entered','preprocessing_entered',
                    'diagnostic_completed','keygen_runs','security_acceptance')},artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=timing,
            required_artifact_bytes=artifact_bytes(out))
    return result


def execute(args):
    with serial_lock(paths.RESULTS_DIR):return _execute(args)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--go-result',type=Path,required=True)
    parser.add_argument('--clear-result',type=Path,required=True)
    parser.add_argument('--go',type=Path,default=paths.WORKSPACE/'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/go')
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--run',action='store_true',help='Build and attempt original initialization under the fixed finite policy')
    result=execute(parser.parse_args(argv))
    print(json.dumps({key:result[key] for key in ('status','attempts','completed','main_entered','preprocessing_entered')}))
    return 0 if result['status'] in ('ok','prepared') else 1


if __name__=='__main__':raise SystemExit(main())
