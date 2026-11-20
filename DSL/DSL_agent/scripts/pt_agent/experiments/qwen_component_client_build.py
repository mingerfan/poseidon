"""Offline build of the own generic Qwen multi-component library only.

No27-artifact manifest is fabricated or admitted here. A successful archive
compiles the concrete52-call/3-bootstrap client code; original artifact linkage,
parameter constructors and encrypted execution remain separate future work.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes,persist_case_completion
from .heir_initializer_probe import durable_json,serial_lock
from .r10_paper_metrics import StageRecorder,run_process,read_events,summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes,_latest_rss_sample

ROUND='QWEN-GENERIC-COMPONENT-LIBRARY-BUILD'
MODE='constructor-free-raw43-shared-native-multi-io-library-build-only'
MODULE='pt_agent.experiments.qwen_component_client_build'
GO_SOURCE=Path(__file__).with_name('qwen_component_client')/'runtime.go'
GIB=1024**3
TOTAL_SECONDS=900
BUILD_SECONDS=600
CASE_BYTES=8*GIB
RESERVE=1024**2
ZERO=('api_calls','downloads','installations','sample_reads','weight_array_reads','models',
 'Count_runs','Emit_runs','native_compiles','generated_code_runs','context_constructions','keygen_runs',
 'encryptions','decryptions','real_ckks_attempts','real_ckks_completed','extra_integrity_checks')


def read(path):return json.loads(Path(path).read_text())


def clean(observed):
    if (type(observed.get('exit_code'))is not int or observed['exit_code']!=0 or observed.get('timed_out')is not False
            or observed.get('signal')is not None or observed.get('signals_sent')!=[]or observed.get('remaining_pids')!=[]):
        raise RuntimeError('Offline library build did not finish cleanly; actual observation retained')


def file_record(path):
    path=Path(path).resolve()
    if not path.is_file()or path.stat().st_size<=0:raise ValueError('Required existing source is missing: '+str(path))
    return dict(path=str(path),bytes=path.stat().st_size)


def source_gate(args):
    prepared=args.prepared.resolve();record=read(prepared/'result.json')
    if record.get('round')!='R31'or record.get('phase')!='prepare'or record.get('status')!='ok':
        raise ValueError('Successful original isolated R31 preparation required')
    work=prepared/'work';module=work/'module';sdk=args.go_sdk.resolve()
    if (sdk/'VERSION').read_text().splitlines()[0]!='go1.25.13'or not os.access(sdk/'bin/go',os.X_OK):
        raise ValueError('Use existing Go1.25.13 SDK')
    mod=(module/'go.mod').read_text()
    replacements=re.findall(r'^replace github\.com/tuneinsight/lattigo/v6 => (.+)$',mod,re.M)
    if (not mod.startswith('module pt-agent-r31-plan\n')or not re.search(r'^go 1\.25\.0\s*$',mod,re.M)
            or not re.search(r'^require github\.com/tuneinsight/lattigo/v6 v6\.0\.0\s*$',mod,re.M)
            or len(replacements)!=1 or Path(replacements[0]).resolve()!=Path(record.get('lattigo_source','')).resolve()
            or re.search(r'^\s*toolchain\b',mod,re.M)):
        raise ValueError('Retain original R31 module version and its existing isolated Lattigo replacement')
    root=Path(replacements[0]).resolve()
    for target in (work/'module-cache',work/'build-cache',root):
        if not target.is_dir():raise ValueError('Existing original offline cache/source is unavailable')
    text=GO_SOURCE.read_text()
    forbidden=r'\b(?:NewParametersFromLiteral|NewKeyGenerator|NewEncryptor|NewDecryptor|Gen\w*Key\w*|EncryptNew|DecryptNew)\s*\('
    if (not re.search(r'^package qwengeneration\s*$',text,re.M)or re.search(forbidden,text)
            or re.search(r'\bfunc\s+(?:main|init)\s*\(',text)):
        raise ValueError('Constructor-free own library source required')
    return dict(prepared=str(prepared),sdk=str(sdk),go=file_record(sdk/'bin/go'),go_version='1.25.13',
        go_mod=file_record(module/'go.mod'),go_sum=file_record(module/'go.sum'),lattigo_source=str(root),
        lattigo_version='6.0.0',module_cache=str(work/'module-cache'),build_cache=str(work/'build-cache'),
        own_go=file_record(GO_SOURCE),generic_library_only=True,actual_component_artifacts_admitted=0,
        original_libraries_modified=False,parameters_selected=False,source_integrity_comparisons=0)


def environment(out):
    env={k:v for k,v in os.environ.items()if k not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')
         and not k.startswith(('LD_','NIX_','__NIX_','GO'))}
    env.update(GOENV='off',GOWORK='off',GOTOOLCHAIN='local',GOPROXY='off',GOSUMDB='off',CGO_ENABLED='0',GOMAXPROCS='1',GOMEMLIMIT='2GiB',
        GOPATH=str(out/'work/gopath'),GOMODCACHE=str(out/'work/module-cache'),GOCACHE=str(out/'work/build-cache'),
        GOTMPDIR=str(out/'work/tmp'),TMPDIR=str(out/'work/tmp'),TEST_TELEMETRY_DIR=str(out/'work/telemetry'),
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    return env


def prepare(out,source):
    work=out/'work';module=work/'module';module.mkdir(parents=True)
    for name in ('tmp','telemetry','gopath'):(work/name).mkdir()
    (work/'telemetry/mode').write_text('off\n')
    mod=Path(source['go_mod']['path']).read_text()
    (module/'go.mod').write_text('module pt-agent-qwen-native-components\n'+mod.split('\n',1)[1])
    shutil.copyfile(source['go_sum']['path'],module/'go.sum')
    shutil.copyfile(source['own_go']['path'],module/'runtime.go')
    snapshots=out/'source';snapshots.mkdir();shutil.copyfile(__file__,snapshots/Path(__file__).name)
    return [str(Path(source['sdk'])/'bin/go'),'build','-p=1','-mod=readonly','-buildvcs=false',
        '-buildmode=archive','-o',str(work/'qwen-generation.a'),'.']


def budget(out):
    size=artifact_bytes(out);free=shutil.disk_usage(out).free
    if size+RESERVE>CASE_BYTES or free<10*GIB:raise RuntimeError('Preserve8GiB case and10GiB free-disk allowance')
    return dict(artifact_bytes=size,free_bytes=free)


def physical(sample,available):
    rss=None if sample is None else sample['subtree_rss_bytes'];reasons=[]
    if rss is not None and rss>4*GIB:reasons.append('Build subtree RSS exceeds4GiB')
    if available<2*GIB:reasons.append('Host available RAM below2GiB')
    return dict(stop=bool(reasons),reasons=reasons,subtree_RSS_bytes=rss,host_available_bytes=available)


def close(out,result,started,recorder,registry):
    result['timing']=summarize_events(read_events(out/'events.jsonl'))
    for _ in range(20):
        size=artifact_bytes(out)
        if size+RESERVE>CASE_BYTES:result.update(status='error',error='Case allowance including completion reserve exhausted')
        result['generic_library_compile_accepted']=result['status']=='ok'and result['generic_library_build_passed']is True
        # No current run can convert this generic archive into27 actual proofs.
        result.update(linked_native_program_verified=False,full_qwen_compile_verified=False,whole_generation_verified=False)
        durable_json(out/'result.json',result);size=artifact_bytes(out)
        if result.get('artifact_bytes')==size:break
        result['artifact_bytes']=size
    else:raise RuntimeError('Terminal metadata size failed to stabilize')
    row=dict(run_id=out.name,round=ROUND,model='Qwen2-0.5B',mode=MODE,git_rev=None,
        metrics={k:result[k]for k in ('status','attempted_builds','completed_builds','generic_library_build_passed',
            'generic_library_compile_accepted','linked_native_program_verified','full_qwen_compile_verified','whole_generation_verified')},
        artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with Path(registry).open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):raise ValueError('Preserve existing registry identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    marker=persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
    if marker['case_artifact_bytes']>CASE_BYTES:raise RuntimeError('Completion exceeded reserved case allowance')


def execute(args):
    root=paths.RESULTS_DIR.resolve();out=args.out.resolve();started=time.perf_counter_ns();deadline=time.monotonic()+TOTAL_SECONDS
    if out.exists()or out.is_symlink()or out==root or not out.is_relative_to(root):raise ValueError('Use one fresh configured result case')
    with serial_lock(root):
        out.mkdir();recorder=StageRecorder(out/'events.jsonl');active=[None];stop=threading.Event();errors=[]
        result=dict(round=ROUND,mode=MODE,model='Qwen2-0.5B',status='running',error=None,attempted_builds=0,completed_builds=0,
            generic_library_build_passed=False,generic_library_compile_accepted=False,linked_native_program_verified=False,
            full_qwen_compile_verified=False,whole_generation_verified=False,actual_component_artifacts_admitted=0,
            actual_component_manifest=None,parameters_selected=False,security_acceptance=None,numerical_acceptance=None,
            CKKS_latency=None,scope='Generic native52-call/3-connector library archive only; original27 source artifacts and actual execution remain unadmitted',
            limits=dict(total_seconds=900,build_seconds=600,AS_gib=4,RSS_gib=4,minimum_available_gib=2,file_gib=8,
                case_gib=8,minimum_free_disk_gib=10,concurrency=1,retries=0),**dict.fromkeys(ZERO,0))
        old={sig:signal.getsignal(sig)for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
        def receive(sig,frame):
            result['termination']=dict(signal=sig,cause=errors[-1]if errors else'incoming_signal')
            raise InterruptedError('Finite library build interrupted')
        for sig in old:signal.signal(sig,receive)
        def check():
            if errors:raise RuntimeError(errors[-1])
            if time.monotonic()>=deadline:raise TimeoutError('900-second complete library allowance exhausted')
            budget(out);observed=physical(_latest_rss_sample(active[0])if active[0]else None,_available_ram_bytes())
            if observed['stop']:result['resource_guard']=observed;raise RuntimeError(str(observed))
        def watch():
            while not stop.wait(.25):
                try:check()
                except BaseException as exc:
                    errors.append(type(exc).__name__+': '+str(exc));durable_json(out/'guard-failure.json',dict(error=errors[-1]));os.kill(os.getpid(),signal.SIGTERM);return
        watcher=threading.Thread(target=watch,daemon=True)
        try:
            signal.setitimer(signal.ITIMER_REAL,TOTAL_SECONDS);watcher.start()
            with recorder.span('source-and-own-library-module'):
                check();source=source_gate(args);result['source']=source;command=prepare(out,source)
                result['command']=command;durable_json(out/'source.json',source)
            result['status']='prepared';durable_json(out/'result.json',result)
            if args.run:
                with recorder.span('private-existing-cache-copy'):
                    for name,key in (('module-cache','module_cache'),('build-cache','build_cache')):
                        check();shutil.copytree(source[key],out/'work'/name);check()
                result['attempted_builds']=1;durable_json(out/'result.json',result);active[0]=out/'build-resources.jsonl'
                with recorder.span('offline-generic-library-build',argv=command):
                    observed=run_process(command,cwd=out/'work/module',log=out/'build.log',resources=active[0],
                        timeout=min(BUILD_SECONDS,max(.01,deadline-time.monotonic())),mem_gib=4,fsize_gib=8,env=environment(out))
                active[0]=None;result['observation']=observed;durable_json(out/'result.json',result);clean(observed)
                result['completed_builds']=1
                archive=out/'work/qwen-generation.a'
                if archive.is_symlink()or not archive.is_file()or archive.stat().st_size<=0:raise RuntimeError('Actual Go library archive missing')
                result['archive']=file_record(archive);result['generic_library_build_passed']=True;check();result['status']='ok'
        except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
        finally:
            active[0]=None;stop.set()
            if watcher.is_alive():watcher.join(timeout=3)
            signal.setitimer(signal.ITIMER_REAL,0)
            if errors:result.update(status='error',error=errors[-1])
            try:close(out,result,started,recorder,args.registry or root/'registry.jsonl')
            finally:
                for sig,handler in old.items():signal.signal(sig,handler)
        return result


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--registry',type=Path)
    p.add_argument('--prepared',type=Path,default=paths.RESULTS_DIR/'lattigo-runtime-plan-prepare-r31-20261116')
    p.add_argument('--go-sdk',type=Path,default=paths.ORBIT_ROOT.parent/'toolchains/go125/go')
    return p


def main(argv=None):
    args=parser().parse_args(argv);result=execute(args)
    print(json.dumps({key:result[key]for key in ('status','error','generic_library_build_passed','linked_native_program_verified','whole_generation_verified')}))
    return int(result['status']not in ('ok','prepared'))


if __name__=='__main__':raise SystemExit(main())
