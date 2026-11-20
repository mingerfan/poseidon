"""Retain a closed, incomplete official DaCapo trace in a reversible archive."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import threading
import time

from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
from .completion_metrics import artifact_bytes,persist_case_completion
from .r34_archive_text import archive_one,archive_state


def source_gate(source_result,results_root):
    source_result=Path(source_result).resolve();root=Path(results_root).resolve()
    if not source_result.is_relative_to(root):raise ValueError('Use a recorded results case')
    case=source_result.parent;record=json.loads(source_result.read_text())
    complete=json.loads((case/'completion.json').read_text())
    obs=record.get('trace_process') or {}
    if (source_result.name!='result.json' or record.get('mode')!='original-source-native-profile-compile'
            or record.get('status')!='error' or complete.get('status')!='error'
            or complete.get('registry_persisted') is not True or complete.get('required_artifacts_persisted') is not True
            or record.get('trace_attempts')!=1 or record.get('trace_completed')!=0
            or record.get('compile_attempts')!=0 or record.get('native_compile_attempts')!=0
            or obs.get('remaining_pids') or obs.get('exit_code') in (None,0)):
        raise ValueError('Only a closed unsuccessful trace with no compilation is eligible')
    names={'resnet20':'ResNet','vgg16':'VGG16','alexnet':'AlexNet'}
    name=names.get(record.get('original_model_selector'))
    if name is None:raise ValueError('Unknown original model')
    source=case/'work/trace/traced'/(name+'.mlir')
    if source.is_symlink() or source.resolve()!=source or not source.is_file() or source.stat().st_size<=0:
        raise ValueError('The explicit original partial trace must be a regular file')
    if source.with_name(source.name+'.gz').exists() or source.with_name(source.name+'.gz.partial').exists():
        raise ValueError('Preserve prior archives and partial attempts')
    return source,record


def run(args):
    from .. import paths
    minimum_free_gib=getattr(args,'minimum_free_gib',10)
    if type(minimum_free_gib)not in(int,float)or not 4<=minimum_free_gib<=10:
        raise ValueError('Use a finite disk reserve from4to10GiB')
    root=paths.RESULTS_DIR.resolve();out=Path(args.out).resolve()
    if out.exists() or not out.is_relative_to(root) or out==root:raise ValueError('A new results case is required')
    source,original=source_gate(args.source_result,root)
    if out.is_relative_to(Path(args.source_result).resolve().parent):raise ValueError('Keep archive accounting outside the source case')
    out.mkdir();started=time.perf_counter_ns();recorder=StageRecorder(out/'events.jsonl')
    result=dict(status='running',mode='closed-incomplete-trace-reversible-gzip',source_result=str(Path(args.source_result).resolve()),
        source_semantics='Incomplete trace retained as evidence; never a valid compiler input',source_model=original['model'],
        source_original_error=original.get('error'),original_trace_process=original['trace_process'],
        source_original_completion_unchanged=True,extra_integrity_checks=0,api_calls=0,model_runs=0,compiler_runs=0,
        reserve_disk_gib=minimum_free_gib,address_space_gib=.5,files=[],error=None,guard_stop=None)
    target=source.with_name(source.name+'.gz');temporary=source.with_name(source.name+'.gz.partial')
    stop=threading.Event();deadline=time.monotonic()+600
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupted(signum,frame):raise RuntimeError('Archive interrupted by signal '+str(signum))
    def check():
        if shutil.disk_usage(root).free<minimum_free_gib*1024**3:raise RuntimeError('Disk reserve unavailable')
        if time.monotonic()>deadline:raise RuntimeError('Archive wall allowance exhausted')
    def monitor():
        while not stop.wait(.1):
            try:check()
            except BaseException as exc:
                result['guard_stop']=str(exc)
                os.kill(os.getpid(),signal.SIGTERM);return
    for sig in previous:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,600);watcher=threading.Thread(target=monitor,daemon=True)
    try:
        limit=512*1024**2;_,hard=resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,(limit if hard==resource.RLIM_INFINITY else min(limit,hard),hard))
        check();watcher.start()
        result['in_progress']=archive_state(source,target,temporary);write_json(out/'result.json',result)
        with recorder.span('compress_closed_partial_trace'):
            result['files']=[archive_one(source,target,temporary)]
        result['status']='ok';result['in_progress']=None
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc),in_progress=archive_state(source,target,temporary))
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=1)
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in previous.items():signal.signal(sig,handler)
        result['released_bytes']=sum(r['released_bytes']for r in result['files'])
        result['timing']=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns())
        result['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(out/'result.json',result)
        revision=(Path(args.source_result).parent/'git.log').read_text().strip()
        row=dict(run_id=out.name,round='R34-CLOSED-TRACE',model=original['model'],mode=result['mode'],git_rev=revision,
            metrics={k:result[k]for k in ('status','released_bytes','extra_integrity_checks')},artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with (root/'registry.jsonl').open('a')as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(out,round_label='R34-CLOSED-TRACE',started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-result',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--minimum-free-gib',type=float,default=10)
    args=parser.parse_args(argv)
    if not 4<=args.minimum_free_gib<=10:parser.error('Use a finite disk reserve from4to10GiB')
    result=run(args);print(json.dumps({k:result[k]for k in ('status','released_bytes','error')}));return 0 if result['status']=='ok'else 1


if __name__=='__main__':raise SystemExit(main())
