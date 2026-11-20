"""One low-priority reversible archive of the fixed closed BERT raw trace.

Only original R34 archive_one reads compiler text. Admission reads JSON/stat
metadata, preserving all executable/data/scientific artifacts. This independent
small archive never acquires or signals the global experiment lease.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import threading
import time

from .. import paths
from . import r34_archive_text as original
from .completion_metrics import artifact_bytes,persist_case_completion
from .dacapo_native_binding import content_filename
from .r10_paper_metrics import StageRecorder,_process,read_events,summarize_events,write_json
from .r27_orbit_lazy_backend import _available_ram_bytes

MODULE='pt_agent.experiments.bert_closed_trace_archive'
ROUND='BERT-CLOSED-TRACE-ARCHIVE'
MODEL='bert-tiny-sst2'
MODE='one-closed-BERT-raw-trace-reversible-gzip-low-priority'
CASES=dict(native='bert-sst2-three-round-frozen-compile-20261117',
    master='bert-sst2-three-round-compile-first-20261117',
    report='bert-sst2-retained-three-round-isolated-report-20261117')
REGISTRY_FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
GIB=1024**3
SECONDS=1800
META_BYTES=16*1024**2
archive_one=original.archive_one


def read(path):return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def metadata(path):
    path=Path(path)
    require(not path.is_symlink() and path.resolve()==path, 'Retain canonical nonsymlink source artifacts')
    value=path.stat()
    require(stat.S_ISREG(value.st_mode), 'Require a regular original artifact')
    return dict(path=str(path),device=value.st_dev,inode=value.st_ino,bytes=value.st_size,mtime_ns=value.st_mtime_ns)


def audited_stat(saved,expected):
    current=metadata(expected)
    require(all(saved.get(key)==current[key] for key in current), 'Saved/current artifact stat provenance differs')
    return current


def _clean(row):
    return (isinstance(row,dict) and type(row.get('exit_code')) is int and row['exit_code']==0
        and row.get('timed_out') is False and row.get('signal') is None
        and row.get('remaining_pids')==[] and row.get('signals_sent')==[])


def source_gate(audit_path):
    """No model/source gate, compiler text, RuntimePlan body or payload reads."""
    root=paths.RESULTS_DIR.resolve()
    require(metadata(audit_path)['bytes']<2*1024**2, 'Require the bounded original metadata audit')
    audit=read(audit_path)
    require(audit.get('must_fix')==[] and audit.get('checks')
        and all(check.get('passed') is True for check in audit['checks']), 'Original bounded metadata audit must be accepted')
    records={};births={};cases={role:root/name for role,name in CASES.items()}
    wanted={path.name:path for path in cases.values()};rows={name:[] for name in wanted}
    with (root/'registry.jsonl').open() as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row=json.loads(line)
                if row.get('run_id') in rows:rows[row['run_id']].append(row)
    for role,case in cases.items():
        require(case.resolve()==case and not case.is_symlink(), 'Fixed BERT cases must remain canonical')
        saved=audit['case_summaries'][role]
        require(saved['case']==str(case), 'Only the three fixed audited BERT cases are eligible')
        result=read(case/'result.json');done=read(case/'completion.json')
        require(result.get('model')==MODEL and result.get('status')=='ok' and done.get('status')=='ok'
            and done.get('round')==result.get('round')
            and done.get('registry_persisted') is True and done.get('required_artifacts_persisted') is True
            and result.get('termination') is None and result.get('terminal_budget_error') is None,
            'All original BERT native/master/report cases need clean durable success')
        for key in ('attempted','completed','compiled_bootstraps','report_samples'):
            if saved.get(key) is not None:require(result.get(key)==saved[key], 'Actual BERT completion differs from the original audit')
        selected=rows[case.name]
        require(len(selected)==1 and set(selected[0])==REGISTRY_FIELDS
            and selected[0].get('artifacts_dir')==str(case)
            and all(selected[0].get(k)==result.get(k) for k in ('round','model','mode','metrics')),
            'Require each original unique matching eight-field registry row')
        if role=='report':
            require(result.get('attempted_report')==1 and result.get('completed_report')==1
                and result.get('report_accepted') is True and result.get('report_samples')==872
                and _clean(result.get('process')), 'The original872 report must have completed once cleanly')
        else:
            expected=['emit','trace','compile'] if role=='native' else ['math','compile']
            require([r.get('phase')for r in result.get('processes',[])]==expected
                and all(_clean(r.get('observation'))for r in result['processes']), 'Original BERT producer phase closure is incomplete')
        records[role]=dict(case=str(case),round=result['round'],mode=result['mode'],status='ok',registry=selected[0])
        for log in case.rglob('*resources.jsonl'):
            with log.open() as stream:
                for line in stream:
                    sample=json.loads(line)
                    for item in [*sample.get('processes',[]),sample.get('outer_process')]:
                        if item is not None:births[int(item['pid'])]=int(item['starttime'])
    master=read(cases['master']/'result.json');report=read(cases['report']/'result.json')
    require(master.get('compile_case')==str(cases['native']) and report.get('compile_case')==str(cases['master'])
        and report.get('source',{}).get('native_case')==str(cases['native']), 'Original BERT source lineage differs')
    audited={int(r['pid']):int(r['starttime'])for r in audit['recorded_births']}
    require(births and births==audited, 'All original audited process births need complete recorded evidence')
    for pid,birth in births.items():
        current=_process(pid)
        require(not current or current['starttime']!=birth or current['state'] in ('Z','X'), 'An original BERT process birth remains active')
    emit=cases['native']/'work/emit';raw=emit/'candidate_trace.mlir'
    source=audited_stat(audit['raw'],raw)
    require(0<source['bytes']<32*GIB, 'Single raw trace must fit the finite32GiB file allowance')
    archive=raw.with_name(raw.name+'.gz');temporary=archive.with_name(archive.name+'.partial')
    require(not any(p.exists() or p.is_symlink() for p in (archive,temporary)), 'Preserve existing archive/partial attempts')
    preserved=[audited_stat(audit['runtime_plan'],emit/'lowered._hecate_golden.runtime-plan.json')]
    manifest=emit/'lowered._hecate_golden.bundle/manifest.json'
    preserved.append(audited_stat(audit['bundle']['manifest'],manifest))
    expected={str(emit/name)for name in ('constants.npz','candidate_trace.py','lowered.mlir')}
    require({r['path']for r in audit['preserved_emit_artifacts']}==expected, 'All original core executable artifacts must stay present')
    preserved.extend(audited_stat(row,Path(row['path']))for row in audit['preserved_emit_artifacts'])
    bundle=read(manifest)
    require(bundle.get('bundle_format_version')==1 and bundle.get('version')==1, 'Retain original bundle manifest format')
    seen=set();total=0
    for blob in bundle['blobs']:
        name=content_filename(blob['content'])  # Existing opaque locator only.
        require(name not in seen and type(blob['byte_length']) is int and blob['byte_length']>0, 'Original bundle locator/length invalid')
        seen.add(name);row=metadata(manifest.parent/'data'/name)
        require(row['bytes']==blob['byte_length'], 'Preserve original executable bundle blob length')
        preserved.append(row);total+=row['bytes']
    require(len(seen)==audit['bundle']['blob_count']==audit['bundle']['referenced_blob_count']
        and total==audit['bundle']['blob_bytes'], 'Original bundle inventory differs')
    return dict(cases=records,source=source,archive=str(archive),temporary=str(temporary),
        preserved=preserved,recorded_birth_count=len(births),process_births={str(k):v for k,v in births.items()},
        bundle_blobs=len(seen),bundle_bytes=total,
        source_binding_scope='Saved/current stat metadata and closed result lineage only; content not revalidated',
        payload_reads_for_admission=0,hash_CRC_readback_checks=0)


@contextmanager
def local_lock(root):
    with (Path(root)/'bert-closed-trace-archive.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def files(source):
    result={}
    for key,path in (('source',source['source']['path']),('archive',source['archive']),('temporary',source['temporary'])):
        value=Path(path)
        result[key]=dict(exists=value.exists(),bytes=value.stat().st_size if value.exists() else None)
    return result


def resources(args,source,deadline,*,admission=False):
    state=files(source);free=shutil.disk_usage(paths.RESULTS_DIR).free;available=_available_ram_bytes()
    reservation=source['source']['bytes']+(source['source']['bytes']+99)//100
    reasons=[]
    if time.monotonic()>=deadline:reasons.append('1800-second wall allowance exhausted')
    if free<10*GIB+(reservation if admission else 0):reasons.append('Temporary allocation plus10GiB free reserve unavailable')
    if available<1.25*GIB:reasons.append('Available physical RAM below1.25GiB')
    if sum(state[k]['bytes']or 0 for k in ('archive','temporary'))>reservation:reasons.append('Worst temporary archive allowance exceeded')
    if artifact_bytes(args.out)+1024**2>META_BYTES:reasons.append('Finite metadata case allowance exhausted')
    return dict(stop=bool(reasons),reasons=reasons,files=state,free_bytes=free,available_bytes=available,
        worst_temporary_reservation_bytes=reservation)


def process_limits():
    for name,allowance in ((resource.RLIMIT_AS,GIB//2),(resource.RLIMIT_FSIZE,32*GIB)):
        _,hard=resource.getrlimit(name);bound=allowance if hard==resource.RLIM_INFINITY else min(hard,allowance)
        resource.setrlimit(name,(bound,bound))
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))


def register(out,result):
    row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=None,
        metrics={k:result[k]for k in ('status','attempted','completed','archive_accepted','released_bytes')},
        artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.seek(0)
        require(not any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()), 'Preserve existing archive registry identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def execute(args):
    root=paths.RESULTS_DIR.resolve();out=Path(args.out)
    require(out.parent==root and out.resolve()==out and not out.exists() and not out.is_symlink()
        and out.name.startswith('bert-closed-trace-archive-') and type(args.seconds) is int and 0<args.seconds<=SECONDS,
        'Use a fresh direct archive metadata case and at most1800seconds')
    out.mkdir();began=time.perf_counter_ns();deadline=time.monotonic()+args.seconds
    recorder=StageRecorder(out/'events.jsonl');stop=threading.Event();watcher=None;handlers={};old_timer=None
    timer_started=time.monotonic()
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,attempted=0,completed=0,
        archive_accepted=False,released_bytes=0,source=None,archive_result=None,guard=None,incoming_signal=None,
        scientific_result_writes=0,models=0,API_calls=0,hash_CRC_readback_checks=0,
        limits=dict(AS_GiB=.5,file_GiB=32,seconds=args.seconds,nice=19,free_GiB=10,minimum_available_GiB=1.25,
            metadata_case_bytes=META_BYTES,concurrency=1,retries=0,global_heavy_lease_used=False))
    def interrupt(signum,frame):
        if result['guard'] is None:result['incoming_signal']=signum
        raise InterruptedError('Archive interrupted by own signal '+str(signum))
    def inspect_resources(admission=False):
        observed=resources(args,result['source'],deadline,admission=admission)
        with (out/'resources.jsonl').open('a') as stream:stream.write(json.dumps(observed)+'\n')
        if observed['stop']:
            result['guard']=observed;raise RuntimeError('; '.join(observed['reasons']))
        return observed
    def monitor():
        while not stop.wait(.25):
            try:inspect_resources()
            except BaseException as exc:
                result['guard']=result['guard']or dict(error=type(exc).__name__)
                os.kill(os.getpid(),signal.SIGTERM)
                return
    try:
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM,signal.SIGXFSZ):
            handlers[sig]=signal.signal(sig,interrupt)
        old_timer=signal.setitimer(signal.ITIMER_REAL,args.seconds)
        with local_lock(root):
            with recorder.span('closed_BERT_trace_archive_admission'):
                result['source']=source_gate(args.audit)
                result['resource_admission']=inspect_resources(admission=True)
                snapshots=out/'source';snapshots.mkdir()
                for path in (Path(__file__),Path(original.__file__)):shutil.copyfile(path,snapshots/path.name)
                require(source_gate(args.audit)==result['source'], 'Original source metadata changed before compression')
            watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
            with recorder.span('original_R34_archive_one'):
                result['attempted']=1;write_json(out/'result.json',result)
                source=result['source'];row=archive_one(Path(source['source']['path']),Path(source['archive']),Path(source['temporary']))
                result['archive_result']=row
                require(row.get('status')in ('compressed','kept_uncompressed'), 'Original primitive must finish once')
                expected=(False,True,False)if row['status']=='compressed'else(True,False,False)
                state=files(source)
                require(tuple(state[k]['exists']for k in ('source','archive','temporary'))==expected, 'Original completed archive file state differs')
                result.update(completed=1,released_bytes=row['released_bytes'])
            result['resource_completion']=inspect_resources()
            require(all(metadata(Path(row['path']))==row for row in source['preserved']), 'Preserved executable artifact stat changed')
            result.update(status='ok',archive_accepted=True)
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc),archive_accepted=False)
    finally:
        # Cleanup itself must not be interrupted by a queued monitor signal or
        # the timer while join is waiting. This entry owns only these handlers.
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL,0)
        stop.set()
        if watcher is not None:watcher.join(timeout=3)
        result['monitor_stopped']=watcher is None or not watcher.is_alive()
        result['signal_cleanup_policy']='Restore prior handlers/timer only after terminal persistence and confirmed monitor closure'
        result['late_monitor_signals_ignored_until_exit']=not result['monitor_stopped']
        if not result['monitor_stopped']:
            result.update(status='error',archive_accepted=False,
                error=result['error']or'Archive monitor remained alive after bounded3-second join')
        try:
            if result['source'] is not None:
                try:result['resource_final']=inspect_resources()
                except BaseException as exc:result.update(status='error',archive_accepted=False,error=result['error']or type(exc).__name__)
                result['final_files']=files(result['source'])
                result['restore_argv']=['gzip','-dk','--',result['source']['archive']]
                result['restore_mtime_ns']=result['source']['source']['mtime_ns']
            if result['guard'] or result['incoming_signal'] or not result['monitor_stopped'] or time.monotonic()>=deadline:
                result.update(status='error',archive_accepted=False)
            result['timing']=summarize_events(read_events(recorder.path));write_json(out/'result.json',result)
            register(out,result)
            persist_case_completion(out,round_label=ROUND,started_ns=began,status=result['status'],
                timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        finally:
            if result['monitor_stopped']:
                for sig,handler in handlers.items():signal.signal(sig,handler)
                if old_timer and old_timer[0]>0:
                    signal.setitimer(signal.ITIMER_REAL,max(.001,old_timer[0]-(time.monotonic()-timer_started)),old_timer[1])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=SECONDS);args=parser.parse_args(argv)
    args.audit=args.audit.expanduser().absolute();args.out=args.out.expanduser().absolute()
    process_limits()
    return int(execute(args)['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
