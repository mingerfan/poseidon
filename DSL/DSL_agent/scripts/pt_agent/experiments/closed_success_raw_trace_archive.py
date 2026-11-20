"""Reversibly archive explicit raw traces of closed successful scientific cases.

This direct resource job takes the original global heavy-work lease itself.
Only raw text is compressed, once per source, using the unchanged R34 primitive.
Every other source artifact, scientific clock and registry record is retained.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time

from .. import paths
from . import a_group_future_closed_archive as metadata
from .autonomous_serial_batch import acquire_lease
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, _process, read_events, summarize_events
from .r34_archive_text import archive_one, archive_state

GIB = 1024**3
ROUND = 'CLOSED-SUCCESS-RAW-TRACE-ARCHIVE'
MODE = 'resource-only-explicit-closed-raw-gzip'
require, read, durable, snapshot = metadata.require, metadata.read, metadata.durable, metadata.snapshot


def producer_births(case):
    births = set()
    for path in case.rglob('*resources.jsonl'):
        with path.open() as stream:
            for line in stream:
                if not line.strip(): continue
                value = json.loads(line); items = list(value.get('processes') or [])
                if value.get('outer_process'): items.append(value['outer_process'])
                for item in items:
                    pid,birth = item.get('pid'),item.get('starttime')
                    require(type(pid) is int and pid > 0 and type(birth) is int and birth > 0,
                        'Every saved producer identity must have a positive PID/starttime')
                    births.add((pid,birth))
    require(births and not any(metadata.same_birth(pid,birth) for pid,birth in births),
        'All positive saved original producer births must be closed')
    return len(births)


def source_gate(manifest):
    require(manifest.get('schema') == 'closed-success-raw-trace-archive-v1'
        and manifest.get('no_future_raw_consumers') is True,
        'Explicit inactive raw source authority required')
    require(manifest.get('host_pid_namespace') == os.readlink('/proc/self/ns/pid'),
        'Use the approved host process namespace for producer closure')
    rows = manifest.get('files')
    require(isinstance(rows, list) and 1 <= len(rows) <= 8, 'A finite explicit source list required')
    root = paths.RESULTS_DIR.resolve()
    selected, seen, cases = [], set(), set()
    for row in rows:
        case, source = Path(row['case']), Path(row['source'])
        require(case.parent == root and case.resolve() == case and not case.is_symlink(),
            'Scientific source must be an original direct results child')
        require(source.is_relative_to(case) and source.name == 'candidate_trace.mlir'
            and str(source) not in seen and str(case) not in cases,
            'One unique explicit raw compiler trace per source case required')
        seen.add(str(source)); cases.add(str(case))
        closed, result = metadata.closed(case, case.name)
        require(result.get('status') == 'ok' and result.get('error') is None
            and result.get('monitor_error') is None and result.get('guard_error') is None
            and result.get('guard_trigger') is None and result.get('termination') is None,
            'Only successfully closed original scientific producers are eligible')
        require(result.get('round') == row['expected_round'] and result.get('model') == row['expected_model']
            and result.get('compiled_bootstraps') == row['compiled_bootstraps'],
            'Original scientific identity differs from the explicit manifest')
        attempted, completed = result.get('attempted') or {}, result.get('completed') or {}
        require(attempted == completed and attempted.get('compile') == 1
            and attempted.get('trace') == 1, 'Original trace and compile must complete exactly once')
        processes = result.get('processes') or []
        require(processes, 'Saved actual scientific process observations required')
        for process in processes:
            observed = process.get('observation') or {}
            require(observed.get('exit_code') == 0 and observed.get('timed_out') is False
                and observed.get('signal') is None and observed.get('signals_sent') == []
                and observed.get('remaining_pids') == [] and not observed.get('cleanup_unknown'),
                'Original producer cleanup must be explicitly successful')
        births = producer_births(case)
        require(births > 0, 'Positive saved original producer birth evidence required')
        raw_stat = snapshot(source, archive_source=True)
        require(raw_stat == row['stat'], 'Original raw path/stat changed')
        target = source.with_name(source.name+'.gz')
        temporary = target.with_name(target.name+'.partial')
        require(not os.path.lexists(target) and not os.path.lexists(temporary),
            'Retain existing archives and partial destinations')
        plans = list(source.parent.glob('*.runtime-plan.json'))
        require(plans, 'Retained actual compiled RuntimePlan required')
        for plan in plans:
            bundle = plan.with_name(plan.name.removesuffix('.runtime-plan.json')+'.bundle')
            require(plan.is_file() and not plan.is_symlink() and bundle.is_dir()
                and not bundle.is_symlink() and (bundle/'manifest.json').is_file()
                and (bundle/'data').is_dir(), 'Actual compiled RuntimePlan and bundle must remain present')
        preserved = [snapshot(path) for path in case.rglob('*') if path != source]
        selected.append(dict(case=str(case), source=str(source), target=str(target), temporary=str(temporary),
            stat=raw_stat, preserved=preserved, producer_births_closed=births, scientific_status=closed['status'],
            original_compiled_bootstraps=result['compiled_bootstraps'], original_completion=str(case/'completion.json')))
    return selected


@contextmanager
def original_heavy_lease(out, result):
    lock = paths.RESULTS_DIR/'autonomous-heavy-experiments.lock'
    with lock.open('a') as stream:
        acquire_lease(stream, 0)
        own = _process(os.getpid())
        require(own and type(own.get('starttime')) is int and own['starttime'] > 0,
            'Actual resource-owner birth must be available')
        result['heavy_lease'] = dict(path=str(lock), owner_pid=os.getpid(), owner_starttime=own['starttime'],
            acquired=True, released=False, original_shared_global_lock=True)
        durable(out/'supervisor.json', dict(pid=os.getpid(), process=own, lease=result['heavy_lease']))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
            result['heavy_lease']['released'] = True


def register(out, result):
    row = dict(run_id=out.name, round=ROUND, model='explicit-closed-raw-traces', mode=MODE,
        git_rev=None, metrics={key:result[key] for key in ('status','completed_files','released_bytes')},
        artifacts_dir=str(out), ts=result['ended_utc'])
    with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
        require(not any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()),
            'Preserve every original archive identity')
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def run(args):
    require(args.run and 1 <= args.seconds <= 1200, 'Explicit finite resource-only execution required')
    out = args.out.absolute()
    require(out.parent == paths.RESULTS_DIR.resolve() and not out.exists(), 'Fresh direct resource-case output required')
    out.mkdir(); started = time.perf_counter_ns(); rec = StageRecorder(out/'events.jsonl')
    result = dict(round=ROUND, mode=MODE, status='running', error=None, files=[], completed_files=0,
        released_bytes=0, in_progress=None, scientific_source_writes=0, RuntimePlan_bundle_weight_writes=0,
        model_runs=0, API_calls=0, native_compiles=0, CKKS_runs=0, integrity_or_readback_checks=0,
        limits=dict(AS_gib=1,file_gib=16,wall_seconds=args.seconds,minimum_free_gib=10,concurrency=1),
        clock_scope='Independent metadata admission and serial resource compression; original scientific clocks retained')
    signals = (signal.SIGTERM,signal.SIGINT,signal.SIGALRM,signal.SIGXFSZ)
    old = {number:signal.getsignal(number) for number in signals}
    old_timer = signal.getitimer(signal.ITIMER_REAL); timer_start = time.monotonic()
    def interrupt(number, frame):
        result['incoming_signal'] = number
        raise InterruptedError('Own resource archive interrupted by signal'+str(number))
    try:
        for number in signals: signal.signal(number,interrupt)
        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        limit = GIB if hard == resource.RLIM_INFINITY else min(GIB,hard)
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
        hard_file = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
        file_limit = 16*GIB if hard_file == resource.RLIM_INFINITY else min(16*GIB,hard_file)
        resource.setrlimit(resource.RLIMIT_FSIZE,(file_limit,file_limit))
        signal.setitimer(signal.ITIMER_REAL,args.seconds)
        with original_heavy_lease(out,result):
            with rec.span('closed_raw_resource_archive'):
                manifest = read(args.manifest); durable(out/'manifest.json',manifest)
                with rec.span('closed_metadata_admission'):
                    selected = source_gate(manifest)
                    result['admitted_sources'] = [{key:value for key,value in row.items() if key != 'preserved'} for row in selected]
                for row in selected:
                    source,target,temporary = map(Path,(row['source'],row['target'],row['temporary']))
                    require(snapshot(source,archive_source=True) == row['stat'],'Raw stat changed after closure admission')
                    require(shutil.disk_usage(source.parent).free >= 10*GIB+row['stat']['bytes']*101//100+1024**2,
                        'Reserve10GiB plus worst temporary gzip storage before each source')
                    require(all(snapshot(Path(item['path'])) == item for item in row['preserved']),
                        'A retained scientific artifact changed before compression')
                    result['in_progress'] = archive_state(source,target,temporary); durable(out/'progress.json',result)
                    with rec.span('gzip_one_closed_raw',source=str(source)):
                        observed = archive_one(source,target,temporary)
                    require(all(snapshot(Path(item['path'])) == item for item in row['preserved']),
                        'A retained scientific artifact changed during resource compression')
                    require(observed['status'] in ('compressed','kept_uncompressed'),'Original R34 archive must finish once')
                    result['files'].append(dict(observed,producer_births_closed=row['producer_births_closed'],
                        retained_non_target_artifacts=len(row['preserved']),original_compiled_bootstraps=row['original_compiled_bootstraps']))
                    result.update(completed_files=len(result['files']),released_bytes=sum(item['released_bytes'] for item in result['files']),in_progress=None)
                    durable(out/'progress.json',result)
                result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
        if result['in_progress']: result['interrupted_archive_state'] = archive_state(source,target,temporary)
    finally:
        for number in signals: signal.signal(number,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL,0)
        result['ended_utc'] = datetime.now(timezone.utc).isoformat()
        result['timing'] = summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns())
        result['resource_elapsed_before_persistence_s'] = (time.perf_counter_ns()-started)/1e9
        result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        durable(out/'result.json',result); register(out,result)
        marker = persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        durable(out/'completion.json',marker)
        for number,handler in old.items(): signal.signal(number,handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL,max(.001,old_timer[0]-(time.monotonic()-timer_start)),old_timer[1])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seconds',type=int,default=1200)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({key:result[key] for key in ('status','error','completed_files','released_bytes')}))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__': raise SystemExit(main())
