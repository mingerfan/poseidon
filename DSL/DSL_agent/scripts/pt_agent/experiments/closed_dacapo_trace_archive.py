"""Archive one closed successful Alex fresh trace under its serial owner.

The original R34 stream primitive is reused after this entry's explicit scope
gate. Scientific results, executable plans/bundles, weights and samples stay
untouched. No archive readback or integrity comparison is performed.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import sys
import threading
import time

from .. import paths
from . import dacapo_compile_continuation as serial
from . import orbit_static_parse_cpu_overlap as lease_tools
from . import r34_archive_text as original
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events

MODULE = 'pt_agent.experiments.closed_dacapo_trace_archive'
ROUND = 'CLOSED-DACAPO-TRACE-ARCHIVE'
MODE = 'closed-success-Alex-fresh-trace-reversible-gzip-once'
MODEL = 'dacapo-original-alexnet-silu-avgpool'
CASE = 'dacapo-original-alexnet-fresh-cold-resource-adapter-20261117'
GIB, SECONDS, SOURCE_CASE_BYTES, META_CASE_BYTES = 1024**3, 1800, 32*1024**3, 4*1024**3
RESERVE = 1024**2
archive_one = original.archive_one
read, save = serial.read, serial.durable


def lease_observation(expected=None):
    table = lease_tools._scan(); own = os.getpid(); row = table.get(own) or {}
    if row.get('state') not in ('R', 'S', 'D'):
        raise ValueError('The archive process must have a current live process identity')
    ancestors, seen, pid = [], {own}, row.get('ppid')
    while pid in table and pid not in seen:
        seen.add(pid); ancestors.append(pid); pid = table[pid].get('ppid')
    owners = lease_tools._lease_owners(paths.RESULTS_DIR/'autonomous-heavy-experiments.lock')
    if len(owners) != 1 or owners[0] not in ancestors:
        raise ValueError('Require one original strict ancestor holding the global heavy lease')
    owner = table[owners[0]]
    identity = dict(pid=owners[0], starttime=owner.get('starttime'))
    if (owner.get('state') not in ('R', 'S', 'D') or type(identity['starttime']) is not int
            or identity['starttime'] <= 0 or expected is not None and identity != expected):
        raise ValueError('The original live ancestor heavy-lease identity changed')
    family = lease_tools._family(table, own)
    foreign = [p for p, value in table.items() if p not in family
        and value.get('state') not in ('Z', 'X')
        and lease_tools._process_name(p) in lease_tools.COMPILERS]
    if foreign:
        raise ValueError('Another native compiler is active outside this archive case')
    return dict(owner=identity, own_pid=own, foreign_compilers=foreign)


def queue_consumers(args, source, owner):
    """Inspect all declared pending jobs, including the actual owner's queue."""
    if not args.queue_case:
        raise ValueError('Declare the owner queue and all known later queue cases')
    records, owner_found = [], False
    for case in args.queue_case:
        case = Path(case)
        if case.is_symlink() or case.resolve() != case or case.parent != paths.RESULTS_DIR.resolve():
            raise ValueError('Queue cases must be direct original result directories')
        manifest = read(case/'manifest.json')
        supervisor = read(case/'supervisor.json') if (case/'supervisor.json').is_file() else {}
        proc = supervisor.get('process') or {}
        if supervisor.get('pid') == owner['pid'] and proc.get('starttime') == owner['starttime']:
            owner_found = True
        state = (read(case/'result.json') if (case/'result.json').is_file() else
            read(case/'progress.json') if (case/'progress.json').is_file() else {'jobs':[]})
        done = {row['id']: row for row in state.get('jobs', [])}
        ids = [row['id'] for row in manifest['jobs']]
        if len(set(ids)) != len(ids):
            raise ValueError('A declared queue contains duplicate job identities')
        for job in manifest['jobs']:
            argv = job.get('argv', [])
            if not isinstance(argv, list) or any(not isinstance(x, str) for x in argv):
                raise ValueError('Only explicit string job arguments can be checked')
            row = done.get(job['id']) or {}
            closed = (row.get('status') in ('ok', 'failed', 'dependency_skipped', 'deadline_skipped')
                and not row.get('cleanup_unknown') and not (row.get('process') or {}).get('remaining_pids'))
            own_job = (argv.count(MODULE) == 1 and '-m' in argv
                and argv.index('-m')+1 < len(argv) and argv[argv.index('-m')+1] == MODULE
                and Path(job.get('out', '')).absolute() == args.out
                and argv.count('--out') == 1 and argv.index('--out')+1 < len(argv)
                and argv[argv.index('--out')+1] == str(args.out)
                and argv.count('--source-case') == 1 and argv.index('--source-case')+1 < len(argv)
                and argv[argv.index('--source-case')+1] == source['source_case'])
            reference = (any(source['source_case'] in value for value in argv)
                or source['source_case'] in job.get('requires_case_statuses', {})
                or source['source_case'] in job.get('requires_case_fields', {}))
            if reference and not closed and not own_job:
                raise ValueError('A pending or running declared job still references the Alex trace source: '+job['id'])
            records.append(dict(queue=str(case), job=job['id'], closed=closed,
                source_reference=reference, archive_self=own_job))
    if not owner_found:
        raise ValueError('The actual heavy-lease owner queue must be included in the declared queues')
    return records


def source_gate(source_case):
    root = paths.RESULTS_DIR.resolve(); case = Path(source_case)
    if case.is_symlink() or case != root/CASE or case.resolve() != case:
        raise ValueError('Only the exact original closed Alex fresh case is eligible')
    result, completion = read(case/'result.json'), read(case/'completion.json')
    worker = result.get('trace_worker') or {}
    native = result.get('compile_worker') or {}
    if (result.get('round') != 'DACAPO-ORIGINAL' or result.get('mode') != 'original-source-native-profile-compile'
            or result.get('model') != MODEL or result.get('status') != 'ok'
            or result.get('source_checked') is not True
            or result.get('original_source_variant') is not True or result.get('r29_cheb63_variant') is not False
            or any(result.get(k) != 1 for k in ('trace_attempts', 'trace_completed', 'compile_attempts',
                'compile_completed', 'native_compile_attempts', 'native_compile_completed'))
            or result.get('cold_trace_compile_completed') is not True
            or result.get('original_trace_reused') is not False or result.get('prior_times_included') is not False
            or worker.get('status') != 'ok' or worker.get('source_unmodified') is not True
            or native.get('status') != 'ok' or completion.get('status') != 'ok'
            or completion.get('round') != 'DACAPO-ORIGINAL'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True):
        raise ValueError('Require the complete successful original fresh trace/native source and closure')
    observations = [result.get('trace_process') or {}, result.get('compile_process') or {}, native.get('process') or {}]
    if any(not serial.original._clean_process(row) for row in observations):
        raise ValueError('All original source phases must have clean process closure')
    table = lease_tools._scan()
    recorded = {int(pid) for row in observations for pid in row.get('peak_rss_kib_by_pid', {})}
    if any(pid in table and table[pid].get('state') not in ('Z', 'X') for pid in recorded):
        raise ValueError('An original source process is still active')
    trace = case/'work/trace/traced/AlexNet.mlir'; metadata = worker.get('mlir') or {}
    if (Path(metadata.get('path', '')) != trace or trace.resolve() != trace
            or trace.is_symlink() or not trace.is_file() or not stat.S_ISREG(trace.stat().st_mode)):
        raise ValueError('Only this closed raw Alex trace is eligible; executable/data paths are excluded')
    info = trace.stat()
    if not 0 < info.st_size <= 16*GIB:
        raise ValueError('The single trace must fit its finite per-file allowance')
    if (type(metadata.get('size_bytes')) is not int or type(metadata.get('mtime_ns')) is not int
            or metadata['size_bytes'] != info.st_size or metadata['mtime_ns'] != info.st_mtime_ns):
        raise ValueError('Saved and current trace stat size/mtime provenance must match')
    target = trace.with_name(trace.name+'.gz'); temporary = target.with_name(target.name+'.partial')
    if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError('Preserve every existing archive or partial attempt')
    plan = case/'work/compile/lowered._hecate_AlexNet.runtime-plan.json'
    bundle = case/'work/compile/lowered._hecate_AlexNet.bundle'
    if ((result.get('plan') or {}).get('path') != str(plan) or not plan.is_file()
            or not bundle.is_dir() or plan.is_symlink() or bundle.is_symlink()
            or plan.resolve() != plan or bundle.resolve() != bundle):
        raise ValueError('Retain this successful source RuntimePlan and bundle independently')
    source_bytes = completion.get('case_artifact_bytes')
    if type(source_bytes) is not int or not info.st_size <= source_bytes < SOURCE_CASE_BYTES:
        raise ValueError('A bounded original-case resource inventory is required')
    return dict(source_case=str(case), source=str(trace), archive=str(target), temporary=str(temporary),
        raw_bytes=info.st_size, raw_mtime_ns=info.st_mtime_ns, raw_atime_ns=info.st_atime_ns,
        source_case_bytes=source_bytes, other_source_case_bytes=source_bytes-info.st_size,
        RuntimePlan=str(plan), bundle=str(bundle), raw_content_read_for_admission=False,
        source_binding_scope='saved and current stat metadata only; original content not revalidated')


def file_states(source):
    result = {}
    for key in ('source', 'archive', 'temporary'):
        path = Path(source[key])
        try:
            value = path.stat(); result[key] = dict(path=str(path), exists=True, bytes=value.st_size)
        except FileNotFoundError:
            result[key] = dict(path=str(path), exists=False, bytes=None)
    return result


def resources(args, source, deadline, owner, *, admission=False):
    lease = lease_observation(owner); queues = queue_consumers(args, source, owner)
    state = file_states(source); free = shutil.disk_usage(paths.RESULTS_DIR).free
    source_bytes = source['other_source_case_bytes']+sum(v['bytes'] or 0 for v in state.values())
    available = serial.available_ram(); reasons = []
    if time.monotonic() >= deadline: reasons.append('Archive wall allowance exhausted')
    if free < 10*GIB: reasons.append('10GiB free-disk reserve exhausted')
    if available < 1.25*GIB: reasons.append('Available physical memory below1.25GiB')
    if source_bytes > SOURCE_CASE_BYTES: reasons.append('Original case plus temporary archive exceeds32GiB')
    if artifact_bytes(args.out)+RESERVE > META_CASE_BYTES: reasons.append('Archive metadata case allowance exhausted')
    temporary_reservation = min(16*GIB, SOURCE_CASE_BYTES-source['source_case_bytes'])
    if admission and free < 10*GIB+temporary_reservation:
        reasons.append('Temporary archive allocation and10GiB reserve are unavailable')
    return dict(stop=bool(reasons), reasons=reasons, files=state, free_bytes=free,
        available_RAM_bytes=available, source_case_accounted_bytes=source_bytes,
        temporary_reservation_bytes=temporary_reservation, lease=lease, declared_jobs_checked=len(queues))


def process_limits():
    for name, allowance in ((resource.RLIMIT_AS, GIB//2), (resource.RLIMIT_FSIZE, 16*GIB)):
        _, hard = resource.getrlimit(name)
        bound = allowance if hard == resource.RLIM_INFINITY else min(allowance, hard)
        resource.setrlimit(name, (bound, bound))


def register(args, result):
    row = dict(run_id=args.out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=None,
        metrics={k:result[k] for k in ('status', 'attempted', 'completed', 'archive_accepted', 'released_bytes')},
        artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
        for line in stream:
            if args.out.name in line and json.loads(line).get('run_id') == args.out.name:
                raise ValueError('Preserve the existing archive registry identity')
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno()); fcntl.flock(stream, fcntl.LOCK_UN)


def execute(args):
    root = paths.RESULTS_DIR.resolve()
    if (args.out.parent != root or args.out.exists() or args.out.is_symlink()
            or not args.out.name.startswith('closed-dacapo-alex-trace-archive-')
            or not 0 < args.seconds <= SECONDS):
        raise ValueError('Use a new direct archive-audit case and at most1800seconds')
    args.out.mkdir(); began = time.perf_counter_ns(); deadline = time.monotonic()+args.seconds
    recorder = StageRecorder(args.out/'events.jsonl'); stop = threading.Event(); watcher = None
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
        attempted=0, completed=0, archive_accepted=False, released_bytes=0,
        source=None, archive_result=None, final_files=None, guard=None, incoming_signal=None,
        timeout_observed=False, scientific_result_writes=0, RuntimePlan_bundle_weight_sample_writes=0,
        models=0, API_calls=0, CRC_or_hash_or_readback_checks=0, extra_integrity_checks=0,
        limits=dict(AS_GiB=.5, file_GiB=16, seconds=args.seconds, source_case_GiB=32,
            metadata_case_GiB=4, free_GiB=10, minimum_available_GiB=1.25, concurrency=1, retries=0))
    handlers = {}; old_timer = None; timer_started = time.monotonic()
    def interrupt(signum, frame):
        if signum == signal.SIGALRM: result['timeout_observed'] = True
        elif result['guard'] is None: result['incoming_signal'] = signum
        raise InterruptedError('Archive interrupted by signal'+str(signum))
    def inspect(admission=False):
        observed = resources(args, result['source'], deadline, result['lease']['owner'], admission=admission)
        with (args.out/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(), **observed), allow_nan=False)+'\n')
        if observed['stop']:
            result['guard'] = observed; raise RuntimeError('; '.join(observed['reasons']))
        return observed
    def monitor():
        while not stop.wait(.25):
            try: inspect()
            except BaseException as exc:
                result['guard'] = result['guard'] or dict(error=type(exc).__name__+': '+str(exc))
                try: save(args.out/'guard.json', result['guard'])
                finally: os.kill(os.getpid(), signal.SIGTERM)
                return
    try:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGXFSZ):
            handlers[signum] = signal.signal(signum, interrupt)
        old_timer = signal.setitimer(signal.ITIMER_REAL, args.seconds)
        with serial.serial_lease(root):
            with recorder.span('closed_trace_archive_admission'):
                result['lease'] = lease_observation()
                result['source'] = source_gate(args.source_case)
                result['queue_admission'] = queue_consumers(args, result['source'], result['lease']['owner'])
                result['resource_admission'] = inspect(admission=True)
                snapshots = args.out/'source'; snapshots.mkdir()
                for path in (Path(__file__), Path(original.__file__)):
                    shutil.copyfile(path, snapshots/path.name)
                if source_gate(args.source_case) != result['source']:
                    raise ValueError('Source metadata changed before the single archive attempt')
            watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
            with recorder.span('original_archive_one'):
                result['attempted'] = 1; save(args.out/'result.json', result)
                source = result['source']
                row = archive_one(Path(source['source']), Path(source['archive']), Path(source['temporary']))
                result['archive_result'] = row
                if row.get('status') not in ('compressed', 'kept_uncompressed'):
                    raise ValueError('The original archive primitive did not return a completed state')
                result.update(completed=1, released_bytes=row['released_bytes'])
                states = file_states(source)
                expected = ((False, True, False) if row['status'] == 'compressed' else (True, False, False))
                if tuple(states[key]['exists'] for key in ('source', 'archive', 'temporary')) != expected:
                    raise ValueError('The completed archive primitive has an inconsistent file state')
            result['resource_completion'] = inspect()
            result.update(status='ok', archive_accepted=True)
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc), archive_accepted=False)
    finally:
        stop.set()
        if watcher is not None: watcher.join(timeout=3)
        result['monitor_stopped'] = watcher is None or not watcher.is_alive()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum in handlers: signal.signal(signum, signal.SIG_IGN)
        try:
            if result['source'] is not None:
                try:
                    result['resource_final'] = inspect()
                except BaseException as exc:
                    result.update(status='error', archive_accepted=False,
                        error=result['error'] or type(exc).__name__+': '+str(exc))
                result['final_files'] = file_states(result['source'])
                result['restore_argv'] = ['gzip', '-dk', '--', result['source']['archive']]
                result['restore_mtime_ns'] = result['source']['raw_mtime_ns']
                result['restore_metadata_scope'] = 'Future stat-bound consumers may explicitly restore this recorded mtime after decompression; no restore is performed here'
            if result['guard'] or result['incoming_signal'] or result['timeout_observed'] or not result['monitor_stopped'] or time.monotonic() >= deadline:
                result.update(status='error', archive_accepted=False)
            result['timing'] = summarize_events(read_events(recorder.path))
            save(args.out/'result.json', result)
            register(args, result)
            persist_case_completion(args.out, round_label=ROUND, started_ns=began,
                status=result['status'], timing=result['timing'], required_artifact_bytes=artifact_bytes(args.out))
        finally:
            for signum, handler in handlers.items(): signal.signal(signum, handler)
            if old_timer and old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, max(.001, old_timer[0]-(time.monotonic()-timer_started)), old_timer[1])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case', type=Path, required=True)
    parser.add_argument('--queue-case', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=SECONDS)
    args = parser.parse_args(argv)
    args.source_case = args.source_case.expanduser().absolute()
    args.queue_case = [x.expanduser().absolute() for x in args.queue_case]
    args.out = args.out.expanduser().absolute()
    process_limits()
    return 0 if execute(args)['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
