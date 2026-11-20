"""Retain accepted Qwen component traces as separately timed gzip files.

This observer owns no compiler slot. A distinct component may compile while its
single owned archive child processes two closed-component text files. Original
constants, RuntimePlans, bundles, scientific metadata and clocks stay in place.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json
from .r34_archive_text import archive_one, archive_state

MODULE = 'pt_agent.experiments.qwen_closed_component_trace_archive'
ROUND = 'QWEN-CLOSED-COMPONENT-TRACE-ARCHIVE'
MODE = 'accepted-component-two-MLIR-gzip-overlap-observer'
MODEL = 'Qwen2-0.5B'
GIB, RESERVE = 1024**3, 1024**2
WATCH_PID, WATCH_START = 199735, 4311840
DEADLINE = datetime(2026, 11, 17, 10, tzinfo=timezone.utc)
PREFIX = 'qwen-retained13-component-expanded-wall-'
CASE_NAMES = dict(source='qwen-retained13-component-expanded-wall-recovery-20261117',
    probe=PREFIX+'frozen-layer0-native-abi-probe-20261117',
    remaining=PREFIX+'frozen-remaining23-head-native-compile-20261117',
    kernel=PREFIX+'frozen-first-terminal-native-kernels-20261117',
    watch='deadline-qwen-retained-expanded-wall-serial-batch-20261117')
NAMES = ('layer0', *('layer'+str(i) for i in range(1, 24)), 'head', 'first', 'terminal')
TRACE_NAMES = ('candidate_trace.mlir', 'candidate_trace.segment.earth.mlir')
REGISTRY_FIELDS = {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'}
TERMINAL = {'ok', 'error', 'completed_with_observations'}


class Pending(ValueError):
    pass


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Object metadata required')
    return value


def durable(path, value):
    write_json(path, value)
    with Path(path).open() as stream:
        os.fsync(stream.fileno())


def process(pid):
    try:
        raw = (Path('/proc')/str(pid)/'stat').read_text()
    except FileNotFoundError:
        return None
    tail = raw[raw.rfind(')')+2:].split()
    return dict(pid=int(pid), state=tail[0], ppid=int(tail[1]), starttime=int(tail[19]))


def same_live(pid, birth):
    now = process(pid)
    return now is not None and now['starttime'] == birth and now['state'] not in ('Z', 'X')


def lease_owners(root):
    info = (Path(root)/'autonomous-heavy-experiments.lock').stat()
    owners = []
    with Path('/proc/locks').open() as stream:
        for line in stream:
            fields = line.split()
            if len(fields) < 6 or fields[1] != 'FLOCK' or fields[3] != 'WRITE':
                continue
            dev = fields[5].split(':')
            if (len(dev) == 3 and int(dev[0], 16) == os.major(info.st_dev)
                    and int(dev[1], 16) == os.minor(info.st_dev) and int(dev[2]) == info.st_ino):
                owners.append(int(fields[4]))
    return sorted(set(owners))


def argument(argv, name):
    if argv.count(name) != 1 or argv.index(name)+1 >= len(argv):
        raise ValueError('One explicit argument required: '+name)
    return argv[argv.index(name)+1]


def module_name(argv):
    return argument(argv, '-m')


def validate_args(args):
    root = Path(args.results_root).resolve()
    for name, suffix in CASE_NAMES.items():
        value = Path(getattr(args, name+'_case'))
        if value.is_symlink() or value != root/suffix or value.resolve() != value:
            raise ValueError('Only this explicit pinned component family is eligible: '+name)
    if args.watch_pid != WATCH_PID or args.watch_starttime != WATCH_START:
        raise ValueError('Require the original pinned producer identity')
    if (not math.isfinite(args.poll_seconds) or not 1 <= args.poll_seconds <= 60
            or type(args.seconds) is not int or not 1 <= args.seconds <= 36000):
        raise ValueError('Use finite observation intervals and wall allowance')
    return root


def consumer_proof(args):
    """Known source/plan-only consumers retain their complete original argv."""
    manifest = read(args.watch_case/'manifest.json')
    if datetime.fromisoformat(manifest['deadline_utc']) != DEADLINE:
        raise ValueError('Retain the original absolute deadline')
    expected = {
        'pt_agent.experiments.qwen_retained_component_replay': args.source_case,
        'pt_agent.experiments.qwen_layer_native_probe': args.probe_case,
        'pt_agent.experiments.qwen_generation_kernel_compile': args.kernel_case,
        'pt_agent.experiments.qwen_remaining_prefill_compile': args.remaining_case,
        'pt_agent.experiments.qwen_component_link_validation': None}
    seen = set(); rows = []
    for job in manifest['jobs']:
        argv = job['argv']
        if not isinstance(argv, list) or any(not isinstance(v, str) for v in argv):
            raise ValueError('Explicit original argv required')
        module = module_name(argv)
        if module not in expected or module in seen:
            raise ValueError('An unreviewed producer or trace consumer was declared')
        seen.add(module)
        if Path(argument(argv, '--out')) != Path(job['out']):
            raise ValueError('Producer output identity differs')
        if expected[module] is not None and Path(job['out']) != expected[module]:
            raise ValueError('Declared component case differs')
        if module != 'pt_agent.experiments.qwen_retained_component_replay':
            if Path(argument(argv, '--source-case')) != args.source_case:
                raise ValueError('Original C2 source differs')
        if any(any(name in cell for name in TRACE_NAMES) for cell in argv):
            raise ValueError('An explicit future raw/marked trace consumer remains')
        rows.append(dict(id=job['id'], module=module, out=job['out']))
    if seen != set(expected):
        raise ValueError('Require the complete original five-job family')
    prefixes = tuple(str(getattr(args, key+'_case')) for key in ('source', 'probe', 'remaining', 'kernel'))
    for path in args.consumer_manifest:
        for job in read(path).get('jobs', []):
            argv = job.get('argv', [])
            if (not isinstance(argv, list) or any(not isinstance(v, str) for v in argv)
                    or any(any(name in cell for name in TRACE_NAMES) for cell in argv)
                    or any(cell.startswith(prefixes) for cell in argv)):
                raise ValueError('Additional pending source/trace consumer requires a separate reviewed policy')
    return rows


def watch_proof(args):
    saved = read(args.watch_case/'supervisor.json')
    if saved.get('pid') != WATCH_PID or (saved.get('process') or {}).get('starttime') != WATCH_START:
        raise ValueError('Producer supervisor identity differs')
    live = same_live(WATCH_PID, WATCH_START)
    owners = lease_owners(args.results_root)
    if live:
        if owners != [WATCH_PID]:
            raise ValueError('Live original producer must retain its unique heavy lease')
        return dict(pid=WATCH_PID, starttime=WATCH_START, live=True, closed=False, lease_owners=owners)
    result_path = args.watch_case/'result.json'; complete_path = args.watch_case/'completion.json'
    if not result_path.is_file() or not complete_path.is_file():
        raise ValueError('Disappeared producer requires durable closure')
    result, complete = read(result_path), read(complete_path)
    if (result.get('status') not in TERMINAL or complete.get('status') != result['status']
            or complete.get('registry_persisted') is not True or complete.get('required_artifacts_persisted') is not True
            or result.get('signal') is not None or owners):
        raise ValueError('Closed producer metadata or current heavy owner differs')
    for job in result.get('jobs', []):
        observed = job.get('process') or {}
        if (job.get('status') not in ('ok', 'failed', 'dependency_skipped', 'deadline_skipped')
                or job.get('cleanup_unknown') or observed.get('remaining_pids') or observed.get('signal') is not None):
            raise ValueError('Producer has incomplete child cleanup')
    return dict(pid=WATCH_PID, starttime=WATCH_START, live=False, closed=True,
        status=result['status'], lease_owners=owners)


def component_case(args, name):
    if name not in NAMES:
        raise ValueError('Unknown original component')
    if name == 'layer0':
        return args.probe_case
    return (args.kernel_case if name in ('first', 'terminal') else args.remaining_case)/name


def original_acceptance(args, name):
    """Lazy metadata-only invocation of the original complete artifact gates."""
    from . import qwen_remaining_prefill_compile as remaining
    if name in ('first', 'terminal'):
        from . import qwen_generation_kernel_compile as kernel
        checked = kernel.admission(args.source_case)
        return kernel.accept_component(component_case(args, name), name, checked)
    checked = remaining.admission(args.source_case, args.probe_case)
    if name == 'layer0':
        return dict(status='ok', runtime_plan=checked['first_layer_runtime_plan'],
            native=checked['first_layer_native'], source_binding=checked['bindings']['layers'][0])
    return remaining.accept_component(component_case(args, name), name, checked)


def registry_proof(args, case, record):
    matches = []
    with (args.results_root/'registry.jsonl').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if Path(row.get('artifacts_dir', '')).resolve() == case:
                    matches.append(row)
    if len(matches) != 1:
        raise ValueError('One original component registry row required')
    row = matches[0]
    if (set(row) != REGISTRY_FIELDS or row['run_id'] != case.name
            or any(row[k] != record[k] for k in ('round', 'model', 'mode'))
            or row['metrics'].get('status') != 'ok'):
        raise ValueError('Original eight-field component registry identity differs')
    return dict(run_id=row['run_id'], round=row['round'], model=row['model'], mode=row['mode'],
        artifacts_dir=str(case), status='ok', field_count=8)


def birth_proof(case):
    births = set(); files = []
    for phase in ('trace', 'compile'):
        if not (case/(phase+'-resources.jsonl')).is_file():
            raise ValueError('Both original trace/native resource records are required')
    for path in case.rglob('*resources.jsonl'):
        if path.is_symlink():
            raise ValueError('Process resource metadata must remain regular')
        files.append(str(path))
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                for proc in row.get('processes', [])+[row.get('outer_process', {})]:
                    if type(proc.get('pid')) is int and type(proc.get('starttime')) is int:
                        births.add((proc['pid'], proc['starttime']))
    if not files or not births:
        raise ValueError('Saved trace/native birth metadata is required')
    for pid, start in births:
        if same_live(pid, start):
            raise Pending('A saved component process identity remains live')
    return dict(resources=files, identities=[dict(pid=p, starttime=s) for p, s in sorted(births)],
        same_birth_live=0)


def accepted_component(args, name):
    case = component_case(args, name)
    result_path, complete_path = case/'result.json', case/'completion.json'
    if not result_path.is_file() or not complete_path.is_file():
        raise Pending('Component has no complete durable result yet')
    record, complete = read(result_path), read(complete_path)
    if record.get('status') == 'running':
        raise Pending('Component is still running')
    if (record.get('status') != 'ok' or complete.get('status') != 'ok'
            or complete.get('registry_persisted') is not True or complete.get('required_artifacts_persisted') is not True):
        raise ValueError('Only actual accepted and durably closed components are eligible')
    accepted = original_acceptance(args, name)
    plan = Path(accepted['runtime_plan'])
    if (accepted.get('status') != 'ok' or plan.parent != case/'codegen/emit'
            or not plan.is_file() or plan.is_symlink()):
        raise ValueError('Accepted actual RuntimePlan path differs')
    registry = registry_proof(args, case, record)
    births = birth_proof(case)
    marker = record['phase_results']['trace']['trace_marker']
    if (Path(marker['raw_trace']) != case/'codegen/emit'/TRACE_NAMES[0]
            or Path(marker['marked_trace']) != case/'codegen/emit'/TRACE_NAMES[1]
            or marker.get('arithmetic_rewritten') is not False or marker.get('raw_trace_retained') is not True):
        raise ValueError('Actual original raw/marked trace provenance differs')
    value = dict(name=name, case=str(case), runtime_plan=str(plan), acceptance=accepted,
        registry=registry, births=births, trace_marker=marker,
        source_binding_scope='Original accepted source/plan/native metadata; trace stat metadata only, original trace content is not revalidated')
    return json.loads(json.dumps(value, allow_nan=False))


def file_proof(component, filename):
    if filename not in TRACE_NAMES:
        raise ValueError('Only the two original component MLIR text files are eligible')
    source = Path(component['case'])/'codegen/emit'/filename
    info = source.lstat()
    if source.is_symlink() or not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 64*GIB:
        raise ValueError('A regular original bounded trace is required')
    target = source.with_name(source.name+'.gz'); temporary = target.with_name(target.name+'.partial')
    if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError('Preserve an existing archive, source conflict or partial attempt')
    return dict(source=str(source), archive=str(target), temporary=str(temporary),
        size_bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def files_state(proof):
    result = {}
    for name in ('source', 'archive', 'temporary'):
        path = Path(proof[name]); exists = path.exists()
        result[name] = dict(path=str(path), exists=exists, bytes=path.stat().st_size if exists else None)
    return result


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise ValueError('Current host memory observation unavailable')


def disk_guard(args, name, proof, *, admission=False):
    free = shutil.disk_usage(args.results_root).free
    batch = artifact_bytes(args.kernel_case) if name in ('first', 'terminal') else (
        artifact_bytes(args.probe_case)+artifact_bytes(args.remaining_case))
    temporary_reserve = proof['size_bytes']+max(RESERVE, math.ceil(proof['size_bytes']/100))
    needed = temporary_reserve if admission else 0
    if (free < 10*GIB+needed or batch+needed+RESERVE > 128*GIB
            or artifact_bytes(args.out)+RESERVE > GIB or available_ram() < 1.25*GIB):
        raise Pending('Actual disk, temporary, batch, metadata or host RAM reserve is unavailable')
    return dict(free_bytes=free, batch_bytes=batch, reserved_temporary_bytes=needed,
        minimum_free_bytes=10*GIB, minimum_available_RAM_bytes=int(1.25*GIB))


def args_record(args):
    value = vars(args).copy()
    for key, item in list(value.items()):
        if isinstance(item, Path):
            value[key] = str(item)
        elif key == 'consumer_manifest':
            value[key] = [str(p) for p in item]
    return value


def restore_args(value):
    args = argparse.Namespace(**value)
    for name in ('results_root', 'out', *[n+'_case' for n in CASE_NAMES]):
        setattr(args, name, Path(getattr(args, name)))
    args.consumer_manifest = [Path(p) for p in args.consumer_manifest]
    return args


def apply_limits():
    for kind, cap in ((resource.RLIMIT_AS, GIB), (resource.RLIMIT_FSIZE, 64*GIB)):
        _, hard = resource.getrlimit(kind)
        bound = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
        resource.setrlimit(kind, (bound, bound))


def archive_worker(request_path, response_path):
    request = read(request_path); args = restore_args(request['args']); validate_args(args)
    stem = args.out/(request['component']+'-'+Path(request['file']['source']).name)
    if (Path(request_path) != stem.with_name(stem.name+'.request.json')
            or Path(response_path) != stem.with_name(stem.name+'.response.json')
            or args.out.parent != args.results_root or args.out.is_symlink()):
        raise ValueError('Owned worker request/response paths differ')
    apply_limits(); proof = request['file']; began = time.perf_counter_ns()
    result = dict(status='error', error=None, archive_result=None, final_files=None)
    old = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    def incoming(sig, frame):
        raise InterruptedError('Own archive child interrupted: '+str(sig))
    for sig in old:
        signal.signal(sig, incoming)
    try:
        if datetime.now(timezone.utc) >= DEADLINE:
            raise TimeoutError('Original absolute deadline reached')
        signal.setitimer(signal.ITIMER_REAL, min(1800., max(.001,
            (DEADLINE-datetime.now(timezone.utc)).total_seconds()-10)))
        consumer_proof(args); watch_proof(args)
        actual = accepted_component(args, request['component'])
        if actual != request['accepted'] or file_proof(actual, Path(proof['source']).name) != proof:
            raise ValueError('Closed source/stat metadata changed before compression')
        disk_guard(args, request['component'], proof, admission=True)
        result['archive_result'] = archive_one(Path(proof['source']), Path(proof['archive']), Path(proof['temporary']))
        result['status'] = 'ok'
    except BaseException as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        result['final_files'] = files_state(proof)
        result['elapsed_seconds'] = (time.perf_counter_ns()-began)/1e9
        durable(response_path, result)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    return 0 if result['status'] == 'ok' else 1


def stop_owned(child, birth):
    """The gzip worker is a leaf; no controller/compiler process is targeted."""
    if child is None or child.poll() is not None:
        return []
    sent = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if same_live(child.pid, birth):
            os.kill(child.pid, sig); sent.append(dict(pid=child.pid, starttime=birth, signal=sig))
        try:
            child.wait(timeout=3); break
        except subprocess.TimeoutExpired:
            continue
    return sent


def run_archive(args, component, filename, deadline):
    proof = file_proof(component, filename); disk_guard(args, component['name'], proof, admission=True)
    stem = args.out/(component['name']+'-'+filename)
    request, response = stem.with_name(stem.name+'.request.json'), stem.with_name(stem.name+'.response.json')
    if request.exists() or response.exists():
        raise ValueError('Preserve an earlier owned compression attempt')
    durable(request, dict(args=args_record(args), component=component['name'], accepted=component, file=proof))
    argv = [sys.executable, '-B', '-m', MODULE, '--archive-request', str(request), '--archive-response', str(response)]
    began = time.perf_counter_ns(); child = None; birth = None; sent = []; failure = None; interrupted = None
    try:
        with stem.with_name(stem.name+'.log').open('x') as log:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            child = subprocess.Popen(argv, cwd=paths.REPO/'DSL/DSL_agent/scripts', env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = process(child.pid)
            if identity is None:
                raise ValueError('Owned archive child identity was not observed')
            birth = identity['starttime']
            while child.poll() is None:
                if time.monotonic() >= deadline or (time.perf_counter_ns()-began)/1e9 >= 1800:
                    raise TimeoutError('Own compression allowance or fixed deadline reached')
                watch_proof(args); disk_guard(args, component['name'], proof)
                time.sleep(.25)
    except BaseException as exc:
        failure = type(exc).__name__+': '+str(exc)
        if isinstance(exc, (KeyboardInterrupt, InterruptedError)):
            interrupted = exc
    finally:
        cleanup_handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        for sig in cleanup_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if child is not None and birth is not None:
                sent = stop_owned(child, birth)
            result = dict(component=component['name'], file=proof, argv=argv, pid=child.pid if child else None,
                starttime=birth, exit_code=child.returncode if child else None, error=failure, signals_sent=sent,
                final_files=files_state(proof),elapsed_seconds=(time.perf_counter_ns()-began)/1e9,
                clock_scope='Separate archive I/O may overlap a different component native; original compiler clocks unchanged')
            if response.is_file():
                result['worker'] = read(response)
            result['status'] = 'ok' if failure is None and child is not None and child.returncode == 0 and (
                result.get('worker') or {}).get('status') == 'ok' else 'error'
            durable(stem.with_name(stem.name+'.observation.json'), result)
        except BaseException:
            if interrupted is not None:
                raise interrupted
            raise
        finally:
            for sig, handler in cleanup_handlers.items():
                signal.signal(sig, handler)
    if interrupted is not None:
        raise interrupted
    return result


def sync_archive_observations(out, result):
    """Recover real launched/closed facts even when an interrupt unwinds a call."""
    rows = [read(path) for path in sorted(out.glob('*.observation.json'))]
    result['archives'] = rows
    result['attempted_archives'] = sum(type(row.get('pid')) is int for row in rows)
    result['completed_archives'] = sum(type(row.get('pid')) is int and row.get('status') == 'ok'
        for row in rows)


def execute(args):
    root = validate_args(args); out = args.out
    if out.exists() or out.is_symlink() or out.parent != root or out.resolve() != out:
        raise ValueError('Use a fresh direct observer result case')
    until = min(time.monotonic()+args.seconds,
        time.monotonic()+max(0., (DEADLINE-datetime.now(timezone.utc)).total_seconds())-30)
    if until <= time.monotonic():
        raise ValueError('Leave30seconds for deadline closure')
    out.mkdir(); started = time.perf_counter_ns()
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
        absolute_deadline_utc=DEADLINE.isoformat(), components={}, archives=[],
        archive_admissions=0, attempted_archives=0, completed_archives=0, source_cases_unchanged=True,
        timing_scope='Independent observer and archive I/O; may overlap another component native',
        model_runs=0, Count_runs=0, compiler_runs=0, API_calls=0, extra_integrity_checks=0,
        archive_reopens=0, native_compiler_concurrency_changed=False)
    recorder = StageRecorder(out/'events.jsonl'); old = {}
    def incoming(sig, frame):
        raise InterruptedError('Observer signal '+str(sig))
    lock = (root/'.qwen-closed-component-trace-archive.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        apply_limits(); result['consumer_proof'] = consumer_proof(args)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            old[sig] = signal.signal(sig, incoming)
        signal.setitimer(signal.ITIMER_REAL, max(.001, until-time.monotonic()))
        while time.monotonic() < until:
            watch = watch_proof(args); result['producer'] = watch
            for name in NAMES:
                if time.monotonic() >= until:
                    break
                if result['components'].get(name, {}).get('terminal'):
                    continue
                progress = result['components'].setdefault(name, dict(status='pending', terminal=False,
                    files=[], file_progress={}))
                try:
                    accepted = accepted_component(args, name)
                    progress.update(status='archiving', case=accepted['case'])
                    for filename in TRACE_NAMES:
                        if filename in progress['file_progress']:
                            continue
                        if time.monotonic() >= until:
                            raise TimeoutError('Archive admission stopped for deadline closure')
                        result['archive_admissions'] += 1
                        with recorder.span('archive-'+name+'-'+filename):
                            row = run_archive(args, accepted, filename, until)
                        progress['files'].append(row); sync_archive_observations(out, result)
                        if row['status'] != 'ok':
                            raise ValueError('Owned archive attempt failed; actual states retained')
                        progress['file_progress'][filename] = row
                        durable(out/'result.json', result)
                    progress.update(status='archived', terminal=True)
                except Pending as exc:
                    progress.update(status='pending', terminal=False, reason=str(exc))
                except (InterruptedError, KeyboardInterrupt):
                    raise
                except BaseException as exc:
                    progress.update(status='rejected_or_archive_error', terminal=True,
                        error=type(exc).__name__+': '+str(exc))
                durable(out/'result.json', result)
            if watch['closed']:
                break
            time.sleep(min(args.poll_seconds, max(0., until-time.monotonic())))
        result['status'] = 'completed_with_observations'
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        sync_archive_observations(out, result)
        result['timing'] = summarize_events(read_events(recorder.path))
        result['artifact_bytes'] = artifact_bytes(out)
        durable(out/'result.json', result)
        row = dict(run_id=out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=None,
            metrics={k:result[k] for k in ('status', 'attempted_archives', 'completed_archives')},
            artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
        with (root/'registry.jsonl').open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
            if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
                raise ValueError('Preserve an existing observer registry identity')
            stream.seek(0, os.SEEK_END); stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
        persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
            timing=result['timing'], required_artifact_bytes=artifact_bytes(out))
        for sig, handler in old.items():
            signal.signal(sig, handler)
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--out', type=Path)
    for name, suffix in CASE_NAMES.items():
        value.add_argument('--'+name+'-case', type=Path, default=paths.RESULTS_DIR/suffix)
    value.add_argument('--watch-pid', type=int, default=WATCH_PID)
    value.add_argument('--watch-starttime', type=int, default=WATCH_START)
    value.add_argument('--poll-seconds', type=float, default=5.)
    value.add_argument('--seconds', type=int, default=36000)
    value.add_argument('--consumer-manifest', type=Path, action='append', default=[])
    value.add_argument('--archive-request', type=Path)
    value.add_argument('--archive-response', type=Path)
    value.add_argument('--run', action='store_true')
    value.set_defaults(results_root=paths.RESULTS_DIR)
    return value


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.archive_request is not None:
        if args.archive_response is None:
            p.error('Owned archive worker requires its response path')
        return archive_worker(args.archive_request, args.archive_response)
    if args.out is None or not args.run:
        p.error('An explicit fresh output and --run are required')
    result = execute(args)
    print(json.dumps({k:result[k] for k in ('status', 'attempted_archives', 'completed_archives')}))
    return 0 if result['status'] == 'completed_with_observations' else 1


if __name__ == '__main__':
    raise SystemExit(main())
