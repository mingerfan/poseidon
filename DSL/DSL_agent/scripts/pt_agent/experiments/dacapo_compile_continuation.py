"""Compile one closed original DaCapo trace in a new, bounded result case.

The saved compiler argv is checked against the original export contract. Only
its -o destination changes. No trace, model loading, key generation, or API
request is performed by this entry. The original failed case remains in place.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from types import SimpleNamespace

from .. import paths
from . import dacapo_original_compile as original
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

GIB = 1024**3
ROUND = 'DACAPO-ORIGINAL-COMPILE-CONTINUATION'
MODE = 'saved-original-trace-single-compile'


def read(path):
    return json.loads(Path(path).read_text())


def durable(path, value):
    write_json(path, value)
    with Path(path).open('r') as stream:
        os.fsync(stream.fileno())
    directory = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def serial_lease(root):
    with (Path(root) / '.dacapo-original-compile-continuation.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another original DaCapo compile continuation owns this slot') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_gate(source_case, output):
    """Read small metadata only; the saved trace and checkpoints are stat-only."""
    case = Path(source_case).resolve()
    old = read(case / 'result.json')
    closed = read(case / 'completion.json')
    trace_worker = old.get('trace_worker') or {}
    trace_process = old.get('trace_process') or {}
    compile_process = old.get('compile_process') or {}
    if (old.get('round') != original.ROUND or old.get('status') != 'error'
            or old.get('source_checked') is not True
            or old.get('trace_attempts') != 1 or old.get('trace_completed') != 1
            or old.get('compile_attempts') != 1 or old.get('compile_completed') != 0
            or trace_worker.get('status') != 'ok' or trace_worker.get('trace_completed') != 1
            or not original._clean_process(trace_process)
            or compile_process.get('remaining_pids') != []
            or closed.get('round') != original.ROUND or closed.get('status') != 'error'
            or closed.get('required_artifacts_persisted') is not True
            or closed.get('registry_persisted') is not True):
        raise ValueError('A closed original source case with a clean saved trace and failed compile is required')
    models = [key for key, row in original.MODEL_SOURCES.items() if row['result_model'] == old.get('model')]
    if len(models) != 1:
        raise ValueError('The saved case does not identify one official original CNN')
    model = models[0]
    relative = Path('work/trace/traced') / (original.MODEL_SOURCES[model]['entry'] + '.mlir')
    trace = case / relative
    metadata = trace_worker.get('mlir') or {}
    if (Path(metadata.get('path', '')).resolve() != trace
            or not trace.is_file() or trace.is_symlink()
            or trace.stat().st_size != metadata.get('size_bytes') or trace.stat().st_size <= 0
            or trace.stat().st_mtime_ns != metadata.get('mtime_ns')):
        raise ValueError('The saved original trace path or file metadata differs')
    profile = read(case / 'compiler-config.json')
    original.validate_compiler_profile(profile)
    if profile != read(old['sources']['compiler_profile']['path']):
        raise ValueError('The saved compiler configuration differs from the official source profile')
    imported_path = Path(old['sources']['imported_spec']['path'])
    contract, _ = original.make_export_contract(read(imported_path), profile, waterline=40)
    if read(case / 'export-contract.json') != contract:
        raise ValueError('The saved original N17/scale51 export contract differs')
    command = read(case / 'compile-argv.json')
    if not isinstance(command, list) or len(command) < 2 or not all(isinstance(v, str) for v in command):
        raise ValueError('The saved compiler argv is malformed')
    compiler = Path(command[0]).resolve()
    if not compiler.is_file():
        raise FileNotFoundError(str(compiler))
    diagnostic = old.get('diagnostic_elide_final_ir', False)
    if type(diagnostic) is not bool:
        raise ValueError('The original diagnostic output choice must be explicit boolean metadata')
    expected = original.compiler_argv(command, binary=compiler, source=trace,
        profile=case / 'compiler-config.json', output=case / 'work/compile/lowered.mlir',
        waterline=40, diagnostic_elide_final_ir=diagnostic)
    if command != expected:
        raise ValueError('Only the unchanged original optimized export argv may continue')
    trace_argv = old.get('trace_argv')
    if (not isinstance(trace_argv, list) or not all(isinstance(v, str) for v in trace_argv)
            or trace_argv.count('--python') != 1):
        raise ValueError('The original trusted Nix interpreter argv is missing')
    intake_python = trace_argv[trace_argv.index('--python') + 1]
    changed = list(command)
    changed[changed.index('-o') + 1] = str(Path(output).resolve())
    return dict(model=model, result_model=old['model'], original_case=str(case),
        original_status=old['status'], original_error=old.get('error'),
        original_completion=closed, original_compile_process=compile_process,
        original_native_compile=(old.get('compile_worker') or {}).get('process'),
        trace_metadata=original._metadata(trace), trace_content_read=False,
        checkpoint_content_read=False, original_sources=old.get('sources'),
        compiler=str(compiler), intake_python=intake_python,
        compiler_profile=str(case / 'compiler-config.json'), contract=contract,
        diagnostic_elide_final_ir=diagnostic, original_argv=command, argv=changed,
        argv_change=dict(index=command.index('-o') + 1, old=command[command.index('-o') + 1],
                         new=changed[command.index('-o') + 1]),
        original_case_modified=False, original_trace_reused=True,
        source_gate_scope='Saved result/closure, original compiler/profile/contract and stat metadata; no digest or trace/checkpoint payload inspection')


def latest_resource(path):
    """Bounded tail of the outer R10 sampler, including all Nix descendants."""
    if path is None:
        return None
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 262144))
            lines = stream.read().decode(errors='replace').splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get('processes'), list):
            subtree = sum(p.get('vmrss_kib', 0) * 1024 for p in row['processes'])
            outer = (row.get('outer_process') or {}).get('vmrss_kib', 0) * 1024
            return dict(sample_ts_ns=row.get('ts_ns'), subtree_rss_bytes=subtree,
                supervisor_rss_bytes=outer, subtree_plus_supervisor_rss_bytes=subtree + outer,
                processes=row['processes'])
    return None


def available_ram():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def status(result, fallback='error'):
    if result.get('guard_trigger') is not None:
        return 'resource_stopped'
    if result.get('wall_timeout_observed'):
        return 'timeout'
    if result.get('incoming_signal') is not None:
        return 'interrupted'
    if result.get('monitor_error') is not None:
        return 'supervision_error'
    return fallback


def receive_signal(result, signum):
    if signum == signal.SIGALRM:
        result['wall_timeout_observed'] = True
        result['timeout_signal'] = signum
    elif result.get('guard_trigger') is not None or result.get('monitor_error') is not None:
        result['guard_stop_signal'] = signum
    elif result.get('wall_timeout_observed'):
        result['timeout_stop_signal'] = signum
    else:
        result['incoming_signal'] = signum
    raise InterruptedError('Compile continuation supervisor received signal ' + str(signum))


def check_resources(args, result, resource_file, deadline):
    if time.monotonic() >= deadline:
        result['wall_timeout_observed'] = True
        raise TimeoutError('Compile continuation total wall allowance exhausted')
    sample = latest_resource(resource_file)
    observed = dict(ts_ns=time.perf_counter_ns(), latest_process_sample=sample,
        available_ram_bytes=available_ram(), disk_free_bytes=shutil.disk_usage(args.out).free,
        artifact_bytes=artifact_bytes(args.out))
    reasons = []
    rss_gib = getattr(args, 'rss_gib', 14)
    if sample is not None and sample['subtree_plus_supervisor_rss_bytes'] > rss_gib * GIB:
        reasons.append(f'subtree including Nix parents and supervisor exceeds{rss_gib:g}GiB RSS')
    if observed['available_ram_bytes'] < 1.25 * GIB:
        reasons.append('available physical memory below1.25GiB')
    if observed['disk_free_bytes'] < args.minimum_free_gib * GIB:
        reasons.append('remaining disk below the explicit reserve')
    if observed['artifact_bytes'] > args.case_gib * GIB:
        reasons.append('new case artifacts exceed the explicit allowance')
    if reasons:
        result['guard_trigger'] = dict(observed, reasons=reasons)
        # The original cause is stored before this write, so even a full disk
        # cannot erase its classification or prevent the monitor's finally kill.
        durable(args.out / 'resource-guard.json', result['guard_trigger'])
        raise RuntimeError('; '.join(reasons))
    return observed


def register(registry, result, out):
    row = dict(run_id=out.name, round=ROUND, model=result.get('model'), mode=MODE, git_rev=None,
        metrics={key: result.get(key) for key in ('status', 'compile_attempts', 'compile_completed',
            'native_compile_attempts', 'native_compile_completed', 'bootstrap_count',
            'export_contract_compatible', 'actual_ckks_latency_seconds', 'security_acceptance')},
        artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise RuntimeError('This continuation run id already exists in the registry')
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    out = Path(args.out).resolve()
    source = Path(args.source_case).resolve()
    root = paths.RESULTS_DIR.resolve()
    if (out.exists() or out.is_symlink() or out == root or not out.is_relative_to(root)
            or source == root or not source.is_relative_to(root)
            or out.is_relative_to(source) or source.is_relative_to(out)):
        raise ValueError('A new result case separate from the original case is required')
    args.out = out
    started = time.perf_counter_ns()
    deadline = time.monotonic() + args.compile_seconds + 240
    out.mkdir(parents=True)
    emit = out / 'emit'
    emit.mkdir()
    result = dict(round=ROUND, mode=MODE, model=None, status='running', error=None,
        compile_attempts=0, compile_completed=0, native_compile_attempts=None, native_compile_completed=None,
        trace_attempts=0, trace_completed=0, original_trace_reused=False, processes=[],
        incoming_signal=None, timeout_signal=None, timeout_stop_signal=None, guard_stop_signal=None,
        guard_trigger=None, monitor_error=None, wall_timeout_observed=False,
        api_calls=0, checkpoint_reads=0, model_forwards=0, report_samples=0, keygen_runs=0,
        encryptions=0, decryptions=0, real_ckks_attempts=0, downloads=0, installations=0,
        predicted_latency_seconds=None, actual_ckks_latency_seconds=None,
        accuracy_acceptance=None, security_acceptance=None, native_execution_ready=False,
        original_case_modified=False, cold_end_to_end_run=False,
        limits=dict(memory_gib=args.memory_gib, memory_scope='virtual address space per child',
            subtree_plus_supervisor_RSS_gib=getattr(args, 'rss_gib', 14), minimum_available_RAM_gib=1.25,
            file_gib=args.file_gib, case_gib=args.case_gib,
            minimum_free_disk_gib=args.minimum_free_gib, compile_seconds=args.compile_seconds,
            wrapper_seconds=args.compile_seconds + 120, total_seconds=args.compile_seconds + 240,
            concurrency=1))
    recorder = StageRecorder(out / 'events.jsonl')
    stop = threading.Event()
    active = [None]
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    prior_timer = signal.getitimer(signal.ITIMER_REAL)
    def interrupted(signum, frame):
        receive_signal(result, signum)
    def monitor():
        while not stop.wait(.2):
            try:
                row = check_resources(args, result, active[0], deadline)
                with (out / 'physical-resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
            except BaseException as exc:
                if result.get('guard_trigger') is None and not result['wall_timeout_observed']:
                    result['monitor_error'] = type(exc).__name__ + ': ' + str(exc)
                try:
                    durable(out / 'monitor-stop.json', dict(status=status(result),
                        guard_trigger=result['guard_trigger'], monitor_error=result['monitor_error'],
                        error=type(exc).__name__ + ': ' + str(exc)))
                except BaseException as log_error:
                    result['monitor_record_error'] = type(log_error).__name__ + ': ' + str(log_error)
                finally:
                    try:
                        os.kill(os.getpid(), signal.SIGTERM)
                    except OSError as kill_error:
                        result['monitor_signal_error'] = str(kill_error)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    for sig in old:
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, args.compile_seconds + 240)
    try:
        with serial_lease(root):
            with recorder.span('saved-original-source-gate'):
                provenance = source_gate(source, emit / 'lowered.mlir')
                result.update(model=provenance['result_model'], source=provenance,
                    original_trace_reused=True,
                    diagnostic_elide_final_ir=provenance['diagnostic_elide_final_ir'],
                    compiler_output_variant=original.compiler_output_variant(SimpleNamespace(
                        diagnostic_elide_final_ir=provenance['diagnostic_elide_final_ir'])))
                durable(out / 'source-gate.json', provenance)
                durable(out / 'compile-argv.json', provenance['argv'])
                shutil.copy2(__file__, out / 'driver.py')
            check_resources(args, result, None, deadline)
            watcher.start()
            worker_args = SimpleNamespace(model=provenance['model'], compile_seconds=args.compile_seconds,
                compiler=Path(provenance['compiler']), memory_gib=args.memory_gib, file_gib=args.file_gib,
                intake_python=Path(provenance['intake_python']),
                diagnostic_elide_final_ir=provenance['diagnostic_elide_final_ir'])
            command = original._nix_worker(worker_args, 'compile', emit, command_file=out / 'compile-argv.json')
            process = dict(stage='original.compile.optimized.continuation', argv=command,
                observation=None, supervision_exception=None)
            result['processes'].append(process)
            result['compile_attempts'] = 1
            durable(out / 'result.json', result)
            resource_file = out / 'compile-resources.jsonl'
            with recorder.span('original.compile.optimized.continuation', argv=command):
                active[0] = resource_file
                try:
                    observation = run_process(command, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                        log=out / 'compile.log', resources=resource_file,
                        timeout=args.compile_seconds + 120, mem_gib=args.memory_gib, fsize_gib=args.file_gib,
                        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                 MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
                    process['observation'] = observation
                    result['wall_timeout_observed'] |= bool(observation.get('timed_out'))
                    if observation.get('signal') is not None:
                        field = 'guard_stop_signal' if (result.get('guard_trigger') is not None
                            or result.get('monitor_error') is not None) else (
                                'timeout_stop_signal' if result['wall_timeout_observed'] else 'incoming_signal')
                        result[field] = observation['signal']
                except BaseException as exc:
                    process['supervision_exception'] = type(exc).__name__ + ': ' + str(exc)
                    process['last_resource_sample'] = latest_resource(resource_file)
                    raise
                finally:
                    active[0] = None
                worker_path = emit / 'worker_result.json'
                worker = read(worker_path) if worker_path.is_file() else None
                result['compile_worker'] = worker
                result['native_compile_attempts'] = worker.get('compile_attempts') if worker is not None else None
                result['native_compile_completed'] = worker.get('compile_completed') if worker is not None else None
                native = (worker or {}).get('process') or {}
                result['wall_timeout_observed'] |= bool(native.get('timed_out'))
                if not original._clean_process(observation) or not worker or worker.get('status') != 'ok':
                    raise RuntimeError('Original native compile continuation failed; saved trace retained')
                if (worker.get('compile_attempts') != 1 or worker.get('compile_completed') != 1
                        or worker.get('file_gib') != args.file_gib or not original._clean_process(native)):
                    raise RuntimeError('The native worker did not attest one clean bounded compile')
                native_limits = native.get('limits') or {}
                if native_limits != dict(memory_gib=args.memory_gib, file_gib=args.file_gib,
                                         wall_seconds=args.compile_seconds):
                    raise RuntimeError('The native compiler observation differs from the declared resource allowances')
                result['compile_completed'] = 1
            with recorder.span('original.export.inspect'):
                plans = list(emit.glob('lowered.*.runtime-plan.json'))
                if len(plans) != 1:
                    raise RuntimeError('One complete exported RuntimePlan is required')
                plan = read(plans[0])
                result['plan'] = original._metadata(plans[0])
                result['compiler_output'] = original.compile_artifact_metadata(emit, plans[0], plan,
                    elided=provenance['diagnostic_elide_final_ir'])
                observed = original.observe_plan(plan, provenance['contract'])
                result.update(observed)
                durable(out / 'plan-observation.json', observed)
                if not observed['export_contract_compatible']:
                    raise RuntimeError('Exported Boot metadata differs from the unchanged original contract')
            check_resources(args, result, resource_file, deadline)
            result['status'] = 'ok'
    except BaseException as exc:
        result.update(status=status(result), error=type(exc).__name__ + ': ' + str(exc)[:4000])
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=3)
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if any(result.get(key) is not None for key in ('guard_trigger', 'incoming_signal', 'monitor_error')) or result['wall_timeout_observed']:
                result['status'] = status(result)
            worker_path = emit / 'worker_result.json'
            if 'compile_worker' not in result and worker_path.is_file():
                result['compile_worker'] = read(worker_path)
                result['native_compile_attempts'] = result['compile_worker'].get('compile_attempts')
                result['native_compile_completed'] = result['compile_worker'].get('compile_completed')
            timing = summarize_events(read_events(out / 'events.jsonl'), time.perf_counter_ns())
            result['timing'] = timing
            result['recorded_elapsed_seconds'] = (time.perf_counter_ns() - started) / 1e9
            result['timing_scope'] = 'Continuation entry through required persistence/registry; original trace and failed attempts retain separate clocks'
            durable(out / 'result.json', result)
            register(args.registry, result, out)
            persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=timing, required_artifact_bytes=artifact_bytes(out))
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
            if prior_timer[0]:
                signal.setitimer(signal.ITIMER_REAL, *prior_timer)
    return 0 if result['status'] == 'ok' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=paths.RESULTS_DIR / 'registry.jsonl')
    parser.add_argument('--memory-gib', type=float, default=24)
    parser.add_argument('--rss-gib', type=float, default=14,
        help='Finite tree-plus-supervisor RSS allowance14..16GiB; independent available-RAM guard remains1.25GiB')
    parser.add_argument('--file-gib', type=float, default=16)
    parser.add_argument('--case-gib', type=float, default=8)
    parser.add_argument('--minimum-free-gib', type=float, default=10)
    parser.add_argument('--compile-seconds', type=int, default=7200)
    args = parser.parse_args(argv)
    values = (args.memory_gib, args.rss_gib, args.file_gib, args.case_gib, args.minimum_free_gib)
    if (not all(math.isfinite(v) for v in values) or not 0 < args.memory_gib <= 32
            or not 14 <= args.rss_gib <= 16
            or not 0 < args.file_gib <= 16 or not 0 < args.case_gib <= 32
            or not 4 <= args.minimum_free_gib <= 10 or not 0 < args.compile_seconds <= 7200):
        parser.error('Use finite AS<=32GiB, RSS14..16GiB, file<=16GiB, new case<=32GiB, reserve4..10GiB and compile<=7200seconds')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
