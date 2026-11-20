"""Fresh original DaCapo trace and compile with explicit per-phase resources.

Only the two ordinary original entry functions receive private globals. Their
source checks, workers, compiler command, output inspection and completion stay
unchanged. No old trace, payload or elapsed time is used by this entry.
"""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from types import FunctionType, SimpleNamespace

from .. import paths
from . import dacapo_original_compile as original
from . import dacapo_compile_continuation as supervision


MODULE = 'pt_agent.experiments.dacapo_fresh_cold'
VARIANT = 'dacapo-original-fresh-cold-per-phase-as-v1'
GIB = 1024**3
TOTAL_SECONDS = 9720
QUEUE_SECONDS = 9960
POLICIES = dict(vgg16=dict(native_AS_gib=24, RSS_gib=14),
                alexnet=dict(native_AS_gib=32, RSS_gib=16))


def _ordinary(function, namespace):
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def resource_policy(args):
    """The existing main memory option remains the trace allowance."""
    model = getattr(args, 'model', None)
    if model not in POLICIES:
        raise ValueError('Fresh cold resources are limited to original VGG16 and AlexNet')
    expected = dict(memory_gib=12, file_gib=16, case_gib=32, planning_gib=24,
                    waterline=40, trace_seconds=1800, compile_seconds=7200)
    for name, wanted in expected.items():
        value = getattr(args, name, None)
        if (type(value) not in (int, float) or not math.isfinite(value) or value != wanted):
            raise ValueError('Fresh cold policy requires '+name+'='+str(wanted))
    if getattr(args, 'diagnostic_elide_final_ir', None) is not True:
        raise ValueError('Preserve the successful original diagnostic printer setting')
    raw_out = Path(args.out);out = raw_out.resolve();root = paths.RESULTS_DIR.resolve()
    if raw_out.is_symlink() or out.exists() or out == root or not out.is_relative_to(root):
        raise ValueError('A new isolated cold case under the results root is required')
    return dict(model=model, trace_AS_gib=12, **POLICIES[model],
        file_gib=16, case_gib=32, minimum_free_disk_gib=10, minimum_available_gib=1.25,
        trace_seconds=1800, native_seconds=7200, trace_Nix_seconds=1860,
        native_Nix_seconds=7260, trace_supervisor_seconds=1920,
        native_supervisor_seconds=7320, monitored_total_seconds=TOTAL_SECONDS,
        required_queue_outer_seconds=QUEUE_SECONDS, concurrency=1, automatic_retries=0)


def phase_worker(args, phase, work, *, command_file=None, policy):
    if phase not in ('trace', 'compile'):
        raise ValueError('Only one original trace and native compile are supported')
    private = SimpleNamespace(**vars(args))
    private.memory_gib = policy['trace_AS_gib' if phase == 'trace' else 'native_AS_gib']
    return original._nix_worker(private, phase, work, command_file=command_file)


def physical_observation(policy, sample, available, free):
    rss = None if sample is None else sample['subtree_plus_supervisor_rss_bytes']
    reasons = []
    if rss is not None and rss > policy['RSS_gib']*GIB:
        reasons.append('subtree including the supervisor exceeds its RSS allowance')
    if available < 1.25*GIB:
        reasons.append('available physical memory below1.25GiB')
    if free < 10*GIB:
        reasons.append('remaining disk below10GiB')
    return dict(stop=bool(reasons), reasons=reasons, subtree_plus_supervisor_rss_bytes=rss,
                available_ram_bytes=available, free_disk_bytes=free,
                sample_ts_ns=None if sample is None else sample.get('sample_ts_ns'))


def _phase_from_command(command):
    phases = [name for name in ('trace', 'compile') if '--worker-'+name in command]
    if len(phases) > 1:
        raise ValueError('Ambiguous original worker phase')
    return phases[0] if phases else None


def _one_argument(command, name):
    if command.count(name) != 1 or command.index(name)+1 >= len(command):
        raise ValueError('One explicit original worker argument required: '+name)
    return command[command.index(name)+1]


def guarded_process(command, *, policy, state, run, **options):
    phase = _phase_from_command(command)
    if phase is None:
        # The original git metadata command has its own existing small budget.
        return run(command, **options)
    memory = policy['trace_AS_gib' if phase == 'trace' else 'native_AS_gib']
    if (float(_one_argument(command, '--memory-gib')) != memory
            or float(_one_argument(command, '--file-gib')) != 16
            or _one_argument(command, '--model') != policy['model']
            or command.count('pt_agent.tools.nixrun') != 1
            or command[command.index('pt_agent.tools.nixrun')-1] != '-m'):
        raise ValueError('Original worker argv and selected resource phase differ')
    if command.count('pt_agent.experiments.dacapo_original_compile') != 1:
        raise ValueError('The unchanged original pure worker must remain the target')
    if options.get('mem_gib') != 12 or options.get('fsize_gib') != 16:
        raise ValueError('The original outer call must retain its trace12/file16 inputs')
    if any(row['phase'] == phase for row in state['phases']):
        raise ValueError('A fresh cold phase is attempted at most once')
    expected_seconds = policy['trace_supervisor_seconds' if phase == 'trace' else 'native_supervisor_seconds']
    if options.get('timeout') != expected_seconds:
        raise ValueError('Original outer phase time allowance differs')
    options = dict(options, mem_gib=memory,
                   timeout=min(expected_seconds, max(.01, state['deadline']-time.monotonic())))
    resources = Path(options['resources']);directory = resources.parent
    row = dict(phase=phase, AS_gib=memory, RSS_gib=policy['RSS_gib'], file_gib=16,
        maximum_seconds=expected_seconds, timeout_seconds=options['timeout'],
        worker_argv=list(command), status='admission', process_launched=False,
        observation=None, samples=0, guard_trigger=None, monitor_error=None)
    state['phases'].append(row)
    stop = threading.Event()
    def inspect():
        if time.monotonic() >= state['deadline']:
            state['total_timeout_observed'] = True
            raise TimeoutError('Fresh cold monitored9720s allowance exhausted')
        observed = physical_observation(policy, supervision.latest_resource(resources),
            supervision.available_ram(), shutil.disk_usage(directory).free)
        row['samples'] += 1;row['latest_physical_observation'] = observed
        with (directory/('physical-'+phase+'.jsonl')).open('a') as stream:
            stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(), **observed), allow_nan=False)+'\n')
            stream.flush()
        if observed['stop']:
            row['guard_trigger'] = observed;state['guard_trigger'] = observed
            original.write_json(directory/('physical-'+phase+'-guard.json'), observed)
            raise RuntimeError('; '.join(observed['reasons']))
    inspect()
    def monitor():
        while not stop.wait(.2):
            try:
                inspect()
            except BaseException as exc:
                row['monitor_error'] = type(exc).__name__+': '+str(exc)
                state['monitor_error'] = row['monitor_error']
                try:
                    original.write_json(directory/('physical-'+phase+'-stop.json'), row)
                finally:
                    try:
                        os.kill(os.getpid(), signal.SIGTERM)
                    except OSError as signal_error:
                        row['monitor_signal_error'] = str(signal_error)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        row.update(status='running', process_launched=True)
        observation = run(command, **options)
        row['observation'] = observation
        row['status'] = 'ok' if original._clean_process(observation) else 'error'
        if state.get('guard_trigger') or state.get('monitor_error') or state['total_timeout_observed']:
            raise RuntimeError('Fresh cold physical or total-time supervision stopped this phase')
        inspect()
        return observation
    except BaseException as exc:
        row.update(status='error', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        stop.set();watcher.join(timeout=3)
        row['watcher_stopped'] = not watcher.is_alive()


def _completed_admission(result, state, policy):
    if (result.get('source_checked') is not True or any(result.get(name) != 1 for name in
            ('trace_attempts', 'trace_completed', 'compile_attempts', 'compile_completed',
             'native_compile_attempts', 'native_compile_completed'))
            or result.get('export_contract_compatible') is not True):
        raise ValueError('One fresh source-admitted trace and native compilation must complete')
    if len(state['phases']) != 2 or [row['phase'] for row in state['phases']] != ['trace', 'compile']:
        raise ValueError('Both original phases must have one supervised process')
    for row, observation in zip(state['phases'], (result.get('trace_process'), result.get('compile_process'))):
        if (row.get('status') != 'ok' or row.get('watcher_stopped') is not True
                or not original._clean_process(observation or {})
                or row['observation'] != observation
                or (observation or {}).get('limits') != dict(memory_gib=row['AS_gib'],
                    file_gib=16, wall_seconds=row['timeout_seconds'])):
            raise ValueError('A supervised outer process or actual resource allowance differs')
    worker = result.get('compile_worker') or {};native = worker.get('process') or {}
    if (worker.get('status') != 'ok' or worker.get('compile_attempts') != 1
            or worker.get('compile_completed') != 1 or not original._clean_process(native)
            or native.get('limits') != dict(memory_gib=policy['native_AS_gib'],
                file_gib=16, wall_seconds=7200)):
        raise ValueError('The native worker must attest the same original bounded compile')
    return dict(passed=True, trace_runs=1, native_compile_runs=1,
                previous_trace_or_payload_used=False, prior_elapsed_times_added=False)


def _recover_worker_metadata(result, out, state):
    """Retain this case's observed worker counts even when a guard interrupted return."""
    for row in state['phases']:
        phase = row['phase']
        process_field = 'trace_process' if phase == 'trace' else 'compile_process'
        if process_field not in result and isinstance(row.get('observation'), dict):
            result[process_field] = row['observation']
        target = Path(out)/'work'/phase/'worker_result.json'
        if target.is_file():
            try:
                worker = original._read(target)
                if not isinstance(worker, dict):
                    raise ValueError('Current worker record must be an object')
                result['trace_worker' if phase == 'trace' else 'compile_worker'] = worker
            except BaseException as exc:
                row['worker_record_error'] = type(exc).__name__+': '+str(exc)
                worker = None
        else:
            worker = None
        if phase == 'compile':
            for field in ('compile_attempts', 'compile_completed'):
                value = (worker or {}).get(field)
                result['native_'+field] = value if type(value) is int and value in (0, 1) else None
        row['current_worker_record_available'] = worker is not None


def run(args):
    policy = resource_policy(args)
    state = dict(variant=VARIANT, policy=policy, phases=[],
        deadline=time.monotonic()+TOTAL_SECONDS, total_timeout_observed=False,
        guard_trigger=None, monitor_error=None, original_modules_modified=False,
        source_adapter_snapshot=None, previous_trace_or_payload_used=False,
        prior_elapsed_times_added=False, model_or_compiler_feedback_to_Agent=False)
    namespace = dict(vars(original))
    def nix_worker(value, phase, work, *, command_file=None):
        if state['source_adapter_snapshot'] is None:
            snapshot = Path(args.out)/'source'/Path(__file__).name
            shutil.copyfile(__file__, snapshot)
            state['source_adapter_snapshot'] = dict(path=str(snapshot), bytes=snapshot.stat().st_size)
        return phase_worker(value, phase, work, command_file=command_file, policy=policy)
    def process(command, **options):
        return guarded_process(command, policy=policy, state=state,
                               run=original.run_process, **options)
    def register(registry, result, out, revision):
        _recover_worker_metadata(result, out, state)
        if time.monotonic() >= state['deadline']:
            state['total_timeout_observed'] = True
        if state['guard_trigger'] is not None:
            result.update(status='resource_stopped', error='Fresh cold physical resource guard stopped this attempt')
        elif state['total_timeout_observed']:
            result.update(status='timeout', error='Fresh cold monitored total-time allowance exhausted')
        elif state['monitor_error'] is not None:
            result.update(status='supervision_error', error=state['monitor_error'])
        if result['status'] == 'ok':
            try:
                result['fresh_compile_admission'] = _completed_admission(result, state, policy)
            except BaseException as exc:
                result.update(status='error', error=type(exc).__name__+': '+str(exc))
        result.update(resource_adapter=state, cold_end_to_end_run=True,
            cold_trace_compile_completed=result['status'] == 'ok', original_trace_reused=False,
            prior_times_included=False,
            cold_clock_scope='Original continuous source preflight, fresh trace, native compile, export inspection, required result/registry persistence and completion; previous attempts and nested child clocks are separate')
        result['limits'].update(trace_AS_gib=12, native_AS_gib=policy['native_AS_gib'],
            subtree_plus_supervisor_RSS_gib=policy['RSS_gib'], minimum_available_RAM_gib=1.25,
            trace_seconds=1800, native_seconds=7200, monitored_total_seconds=TOTAL_SECONDS,
            required_queue_outer_seconds=QUEUE_SECONDS, concurrency=1,
            memory_scope='memory_gib is the original trace input; trace_AS_gib/native_AS_gib are the observed phase choices')
        original.write_json(Path(out)/'result.json', result)
        return original._register(registry, result, out, revision)
    namespace.update(_nix_worker=nix_worker, run_process=process, _register=register)
    private_run = _ordinary(original.run, namespace)
    with supervision.serial_lease(paths.RESULTS_DIR.resolve()):
        return private_run(args)


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if '--worker-trace' in values or '--worker-compile' in values:
        raise ValueError('Fresh cold wrapper exposes only the complete original outer entry')
    namespace = dict(vars(original), run=run)
    return _ordinary(original.main, namespace)(values)


if __name__ == '__main__':
    raise SystemExit(main())
