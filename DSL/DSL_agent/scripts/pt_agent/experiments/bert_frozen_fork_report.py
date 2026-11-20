"""One isolated native-validation report of an accepted frozen BERT fork.

The original C2 and compile cases stay read-only.  A fresh metadata shell owns
the original report worker's access marker and numerical outputs.  Only the
system outer reads the global registry; the unchanged recorded worker receives
all paths explicitly inside Nix.
"""
import argparse
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
from . import bert_budget_fork_compile as compiled
from . import bert_sst2 as original
from . import bert_serial_continuation as serial
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, _scan, read_events, run_process, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes
from .r31_lattigo_runtime_plan import physical_memory_observation

ROUND = 'BERT-FROZEN-FORK-REPORT'
MODEL = original.MODEL
MODE = 'isolated-frozen-budget-fork-native-validation872-report-once'
MODULE = 'pt_agent.experiments.bert_frozen_fork_report'
GIB, SECONDS = 1024**3, 1800
CASE_BYTES, RESERVE = 4*GIB, 1024**2
ZERO = ('new_agent_steps', 'new_tune_commands', 'new_confirmation_commands', 'api_calls',
        'compiler_runs', 'keygen_runs', 'ckks_runs', 'native_test_reads', 'integrity_checks',
        'downloads', 'installations')
read, save = serial.read, serial.durable_json


def registry_row(case, result):
    rows = []
    with (paths.RESULTS_DIR/'registry.jsonl').open() as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == case.name: rows.append(row)
    if (len(rows) != 1 or set(rows[0]) != serial.REGISTRY_FIELDS
            or any(rows[0].get(k) != result.get(k) for k in ('round', 'model', 'mode', 'metrics'))
            or Path(rows[0].get('artifacts_dir', '')).resolve() != case):
        raise ValueError('Require one matching eight-field closed-source registry row')
    return rows[0]


def closed_case(case):
    result, completion = read(case/'result.json'), read(case/'completion.json')
    if (result.get('status') != 'ok' or completion.get('status') != 'ok'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True):
        raise ValueError('Accepted C2 and compile sources must have durable clean closure')
    return result, registry_row(case, result)


def admission(args):
    """Read metadata only; reuse the original complete C2/checkpoint/math gates."""
    root = paths.RESULTS_DIR.resolve()
    if any(p.parent != root for p in (args.source_case, args.compile_case, args.out)):
        raise ValueError('Use separate direct result cases for the frozen source, compile and report')
    if len({args.source_case, args.compile_case, args.out}) != 3:
        raise ValueError('Preserve original source cases with a separate report output')
    master, master_row = closed_case(args.compile_case)
    if (master.get('round') != compiled.ROUND or master.get('mode') != compiled.MODE
            or master.get('model') != MODEL or master.get('math_passed') is not True
            or master.get('compile_completed') is not True or master.get('compile_accepted') is not True
            or master.get('attempted') != dict(math=1, compile=1)
            or master.get('completed') != dict(math=1, compile=1)
            or master.get('report_deferred') is not True
            or any(master.get(k) != 0 for k in compiled.base.ZERO)):
        raise ValueError('Require the accepted same-source compile-first result')
    proof = args.compile_case/'outer-admission.json'
    if not proof.is_file(): raise ValueError('Retain the actual original compile admission proof')
    lineage = (master.get('source') or {}).get('fork_provenance') or {}
    binding = SimpleNamespace(source_case=args.source_case, out=args.compile_case,
        original_source_case=Path(lineage['original_source_case']).resolve(),
        continuation_case=Path(lineage['continuation_case']).resolve(),
        boundary_snapshot=Path(lineage['boundary_snapshot']).resolve(),
        checkpoint_audit=Path((master.get('source') or {})['checkpoint_audit']).resolve(),
        worker_phase=None, delegate_compile=False, admission_proof=None)
    # Existing proof must already exist: metadata_source can only re-read it.
    scope = compiled.namespace(binding)
    source = scope['metadata_source'](args.source_case, binding.checkpoint_audit)
    if source != master['source']:
        raise ValueError('Frozen source differs from the actual completed compiler source')
    math_result = Path(master['math_result']).resolve()
    if not math_result.is_relative_to(args.compile_case):
        raise ValueError('Use the actual compiler master calibration-math proof')
    expected = scope['compile_source'](args.source_case, binding.checkpoint_audit, math_result)
    native_case = Path(master['compile_case']).resolve()
    if native_case.parent != root or native_case in (args.source_case, args.compile_case, args.out):
        raise ValueError('Actual native compiler case must remain distinct')
    native, native_row = closed_case(native_case)
    if (native.get('round') != compiled.ROUND or native.get('mode') != compiled.MODE
            or native.get('model') != MODEL or native.get('source') != expected
            or native.get('attempted') != dict(emit=1, trace=1, compile=1)
            or native.get('completed') != dict(emit=1, trace=1, compile=1)
            or any(native.get(k) != 0 for k in compiled.base.original.ZERO_FIELDS)
            or type(master.get('compiled_bootstraps')) is not int or master['compiled_bootstraps'] < 0
            or native.get('compiled_bootstraps') != master['compiled_bootstraps']):
        raise ValueError('All three actual compiler phases must match this frozen source and math')
    for result in (master, native):
        for row in result.get('processes', []):
            compiled.fork._closed_process(row.get('observation'), successful=True)
    if ([row.get('phase') for row in master.get('processes', [])] != ['math', 'compile']
            or [row.get('phase') for row in native.get('processes', [])] != ['emit', 'trace', 'compile']):
        raise ValueError('Retain all actual math/delegate and Emit/trace/native process observations')
    raw_model, raw_inputs = Path(source['original_model']).resolve(), Path(source['sample_source']['source']).resolve()
    for name, value in (('intake_python', args.intake_python), ('data_python', args.data_python),
                        ('metadata_template', args.metadata_template)):
        if not (value.is_file() or value.is_symlink()): raise ValueError('Required existing path is absent: '+name)
    if (args.metadata_template != Path(lineage['metadata_template']).resolve()
            or args.data_python.absolute() != Path(source['sample_source']['reader_python']).absolute()):
        raise ValueError('Keep the original metadata template and saved independent SST-2 reader')
    original_command = read(args.source_case/'phases/001-confirm/command.json')
    for flag, value in (('--python', args.intake_python), ('--data-python', args.data_python),
                         ('--ledger', args.ledger), ('--allowance', args.allowance)):
        if value.resolve() != compiled._one(original_command, flag):
            raise ValueError('Retain the original confirmation command path: '+flag)
    from .r29_local_compile_matrix import compiler_argv
    compiler_argv(read(args.metadata_template), 'pars', binary='/compiler', source='/input', profile='/profile', output='/lowered')
    return dict(source_case=str(args.source_case), compile_case=str(args.compile_case), native_case=str(native_case),
        compiled_bootstraps=master['compiled_bootstraps'], model=str(raw_model), inputs=str(raw_inputs),
        original_model_info=source['fork_provenance']['original_model_info'], model_info=source['model_info'],
        program=source['program'], frozen=source['frozen'], task_policy=source['task_policy'],
        split_version=source['split_version'], report_samples=872,
        source=source, compiler_registry=master_row, native_registry=native_row,
        original_report_access_present=False, report_feedback_permitted=False, algorithm_changed=False)


def prepare_shell(args, source):
    model, samples, session = args.out/'model', args.out/'model/samples', args.out/'session'
    samples.mkdir(parents=True); session.mkdir()
    for name in ('model_info.json', 'calibration.json', 'samples/source.json'):
        shutil.copyfile(args.source_case/'model'/name, model/name)
    for name in ('graph', 'samples/calibration.npz', 'samples/heldout.npz', 'samples/confirm.npz'):
        target = (args.source_case/'model'/name).resolve(strict=True)
        (model/name).symlink_to(target, target_is_directory=name == 'graph')
    for name in ('SESSION.json', 'AGENT.json', 'program.ptdsl'):
        shutil.copyfile(args.source_case/'session'/name, session/name)
    marker = read(args.source_case/'session/FROZEN.json')
    marker['program'] = str(session/'program.ptdsl')
    save(session/'FROZEN.json', marker)
    return shell_bindings(args)


def shell_bindings(args):
    for name in ('model_info.json', 'calibration.json', 'samples/source.json'):
        if read(args.out/'model'/name) != read(args.source_case/'model'/name):
            raise ValueError('Copied report metadata changed: '+name)
    aliases = {}
    for name in ('graph', 'samples/calibration.npz', 'samples/heldout.npz', 'samples/confirm.npz'):
        local, source = args.out/'model'/name, args.source_case/'model'/name
        if not local.is_symlink() or local.resolve() != source.resolve():
            raise ValueError('Report read-only reference changed: '+name)
        aliases[name] = str(local.resolve())
    for name in ('SESSION.json', 'AGENT.json'):
        if read(args.out/'session'/name) != read(args.source_case/'session'/name):
            raise ValueError('Copied report frozen session changed')
    marker = read(args.out/'session/FROZEN.json'); original_marker = read(args.source_case/'session/FROZEN.json')
    original_marker['program'] = str(args.out/'session/program.ptdsl')
    if (marker != original_marker or (args.out/'session/program.ptdsl').read_text() !=
            (args.source_case/'session/program.ptdsl').read_text()):
        raise ValueError('Only the copied frozen marker program path may change')
    serial.frozen_state(args.out)
    if (args.source_case/'session/REPORT_ACCESS.json').exists() or (args.source_case/'model/samples/report.npz').exists():
        raise ValueError('Original C2 report capability must stay unused')
    if (args.out/'model/samples/report.npz').is_symlink():
        raise ValueError('New report caches must remain within the independent case')
    return dict(aliases=aliases, alias_access='Read-only in the unchanged report worker',
        frozen_program_path_rebased_only=True, original_case_writes=0)


def command(args, source, remaining):
    argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(max(1, int(remaining))),
        '--python', str(args.intake_python), '--', '-B', '-m', 'pt_agent.experiments.bert_sst2_recorded',
        '--delegate-worker', '--metadata-template', str(args.metadata_template), '--phase', 'report',
        '--work-dir', str(args.out/'worker'), '--model-dir', str(args.out/'model'),
        '--session-dir', str(args.out/'session'), '--model', source['model'], '--inputs', source['inputs']]
    for name in ('ledger', 'allowance', 'data_python'):
        argv += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    return argv


def check_report(args, worker):
    original.check_worker('report', worker)
    rows = worker.get('frozen_math') or []
    if (len(rows) != 3 or [row.get('sample') for row in rows] != [0, 1, 2]
            or any(len(row.get('checks', [])) != 3 or any(check.get('passed') is not True
                or check.get('criterion') != compiled.base.original.CRITERION
                or any(type(check.get(k)) not in (int, float) or not math.isfinite(check[k]) or check[k] < 0
                    for k in ('max_absolute_error', 'max_tolerance_ratio'))
                or check['max_tolerance_ratio'] > 1 for check in row['checks']) for row in rows)
            or worker.get('task_policy') != {'acc_drop_pp': .5}
            or type(worker.get('task_policy_passed')) is not bool or worker.get('llm_calls') != 0):
        raise ValueError('Require all original strict mathematics and the unchanged report task policy')
    access = read(args.out/'session/REPORT_ACCESS.json')
    if (access.get('complete') is not True or access.get('report_samples') != 872
            or access.get('program') != str(args.out/'session/program.ptdsl')
            or access.get('result') != str(args.out/'worker/worker_result.json')
            or not (args.out/'worker/report-outputs.npz').is_file()):
        raise ValueError('The single new report capability and output must be completely persisted')


def resources(args):
    table = _scan(); owned = {os.getpid()}
    while True:
        expanded = owned | {pid for pid, row in table.items() if row.get('ppid') in owned}
        if expanded == owned: break
        owned = expanded
    sample = dict(subtree_rss_bytes=sum(table.get(pid, {}).get('vmrss_kib', 0)*1024 for pid in owned))
    observation = physical_memory_observation(sample, _available_ram_bytes(), rss_gib=4, minimum_available_gib=1.25)
    observation.update(case_bytes=artifact_bytes(args.out), disk_free_bytes=shutil.disk_usage(args.out).free)
    if observation['case_bytes']+RESERVE > CASE_BYTES: observation['reasons'].append('Case allowance with completion reserve exhausted')
    if observation['disk_free_bytes'] < 10*GIB: observation['reasons'].append('10GiB disk reserve exhausted')
    observation['stop'] = bool(observation['reasons'])
    return observation


def claim(args):
    directory = paths.RESULTS_DIR/'bert-frozen-report-claims'; directory.mkdir(exist_ok=True)
    target = directory/(args.source_case.name+'.json')
    with target.open('x') as stream:
        json.dump(dict(source_case=str(args.source_case), report_case=str(args.out), retries=0, mode=MODE), stream)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return str(target)


def execute(args):
    if args.out.exists(): raise FileExistsError('Preserve the prior isolated report case')
    args.out.mkdir(parents=True); started = time.perf_counter_ns(); deadline = time.monotonic()+SECONDS
    recorder = StageRecorder(args.out/'events.jsonl'); stop = threading.Event(); watcher = None
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
        source_case=str(args.source_case), compile_case=str(args.compile_case), admission_passed=False,
        attempted_report=0, completed_report=0, report_accepted=False, task_policy_passed=None,
        report_samples=0, source=None, process=None, worker=None, guard_trigger=None, incoming_signal=None,
        wall_timeout_observed=False, accuracy_acceptance=None, security_acceptance=None, CKKS_latency=None,
        guard_stop_signal=None,
        resource_admission=None, resource_completion=None,
        resource_limits=dict(AS_gib=4, tree_RSS_gib=4, host_available_gib=1.25, file_gib=8,
            case_gib=4, free_gib=10, seconds=SECONDS, completion_reserve_bytes=RESERVE), **dict.fromkeys(ZERO, 0))
    handlers = {}
    def interrupt(signum, frame):
        if result['guard_trigger'] is None: result['incoming_signal'] = signum
        raise InterruptedError('Isolated report outer received signal'+str(signum))
    def monitor():
        while not stop.wait(.25):
            try:
                observation = resources(args)
                if time.monotonic() >= deadline:
                    observation['stop'] = True; observation['reasons'].append('Report wall allowance exhausted')
                    result['wall_timeout_observed'] = True
                if not observation['stop']: continue
                result['guard_trigger'] = observation
                result['guard_stop_signal'] = signal.SIGTERM
                try:
                    with (args.out/'guard.jsonl').open('a') as stream: stream.write(json.dumps(observation)+'\n')
                finally: os.kill(os.getpid(), signal.SIGTERM)
                return
            except BaseException as exc:
                result['guard_trigger'] = dict(error=type(exc).__name__+': '+str(exc))
                result['guard_stop_signal'] = signal.SIGTERM
                os.kill(os.getpid(), signal.SIGTERM); return
    try:
        for signum in (signal.SIGINT, signal.SIGTERM): handlers[signum] = signal.signal(signum, interrupt)
        with serial.serial_lease(paths.RESULTS_DIR/'bert-serial-continuation.lock'), recorder.span('isolated_frozen_report'):
            source = admission(args); result.update(source=source, admission_passed=True)
            if serial.foreign_heavy_processes(): raise RuntimeError('Another numeric or compiler process is active')
            result['resource_admission'] = resources(args)
            if result['resource_admission']['stop']: raise RuntimeError('Report resource admission failed')
            result['exclusive_report_claim'] = claim(args)
            for module, name in ((sys.modules[__name__], 'driver-source.py'), (original, 'original-outer-source.py'),
                                 (compiled, 'original-fork-compile-source.py')):
                shutil.copyfile(module.__file__, args.out/name)
            shutil.copyfile(Path(original.__file__).with_name('bert_sst2_worker.py'), args.out/'original-report-worker.py')
            result['shell'] = prepare_shell(args, source)
            save(args.out/'admission.json', source)
            work = args.out/'worker'; work.mkdir(); (work/'tmp').mkdir()
            argv = command(args, source, deadline-time.monotonic()); save(args.out/'command.json', argv)
            if admission(args) != source: raise ValueError('Source changed before report admission')
            result['attempted_report'] = 1; save(args.out/'result.json', result)
            watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
            with recorder.span('original_recorded_report_worker_once'):
                result['process'] = run_process(argv, cwd=paths.REPO/'DSL/DSL_agent/scripts', log=work/'worker.log',
                    resources=work/'resources.jsonl', timeout=max(.01, deadline-time.monotonic()), mem_gib=4,
                    fsize_gib=8, env=original.child_environment(args, work))
            if (work/'worker_result.json').exists(): result['worker'] = read(work/'worker_result.json')
            result['wall_timeout_observed'] |= bool((result['process'] or {}).get('timed_out'))
            compiled.fork._closed_process(result['process'], successful=True)
            check_report(args, result['worker'] or {})
            result.update(completed_report=1, report_samples=872, task_policy_passed=result['worker']['task_policy_passed'])
            result['shell_recheck'] = shell_bindings(args)
            if admission(args) != source: raise ValueError('Original C2/compile provenance changed during report')
            result['resource_completion'] = resources(args)
            if result['guard_trigger'] or result['resource_completion']['stop']:
                raise RuntimeError('Report resource allowance exhausted')
            result.update(status='ok', report_accepted=True)
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc), report_accepted=False)
    finally:
        stop.set()
        if watcher is not None: watcher.join(timeout=3)
        for signum in handlers: signal.signal(signum, signal.SIG_IGN)
        try:
            if result['guard_trigger']: result.update(status='error', report_accepted=False)
            result['timing'] = summarize_events(read_events(recorder.path))
            for _ in range(20):
                size = artifact_bytes(args.out)
                result['terminal_budget'] = dict(bytes_before_completion=size, reserved_bytes=RESERVE, case_limit_bytes=CASE_BYTES)
                if size+RESERVE > CASE_BYTES: result.update(status='error', report_accepted=False, error='Report terminal metadata exceeds reserved case allowance')
                result['metrics'] = {k:result[k] for k in ('status', 'admission_passed', 'attempted_report', 'completed_report',
                    'report_accepted', 'report_samples', 'task_policy_passed', *ZERO)}
                save(args.out/'result.json', result)
                if artifact_bytes(args.out) == size: break
            else: raise RuntimeError('Report terminal artifact accounting did not converge')
            from ..registry import git_rev
            row = dict(run_id=args.out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=git_rev(),
                metrics=result['metrics'], artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat())
            with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX); stream.seek(0)
                if any(json.loads(line).get('run_id') == args.out.name for line in stream if line.strip()):
                    raise RuntimeError('Duplicate isolated report registry identity')
                stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            persist_case_completion(args.out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=result['timing'], required_artifact_bytes=artifact_bytes(args.out))
            if ((args.out/'completion.json').stat().st_size > RESERVE
                    or artifact_bytes(args.out) > CASE_BYTES and result['status'] != 'error'):
                raise RuntimeError('Report completion exceeded its reserved allowance')
        finally:
            for signum, handler in handlers.items(): signal.signal(signum, handler)
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source-case', 'compile-case', 'out', 'metadata-template'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--intake-python', type=Path, default=paths.INTAKE_PYTHON)
    cli.add_argument('--data-python', type=Path, default=paths.PAPER_DATA_PYTHON)
    cli.add_argument('--ledger', type=Path, default=paths.RESULTS_DIR/'llm_calls.jsonl')
    cli.add_argument('--allowance', type=Path, default=paths.RESULTS_DIR/'autonomous-api-allowance.json')
    return cli


def main(argv=None):
    cli = parser(); args = cli.parse_args(argv)
    if args.data_python is None: cli.error('Pass the original independent data interpreter explicitly')
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.expanduser().absolute() if name.endswith('python') else value.expanduser().resolve())
    result = execute(args)
    print(json.dumps({k:result.get(k) for k in ('status', 'completed_report', 'report_accepted', 'task_policy_passed', 'error')}), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__': raise SystemExit(main())
