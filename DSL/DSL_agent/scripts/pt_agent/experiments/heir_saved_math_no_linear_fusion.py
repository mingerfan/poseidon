"""One saved-math HEIR continuation with only linear fusion disabled.

The original closed host JSON cases, source MLIR and strict train0..2 proof stay
read-only. A new native invocation removes one named pass, retains all remaining
passes and requested parameters, and prints only a failing pass with literals
elided. This entry performs no export, numerical forward, Go or CKKS execution.
"""
import argparse
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
from . import heir_host_go_compile as host_sources
from . import heir_vgg_fused_pipeline as original
from . import heir_alexnet_host_serial_json as serial
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_initializer_probe import (durable_json, serial_lock, receive_signal,
    observe_termination, interruption_status)
from .r10_paper_metrics import StageRecorder, read_events, summarize_events, run_process, _scan
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample

MODULE = 'pt_agent.experiments.heir_saved_math_no_linear_fusion'
ROUND = 'HEIR-SAVED-MATH-NO-LINEAR-FUSION'
MODE = 'closed-host-JSON-same-source-native-only-no-linear-fusion-failure-IR'
VARIANT = 'remove-one-linalg-fuse-linear-ops-v1'
TOTAL_SECONDS, NATIVE_SECONDS = 7440, 7200
GIB = 1024**3
CASE_BYTES, RESERVE_BYTES = 32*GIB, 1024**2
FAILURE_FLAGS = ('--mlir-print-ir-after-all', '--mlir-print-ir-after-failure',
    '--mlir-elide-elementsattrs-if-larger=16', '--mlir-elide-resource-strings-if-larger=64')
ZERO_FIELDS = (*original.ZERO_FIELDS, 'sample_reads', 'model_forwards', 'export_runs', 'new_math_runs')


def _file(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError('A regular nonempty source artifact is required: '+str(path))
    return dict(path=str(path), resource_size_bytes=path.stat().st_size)


def _same(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def remove_linear_fusion(pipeline):
    """Remove exactly one complete pass entry; keep every other character."""
    if not isinstance(pipeline, str) or pipeline.count('linalg-fuse-linear-ops') != 1:
        raise ValueError('Exactly one original linear-fusion pass is required')
    lines = pipeline.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.strip() == 'linalg-fuse-linear-ops,']
    if len(matches) != 1:
        raise ValueError('The original standalone linear-fusion pass entry differs')
    index = matches[0]
    return ''.join(lines[:index]+lines[index+1:])


def _pipeline_at(checked, native):
    pipeline = checked['pipeline']
    for name in dict.fromkeys(re.findall(r'parameter-file=([^\s}]+)', pipeline)):
        if not _same(name, checked['paths']['native_parameters']):
            raise ValueError('Every original parameter-file reference must have the same source')
        pipeline = pipeline.replace('parameter-file='+name, 'parameter-file='+str(native/'native-parameters.json'))
    return pipeline.replace('${RESOURCES}', str(native/'resources'))


def _compiler_argv(tool, source, native, pipeline):
    from .heir_cnn_preflight import compiler_command
    command = compiler_command(tool, source, native/'model-lattigo.mlir', native/'resources')
    command = [value for value in command if not value.startswith(
        ('--annotate-module=', '--torch-linalg-to-ckks=', '--scheme-to-lattigo'))]
    command.insert(2, '--pass-pipeline='+pipeline)
    return command


def _closed_identities(case):
    """Read only first resource metadata records; never inspect process memory."""
    identities = {}
    for phase in ('export_math', 'native_compile'):
        path = case/'phases'/phase/'resources.jsonl'
        with path.open() as stream:
            first = json.loads(next(line for line in stream if line.strip()))
        for row in [first.get('outer_process'), *first.get('processes', [])]:
            if not isinstance(row, dict) or type(row.get('pid')) is not int or type(row.get('starttime')) is not int:
                raise ValueError('Saved original process birth identities are required')
            identities[row['pid']] = row['starttime']
    table = _scan()
    for pid, birth in identities.items():
        row = table.get(pid)
        if row and row.get('starttime') == birth and row.get('state') not in ('Z', 'X'):
            raise ValueError('An original source process is still live')
    return [dict(pid=pid, starttime=birth, original_identity_live=False) for pid, birth in identities.items()]


def _source_registry(args, record):
    registry = args.registry or paths.RESULTS_DIR/'registry.jsonl'
    with Path(registry).open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        rows = [row for line in stream if line.strip()
                for row in [json.loads(line)] if row.get('run_id') == args.source_case.name]
    if len(rows) != 1 or set(rows[0]) != host_sources.REGISTRY_FIELDS:
        raise ValueError('One exact eight-field original source registry row is required')
    row = rows[0];metrics = row.get('metrics') or {}
    if (any(row.get(key) != record.get(key) for key in ('round', 'mode', 'model'))
            or not _same(row.get('artifacts_dir'), args.source_case)
            or any(metrics.get(key) != record.get(key) for key in
                ('status', 'attempted', 'completed', 'strict_math_passed', 'model_compiled', 'static_native_bootstraps'))
            or metrics.get('model_compiled_accepted') is not False):
        raise ValueError('Original registry must retain the same failed native fact and scope')
    return dict(path=str(Path(registry).resolve()), row=row)


def source_gate(args):
    descriptor = host_sources._host_pipeline(args.source_variant)
    model = host_sources.VARIANTS[args.source_variant]
    pipeline = host_sources._original_pipeline(model)
    case = args.source_case.resolve();record = original.read(case/'result.json')
    if (not case.name.startswith(descriptor.host.OUT_PREFIX)
            or record.get('round') != descriptor.host.ROUND or record.get('mode') != descriptor.host.MODE
            or record.get('model') != host_sources.common.MODELS[model]
            or record.get('variant') != pipeline.VARIANT or record.get('status') != 'error'
            or record.get('runtime_variant') != descriptor.runtime
            or record.get('timing_variant') != descriptor.timing
            or record.get('constant_parser_variant') != descriptor.parser
            or record.get('strict_math_passed') is not True or record.get('quality_selection') is not False
            or record.get('original_model_changed') is not False
            or record.get('model_compiled') is not False or record.get('model_compiled_accepted') is not False
            or record.get('static_native_bootstraps') is not None
            or record.get('guard_trigger') is not None or record.get('incoming_signal') is not None
            or record.get('timeout_signal') is not None or record.get('wall_timeout_observed') is not False
            or any(type(record.get(key)) is not int or record[key] != 0 for key in original.ZERO_FIELDS)
            or any(record.get(key) is not None for key in original.FORMAL)):
        raise ValueError('Exact closed host JSON math-pass/native-failure source is required')
    completion = original.read(case/'completion.json')
    if (completion.get('round') != record['round'] or completion.get('status') != 'error'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True):
        raise ValueError('Original failed case must be durably closed')
    if (record.get('attempted') != dict(export_math=1, native_compile=1)
            or record.get('completed') != dict(export_math=1, native_compile=0)):
        raise ValueError('Exactly one accepted math and one failed native attempt are required')
    if any(type(record[field][phase]) is not int for field in ('attempted', 'completed')
            for phase in ('export_math', 'native_compile')):
        raise ValueError('Original phase counters must be exact integers')
    phases = record.get('phases') or {};math_phase = phases.get('export_math') or {}
    if math_phase.get('status') != 'ok':raise ValueError('Original export/math phase is incomplete')
    original.clean_process(math_phase.get('observation') or {})
    if type((math_phase.get('observation') or {}).get('exit_code')) is not int:
        raise ValueError('The original math process exit code must be an integer')
    native = phases.get('native_compile') or {};observation = native.get('observation') or {}
    if (native.get('status') != 'error' or type(observation.get('exit_code')) is not int
            or observation['exit_code'] != 1 or observation.get('timed_out') is not False
            or observation.get('signal') is not None or observation.get('signals_sent') != []
            or observation.get('remaining_pids') != []):
        raise ValueError('Original native compiler error must have cleanly reaped its process tree')
    if (any(type(observation.get(key)) is not int for key in ('started_ns', 'ended_ns'))
            or observation['ended_ns'] <= observation['started_ns']):
        raise ValueError('Original native process clock endpoints must be ordered integer observations')
    saved = original.read(case/'source.json')
    if record.get('source') != saved or set(saved.get('paths') or {}) != set(original.SOURCE_FIELDS):
        raise ValueError('Complete unchanged original source metadata is required')
    checked = pipeline.source_gate({key: Path(value) for key, value in saved['paths'].items()})
    if checked != saved:raise ValueError('Current original source metadata differs')
    worker = original.read(case/'worker/result.json')
    if worker != record.get('worker'):raise ValueError('Saved math worker and outer source differ')
    audit = pipeline.check_worker(worker, checked)
    if math_phase.get('audit') != audit:raise ValueError('Original strict train0..2 math audit differs')
    runtime = descriptor.numeric.host if model == 'vgg16' else descriptor.host.host
    interpreter = runtime.host_python()
    wanted_runtime = dict(variant='r16a-existing-orion-host-runtime', interpreter=str(interpreter),
        last_observed_import_versions=runtime.OBSERVED_HOST_VERSIONS,
        original_Nix_versions_observed_here=False, original_Nix_runtime_equivalence_claimed=False,
        original_worker_code_unchanged=True, text_or_sample_or_weight_reads_in_import_check=0,
        nix_launcher_used=False, numerical_threads=1, strict_reference_or_tolerance_changed=False)
    if any((record.get('host_runtime') or {}).get(key) != value for key, value in wanted_runtime.items()):
        raise ValueError('Original host numerical runtime provenance differs')
    math_command = [str(interpreter), '-B', '-m', descriptor.numeric.MODULE,
        '--worker', '--out', str(case/'worker'), '--memory-gib', '24']
    for key in original.SOURCE_FIELDS:math_command += ['--'+key.replace('_', '-'), checked['paths'][key]]
    if math_phase.get('argv') != math_command:raise ValueError('Original exact host math argv differs')
    source = case/'worker/model.mlir';old_native = case/'native';request = record.get('native_request') or {}
    explicit = (old_native/'explicit.pipeline').read_text().strip()
    old_command = native.get('argv') or []
    if (not isinstance(old_command, list) or not old_command or not isinstance(old_command[0], str)
            or not _same(old_command[0], args.heir_opt)):
        raise ValueError('Requested compiler must be the same installed original executable')
    if (explicit != _pipeline_at(checked, old_native)
            or original.read(old_native/'native-parameters.json') != checked['native_parameters']
            or request.get('parameters') != checked['native_parameters']
            or not _same(request.get('source_mlir'), source)
            or not _same(request.get('pipeline'), old_native/'explicit.pipeline')
            or request.get('original_model_contract') != checked['contract']
            or request.get('lowered_adapter_contract') != worker.get('lowered_adapter_contract')
            or old_command != _compiler_argv(old_command[0], source, old_native, explicit)
            or (old_native/'model-lattigo.mlir').exists()):
        raise ValueError('Every original native argv/parameter/source field must match the failed case')
    if not args.heir_opt.is_file() or not os.access(args.heir_opt, os.X_OK):
        raise ValueError('The same existing original HEIR executable is required')
    stderr_path = case/'phases/native_compile/worker.log'
    if stderr_path.stat().st_size > 1024**2:raise ValueError('Original native diagnostic exceeds the bounded metadata scope')
    errors = stderr_path.read_text().count('error: unexpected unrealized conversion cast op found')
    if errors <= 0:raise ValueError('The original unresolved native cast failure is required')
    remove_linear_fusion(explicit)
    return dict(model=record['model'], source_variant=args.source_variant, source_case=str(case),
        original_source=checked, original_round=record['round'], original_mode=record['mode'],
        original_source_mlir=_file(source), original_native_argv=native['argv'],
        original_parameters=checked['native_parameters'], inherited_strict_math=audit,
        original_complete_case_seconds=completion.get('complete_case_elapsed_seconds'),
        original_native_process_seconds=(observation['ended_ns']-observation['started_ns'])/1e9,
        original_cast_errors=errors, original_closed_identities=_closed_identities(case),
        original_registry=_source_registry(args, record), original_artifacts_modified=False,
        original_MLIR_payload_read=False, new_math_runs=0)


def native_command(args, out, checked):
    native = out/'native'
    # Keep the recorded executable spelling; CLI canonicalization may resolve
    # the installed bin/heir-opt symlink to its site-packages target.
    tool = checked['original_native_argv'][0]
    if not _same(tool, args.heir_opt):raise ValueError('The admitted original compiler identity changed')
    command = original.native_command(tool, Path(checked['original_source_mlir']['path']),
        native, checked['original_source'])
    pipeline = (native/'explicit.pipeline').read_text().strip()
    if command != _compiler_argv(tool, Path(checked['original_source_mlir']['path']), native, pipeline):
        raise ValueError('The original native command builder changed unexpectedly')
    wanted = _pipeline_at(checked['original_source'], native)
    if pipeline != wanted or original.read(native/'native-parameters.json') != checked['original_parameters']:
        raise ValueError('Only original parameter/storage rebasing is allowed before the one pass removal')
    replacement = remove_linear_fusion(pipeline)
    command[2] = '--pass-pipeline='+replacement
    (native/'explicit.pipeline').write_text(replacement+'\n')
    command.extend(FAILURE_FLAGS)
    return command


def physical_observation(sample, available, free, lease):
    # The original host/disk limits stay exact; this explicit variant has RSS16.
    observed = original.physical_observation(None, available, free)
    rss = max(lease['own_subtree_rss_bytes'],
        (0 if sample is None else sample['subtree_rss_bytes'])+lease['supervisor_rss_bytes'])
    if rss > 16*GIB:observed['reasons'].append('Own supervisor and descendant RSS exceeds16GiB')
    observed.update(subtree_rss_bytes=rss, stop=bool(observed['reasons']), serial_lease=lease,
        sample_ts_ns=None if sample is None else sample.get('sample_ts_ns'))
    return observed


def observe_child(result, observation):
    if observation.get('timed_out'):
        result['wall_timeout_observed'] = True
        result['child_timeout_signal'] = observation.get('signal')
    else:
        observe_termination(result, observation)


def _registry(out, result, registry):
    metrics = {key: result[key] for key in ('status', 'attempted', 'completed', 'strict_math_passed',
        'math_inherited', 'model_compiled', 'model_compiled_accepted', 'static_native_bootstraps', *original.FORMAL)}
    row = dict(run_id=out.name, round=ROUND, model=result['model'], mode=MODE, git_rev=None,
        metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('One new native continuation registry row is required')
        stream.seek(0, os.SEEK_END);stream.write(json.dumps(row, allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def _snapshot(out, args):
    directory = out/'source';directory.mkdir();rows = []
    descriptor = host_sources._host_pipeline(args.source_variant)
    modules = [sys.modules[__name__], original, host_sources, host_sources.common, descriptor.host,
        descriptor.numeric, *descriptor.dependencies, serial]
    seen = set()
    for module in modules:
        source = Path(module.__file__).resolve()
        if source in seen:continue
        seen.add(source);target = directory/source.name;shutil.copyfile(source, target)
        rows.append(dict(original=str(source), snapshot=str(target)))
    return rows


def execute(args):
    root = paths.RESULTS_DIR.resolve();out = args.out.resolve()
    if out.exists() or out == root or not out.is_relative_to(root):
        raise ValueError('One fresh independent case under the results root is required')
    with serial_lock(root):
        out.mkdir(parents=True);(out/'tmp').mkdir();(out/'phases').mkdir()
        started = time.perf_counter_ns();deadline = time.monotonic()+TOTAL_SECONDS
        events = StageRecorder(out/'events.jsonl');active = [None];identity = [None];failures = []
        result = dict(round=ROUND, mode=MODE, variant=VARIANT,
            model=host_sources.common.MODELS[host_sources.VARIANTS[args.source_variant]],
            status='running', error=None, attempted=dict(native_compile=0), completed=dict(native_compile=0), phases={},
            strict_math_passed=None, math_inherited=True, new_math_measured=False,
            model_compiled=False, model_compiled_accepted=False, static_native_bootstraps=None,
            original_model_changed=False, quality_selection=False, incoming_signal=None, timeout_signal=None,
            child_timeout_signal=None,
            wall_timeout_observed=False, guard_trigger=None, guard_stop_signal=None,
            failure_ir_printing=dict(enabled=True, flags=list(FAILURE_FLAGS),
                included_in_native_clock=True, failed_pass_only=True, dense_literals_elided=True,
                possible_printing_work_and_log_size_overhead=True),
            timing_scope='Fresh saved-math native-only serial continuation; old export/math/native clocks remain separate',
            limits=dict(total_seconds=TOTAL_SECONDS, native_seconds=NATIVE_SECONDS, AS_gib=24,
                subtree_RSS_gib=16, minimum_available_gib=1.25, minimum_free_disk_gib=10,
                file_gib=16, case_gib=32, concurrency=1, completion_reserve_bytes=RESERVE_BYTES),
            **{key: 0 for key in ZERO_FIELDS}, **{key: None for key in original.FORMAL})
        stop = threading.Event()
        def check():
            if failures:raise RuntimeError(failures[0])
            if time.monotonic() >= deadline:
                result['wall_timeout_observed'] = True;raise TimeoutError('Finite7440s continuation allowance exhausted')
            lease = serial.lease_observation(identity[0]);identity[0] = lease['owner']
            observed = physical_observation(_latest_rss_sample(active[0]) if active[0] else None,
                _available_ram_bytes(), shutil.disk_usage(out).free, lease)
            result['latest_resource_observation'] = observed
            if observed['stop']:
                result['guard_trigger'] = observed;durable_json(out/'resource-guard.json', observed)
                raise RuntimeError('; '.join(observed['reasons']))
            size = artifact_bytes(out)
            if size > CASE_BYTES-RESERVE_BYTES:
                result['guard_trigger'] = dict(kind='artifact', actual_bytes=size, allowance_bytes=CASE_BYTES)
                raise RuntimeError('32GiB case allowance including completion reserve exhausted')
        def monitor():
            while not stop.wait(.2):
                try:check()
                except BaseException as exc:
                    failures.append(type(exc).__name__+': '+str(exc));result['monitor_stop_reason'] = failures[-1]
                    try:durable_json(out/'monitor-stop.json', dict(reason=failures[-1], guard=result.get('guard_trigger')))
                    finally:
                        try:os.kill(os.getpid(), signal.SIGTERM)
                        except OSError as error:result['monitor_signal_error'] = str(error)
                    return
        watcher = threading.Thread(target=monitor, daemon=True)
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        for sig in old:signal.signal(sig, lambda signum, frame: receive_signal(result, signum))
        signal.setitimer(signal.ITIMER_REAL, TOTAL_SECONDS)
        try:
            with events.span('saved-math-no-linear-fusion-continuation'):
                check();watcher.start()
                with events.span('source-admission'):
                    checked = source_gate(args);result.update(source=checked, strict_math_passed=True)
                    durable_json(out/'source.json', checked);result['source_snapshots'] = _snapshot(out, args)
                    command = native_command(args, out, checked);result['native_argv'] = command
                    result['native_request'] = dict(parameters=checked['original_parameters'],
                        source_mlir=checked['original_source_mlir']['path'], pipeline=str(out/'native/explicit.pipeline'),
                        pass_removed='linalg-fuse-linear-ops', removed_count=1, other_pass_order_unchanged=True)
                result['status'] = 'prepared';durable_json(out/'result.json', result)
                if args.run:
                    check()
                    if source_gate(args) != checked:raise ValueError('Source changed before the one native invocation')
                    directory = out/'phases/native_compile';directory.mkdir()
                    row = dict(argv=command, status='running', observation=None);result['phases']['native_compile'] = row
                    result['attempted']['native_compile'] = 1;durable_json(out/'result.json', result)
                    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')
                        and not k.startswith(('LD_', 'NIX_', '__NIX_'))}
                    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(out/'tmp'))
                    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):env[key] = '1'
                    with events.span('native_compile', argv=command):
                        active[0] = directory/'resources.jsonl'
                        try:
                            observation = run_process(command, cwd=paths.REPO/'DSL/DSL_agent/scripts',
                                log=directory/'worker.log', resources=active[0],
                                timeout=min(NATIVE_SECONDS, max(.01, deadline-time.monotonic())),
                                mem_gib=24, fsize_gib=16, env=env)
                            row['observation'] = observation;observe_child(result, observation)
                            durable_json(directory/'process-return.json', observation)
                        finally:active[0] = None
                        original.clean_process(observation);check()
                        artifact = original.check_native(out/'native', checked['original_source'])
                        row.update(status='ok', artifact=artifact);result['completed']['native_compile'] = 1
                        result.update(status='ok', model_compiled=True,
                            static_native_bootstraps=artifact['static_bootstrap_operations'])
        except BaseException as exc:
            result.update(status=interruption_status(result), error=type(exc).__name__+': '+str(exc)[:4000])
            for row in result['phases'].values():
                if row['status'] == 'running':row.update(status=result['status'], error=result['error'])
        finally:
            stop.set()
            if watcher.is_alive():watcher.join(timeout=3)
            result['monitor_stopped'] = not watcher.is_alive()
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old.items():signal.signal(sig, handler)
            if failures:result.update(status=interruption_status(result), error=failures[0])
            try:check()
            except BaseException as exc:
                result.update(status=interruption_status(result), error=type(exc).__name__+': '+str(exc)[:4000])
            if not result['monitor_stopped']:
                result.update(status='error', error='Resource monitor did not stop after the bounded3s join; native completion facts are retained')
            result['model_compiled_accepted'] = bool(result['status'] == 'ok' and result['model_compiled'])
            result['timing'] = summarize_events(read_events(out/'events.jsonl'), time.perf_counter_ns())
            # Reuse only the ordinary size fixed-point helper in a private namespace.
            from types import FunctionType
            terminal = FunctionType(original.terminal_result.__code__, dict(vars(original),
                CASE_BYTES=CASE_BYTES, COMPLETION_RESERVE=RESERVE_BYTES), 'terminal_result')
            terminal(out, result)
            if result['status'] != 'ok':result['model_compiled_accepted'] = False;terminal(out, result)
            _registry(out, result, args.registry or root/'registry.jsonl')
            marker = persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=result['timing'], required_artifact_bytes=artifact_bytes(out))
            if (out/'completion.json').stat().st_size > RESERVE_BYTES or marker['case_artifact_bytes'] > CASE_BYTES:
                raise RuntimeError('Completion exceeded the recorded reserved case allowance')
        return result


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument('--source-case', type=Path, required=True)
    command.add_argument('--source-variant', choices=tuple(host_sources.VARIANTS), required=True)
    command.add_argument('--out', type=Path, required=True)
    command.add_argument('--heir-opt', type=Path, required=True)
    command.add_argument('--registry', type=Path)
    command.add_argument('--run', action='store_true')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):setattr(args, key, value.expanduser().resolve())
    result = execute(args)
    print(json.dumps({key: result.get(key) for key in ('status', 'error', 'model_compiled', 'model_compiled_accepted')}))
    return int(result['status'] not in ('prepared', 'ok'))


if __name__ == '__main__':raise SystemExit(main())
