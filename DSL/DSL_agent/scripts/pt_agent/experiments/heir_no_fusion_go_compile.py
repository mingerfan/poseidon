"""Explicit accepted no-linear-fusion source, then the original offline Go build.

The saved original host math and current native artifact have separate lineage.
Only provenance and terminal recording are adapted in private ordinary-function
globals; original translators, SDK/cache, build argv and numerical policy stay.
No generated initializer, model, context, key or ciphertext operation executes.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from .. import paths
from . import heir_common_go_compile as common
from . import heir_host_go_compile as host_sources
from . import heir_saved_math_no_linear_fusion as native

MODULE = 'pt_agent.experiments.heir_no_fusion_go_compile'
ROUND = 'HEIR-NO-LINEAR-FUSION-GO-COMPILE'
MODE = 'accepted-saved-math-no-linear-fusion-native-split-offline-Go-library'


def _accepted_registry(args, record):
    registry = args.registry or paths.RESULTS_DIR/'registry.jsonl'
    with Path(registry).open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        rows = [row for line in stream if line.strip()
                for row in [json.loads(line)] if row.get('run_id') == args.source_case.name]
    if len(rows) != 1 or set(rows[0]) != host_sources.REGISTRY_FIELDS:
        raise ValueError('One accepted no-fusion eight-field registry row is required')
    row = rows[0];metrics = row.get('metrics') or {}
    keys = ('status', 'attempted', 'completed', 'strict_math_passed', 'math_inherited',
        'model_compiled', 'model_compiled_accepted', 'static_native_bootstraps', *native.original.FORMAL)
    if (any(row.get(key) != record.get(key) for key in ('round', 'mode', 'model'))
            or not common._same(row.get('artifacts_dir'), args.source_case)
            or any(metrics.get(key) != record.get(key) for key in keys)):
        raise ValueError('Accepted no-fusion registry and actual durable result differ')
    return dict(path=str(Path(registry).resolve()), row=row)


def _closed_native_processes(case):
    with (case/'phases/native_compile/resources.jsonl').open() as stream:
        first = json.loads(next(line for line in stream if line.strip()))
    identities = {}
    for row in [first.get('outer_process'), *first.get('processes', [])]:
        if not isinstance(row, dict) or type(row.get('pid')) is not int or type(row.get('starttime')) is not int:
            raise ValueError('Actual source-native PID/birth metadata is required')
        identities[row['pid']] = row['starttime']
    table = native._scan()
    for pid, birth in identities.items():
        observed = table.get(pid)
        if observed and observed.get('starttime') == birth and observed.get('state') not in ('Z', 'X'):
            raise ValueError('The accepted source-native case still has a live original identity')
    return [dict(pid=pid, starttime=birth, original_identity_live=False) for pid, birth in identities.items()]


def model_source_gate(args):
    if host_sources.VARIANTS.get(args.source_variant) != args.model:
        raise ValueError('Original model and host-math source variant must match')
    case = args.source_case.resolve();record = common.read(case/'result.json')
    if (record.get('round') != native.ROUND or record.get('mode') != native.MODE
            or record.get('variant') != native.VARIANT or record.get('model') != common.MODELS[args.model]
            or record.get('status') != 'ok' or record.get('model_compiled') is not True
            or record.get('model_compiled_accepted') is not True or record.get('monitor_stopped') is not True
            or record.get('strict_math_passed') is not True or record.get('math_inherited') is not True
            or record.get('new_math_measured') is not False or record.get('quality_selection') is not False
            or record.get('original_model_changed') is not False
            or any(record.get(key) is not None for key in
                   ('guard_trigger', 'incoming_signal', 'timeout_signal', 'child_timeout_signal', 'guard_stop_signal'))
            or record.get('wall_timeout_observed') is not False
            or any(type(record.get(key)) is not int or record[key] != 0 for key in native.ZERO_FIELDS)
            or any(record.get(key) is not None for key in native.original.FORMAL)):
        raise ValueError('Closed accepted no-fusion native source with inherited strict math is required')
    if (record.get('attempted') != dict(native_compile=1) or record.get('completed') != dict(native_compile=1)
            or any(type(record[field]['native_compile']) is not int for field in ('attempted', 'completed'))
            or set(record.get('phases') or {}) != {'native_compile'}):
        raise ValueError('Exactly one actual native phase and no invented math/export phases are required')
    phase = record['phases']['native_compile']
    if phase.get('status') != 'ok':raise ValueError('The actual source native phase is incomplete')
    common.clean(phase.get('observation') or {})
    limits = dict(total_seconds=native.TOTAL_SECONDS, native_seconds=native.NATIVE_SECONDS,
        AS_gib=24, subtree_RSS_gib=16, minimum_available_gib=1.25, minimum_free_disk_gib=10,
        file_gib=16, case_gib=32, concurrency=1, completion_reserve_bytes=native.RESERVE_BYTES)
    if any(type((record.get('limits') or {}).get(key)) is not type(value)
            or record['limits'][key] != value for key, value in limits.items()):
        raise ValueError('The exact bounded no-fusion source resource policy is required')
    observed_limits = (phase.get('observation') or {}).get('limits') or {}
    wall = observed_limits.get('wall_seconds')
    if (observed_limits.get('memory_gib') != 24 or observed_limits.get('file_gib') != 16
            or type(wall) not in (int, float) or not 0 < wall <= native.NATIVE_SECONDS):
        raise ValueError('The original actual native supervision limits differ')
    completion = common.read(case/'completion.json')
    if (completion.get('round') != native.ROUND or completion.get('status') != 'ok'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True):
        raise ValueError('The accepted no-fusion source must be durably closed')
    saved = common.read(case/'source.json')
    if saved != record.get('source') or saved.get('source_variant') != args.source_variant:
        raise ValueError('Exact no-fusion saved/outer source lineage is required')
    old_argv = saved.get('original_native_argv') or []
    if not old_argv or not isinstance(old_argv[0], str) or not saved.get('source_case'):
        raise ValueError('The original failed host source/compiler identity is required')
    original_args = SimpleNamespace(source_case=Path(saved['source_case']), source_variant=args.source_variant,
        heir_opt=Path(old_argv[0]), registry=args.registry)
    inherited = native.source_gate(original_args)
    if inherited != saved or inherited['model'] != record['model'] or inherited.get('new_math_runs') != 0:
        raise ValueError('Original graph/program/probes/parameters and inherited strict-math proof differ')
    # Stat fields are source/resource accounting. This comparison makes no claim
    # of byte/content equivalence and never opens MLIR or external payloads.
    directory = case/'native';source_mlir = Path(inherited['original_source_mlir']['path'])
    wanted = inherited['original_parameters'];original_source = inherited['original_source']
    pipeline = native.remove_linear_fusion(native._pipeline_at(original_source, directory))
    if ((directory/'explicit.pipeline').read_text().strip() != pipeline
            or common.read(directory/'native-parameters.json') != wanted):
        raise ValueError('One removed pass and all original native parameters must be retained')
    expected = native._compiler_argv(old_argv[0], source_mlir, directory, pipeline)+list(native.FAILURE_FLAGS)
    request = record.get('native_request') or {}
    if (record.get('native_argv') != expected or phase.get('argv') != expected
            or request.get('parameters') != wanted
            or not common._same(request.get('source_mlir'), source_mlir)
            or not common._same(request.get('pipeline'), directory/'explicit.pipeline')
            or request.get('pass_removed') != 'linalg-fuse-linear-ops'
            or type(request.get('removed_count')) is not int or request['removed_count'] != 1
            or request.get('other_pass_order_unchanged') is not True):
        raise ValueError('Complete actual no-fusion compiler argv/request differs')
    if Path(old_argv[0]).resolve().parent != args.translate.resolve().parent:
        raise ValueError('The same installed original compiler/translator pair is required')
    artifact = common.read(directory/'model-native-artifact.json')
    if artifact != phase.get('artifact'):raise ValueError('Actual native artifact and saved phase inspection differ')
    parameters = artifact.get('parameters') or []
    count = artifact.get('static_bootstrap_operations')
    if (len(parameters) != 1 or any(parameters[0].get(key) != wanted[field] for key, field in
            (('logN', 'logN'), ('Q', 'Q'), ('P', 'P'), ('log_default_scale', 'logDefaultScale')))
            or type(count) is not int or count < 0 or type(record.get('static_native_bootstraps')) is not int
            or record['static_native_bootstraps'] != count
            or artifact.get('backend_executed') is not False or artifact.get('security_validated') is not False):
        raise ValueError('Full generated native parameters and current static site inventory differ')
    registry = _accepted_registry(args, record);identities = _closed_native_processes(case)
    return dict(source_result=str(case/'result.json'), source_case=str(case), model=record['model'],
        variant=native.VARIANT, input_mlir=common._file(directory/'model-lattigo.mlir'), native_artifact=artifact,
        original_source=original_source, strict_math=inherited['inherited_strict_math'],
        original_model_changed=False, previous_lowering_clock_included=False,
        compiled_mlir_or_external_payload_read_by_gate=False,
        no_fusion_source=dict(source_variant=args.source_variant, round=record['round'], mode=record['mode'],
            accepted_registry=registry, closed_native_identities=identities, original_failed_host_case=inherited['source_case'],
            source_proof=saved, math_inherited=True, new_math_runs=0, monitor_stopped=True,
            native_timing=record.get('timing'), complete_case_seconds=completion.get('complete_case_elapsed_seconds'),
            source_native_observation=phase['observation'], previous_clocks_added_to_Go_clock=False))


def source_gate(args):
    return dict(model=model_source_gate(args), backend=common.backend_gate(args))


def environment(out):
    return dict(common.environment(out), GOENV='off', GOWORK='off')


def append_registry(out, result, registry):
    metrics = {key: result[key] for key in ('status', 'attempted', 'completed', 'go_sources_generated',
        'library_build_passed', 'go_library_compile_accepted', 'monitor_stopped', 'static_native_bootstraps',
        *common.FORMAL_FIELDS)}
    row = dict(run_id=out.name, round=ROUND, model=result['model'], mode=MODE, git_rev=result['git_rev'],
        metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('The Go followup may have one eight-field registry row')
        stream.seek(0, os.SEEK_END);stream.write(json.dumps(row, allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def _namespace(args):
    threads = [];deadline = time.monotonic()+common.TOTAL_SECONDS
    def thread(*positional, **keywords):
        created = common.threading.Thread(*positional, **keywords);threads.append(created);return created
    namespace = dict(vars(common), ROUND=ROUND, MODE=MODE, MODULE=MODULE, source_gate=source_gate,
        environment=environment, append_registry=append_registry, observe_termination=native.observe_child,
        threading=SimpleNamespace(Thread=thread, Event=common.threading.Event))
    prepare_original = host_sources._clone(common.prepare, namespace)
    terminal_original = host_sources._clone(common.terminal, namespace)
    def prepare(out, checked):
        prepare_original(out, checked)
        listing = common.read(out/'source/snapshots.json');seen = {row['source'] for row in listing['files']}
        for module in (sys.modules[__name__], native, host_sources):
            path = Path(module.__file__).resolve()
            if str(path) in seen:continue
            seen.add(str(path));target = out/'source'/path.name;shutil.copyfile(path, target)
            listing['files'].append(dict(source=str(path), snapshot=str(target), size_bytes=target.stat().st_size))
        common.durable_json(out/'source/snapshots.json', listing)
    def durable(path, result):
        if Path(path).name == 'result.json' and result.get('round') == ROUND:
            if result.get('status') not in ('running', 'prepared', 'ok', 'error'):
                result['original_termination_status'] = result['status'];result['status'] = 'error'
            result.update(source_variant=args.source_variant,
                timing_variant='fresh-Go-only-from-accepted-no-linear-fusion-native',
                inherited_math_not_remeasured=True, old_failures_preserved=True,
                previous_native_clock_added=False, stat_accounting_is_content_validation=False)
        common.durable_json(path, result)
    def terminal(out, result):
        result['monitor_stopped'] = bool(len(threads) == 1 and not threads[0].is_alive())
        try:host_sources._terminal_policy(out, result, deadline)
        except BaseException as exc:result.update(status='error', error=type(exc).__name__+': '+str(exc))
        if not result['monitor_stopped']:
            result.update(status='error', error='The private Go resource monitor did not stop; actual build facts are retained')
        terminal_original(out, result)
    namespace.update(prepare=prepare, durable_json=durable, terminal=terminal)
    namespace['execute'] = host_sources._clone(common.execute, namespace)
    return namespace


def execute(args):
    return _namespace(args)['execute'](args)


def parser():
    command = common.parser();command.description = __doc__
    command.add_argument('--source-variant', choices=tuple(host_sources.VARIANTS), required=True)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):setattr(args, name, value.expanduser().resolve())
    result = execute(args)
    print(json.dumps({key: result.get(key) for key in ('model', 'status', 'error',
        'go_sources_generated', 'library_build_passed', 'go_library_compile_accepted')}))
    return int(result['status'] not in ('prepared', 'ok'))


if __name__ == '__main__':raise SystemExit(main())
