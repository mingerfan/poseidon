"""Explicit accepted host-HEIR sources, followed by the original offline Go build.

Only the two recorded host JSON identities are admitted. Original source/math,
native parameters and complete compiler argv are checked by the unchanged own
common Go gate in private globals. No source execution guard is re-entered.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
from types import FunctionType, SimpleNamespace

from .. import paths
from . import heir_common_go_compile as common
from .heir_vgg_fused_pipeline import SOURCE_FIELDS

ROUND = 'HEIR-HOST-GO-COMPILE'
MODE = 'accepted-host-JSON-native-split-lattigo-offline-library-build'
MODULE = 'pt_agent.experiments.heir_host_go_compile'
VARIANTS = {'vgg-rss12-json-overlap': 'vgg16', 'alex-serial-json': 'alexnet'}
REGISTRY_FIELDS = frozenset(('run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'))


def _clone(function, namespace):
    if function.__closure__ is not None:
        raise ValueError('Only original ordinary own functions without closures are cloned')
    cloned = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


def _original_pipeline(model):
    return common._pipeline(model)


def _host_pipeline(variant):
    if variant == 'vgg-rss12-json-overlap':
        from . import heir_vgg_host_cpu_overlap_rss12_json as host
        numeric = host
        dependencies = (host.rss12, host.host, host.host.base, host.math_check)
        runtime = 'host-runtime-cpu-overlap'
        timing = 'overlap-with-identified-BERT-plus-existing-host-runtime'
        parser = 'private-numeric-JSON-original-checker-code'
    elif variant == 'alex-serial-json':
        from . import heir_alexnet_host_serial_json as host
        numeric = host.numeric
        dependencies = (numeric, numeric.common_json, numeric.common_json.math_check,
                        numeric.base, numeric.host, numeric.alex)
        runtime = 'existing-host-runtime-serial'
        timing = 'fresh-serial-host-runtime'
        parser = 'private-Alex-mask-and-common-numeric-JSON'
    else:
        raise ValueError('One explicit reviewed host source variant is required')
    return SimpleNamespace(host=host, numeric=numeric, dependencies=dependencies,
        runtime=runtime, timing=timing, parser=parser)


def _facade(args):
    if VARIANTS.get(args.source_variant) != args.model:
        raise ValueError('Explicit host variant and original model must match')
    original = _original_pipeline(args.model)
    host = _host_pipeline(args.source_variant).host
    # Every original source/check_worker/VARIANT/parameter method remains the
    # same object. Only the expected independently recorded outer identity moves.
    return SimpleNamespace(**dict(vars(original), ROUND=host.ROUND, MODE=host.MODE))


def _registry_source(args, record):
    registry = args.registry or paths.RESULTS_DIR/'registry.jsonl'
    rows = []
    with Path(registry).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == args.source_case.name:
                    rows.append(row)
    if len(rows) != 1 or set(rows[0]) != REGISTRY_FIELDS:
        raise ValueError('One original eight-field outer registry row is required')
    row = rows[0];metrics = row.get('metrics') or {}
    if (row.get('round') != record['round'] or row.get('mode') != record['mode']
            or row.get('model') != record['model']
            or not common._same(row.get('artifacts_dir'), args.source_case)
            or metrics.get('status') != 'ok' or metrics.get('model_compiled_accepted') is not True
            or any(metrics.get(key) != record.get(key) for key in
                   ('attempted', 'completed', 'strict_math_passed', 'model_compiled', 'static_native_bootstraps'))):
        raise ValueError('Original accepted host source registry differs from the durable result')
    return dict(path=str(Path(registry).resolve()), run_id=row['run_id'], unique_original_row=True)


def model_source_gate(args):
    facade = _facade(args);descriptor = _host_pipeline(args.source_variant)
    case = args.source_case.resolve();record = common.read(case/'result.json')
    if (not case.name.startswith(descriptor.host.OUT_PREFIX)
            or record.get('model_compiled_accepted') is not True
            or record.get('runtime_variant') != descriptor.runtime
            or record.get('timing_variant') != descriptor.timing
            or record.get('constant_parser_variant') != descriptor.parser):
        raise ValueError('Accepted exact host JSON/runtime source identity required')
    # Reuse the complete original gate, including source/check_worker, clean
    # phases, exact native primes/scale and complete native compiler command.
    namespace = dict(vars(common), _pipeline=lambda model: facade if model == args.model else _original_pipeline(model))
    checked = _clone(common.model_source_gate, namespace)(args)
    runtime = record.get('host_runtime') or {}
    host_runtime = descriptor.numeric.host if args.model == 'vgg16' else descriptor.host.host
    expected_interpreter = host_runtime.host_python()
    required_runtime = dict(variant='r16a-existing-orion-host-runtime',
        interpreter=str(expected_interpreter), last_observed_import_versions=host_runtime.OBSERVED_HOST_VERSIONS,
        original_Nix_versions_observed_here=False, original_Nix_runtime_equivalence_claimed=False,
        original_worker_code_unchanged=True, text_or_sample_or_weight_reads_in_import_check=0,
        nix_launcher_used=False, numerical_threads=1, strict_reference_or_tolerance_changed=False)
    if any(runtime.get(key) != value for key, value in required_runtime.items()):
        raise ValueError('Saved existing host interpreter/versions/strict numerical provenance differs')
    source_paths = checked['original_source']['paths']
    command = [str(expected_interpreter), '-B', '-m', descriptor.numeric.MODULE,
        '--worker', '--out', str(case/'worker'), '--memory-gib', '24']
    for name in SOURCE_FIELDS:
        command += ['--'+name.replace('_', '-'), source_paths[name]]
    if record['phases']['export_math'].get('argv') != command:
        raise ValueError('The exact original host JSON worker and all explicit source arguments are required')
    registry = _registry_source(args, record)
    completion = common.read(case/'completion.json')
    checked['host_source'] = dict(source_variant=args.source_variant, round=record['round'], mode=record['mode'],
        runtime_variant=descriptor.runtime, timing_variant=descriptor.timing,
        constant_parser_variant=descriptor.parser, host_runtime=runtime, original_registry=registry,
        model_compiled_accepted=True, original_timing=record.get('timing'),
        original_complete_case_elapsed_seconds=completion.get('complete_case_elapsed_seconds'),
        prior_lowering_clock_added_to_Go_clock=False, live_source_execution_guard_reentered=False,
        worker_command=command)
    return checked


def source_gate(args):
    return dict(model=model_source_gate(args), backend=common.backend_gate(args))


def environment(out):
    # Keep the existing offline SDK/cache policy and prevent saved GOENV or a
    # parent go.work file from changing that explicit module contract.
    return dict(common.environment(out), GOENV='off', GOWORK='off')


def append_registry(out, result, registry):
    metrics = {key: result[key] for key in ('status', 'attempted', 'completed', 'go_sources_generated',
        'library_build_passed', 'go_library_compile_accepted', 'static_native_bootstraps', *common.FORMAL_FIELDS)}
    metrics.update(source_variant=result['source_variant'], timing_variant=result['timing_variant'])
    row = dict(run_id=out.name, round=ROUND, model=result['model'], mode=MODE, git_rev=result['git_rev'],
        metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('A Go followup case may have one eight-field registry row')
        stream.seek(0, os.SEEK_END);stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def _terminal_policy(out, result, deadline):
    if result.get('incoming_signal') is not None or result.get('timeout_signal') is not None:
        result.update(status='error', error='Recorded source/tool supervision signal prevents acceptance')
    if time.monotonic() >= deadline:
        result.update(status='error', error='Original1800s Go followup allowance exhausted at closure',
                      wall_timeout_observed=True)
    observed = common.physical_observation(None, common._available_ram_bytes(), shutil.disk_usage(out).free)
    if observed['stop']:
        result.update(status='error', error='; '.join(observed['reasons']), terminal_resource_guard=observed)
    if common.artifact_bytes(out) > common.CASE_BYTES-common.RESERVE_BYTES:
        result.update(status='error', error='Original8GiB case allowance including closure reserve exhausted')


def _namespace(args):
    namespace = dict(vars(common), ROUND=ROUND, MODE=MODE, MODULE=MODULE,
        source_gate=source_gate, environment=environment, append_registry=append_registry,
        _pipeline=lambda model: _facade(args) if model == args.model else _original_pipeline(model))
    deadline = time.monotonic()+common.TOTAL_SECONDS
    original_prepare = _clone(common.prepare, namespace)
    original_terminal = _clone(common.terminal, namespace)
    def durable(path, result):
        if Path(path).name == 'result.json' and result.get('round') == ROUND:
            if result.get('status') not in ('running', 'prepared', 'ok', 'error'):
                result['original_termination_status'] = result['status'];result['status'] = 'error'
            result.update(source_variant=args.source_variant, timing_variant='fresh-Go-only-host-native-continuation',
                previous_host_lowering_clock_added=False, live_source_execution_guard_reentered=False,
                source_identity_bridge='two-explicit-reviewed-host-JSON-identities')
        common.durable_json(path, result)
    def prepare(out, checked):
        original_prepare(out, checked)
        descriptor = _host_pipeline(args.source_variant)
        listing = common.read(out/'source/snapshots.json');seen = {row['source'] for row in listing['files']}
        for module in (SimpleNamespace(__file__=__file__), descriptor.host, descriptor.numeric, *descriptor.dependencies):
            path = Path(module.__file__).resolve()
            if str(path) in seen:continue
            seen.add(str(path));target = out/'source'/path.name;shutil.copyfile(path, target)
            listing['files'].append(dict(source=str(path), snapshot=str(target), size_bytes=target.stat().st_size))
        common.durable_json(out/'source/snapshots.json', listing)
    def terminal(out, result):
        try:_terminal_policy(out, result, deadline)
        except BaseException as exc:result.update(status='error', error=type(exc).__name__+': '+str(exc))
        original_terminal(out, result)
    namespace.update(durable_json=durable, prepare=prepare, terminal=terminal)
    namespace['execute'] = _clone(common.execute, namespace)
    return namespace


def execute(args):
    return _namespace(args)['execute'](args)


def parser():
    command = common.parser();command.description = __doc__
    command.add_argument('--source-variant', choices=tuple(VARIANTS), required=True)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):setattr(args, name, value.expanduser().resolve())
    result = execute(args)
    print(json.dumps({key: result[key] for key in ('model', 'status', 'error', 'go_sources_generated',
        'library_build_passed', 'go_library_compile_accepted')}))
    return int(result['status'] not in ('prepared', 'ok'))


if __name__ == '__main__':raise SystemExit(main())
