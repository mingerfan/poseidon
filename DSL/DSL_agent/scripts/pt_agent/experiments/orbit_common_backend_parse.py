"""Statically consume original Orbit VGG/Alex optimized MLIR with its parser.

Only own ordinary Python orchestration functions receive private globals.  The
existing Go main overlay, official parser/scheduler, initialization admission,
offline environment, resource checks and finite execution phases are reused.
The shared Res interpreter build identifies a backend, not this target model.
"""
import argparse
import json
from pathlib import Path
import shutil
from types import FunctionType, SimpleNamespace

from . import orbit_backend_parse_only as parser_entry
from . import orbit_common_model_frontend as frontend
from . import orbit_common_model_optimizer as optimizer

ROUND = 'ORBIT-COMMON-BACKEND-PARSER'
MODE = 'original-common-earth-parser-and-static-SSA-consumption-only'
MODEL_RECORDS = {key: dict(optimizer.MODEL_RECORDS[key]) for key in ('vgg16', 'alexnet')}
CLIENT = parser_entry.CLIENT
ZERO_FIELDS = parser_entry.ZERO_FIELDS
read = parser_entry.read
check_parsed = parser_entry.check_parsed
build_command = parser_entry.build_command
parser_command = parser_entry.parser_command
native_environment = parser_entry.shared.native_environment


def _same(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _one_counts(record, phases):
    for field in ('attempted', 'completed'):
        for phase in phases:
            value = (record.get(field) or {}).get(phase)
            if type(value) is not int or value != 1:
                raise ValueError('One completed original phase required: '+field+'/'+phase)


def _completion(case):
    row = read(case/'completion.json')
    if (row.get('status') != 'ok' or row.get('required_artifacts_persisted') is not True
            or row.get('registry_persisted') is not True):
        raise ValueError('Original source must have a persisted successful completion')


def _processes(result, phases):
    rows = [row for row in result.get('processes', []) if row.get('phase') in phases]
    if len(rows) != len(phases) or {row['phase'] for row in rows} != set(phases):
        raise ValueError('Every original common-model phase needs its clean process observation')
    for row in rows:
        parser_entry.clean(row.get('observation') or {})


def _phase_records(case, phases):
    for phase, record in phases.items():
        if read(case/'work'/(phase+'-worker')/'phase_result.json') != record:
            raise ValueError('Saved worker and outer phase metadata differ: '+phase)


def _target_source(args):
    """Read source/TDAG JSON and stat MLIR/constants; no payload is opened."""
    selection = MODEL_RECORDS[args.model]
    factory = selection['factory'];case = args.source_case.resolve();work = case/'work'
    result = read(case/'result.json')
    if (result.get('round') != optimizer.ROUND or result.get('mode') != optimizer.MODE
            or result.get('model') != factory or result.get('model_key') != args.model
            or result.get('status') != 'ok' or result.get('native_optimizer_completed') is not True
            or result.get('placement_artifact_checked') is not True):
        raise ValueError('Successful same-model original common optimizer required')
    _completion(case);_one_counts(result, ('optimize', 'validate'));_processes(result, ('optimize', 'validate'))
    phases = result.get('phase_results') or {}
    _phase_records(case, {phase: phases.get(phase) or {} for phase in ('optimize', 'validate')})
    for phase in ('optimize', 'validate'):
        optimizer.validate_phase(phases.get(phase) or {}, phase, dict(model=factory), work)
    profile = (args.orbit_root/'cost_models/profiled_LATTIGONEW_CPU64k_3_16.json').resolve()
    if read(profile).get('polynomialDegree') != 131072:
        raise ValueError('Original 64k profile identity differs')
    for phase in ('optimize', 'validate'):
        if not _same((phases[phase].get('params') or {}).get('profile'), profile):
            raise ValueError('Optimizer and checker must use this original backend profile')
    if not _same(phases['validate'].get('source_record'), work/'optimize-worker/phase_result.json'):
        raise ValueError('Checker must consume this optimizer worker result')
    previous = result.get('source_records') or {}
    front_path = Path(previous.get('source_record', '')).resolve()
    if (front_path.name != 'result.json' or not front_path.is_file()
            or not _same(phases['optimize'].get('source_record'), front_path)):
        raise ValueError('Optimizer must retain its exact successful frontend result')
    source = read(front_path);front_case = front_path.parent
    if (source.get('round') != frontend.ROUND or source.get('mode') != frontend.MODE
            or source.get('model') != factory or source.get('model_key') != args.model
            or source.get('status') != 'ok' or source.get('orbit_model_trace_passed') is not True
            or source.get('tdag_parse_passed') is not True):
        raise ValueError('Successful same-model original frontend required')
    _completion(front_case);_one_counts(source, ('trace', 'parse'));_processes(source, ('trace', 'parse'))
    front_phases = source.get('phase_results') or {}
    _phase_records(front_case, {phase: front_phases.get(phase) or {} for phase in ('trace', 'parse')})
    activations = frontend.worker.MODEL_RECORDS[factory]['activations']
    locations = dict(model=factory, activations=activations, profile=profile)
    for phase in ('trace', 'parse'):
        frontend.validate_phase(front_phases.get(phase) or {}, phase, locations, front_case/'work')
    trace = front_phases['trace'];original_graph = front_phases['parse']['graph_check']
    records = source.get('source_records') or {}
    if (records.get('model') != factory or records.get('original_model_modified') is not False
            or records.get('weights_modified') is not False or records.get('polynomial_modified') is not False
            or not _same(records.get('original_profile'), profile)
            or not _same((previous.get('input_mlir') or {}).get('path'), trace['mlir']['path'])
            or previous['input_mlir'].get('size_bytes') != trace['mlir']['size_bytes']
            or previous.get('original_graph_nodes') != original_graph['node_count']
            or previous.get('original_graph_edges') != original_graph['edge_count']):
        raise ValueError('Original model/weights/Poly and frontend graph lineage differ')
    constants = trace.get('constant_files') or []
    if len(constants) != 1:
        raise ValueError('One original full-model constant artifact required')
    constant = parser_entry.shared._file(constants[0].get('path', ''))
    constant_path = Path(constant['path'])
    if (not constant_path.is_relative_to(front_case/'work/trace-worker/constants')
            or constant_path.suffix != '.cst' or constant['bytes'] != constants[0].get('size_bytes')):
        raise ValueError('Original frontend constant path/size metadata differs')
    validated = phases['validate'];graph = optimizer.worker._placement_record(read(validated['tdag']))
    saved = validated.get('graph_check') or {}
    keys = ('node_count', 'edge_count', 'op_counts', 'inputs', 'outputs',
            'placement_bootstrap_nodes', 'placement_bootstrap_weight_sum')
    if (any(graph[key] != saved.get(key) for key in keys)
            or graph['inputs'] != ['arg0'] or len(graph['outputs']) != 1):
        raise ValueError('Exact saved placed TDAG identity differs')
    expected = {key: graph[key] for key in keys if key != 'placement_bootstrap_weight_sum'}
    expected['output_id'] = int(graph['outputs'][0])
    optimized = parser_entry.shared._file(work/'optimized.mlir')
    return dict(source_result=str(case/'result.json'), frontend_result=str(front_path), model=factory,
        model_key=args.model, optimized_mlir=optimized, constants=constant,
        optimizer_params=validated['params'], optimization_params=phases['optimize']['params'],
        placement_bootstrap_nodes=graph['placement_bootstrap_nodes'],
        placement_bootstrap_weight_sum=graph['placement_bootstrap_weight_sum'], expected_graph=expected,
        prior_optimizer_times_included=False, model_payload_or_inputs_read=False,
        original_model_modified=False, weights_modified=False, polynomial_modified=False,
        scope='This common-model optimized Earth MLIR; independent of the shared Res build model')


def source_gate(args):
    source = _target_source(args);build = args.shared_build_case.resolve()
    result = read(build/'result.json')
    if (result.get('round') != parser_entry.shared.ROUND or result.get('mode') != parser_entry.shared.MODE
            or result.get('status') != 'ok' or result.get('runtime_interpreter_compiled') is not True
            or result.get('go_build_attempts') != 1 or result.get('go_build_completed') != 1
            or result.get('cleanup_unknown') is not False
            or any(result.get(key) != 0 for key in parser_entry.shared.ZERO_FIELDS)):
        raise ValueError('Completed zero-execution shared original interpreter build required')
    _completion(build);parser_entry.clean(result.get('process') or {})
    binary = result.get('binary') or {};actual_binary = parser_entry.shared._file(build/'orbit-fhe')
    if not _same(binary.get('path'), actual_binary['path']) or binary.get('bytes') != actual_binary['bytes']:
        raise ValueError('Original shared binary metadata differs')
    module = (args.orbit_root/'backend/lattigo').resolve()
    if not _same((result.get('package') or {}).get('interpreter_package'), module/'lowering/fhe'):
        raise ValueError('Shared build must identify the same original Go module')
    command = json.loads((build/'build-argv.json').read_text())
    if command != parser_entry.shared.build_command(args.go_sdk, build/'orbit-fhe'):
        raise ValueError('Shared build SDK and official build command differ')
    cache = build/'work/module-cache'
    if not cache.is_dir():raise ValueError('Original isolated shared module cache missing')
    audit = parser_entry.initialization_gate(args.initialization_audit, module, args.go_sdk, cache)
    return dict(optimized_source=source, shared_build_case=str(build), initialization=audit,
        expected_graph=source['expected_graph'], module=str(module), module_cache=str(cache),
        shared_build_model=(result.get('source') or {}).get('model'),
        shared_build_proves_target_model=False, target_model=source['model'],
        scope='Official parser/static scheduling for this model; per-model machine-code generation N/A')


def _clone_plain(function, namespace):
    if not isinstance(function, FunctionType) or hasattr(function, '__wrapped__'):
        raise TypeError('Only undecorated ordinary orchestration functions may be cloned')
    cloned = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


def _namespace(args):
    """Keep context managers and original module globals untouched."""
    namespace = dict(vars(parser_entry))
    namespace.update(__file__=__file__, ROUND=ROUND, MODE=MODE, CLIENT=CLIENT, source_gate=source_gate,
        shared=SimpleNamespace(MODEL=MODEL_RECORDS[args.model]['factory'],
            read=parser_entry.shared.read, native_environment=native_environment))
    for name in ('execute', 'prepare', 'append_registry'):
        namespace[name] = _clone_plain(getattr(parser_entry, name), namespace)
    underlying_prepare = namespace['prepare']

    def prepare(call_args, out, checked):
        work = underlying_prepare(call_args, out, checked)
        shutil.copyfile(parser_entry.__file__, out/'source/underlying-parser-entry.py')
        return work

    namespace['prepare'] = prepare
    return namespace


def execute(args):
    return _namespace(args)['execute'](args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', choices=tuple(MODEL_RECORDS), required=True)
    for name in ('out', 'source-case', 'shared-build-case', 'orbit-root', 'go-sdk', 'initialization-audit'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registry', type=Path);parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):setattr(args, name, value.expanduser().resolve())
    result = execute(args)
    print(json.dumps({key: result[key] for key in ('model', 'status', 'error',
        'compiler_graph_completed', 'backend_parser_consumption_verified')}))
    return int(result['status'] not in ('ok', 'prepared'))


if __name__ == '__main__':raise SystemExit(main())
