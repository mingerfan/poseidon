"""Inner, path-driven workers for the R10 paper diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys


class WorkerFailure(RuntimeError):
    def __init__(self, message, *, status='error'):
        super().__init__(message)
        self.status = status


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def _event_records(path):
    if not Path(path).is_file():
        return []
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # The outer event summarizer retains and diagnoses a partial line.
            continue
    return records


def _real_probes_passed(events_path):
    attempted = [event for event in _event_records(events_path)
                 if event.get('type') in ('begin', 'end')
                 and event.get('stage_id') == 'intake.real_probe']
    if not attempted:
        return None
    completed = [event for event in attempted
                 if event.get('type') == 'end'
                 and event.get('status') == 'ok']
    return len(completed) == 3


def check_qwen_identity(imported) -> dict:
    """Record the original config and reject a cropped Qwen decoder."""
    try:
        cfg = imported.original.config
        config_layers = cfg.num_hidden_layers
        actual_layers = len(imported.original.module.model.layers)
    except (AttributeError, TypeError) as exc:
        raise ValueError('Qwen config and actual decoder layers are required') from exc
    if config_layers != 24 or actual_layers != 24:
        raise ValueError(
            f'Qwen must retain all 24 layers: config={config_layers}, actual={actual_layers}')
    metadata = {'config_layers': config_layers, 'actual_layers': actual_layers}
    for key in ('model_type', 'architectures', 'hidden_size', 'intermediate_size',
                'num_attention_heads', 'num_key_value_heads', 'vocab_size',
                'tie_word_embeddings', 'max_position_embeddings', 'rope_theta',
                'torch_dtype', 'use_cache', 'use_sliding_window'):
        if hasattr(cfg, key):
            metadata[key] = getattr(cfg, key)
    hidden = metadata.get('hidden_size')
    heads = metadata.get('num_attention_heads')
    if isinstance(hidden, int) and isinstance(heads, int) and heads and hidden % heads == 0:
        metadata['head_dim'] = hidden // heads
    if hasattr(cfg, 'to_dict'):
        metadata['original_config'] = cfg.to_dict()
    return metadata


def _prepare_text(args, recorder, result):
    import pyarrow.parquet as parquet

    source = args.inputs.resolve()
    if source.suffix.lower() != '.parquet':
        raise ValueError('prepare-text requires the original training Parquet')
    if source.name.lower().startswith(('test', 'validation')):
        raise ValueError('prepare-text requires training sentences')
    output = args.work_dir / 'samples.jsonl'
    rows_path = args.work_dir / 'source_rows.json'
    count = 0
    with recorder.span('prepare_text', artifact=str(output)):
        reader = parquet.ParquetFile(source)
        with output.open('x') as stream:
            for batch in reader.iter_batches(batch_size=223, columns=['sentence']):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text, str):
                        raise ValueError('Training sentence column contains a non-string value')
                    stream.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                    if count == 223:
                        break
                if count == 223:
                    break
        if count != 223:
            raise ValueError(f'Expected 223 training sentences, found {count}')
        _write_json(rows_path, {'source': str(source), 'column': 'sentence',
                               'row_number_base': 0, 'source_rows': list(range(count))})
    result.update(inputs_jsonl=str(output), source_rows=str(rows_path), sample_count=count,
                  artifacts={'samples': str(output), 'source_rows': str(rows_path)})


def _shape_record(info, metadata):
    graph = info.graph
    inputs = {name: list(graph.tensors[name].shape) for name in graph.inputs}
    outputs = {name: list(graph.tensors[name].shape) for name in graph.outputs}
    length = max(info.sequence['buckets'])
    hidden = metadata['hidden_size']
    vocab = metadata['vocab_size']
    expected = {'inputs_embeds': [1, length, hidden], 'attention_mask': [1, length]}
    if 'sel' in inputs:
        expected['sel'] = [1, length]
    expected_length = 1 if 'sel' in inputs else length
    ok = (inputs == expected and len(outputs) == 1
          and next(iter(outputs.values()), None) == [1, expected_length, vocab])
    return {'ok': ok, 'inputs': inputs, 'outputs': outputs,
            'expected_inputs': expected, 'expected_output': [1, expected_length, vocab]}


def _qwen_import(args, recorder, result):
    from ..data.samples import read
    from ..intake import coverage
    from ..intake.probe import import_model

    with recorder.span('read_samples', artifact=str(args.inputs)):
        samples = read(args.inputs)
    if samples.kind != 'text' or samples.labels is not None or len(samples) != 223:
        raise ValueError('Qwen import resource check requires 223 original unlabelled texts')
    with recorder.span('import_model', artifact=str(args.model)):
        result['import_completed'] = False
        imported = import_model(args.model, samples, observer=recorder)
    info = imported.info
    graph = info.graph
    with recorder.span('import_diagnostics'):
        metadata = check_qwen_identity(imported)
        processor = imported.original.processor
        metadata['tokenizer_model_max_length'] = getattr(processor, 'model_max_length', None)
        metadata['tokenizer_pad_token_id'] = getattr(processor, 'pad_token_id', None)
        metadata['tokenizer_eos_token_id'] = getattr(processor, 'eos_token_id', None)
        metadata['padding_side'] = getattr(processor, 'padding_side', None)
        metadata['truncation_side'] = getattr(processor, 'truncation_side', None)
        errors = list(getattr(graph, 'lowering_errors', ()))
        coverage_report = coverage.report(graph)
        coverage_check = coverage.check(graph, slots=16384)
        normalized = bool(getattr(graph, 'intake_stats', {}).get('transformer_normalized')) and not errors
        shapes = _shape_record(info, metadata)
        weights_nonempty = bool(graph.weights) and any(value.size > 0 for value in graph.weights.values())
        probes_passed = _real_probes_passed(args.work_dir / 'events.jsonl')
        diagnostics = {
            'import_completed': False, 'real_probes_passed': probes_passed,
            'normalization_completed': normalized, 'lowering_errors': errors,
            'coverage_report': coverage_report, 'coverage_check': coverage_check,
            'coverage_slots': 16384, 'static_coverage_ready': normalized and coverage_check['ok'],
            'lowering_ready': None, 'model_metadata': metadata,
            'intake_stats': getattr(graph, 'intake_stats', {}), 'shape_check': shapes,
            'weights_nonempty': weights_nonempty, 'weight_count': len(graph.weights),
            'weight_bytes': sum(value.nbytes for value in graph.weights.values()),
            'sequence': info.sequence,
            'split_counts': {key: len(info.split[key])
                             for key in ('calibration', 'heldout', 'confirm', 'report')},
            'evaluated_splits': [],
        }
        diagnostics_path = args.work_dir / 'import_diagnostics.json'
        _write_json(diagnostics_path, diagnostics)
        result.update(diagnostics)
        result['artifacts'] = {'import_diagnostics': str(diagnostics_path)}
    saved = args.work_dir / 'model_info'
    with recorder.span('save_model_info', artifact=str(saved)):
        info.save(saved)
        required = [saved / 'model_info.json', saved / 'graph/graph.json', saved / 'graph/weights.npz']
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            raise WorkerFailure('ModelInfo/Graph/weights were not fully saved')
    result['artifacts']['model_info'] = str(saved)
    result['import_completed'] = bool(probes_passed and weights_nonempty and shapes['ok'])
    if not result['import_completed']:
        raise WorkerFailure('Saved import did not satisfy real-probe, weight or shape checks')


def _compiler_limit(args, recorder, result):
    from .. import toolchain
    from ..manage.streams import resolve

    record_path = args.export_record.resolve()
    with recorder.span('read_export_record', artifact=str(record_path)):
        report = json.loads(record_path.read_text())
        export = report.get('export', {})
        if not isinstance(export, dict) or not export.get('package') or not export.get('weights'):
            raise WorkerFailure('REPORT must contain export.package and export.weights',
                                status='unsupported_contract')
        package, weights = Path(export['package']), Path(export['weights'])
        if not package.is_absolute() or not weights.is_absolute():
            raise WorkerFailure('Export record must provide explicit package and weights paths',
                                status='unsupported_contract')
        config = json.loads((package / 'config/model.json').read_text())
        if (config.get('name'), config.get('compile_mode'), config.get('boot_profile')) != ('mlp', 'pars', 'qwen48'):
            raise WorkerFailure('Engineering check requires the existing mlp/pars/qwen48 package',
                                status='unsupported_contract')
        if not weights.is_file():
            raise FileNotFoundError(weights)
        for filename in ('trace_model.py', 'compile_model.py'):
            if not (package / filename).is_file():
                raise FileNotFoundError(package / filename)
        result.update(export_record=str(record_path), source_package=str(package),
                      source_weights=str(weights), compile_mode='pars', boot_profile='qwen48')
    toolchain.setup_env()
    trace_dir, compile_dir = args.work_dir / 'trace', args.work_dir / 'compiled'
    trace_command = [sys.executable, '-B', str(package / 'trace_model.py'), '--weights', str(weights),
                     '--output-dir', str(trace_dir)]
    result['trace_command'] = trace_command
    with recorder.span('trace', wall_seconds=3600, artifact=str(trace_dir)):
        trace = toolchain.trusted_run(args.work_dir, trace_command, args.work_dir / 'trace.log',
                                     env_extra=toolchain.package_trace_env(), wall_seconds=3600)
        result['trace'] = trace
        mlir = resolve(trace_dir / 'trace_model.mlir')
        trace.update(mlir=str(mlir), mlir_exists=mlir.is_file(),
                     mlir_bytes=mlir.stat().st_size if mlir.is_file() else None)
        if trace['exit'] != 0 or not mlir.is_file():
            raise WorkerFailure('Trusted package trace failed',
                                status='timeout' if trace.get('timeout') else 'error')
    compile_command = [sys.executable, '-B', str(package / 'compile_model.py'),
                       '--hecate-opt', str(toolchain.host_paths()['hecate_opt']),
                       '--input', str(mlir), '--output-dir', str(compile_dir)]
    result['compile_command'] = compile_command
    with recorder.span('compile', wall_seconds=7200, artifact=str(compile_dir)):
        compiled = toolchain.trusted_run(
            args.work_dir, compile_command, args.work_dir / 'compile.log',
            env_extra={'LD_LIBRARY_PATH': os.environ['HECATE_PYTHON_LIBRARY_PATH']}, wall_seconds=7200)
        result['compile'] = compiled
        plan_path = compile_dir / 'lowered._hecate_mlp.runtime-plan.json'
        compiled['plan_exists'] = plan_path.is_file()
        if compiled['exit'] != 0 or not plan_path.is_file():
            raise WorkerFailure('Trusted package compilation failed',
                                status='timeout' if compiled.get('timeout') else 'error')
    with recorder.span('read_runtime_plan', artifact=str(plan_path)):
        result['plan_stats'] = toolchain.plan_stats(compile_dir, function='mlp')
        result['compiled_bootstraps'] = result['plan_stats']['boots']
        result['artifacts'] = {'runtime_plan': str(plan_path), 'trace': str(mlir)}


def _has_memory_error(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, MemoryError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare-text', 'qwen-import', 'compiler-limit'), required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--export-record', type=Path)
    args = parser.parse_args(argv)
    for name in ({'prepare-text': ('inputs',), 'qwen-import': ('inputs', 'model'),
                  'compiler-limit': ('export_record',)}[args.phase]):
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required for ' + args.phase)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.work_dir / 'worker_result.json'
    if result_path.exists():
        raise FileExistsError('Existing worker result is preserved: ' + str(result_path))
    result = {'phase': args.phase, 'status': 'error', 'pid': os.getpid(), 'ppid': os.getppid(),
              'llm_calls': 0, 'ckks_runs': 0, 'lowering_ready': None, 'import_completed': None,
              'real_probes_passed': None, 'normalization_completed': None,
              'static_coverage_ready': None, 'compiled_bootstraps': None,
              'predicted_latency': None, 'candidates': None}
    try:
        from .r10_paper_metrics import StageRecorder
        recorder = StageRecorder(args.work_dir / 'events.jsonl')
        with recorder.span('worker', phase=args.phase, pid=os.getpid(), ppid=os.getppid()):
            worker = {'prepare-text': _prepare_text, 'qwen-import': _qwen_import,
                      'compiler-limit': _compiler_limit}[args.phase]
            worker(args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = ('resource_limit' if _has_memory_error(exc) else
                            'dependency_missing' if isinstance(exc, ModuleNotFoundError) else
                            getattr(exc, 'status', 'error'))
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if args.phase == 'qwen-import':
            result['real_probes_passed'] = _real_probes_passed(args.work_dir / 'events.jsonl')
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write_json(result_path, result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
