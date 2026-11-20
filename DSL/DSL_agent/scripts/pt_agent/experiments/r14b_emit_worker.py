"""R14b: resume only Emit from the preserved Alex R14 prewrite stop."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import resource
import signal
from types import SimpleNamespace

from . import r14_common_worker as previous
from .r10_paper_metrics import StageRecorder, read_events, write_json
from .r13_common_models import FACTORIES


INTERNAL_SECONDS = 1800
CASE_BYTES = 4 * 1024**3
EMIT_BYTES = 7 * 1024**3 // 2
MODEL = next(key for key, value in FACTORIES.items() if value['name'] == 'alexnet_silu_avgpool')
STAT_FIELDS = ('counts', 'ops_total', 'distinct_consts')
COUNT_KINDS = ('add', 'sub', 'neg', 'mul_cc', 'mul_cp', 'rot', 'boot')
PRIOR_STAGES = ('reference_passed', 'calibration_completed', 'plan_completed',
                'typecheck_passed', 'signature_passed', 'count_completed', 'stats_equal')
PRIOR_ATTEMPTS = ('calibration_attempts', 'plan_attempts', 'count_attempts', 'emit_attempts')
REQUIRED_STAGES = ('source_checked', 'typecheck_passed', 'signature_passed',
                   'stats_equal', 'emit_completed', 'lowering_ready')
WorkerFailure = previous.WorkerFailure


def validate_resume_source(source_result, source_count, source_emit_stats, r13_result) -> dict:
    """Validate saved observations; this function never loads a model or sample."""
    errors = []
    if not all(isinstance(record, dict) for record in
               (source_result, source_count, source_emit_stats, r13_result)):
        return dict(passed=False, model=None, errors=['Resume records must be JSON objects'])
    worker = source_result.get('worker') or {}
    r13_worker = r13_result.get('worker') or {}
    if not isinstance(worker, dict) or not isinstance(r13_worker, dict):
        return dict(passed=False, model=None, errors=['Source workers must be JSON objects'])
    if (source_result.get('round') != 'R14' or worker.get('round') != 'R14'
            or source_result.get('model') != MODEL or worker.get('model') != MODEL
            or source_result.get('status') != 'artifact_budget_exceeded'
            or worker.get('status') != 'artifact_budget_exceeded'
            or worker.get('failed_phase') != 'emit_write'):
        errors.append('Expected the preserved original Alex R14 prewrite budget stop')
    if any(worker.get(name) is not True for name in PRIOR_STAGES):
        errors.append('The original reference, calibration, rule, checks and counts must have passed')
    if worker.get('emit_completed') is not None or worker.get('lowering_ready') is not None:
        errors.append('The original Emit write and readback must remain incomplete')
    if any(type(worker.get(name)) is not int or worker[name] != 1 for name in PRIOR_ATTEMPTS):
        errors.append('Each original preparation stage must have exactly one recorded attempt')
    if worker.get('rule_options') != previous.RULE_OPTIONS:
        errors.append('The saved R14 rule options differ')
    if (r13_result.get('round') != 'R13' or r13_result.get('status') != 'ok'
            or r13_result.get('mode') != 'common-model-intake-preparation'
            or r13_result.get('model') != MODEL or r13_worker.get('round') != 'R13'
            or r13_worker.get('status') != 'ok' or r13_worker.get('model') != MODEL
            or any(r13_worker.get(name) is not True for name in
                   ('import_completed', 'real_probes_passed', 'graph_math_passed',
                    'roundtrip_passed', 'static_coverage'))):
        errors.append('The original successful Alex R13 record is required')
    count = source_count.get('stats')
    records = (('count.json', count), ('worker.count_stats', worker.get('count_stats')),
               ('emit_stats.json', source_emit_stats.get('stats')),
               ('worker.emit_stats', worker.get('emit_stats')))
    for name, stats in records:
        if not isinstance(stats, dict) or any(key not in stats for key in STAT_FIELDS):
            errors.append('Missing saved statistics: ' + name)
            continue
        counts = stats['counts']
        if (not isinstance(counts, dict) or set(counts) != set(COUNT_KINDS)
                or any(type(value) is not int or value < 0 for value in counts.values())
                or type(stats['ops_total']) is not int or stats['ops_total'] != sum(counts.values())
                or type(stats['distinct_consts']) is not int or stats['distinct_consts'] < 0):
            errors.append('Invalid saved operation statistics: ' + name)
        if isinstance(count, dict) and any(stats.get(key) != count.get(key) for key in STAT_FIELDS):
            errors.append('Saved Count and Emit statistics differ: ' + name)
    return dict(passed=not errors, model=MODEL, errors=errors,
                prior_stages={key: worker.get(key) for key in PRIOR_STAGES},
                prior_attempts={key: worker.get(key) for key in PRIOR_ATTEMPTS},
                prior_failed_phase=worker.get('failed_phase'))


def _checked_paths(args, source, r13):
    """Bind argv to the original artifact locations without file digests."""
    source_dir = args.source_result.resolve().parent
    source_worker = source.get('worker') or {}
    artifacts = source_worker.get('artifacts') or {}
    source_r13 = source.get('source_r13') or {}
    r13_dir = args.r13_result.resolve().parent
    for name, actual in (('program', args.program), ('count', args.count), ('emit_stats', args.emit_stats)):
        value = artifacts.get(name)
        if (not isinstance(value, str) or Path(value).resolve() != actual.resolve()
                or not actual.resolve().is_relative_to(source_dir)):
            raise WorkerFailure('Original R14 artifact location differs: ' + name, status='source_failed')
    if (source_r13.get('result') != str(args.r13_result.resolve())
            or source_worker.get('r13_source_result') != str(args.r13_result.resolve())
            or source_r13.get('run') != str(r13_dir)):
        raise WorkerFailure('Original R13 result location differs', status='source_failed')
    info = ((r13.get('worker') or {}).get('artifacts') or {}).get('model_info')
    original_info = (source_r13.get('artifacts') or {}).get('model_info')
    if (not isinstance(info, str) or Path(info).resolve() != args.model_info.resolve()
            or not args.model_info.resolve().is_relative_to(r13_dir)
            or original_info is None or Path(original_info).resolve() != args.model_info.resolve()):
        raise WorkerFailure('Original saved ModelInfo location differs', status='source_failed')
    commands = [process.get('command') for process in source.get('processes', [])
                if isinstance(process, dict) and isinstance(process.get('command'), list)
                and 'pt_agent.experiments.r14_common_worker' in process['command']]
    if len(commands) != 1 or commands[0].count('--frontend') != 1:
        raise WorkerFailure('Expected one original worker command with one frontend option', status='source_failed')
    command = commands[0]
    index = command.index('--frontend')
    if (index + 1 == len(command) or not isinstance(command[index + 1], str)
            or Path(command[index + 1]).resolve() != args.frontend.resolve()):
        raise WorkerFailure('Original frontend location differs', status='source_failed')
    for path in (args.program, args.count, args.emit_stats, args.r13_result, args.frontend):
        if not path.is_file():
            raise FileNotFoundError('Missing resume prerequisite: ' + str(path))
    return dict(source_result=str(args.source_result.resolve()),
                program=str(args.program.resolve()), count=str(args.count.resolve()),
                emit_stats=str(args.emit_stats.resolve()), r13_result=str(args.r13_result.resolve()),
                model_info=str(args.model_info.resolve()), frontend=str(args.frontend.resolve()))


def _observe_budget(args, result, stage, boundary, *, enforce=True):
    size = previous._case_size(args.work_dir.parent)
    result['case_artifact_bytes'] = size
    result['case_budget_observations'].append(dict(stage=stage, boundary=boundary, bytes=size,
                                                  limit_bytes=CASE_BYTES, within_budget=size <= CASE_BYTES))
    if enforce and size > CASE_BYTES:
        raise WorkerFailure('The R14b case exceeded 4 GiB of saved artifacts', status='artifact_budget_exceeded')


@contextmanager
def _phase(args, recorder, result, name):
    _observe_budget(args, result, name, 'before')
    write_json(args.work_dir / 'worker_result.json', result)
    try:
        with recorder.span(name):
            yield
    except BaseException:
        _observe_budget(args, result, name, 'after', enforce=False)
        write_json(args.work_dir / 'worker_result.json', result)
        raise
    else:
        _observe_budget(args, result, name, 'after')
        write_json(args.work_dir / 'worker_result.json', result)


def _emit_budget(text, constants):
    observation = previous._emit_budget(text, constants)
    observation.update(limit_bytes=EMIT_BYTES, within_budget=observation['estimated_bytes'] <= EMIT_BYTES)
    return observation


def _strategy_check(program):
    header = program.header
    errors = []
    if (header.slots != 16384 or header.refresh != 'compiler' or header.placement != 'depth-dp'
            or header.manage != 'compiler' or header.l_max is not None or header.boot_out_level is not None):
        errors.append('The original rule header differs')
    for operation in program.ops:
        if operation.kind == 'act' and (operation.attrs.get('fn') != 'silu'
                or operation.attrs.get('method') != 'cheb' or operation.attrs.get('deg') != 63):
            errors.append('The original SiLU cheb63 strategy differs at op ' + str(operation.id))
    return dict(passed=not errors, errors=errors, header=asdict(header),
                rule_options=dict(previous.RULE_OPTIONS))


def resume_emit(args, recorder, result) -> None:
    import numpy as np
    from ..backends import EmitBackend
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..lower import lower
    from ..models.info import ModelInfo

    def save(name, value):
        path = args.work_dir / (name + '.json')
        write_json(path, value)
        result['artifacts'][name] = str(path)

    with _phase(args, recorder, result, 'source_check'):
        source = json.loads(args.source_result.read_text())
        count_record = json.loads(args.count.read_text())
        emit_record = json.loads(args.emit_stats.read_text())
        r13 = json.loads(args.r13_result.read_text())
        observation = validate_resume_source(source, count_record, emit_record, r13)
        result['model'] = observation['model']
        save('resume_source', observation)
        if not observation['passed']:
            result['source_checked'] = False
            raise WorkerFailure('Saved resume source checks failed: ' + '; '.join(observation['errors']),
                                status='source_failed')
        source_paths = _checked_paths(args, source, r13)
        result['source_resume'] = dict(**source_paths, prior_stages=observation['prior_stages'],
                                     prior_attempts=observation['prior_attempts'],
                                     prior_failed_phase=observation['prior_failed_phase'],
                                     prior_timing=source.get('timing'),
                                     included_in_current_turnaround=False)
        result['count_stats'] = count_record['stats']
        result['r13_source_result'] = str(args.r13_result.resolve())
        result['r13_timing'] = r13.get('timing')
    with _phase(args, recorder, result, 'load_saved_model'):
        info = ModelInfo.load(args.model_info)
        checked_source = previous.check_saved_source(info, r13)
        save('source_check', checked_source)
        result['source_checked'] = checked_source['passed']
        if not checked_source['passed']:
            raise WorkerFailure('Original saved ModelInfo checks failed', status='source_failed')
        for key in ('source', 'inputs', 'split', 'assumptions', 'task', 'metric', 'threshold'):
            result[{'source': 'model_source', 'inputs': 'input_metadata'}.get(key, key)] = getattr(info, key)
    with _phase(args, recorder, result, 'parse_typecheck'):
        text = args.program.read_text()
        program = parse(text)
        checked = check(program, {name: weight.shape for name, weight in info.graph.weights.items()}, graph=info.graph)
        save('typecheck', dict(ok=checked.ok, diags=[asdict(diag) for diag in checked.diags]))
        result['typecheck_passed'] = checked.ok
        if not checked.ok:
            raise WorkerFailure('The saved rule program failed typechecking', status='typecheck_failed')
        strategy = _strategy_check(program)
        save('rule_strategy', strategy)
        if not strategy['passed']:
            raise WorkerFailure('The saved rule strategy differs', status='source_failed')
        result['rule_options'] = dict(previous.RULE_OPTIONS)
    with _phase(args, recorder, result, 'signature_check'):
        signature = previous.check_program_signature(info, program)
        save('signature', signature)
        result['signature_passed'] = signature['passed']
        if not signature['passed']:
            raise WorkerFailure('The saved program signature differs', status='signature_failed')
    emit_dir = args.work_dir / 'emit'
    with _phase(args, recorder, result, 'emit_lower'):
        result['emit_attempts'] = 1
        write_json(args.work_dir / 'worker_result.json', result)
        emitter = EmitBackend(16384, frontend=args.frontend, constants=emit_dir / 'constants.npz', save_dir=emit_dir)
        lower(program, info.graph.weights, emitter, act_mode='program')
        emitted = emitter.stats()
        result['emit_stats'] = emitted
        count = result['count_stats']
        differences = {key: dict(count=count[key], emit=emitted[key])
                       for key in STAT_FIELDS if count[key] != emitted[key]}
        result['stats_equal'] = not differences
        save('emit_stats', dict(stats=emitted, differences=differences, matched=not differences))
        if differences:
            raise WorkerFailure('Saved Count and new Emit statistics differ', status='count_emit_mismatch')
    with _phase(args, recorder, result, 'emit_write'):
        emitted_text, constants = emitter.render()
        budget = _emit_budget(emitted_text, constants)
        result['emit_budget'] = budget
        save('emit_budget', budget)
        if not budget['within_budget']:
            raise WorkerFailure('The Emit artifact estimate exceeds 3.5 GiB', status='artifact_budget_exceeded')
        del emitted_text
        emitter.write(emit_dir)
        result['artifacts'].update(trace_source=str(emit_dir / 'candidate_trace.py'),
                                   constants=str(emit_dir / 'constants.npz'))
        expected = {name: SimpleNamespace(dtype=np.dtype(array.dtype), shape=tuple(array.shape))
                    for name, array in constants.items()}
        del constants, emitter
        gc.collect()
        result['constants_released_before_readback'] = True
    with _phase(args, recorder, result, 'metadata_readback'):
        metadata = previous._readback_emit(emit_dir, expected)
        save('emit_metadata', metadata)
        result['emit_completed'] = True
        result['lowering_ready'] = True


def _write_terminal(args, result):
    """Include the worker result itself in the terminal metadata observation."""
    result_path = args.work_dir / 'worker_result.json'
    others = previous._case_size(args.work_dir.parent) - (result_path.stat().st_size if result_path.exists() else 0)
    observation = dict(stage='worker', boundary='terminal', bytes=None,
                       limit_bytes=CASE_BYTES, within_budget=None)
    result['case_budget_observations'].append(observation)
    while True:
        encoded = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        size = others + len(encoded)
        if size > CASE_BYTES:
            result.update(status='artifact_budget_exceeded',
                          error=dict(type='WorkerFailure', message='The R14b case exceeded 4 GiB'))
        same = result['case_artifact_bytes'] == size and observation['bytes'] == size
        result['case_artifact_bytes'] = observation['bytes'] = size
        observation['within_budget'] = size <= CASE_BYTES
        if same:
            break
    write_json(result_path, result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-result', 'program', 'count', 'emit-stats', 'r13-result',
                 'model-info', 'frontend', 'work-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir / 'worker_result.json').exists():
        raise FileExistsError('Existing R14b worker results are preserved')
    result = dict(round='R14b', model=None, status='error', error=None, failed_phase=None,
                  pid=os.getpid(), ppid=os.getppid(), artifacts={}, case_budget_observations=[],
                  case_artifact_bytes=None, count_stats=None, emit_stats=None,
                  reference_passed=None, calibration_completed=None, plan_completed=None, count_completed=None,
                  compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None,
                  llm_calls=0, real_ckks_attempts=0, real_ckks_completed=0, trace_runs=0,
                  compiler_runs=0, keygen_runs=0, model_imports=0, downloads=0, installations=0,
                  reference_attempts=0, calibration_attempts=0, plan_attempts=0, count_attempts=0,
                  emit_attempts=0, constants_released_before_readback=None,
                  **dict.fromkeys(REQUIRED_STAGES))
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    old_handler = None
    try:
        def timeout(signum, frame):
            raise WorkerFailure('The R14b worker exceeded 1800 seconds', status='timeout')
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(INTERNAL_SECONDS)
        with recorder.span('worker'):
            resume_emit(args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = ('resource_limit' if previous._memory_error(exc) else
                            'dependency_missing' if isinstance(exc, (ModuleNotFoundError, FileNotFoundError)) else
                            getattr(exc, 'status', 'error'))
        result['error'] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        errors = [event for event in read_events(args.work_dir / 'events.jsonl')
                  if event.get('type') == 'end' and event.get('status') == 'error']
        result['failed_phase'] = errors[0]['stage_id'] if errors else 'worker'
    finally:
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write_terminal(args, result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
