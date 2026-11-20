"""R14b outer owner for one saved AlexNet Emit-only continuation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .. import paths
from .r10_paper_metrics import (StageRecorder, UnsupportedSupervision, read_events,
                               run_process, summarize_events, write_json)
from .r14_common_preflight import DependencyMissing, PhaseStopped, _case_size, _git_state


INTERNAL_SECONDS = 1800
OUTER_SECONDS = 1860
MEMORY_GIB = 4
CASE_BUDGET_BYTES = 4 * 1024**3
MODE = 'alex-emit-resume'
MODEL = 'pt_agent.experiments.r13_common_models:alexnet_silu_avgpool'
STAGE_FIELDS = ('source_checked', 'typecheck_passed', 'signature_passed',
                'stats_equal', 'emit_completed', 'lowering_ready')
PRIOR_STAGE_FIELDS = ('reference_passed', 'calibration_completed', 'plan_completed',
                      'count_completed')
ZERO_FIELDS = ('reference_attempts', 'calibration_attempts', 'plan_attempts',
               'count_attempts', 'llm_calls', 'real_ckks_attempts',
               'real_ckks_completed', 'trace_runs', 'compiler_runs', 'keygen_runs',
               'model_imports', 'downloads', 'installations')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance')


def finalize_result(*, worker, processes, timing, status, error, git_state, out) -> dict:
    record = worker or {}
    metrics = {key: record.get(key) for key in STAGE_FIELDS + PRIOR_STAGE_FIELDS}
    metrics.update({key: record.get(key, 0) for key in ZERO_FIELDS})
    metrics.update({key: None for key in UNMEASURED_FIELDS})
    metrics.update(status=status, emit_attempts=record.get('emit_attempts', 0),
                   count_stats=record.get('count_stats'), emit_stats=record.get('emit_stats'))
    metrics['phase_elapsed_ns'] = {
        span['event_id']: dict(stage_id=span['stage_id'],
                              inclusive_ns=span.get('inclusive_ns'),
                              exclusive_ns=span.get('exclusive_ns'))
        for span in timing.get('spans', [])}
    return dict(run_id=Path(out).name, round='R14b', mode=MODE,
                model=record.get('model'), status=status, error=error, git_rev=git_state,
                worker=record, processes=processes, timing=timing, metrics=metrics)


def _observe_budget(out, stage, observations):
    observation = dict(stage=stage, boundary='outer', **_case_size(out),
                       limit_bytes=CASE_BUDGET_BYTES)
    observation['within_budget'] = observation['bytes'] <= CASE_BUDGET_BYTES
    observations.append(observation)
    if not observation['within_budget']:
        raise PhaseStopped('artifact_budget_exceeded',
                           'The complete R14b case exceeds its 4 GiB artifact allowance')


def _source_paths(source_run):
    source_run = Path(source_run).resolve()
    result_path = source_run / 'result.json'
    if not result_path.is_file():
        raise DependencyMissing('Missing original R14 result: ' + str(result_path))
    previous = json.loads(result_path.read_text())
    worker = previous.get('worker') or {}
    if (previous.get('round') != 'R14' or previous.get('model') != MODEL
            or previous.get('status') != 'artifact_budget_exceeded'
            or previous.get('mode') != 'common-model-rule-preparation'
            or worker.get('model') != MODEL or worker.get('round') != 'R14'
            or worker.get('status') != 'artifact_budget_exceeded'
            or worker.get('failed_phase') != 'emit_write'):
        raise PhaseStopped('source_failed', 'R14b requires the original AlexNet prewrite budget stop')
    selected = {'source_result': result_path}
    artifacts = worker.get('artifacts') or {}
    for key in ('program', 'count', 'emit_stats'):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            raise PhaseStopped('source_failed', 'Missing original R14 artifact: ' + key)
        path = Path(value).resolve()
        if not path.is_relative_to(source_run):
            raise PhaseStopped('source_failed', 'Original artifact must remain inside R14 source run: ' + key)
        selected[key] = path
    prior = previous.get('source_r13') or {}
    r13_value = prior.get('result')
    info_value = (prior.get('artifacts') or {}).get('model_info')
    if not isinstance(r13_value, str) or not isinstance(info_value, str):
        raise PhaseStopped('source_failed', 'Missing original R13 result or saved ModelInfo path')
    r13_result = Path(r13_value).resolve()
    r13_run = Path(prior.get('run') or r13_result.parent).resolve()
    model_info = Path(info_value).resolve()
    if (r13_result != r13_run / 'result.json' or not model_info.is_relative_to(r13_run)
            or Path(worker.get('r13_source_result', '')).resolve() != r13_result):
        raise PhaseStopped('source_failed', 'Original R13 result or ModelInfo path differs')
    commands = [item.get('command') for item in previous.get('processes', []) if isinstance(item, dict)]
    if len(commands) != 1 or not isinstance(commands[0], list):
        raise PhaseStopped('source_failed', 'Expected exactly one original worker command')
    command = commands[0]
    if command.count('--frontend') != 1:
        raise PhaseStopped('source_failed', 'Original command must contain one --frontend option')
    index = command.index('--frontend') + 1
    if index >= len(command) or not isinstance(command[index], str) or not command[index]:
        raise PhaseStopped('source_failed', 'Missing original frontend argument')
    selected.update(r13_result=r13_result, model_info=model_info,
                    frontend=Path(command[index]).resolve())
    return previous, selected


def _command(source, work):
    command = ['-m', 'pt_agent.experiments.r14b_emit_worker']
    for option, key in (('--source-result', 'source_result'), ('--program', 'program'),
                        ('--count', 'count'), ('--emit-stats', 'emit_stats'),
                        ('--r13-result', 'r13_result'), ('--model-info', 'model_info'),
                        ('--frontend', 'frontend')):
        command.extend([option, str(source[key])])
    return command + ['--work-dir', str(work)]


def _write_terminal_result(out, result, observations):
    destination = out / 'result.json'
    base = _case_size(out)['bytes'] - (destination.stat().st_size if destination.exists() else 0)
    terminal = dict(stage='terminal', boundary='outer', bytes=base,
                    limit_bytes=CASE_BUDGET_BYTES, within_budget=True)
    observations.append(terminal)
    result['case_budget_observations'] = observations
    for _ in range(8):
        total = base + len((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        within = total <= CASE_BUDGET_BYTES
        changed = (terminal['bytes'] != total or terminal['within_budget'] != within
                   or result['metrics']['status'] != result['status'])
        terminal.update(bytes=total, within_budget=within)
        result['case_artifact_bytes'] = total
        result['metrics'].update(case_artifact_bytes=total, case_artifact_budget_passed=within)
        if not within and result['status'] != 'artifact_budget_exceeded':
            result['status_before_artifact_budget'] = result['status']
            result.update(status='artifact_budget_exceeded',
                          error='The complete R14b case exceeds its 4 GiB artifact allowance')
            changed = True
        result['metrics']['status'] = result['status']
        if not changed:
            write_json(destination, result)
            return
    raise RuntimeError('R14b terminal artifact accounting did not settle')


def execute(args) -> dict:
    root = paths.RESULTS_DIR.resolve()
    out = Path(args.out).absolute()
    if out.resolve() == root or not out.resolve().is_relative_to(root):
        raise ValueError('--out must name a new child directory of RESULTS_DIR')
    source_run = Path(args.source_run).resolve()
    if out.resolve().is_relative_to(source_run):
        raise ValueError('--out must remain separate from the preserved R14 source run')
    # Protect the saved R13 run before creating any new directory there.
    source_path = source_run / 'result.json'
    if source_path.is_file():
        try:
            hint = json.loads(source_path.read_text()).get('source_r13') or {}
        except (OSError, ValueError, AttributeError):
            hint = {}
        if isinstance(hint, dict) and isinstance(hint.get('run'), str):
            if out.resolve().is_relative_to(Path(hint['run']).resolve()):
                raise ValueError('--out must remain separate from the preserved R13 source run')
    if out.exists():
        raise ValueError('Existing run directories are preserved')
    out.mkdir(parents=True)
    work = out / 'worker'
    work.mkdir()
    (out / 'resources.jsonl').touch()
    recorder = StageRecorder(out / 'events.jsonl')
    processes, child_events, observations = [], [], []
    worker, worker_read_error, previous, source = None, None, None, {}
    status, error = 'ok', None
    started_ns = time.perf_counter_ns()
    try:
        with recorder.span('turnaround', source_run=str(source_run)):
            with recorder.span('preflight'):
                _observe_budget(out, 'preflight_start', observations)
                previous, source = _source_paths(source_run)
                interpreter = paths.INTAKE_PYTHON.absolute()
                required = list(source.values()) + [source['model_info'] / 'model_info.json']
                missing = [str(path) for path in required if not path.exists()]
                if not interpreter.is_file() and not interpreter.is_symlink():
                    missing.append(str(interpreter))
                if missing:
                    raise DependencyMissing('Missing local prerequisites: ' + ', '.join(missing))
                if shutil.disk_usage(out).free < 10 * 1024**3:
                    raise DependencyMissing('At least 10 GiB free disk space is required')
                _observe_budget(out, 'preflight_end', observations)
            argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun',
                    '--seconds', str(INTERNAL_SECONDS), '--python', str(interpreter),
                    '--', *_command(source, work)]
            process = None
            with recorder.span('emit_resume', wall_seconds=OUTER_SECONDS, model=MODEL) as span:
                try:
                    process = run_process(
                        argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=out / 'worker.log',
                        resources=out / 'resources.jsonl', timeout=OUTER_SECONDS,
                        mem_gib=MEMORY_GIB, fsize_gib=8,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                             'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
                    processes.append(dict(label='emit_resume', command=argv, **process))
                finally:
                    for event in read_events(work / 'events.jsonl'):
                        if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                            event['parent_id'] = span['event_id']
                        child_events.append(event)
                    result_path = work / 'worker_result.json'
                    if result_path.is_file():
                        try:
                            worker = json.loads(result_path.read_text())
                            if not isinstance(worker, dict):
                                worker = None
                                worker_read_error = 'Worker result must be a JSON object'
                        except (OSError, ValueError) as exc:
                            worker_read_error = f'{type(exc).__name__}: {exc}'[:2000]
                _observe_budget(out, 'worker_return', observations)
                if process is None:
                    raise PhaseStopped('process_failed', 'Worker process observation is unavailable')
                if process['timed_out']:
                    raise PhaseStopped('timeout', 'The outer R14b process exceeded 1860 seconds')
                if process['signal'] or process['remaining_pids']:
                    raise PhaseStopped('process_failed', str({key: process[key] for key in
                                                            ('exit_code', 'signal', 'remaining_pids')}))
                if worker is None:
                    raise PhaseStopped('result_missing', worker_read_error or 'Worker did not write its result')
                if worker.get('status') != 'ok':
                    raise PhaseStopped(worker.get('status', 'error'), str(worker.get('error')))
                if process['exit_code'] != 0:
                    raise PhaseStopped('process_failed', 'Worker exit code: ' + str(process['exit_code']))
                if (worker.get('round') != 'R14b' or worker.get('model') != MODEL
                        or any(worker.get(key) is not True for key in STAGE_FIELDS)
                        or worker.get('emit_attempts') != 1
                        or any(worker.get(key) != 0 for key in ZERO_FIELDS)
                        or any(worker.get(key) is not None for key in UNMEASURED_FIELDS + PRIOR_STAGE_FIELDS)):
                    raise PhaseStopped('incomplete', 'Successful worker result is missing required observations')
    except DependencyMissing as exc:
        status, error = 'dependency_missing', str(exc)
    except PhaseStopped as exc:
        status, error = exc.status, str(exc)
    except UnsupportedSupervision as exc:
        status, error = 'unsupported_process_supervision', str(exc)
    except BaseException as exc:
        status, error = 'error', f'{type(exc).__name__}: {exc}'[:2000]
    finally:
        ended_ns = time.perf_counter_ns()
        timing = summarize_events(read_events(out / 'events.jsonl') + child_events,
                                  termination_observed_ns=ended_ns)
        timing['outer_interval'] = dict(start_ns=started_ns, end_ns=ended_ns,
                                        elapsed_ns=ended_ns - started_ns)
        if status == 'ok' and not timing.get('attribution_ready'):
            status, error = 'timing_invalid', str(timing.get('diagnostics'))
        with (out / 'stages.jsonl').open('w') as stream:
            for stage in timing['spans']:
                stream.write(json.dumps(stage, ensure_ascii=False) + '\n')
        result = finalize_result(worker=worker, processes=processes, timing=timing,
                                 status=status, error=error, git_state=_git_state(), out=out)
        result.update(model=(previous or {}).get('model'), worker_result_read_error=worker_read_error)
        result['source_r14'] = dict(run=str(source_run), result=str(source_path),
                                   artifacts={key: str(value) for key, value in source.items()},
                                   prior_timing=(previous or {}).get('timing'),
                                   included_in_current_turnaround=False)
        _write_terminal_result(out, result, observations)
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round='R14b', model=result['model'],
                                         mode=MODE, git_rev=result['git_rev'], metrics=result['metrics'],
                                         artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()),
                                    ensure_ascii=False) + '\n')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = execute(args)
    print(json.dumps(dict(status=result['status'], out=str(args.out)), ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
