"""R14 outer owner for one bounded saved-model rule-program preparation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .. import paths, toolchain
from .r10_paper_metrics import (StageRecorder, UnsupportedSupervision, read_events,
                               run_process, summarize_events, write_json)
from .r13_common_models import FACTORIES


INTERNAL_SECONDS = 1800
OUTER_SECONDS = 1860
MEMORY_GIB = 4
CASE_BUDGET_BYTES = 4 * 1024**3
MODE = 'common-model-rule-preparation'
STAGE_FIELDS = ('source_checked', 'reference_passed', 'calibration_completed',
                'plan_completed', 'typecheck_passed', 'signature_passed',
                'count_completed', 'stats_equal', 'emit_completed', 'lowering_ready')
ATTEMPT_FIELDS = ('calibration_attempts', 'plan_attempts', 'count_attempts', 'emit_attempts')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance')
ZERO_FIELDS = ('llm_calls', 'real_ckks_attempts', 'real_ckks_completed', 'trace_runs',
               'compiler_runs', 'keygen_runs', 'model_imports', 'downloads', 'installations')


class DependencyMissing(RuntimeError):
    pass


class PhaseStopped(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _git_state():
    def read(*command):
        return subprocess.run(['git', *command], cwd=paths.REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    return dict(revision=read('rev-parse', 'HEAD'), branch=read('branch', '--show-current'),
                working_tree_changed=bool(read('status', '--porcelain')))


def finalize_result(*, worker, processes, timing, status, error, git_state, out):
    """Retain partial observations and separate structure from unmeasured results."""
    record = worker or {}
    metrics = {name: record.get(name) for name in STAGE_FIELDS}
    metrics.update({name: record.get(name, 0) for name in ATTEMPT_FIELDS + ZERO_FIELDS})
    metrics.update({name: None for name in UNMEASURED_FIELDS})
    metrics.update(count_stats=record.get('count_stats'), emit_stats=record.get('emit_stats'),
                   reference_criterion='1e-12+1e-12*abs(reference)')
    metrics['phase_elapsed_ns'] = {
        span['event_id']: dict(stage_id=span['stage_id'],
                              inclusive_ns=span.get('inclusive_ns'),
                              exclusive_ns=span.get('exclusive_ns'))
        for span in timing.get('spans', [])}
    return dict(run_id=Path(out).name, round='R14', mode=MODE, model=record.get('model'),
                status=status, error=error, git_rev=git_state, worker=record,
                processes=processes, timing=timing, metrics=metrics)


def _case_size(out):
    files = [entry for entry in Path(out).rglob('*') if entry.is_file()]
    return dict(bytes=sum(entry.stat().st_size for entry in files), files=len(files))


def _observe_budget(out, stage, observations):
    observation = dict(stage=stage, boundary='outer', **_case_size(out),
                       limit_bytes=CASE_BUDGET_BYTES)
    observation['within_budget'] = observation['bytes'] <= CASE_BUDGET_BYTES
    observations.append(observation)
    if not observation['within_budget']:
        raise PhaseStopped('artifact_budget_exceeded',
                           'The complete R14 case exceeds its 4 GiB artifact allowance')
    return observation


def _saved_source(r13_run):
    r13_run = Path(r13_run).resolve()
    result_path = r13_run / 'result.json'
    if not result_path.is_file():
        raise DependencyMissing('Missing R13 result: ' + str(result_path))
    previous = json.loads(result_path.read_text())
    model, worker = previous.get('model'), previous.get('worker') or {}
    passed = ('import_completed', 'real_probes_passed', 'graph_math_passed',
              'roundtrip_passed', 'static_coverage')
    if (previous.get('round') != 'R13' or previous.get('status') != 'ok'
            or previous.get('mode') != 'common-model-intake-preparation'
            or model not in FACTORIES or worker.get('status') != 'ok'
            or worker.get('model') != model
            or any(worker.get(name) is not True for name in passed)):
        raise PhaseStopped('source_failed', 'R14 requires a successful original R13 model record')
    source = worker.get('model_source') or {}
    expected = FACTORIES[model]
    if (source.get('factory') != model
            or Path(source.get('checkpoint', '')).resolve() != Path(expected['checkpoint']).resolve()
            or Path(source.get('original_source', '')).resolve() != Path(expected['source']).resolve()):
        raise PhaseStopped('source_failed', 'R13 factory or original checkpoint provenance differs')
    artifacts = worker.get('artifacts') or {}
    selected = {}
    for key in ('model_info', 'training_probes', 'training_probe_source'):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            raise PhaseStopped('source_failed', 'Missing R13 artifact location: ' + key)
        selected[key] = Path(value).resolve()
        if not selected[key].is_relative_to(r13_run):
            raise PhaseStopped('source_failed', 'R13 artifact must remain inside its source run: ' + key)
    return model, previous, dict(r13_result=result_path, **selected)


def _command(source, records, selections, frontend, work):
    return ['-m', 'pt_agent.experiments.r14_common_worker',
            '--model-info', str(source['model_info']),
            '--r13-result', str(source['r13_result']),
            '--r13-probes', str(source['training_probes']),
            '--r13-probe-source', str(source['training_probe_source']),
            '--inputs', str(records), '--selections', str(selections),
            '--frontend', str(frontend), '--work-dir', str(work)]


def _write_terminal_result(out, result, observations):
    """Account for the final result's own UTF-8 size without rereading its bytes."""
    existing = out / 'result.json'
    base = _case_size(out)['bytes'] - (existing.stat().st_size if existing.exists() else 0)
    terminal = dict(stage='terminal', boundary='outer', bytes=base, limit_bytes=CASE_BUDGET_BYTES,
                    within_budget=True)
    observations.append(terminal)
    result['case_budget_observations'] = observations
    for _ in range(8):
        total = base + len((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        within = total <= CASE_BUDGET_BYTES
        changed = terminal['bytes'] != total or terminal['within_budget'] != within
        terminal.update(bytes=total, within_budget=within)
        result['case_artifact_bytes'] = total
        result['metrics']['case_artifact_bytes'] = total
        result['metrics']['case_artifact_budget_passed'] = within
        if not within and result['status'] != 'artifact_budget_exceeded':
            result['status_before_artifact_budget'] = result['status']
            result.update(status='artifact_budget_exceeded',
                          error='The complete R14 case exceeds its 4 GiB artifact allowance')
            changed = True
        if not changed:
            break
    else:
        raise RuntimeError('R14 terminal artifact accounting did not settle')
    write_json(existing, result)


def execute(args):
    root = paths.RESULTS_DIR.resolve()
    out = Path(args.out).absolute()
    if out.resolve() == root or not out.resolve().is_relative_to(root):
        raise ValueError('--out must name a new child directory of RESULTS_DIR')
    if out.resolve().is_relative_to(Path(args.r13_run).resolve()):
        raise ValueError('--out must remain separate from the preserved R13 source run')
    if out.exists():
        raise ValueError('Existing run directories are preserved')
    out.mkdir(parents=True)
    work = out / 'worker'
    work.mkdir()
    (out / 'resources.jsonl').touch()
    recorder = StageRecorder(out / 'events.jsonl')
    processes, child_events, observations = [], [], []
    worker, worker_read_error, model, previous, source = None, None, None, None, {}
    status, error = 'ok', None
    started_ns = time.perf_counter_ns()
    try:
        with recorder.span('turnaround', r13_run=str(args.r13_run)):
            with recorder.span('preflight'):
                _observe_budget(out, 'preflight_start', observations)
                model, previous, source = _saved_source(args.r13_run)
                records = (paths.WORK_RESULTS /
                           'agent-cifar-offline-records-r1804/cifar10-records').resolve()
                selections = (paths.WORK_RESULTS /
                              'agent-resnet20-original-accuracy-r1805/frozen-selections.json').resolve()
                frontend = Path(toolchain.host_paths()['frontend']).resolve()
                interpreter = paths.INTAKE_PYTHON.absolute()
                required = [records, records / 'dataset.json', selections, frontend,
                            source['model_info'], source['model_info'] / 'model_info.json',
                            source['training_probes'], source['training_probe_source']]
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
                    '--', *_command(source, records, selections, frontend, work)]
            process = None
            with recorder.span('common_model_rule_preparation', wall_seconds=OUTER_SECONDS,
                               model=model) as span:
                try:
                    process = run_process(
                        argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=out / 'worker.log',
                        resources=out / 'resources.jsonl', timeout=OUTER_SECONDS,
                        mem_gib=MEMORY_GIB, fsize_gib=8,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                             'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
                    processes.append(dict(label='common_model_rule_preparation', command=argv, **process))
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
                        except (OSError, json.JSONDecodeError) as exc:
                            worker_read_error = f'{type(exc).__name__}: {exc}'[:2000]
                _observe_budget(out, 'worker_return', observations)
                if process is None:
                    raise PhaseStopped('process_failed', 'Worker process observation is unavailable')
                if process['timed_out']:
                    raise PhaseStopped('timeout', 'The outer R14 process exceeded 1860 seconds')
                if process['signal'] or process['remaining_pids']:
                    raise PhaseStopped('process_failed', str({key: process[key] for key in
                                                            ('exit_code', 'signal', 'remaining_pids')}))
                if worker is None:
                    raise PhaseStopped('result_missing', worker_read_error or 'Worker did not write its result')
                if worker.get('status') != 'ok':
                    raise PhaseStopped(worker.get('status', 'error'), str(worker.get('error')))
                if process['exit_code'] != 0:
                    raise PhaseStopped('process_failed', 'Worker exit code: ' + str(process['exit_code']))
                if (worker.get('round') != 'R14' or worker.get('model') != model
                        or any(worker.get(name) is not True for name in STAGE_FIELDS)
                        or any(worker.get(name) != 1 for name in ATTEMPT_FIELDS)
                        or any(worker.get(name) != 0 for name in ZERO_FIELDS)
                        or any(worker.get(name) is not None for name in UNMEASURED_FIELDS)):
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
        result['model'] = model
        result['worker_result_read_error'] = worker_read_error
        result['source_r13'] = dict(run=str(Path(args.r13_run).resolve()),
                                    result=str(source.get('r13_result')) if source else None,
                                    model=model, artifacts={key: str(value) for key, value in source.items()},
                                    prior_import_spans=[span for span in (previous or {}).get('timing', {}).get('spans', [])
                                                        if span.get('stage_id') == 'import_model'],
                                    included_in_current_turnaround=False)
        _write_terminal_result(out, result, observations)
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round='R14', model=model, mode=MODE,
                                         git_rev=result['git_rev'], metrics=result['metrics'],
                                         artifacts_dir=str(out),
                                         ts=datetime.now(timezone.utc).isoformat()),
                                    ensure_ascii=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r13-run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = execute(args)
    print(json.dumps(dict(status=result['status'], out=str(args.out)), ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
