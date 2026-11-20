"""R15 outer owner for one saved fixed-rule heldout numerical evaluation."""

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
MODE = 'fixed-rule-heldout-validation'
STAGE_FIELDS = ('source_checked', 'typecheck_passed', 'signature_passed',
                'samples_checked', 'math_probes_passed')
RESULT_FIELDS = ('evaluation_completed', 'task_passed', 'domain_passed',
                 'heldout_acceptance', 'all_finite', 'completed_samples',
                 'engine_counts', 'validation_metrics', 'domain_rows', 'max_u',
                 'source_count_stats', 'max_logit_absolute_difference')
ZERO_FIELDS = ('llm_calls', 'real_ckks_attempts', 'real_ckks_completed', 'trace_runs',
               'compiler_runs', 'keygen_runs', 'model_imports', 'downloads',
               'installations', 'calibration_attempts', 'plan_attempts', 'count_attempts',
               'emit_attempts', 'confirm_samples', 'report_samples', 'freeze_attempts',
               'agent_runs', 'tuner_runs', 'cost_estimates')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance')
EXPORTS = {
    'r14-common-resnet20-silu-20261115': ('R14', 'common-model-rule-preparation', 'resnet20_silu'),
    'r14a-vgg16-emit-resume-20261115': ('R14a', 'vgg-emit-resume', 'vgg16_silu_avgpool'),
    'r14b-alexnet-emit-resume-20261115': ('R14b', 'alex-emit-resume', 'alexnet_silu_avgpool'),
}


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise PhaseStopped('source_failed', 'Expected a saved JSON object: ' + str(path))
    return value


def _path(value, name):
    if not isinstance(value, str) or not value:
        raise PhaseStopped('source_failed', 'Missing saved source path: ' + name)
    return Path(value).resolve()


def resolve_source(export_run) -> tuple[dict, dict]:
    """Resolve the three approved export forms without loading numeric artifacts."""
    run = Path(export_run).resolve()
    expected = EXPORTS.get(run.name)
    if expected is None or run.parent != paths.RESULTS_DIR.resolve():
        raise PhaseStopped('source_failed', 'R15 requires one of the three approved export directories')
    result_path = run / 'result.json'
    if not result_path.is_file():
        raise DependencyMissing('Missing successful export result: ' + str(result_path))
    previous = _json(result_path)
    worker = previous.get('worker') or {}
    revision, mode, short = expected
    model = 'pt_agent.experiments.r13_common_models:' + short
    stages = ('typecheck_passed', 'signature_passed', 'stats_equal',
              'emit_completed', 'lowering_ready')
    if (previous.get('round') != revision or previous.get('mode') != mode
            or previous.get('status') != 'ok' or previous.get('model') != model
            or worker.get('round') != revision or worker.get('status') != 'ok'
            or worker.get('model') != model
            or any(worker.get(k) is not True for k in stages)):
        raise PhaseStopped('source_failed', 'Saved export success, model or stage fields differ')
    selected = dict(export_result=result_path)
    if revision == 'R14':
        artifacts = worker.get('artifacts') or {}
        prior = previous.get('source_r13') or {}
        selected.update(rule_result=result_path,
                        program=_path(artifacts.get('program'), 'program'),
                        count=_path(artifacts.get('count'), 'count'),
                        r13_result=_path(prior.get('result'), 'r13_result'),
                        model_info=_path((prior.get('artifacts') or {}).get('model_info'), 'model_info'))
        rule_run = run
    else:
        resume = worker.get('source_resume') or {}
        prior = previous.get('source_r14') or {}
        prior_artifacts = prior.get('artifacts') or {}
        mapping = dict(rule_result='source_result', program='program', count='count',
                       r13_result='r13_result', model_info='model_info')
        for key, original in mapping.items():
            selected[key] = _path(resume.get(original), original)
            if selected[key] != _path(prior_artifacts.get(original), 'source_r14.' + original):
                raise PhaseStopped('source_failed', 'Resume and outer source locations differ: ' + key)
        if selected['rule_result'] != _path(prior.get('result'), 'source_r14.result'):
            raise PhaseStopped('source_failed', 'Original R14 result differs')
        rule_run = _path(prior.get('run'), 'source_r14.run')
        rule = _json(selected['rule_result'])
        rw = rule.get('worker') or {}
        if (selected['rule_result'] != rule_run / 'result.json'
                or rule.get('round') != 'R14' or rule.get('model') != model
                or rule.get('status') != 'artifact_budget_exceeded'
                or rw.get('status') != 'artifact_budget_exceeded'
                or rw.get('failed_phase') != 'emit_write'):
            raise PhaseStopped('source_failed', 'Resume requires the preserved original emit_write stop')
    for key in ('program', 'count'):
        if not selected[key].is_relative_to(rule_run):
            raise PhaseStopped('source_failed', 'Original R14 artifact is outside its source run: ' + key)
    r13_run = selected['r13_result'].parent
    if (selected['r13_result'].name != 'result.json'
            or not selected['model_info'].is_relative_to(r13_run)):
        raise PhaseStopped('source_failed', 'Original R13 result or ModelInfo directory differs')
    r13 = _json(selected['r13_result'])
    r13_worker = r13.get('worker') or {}
    if (r13.get('round') != 'R13' or r13.get('status') != 'ok'
            or r13.get('model') != model or r13_worker.get('status') != 'ok'
            or r13_worker.get('model') != model
            or selected['model_info'] != _path((r13_worker.get('artifacts') or {}).get('model_info'),
                                                'R13.artifacts.model_info')):
        raise PhaseStopped('source_failed', 'Saved original R13 source or ModelInfo mapping differs')
    selected['records'] = paths.WORK_RESULTS / 'agent-cifar-offline-records-r1804/cifar10-records'
    return previous, selected


def _protected_sources(export_run):
    """Protect saved directories even if later source validation rejects a record."""
    export_run = Path(export_run).resolve()
    protected, pending, visited = {export_run}, [export_run / 'result.json'], set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        if not path.is_file():
            continue
        try:
            record = _json(path)
        except (OSError, ValueError, PhaseStopped):
            continue
        for name in ('source_r13', 'source_r14'):
            prior = record.get(name) or {}
            if not isinstance(prior, dict):
                continue
            if isinstance(prior.get('run'), str):
                protected.add(Path(prior['run']).resolve())
            if isinstance(prior.get('result'), str):
                source = Path(prior['result']).resolve()
                protected.add(source.parent)
                pending.append(source)
        resume = (record.get('worker') or {}).get('source_resume') or {}
        for name in ('source_result', 'r13_result'):
            if isinstance(resume.get(name), str):
                source = Path(resume[name]).resolve()
                protected.add(source.parent)
                pending.append(source)
    return protected


def finalize_result(*, worker, processes, timing, status, error, git_state, out) -> dict:
    record = worker or {}
    metrics = {key: record.get(key) for key in STAGE_FIELDS + RESULT_FIELDS + ZERO_FIELDS}
    metrics.update({key: None for key in UNMEASURED_FIELDS})
    metrics.update(status=status, candidate_frozen=False, acc_drop_pp_unit='percentage points',
                   evaluation_split='heldout', reference_criterion='1e-12+1e-12*abs(reference)',
                   primary_engine='Tensor float64, cache=False', packed_mock_probe_target=3)
    metrics['phase_elapsed_ns'] = {
        span['event_id']: dict(stage_id=span['stage_id'], inclusive_ns=span.get('inclusive_ns'),
                              exclusive_ns=span.get('exclusive_ns'))
        for span in timing.get('spans', [])}
    return dict(run_id=Path(out).name, round='R15', mode=MODE, model=record.get('model'),
                status=status, error=error, git_rev=git_state, worker=record,
                processes=processes, timing=timing, metrics=metrics)


def _observe_budget(out, stage, observations):
    observation = dict(stage=stage, boundary='outer', **_case_size(out), limit_bytes=CASE_BUDGET_BYTES)
    observation['within_budget'] = observation['bytes'] <= CASE_BUDGET_BYTES
    observations.append(observation)
    if not observation['within_budget']:
        raise PhaseStopped('artifact_budget_exceeded', 'The R15 case exceeds 4 GiB')


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
            result.update(status='artifact_budget_exceeded', error='The complete R15 case exceeds 4 GiB')
            changed = True
        result['metrics']['status'] = result['status']
        if not changed:
            write_json(destination, result)
            return
    raise RuntimeError('R15 terminal artifact accounting did not settle')


def _command(source, work):
    command = ['-m', 'pt_agent.experiments.r15_rule_validation_worker']
    for key in ('export_result', 'rule_result', 'program', 'count', 'r13_result', 'model_info', 'records'):
        command.extend(['--' + key.replace('_', '-'), str(source[key])])
    return command + ['--work-dir', str(work)]


def _check_success(worker, model):
    if (worker.get('round') != 'R15' or worker.get('model') != model
            or any(worker.get(key) is not True for key in STAGE_FIELDS)
            or worker.get('evaluation_completed') is not True
            or worker.get('completed_samples') != 1000
            or worker.get('all_finite') is not True or worker.get('candidate_frozen') is not False
            or any(worker.get(key) != 0 for key in ZERO_FIELDS)
            or any(worker.get(key) is not None for key in UNMEASURED_FIELDS)):
        raise PhaseStopped('incomplete', 'Successful R15 worker is missing required observations')
    for key, count in dict(reference_forward=1000, tensor_program=1000,
                           tensor_exact=3, mock_exact=3, mock_program=3).items():
        if (worker.get('engine_counts') or {}).get(key) != dict(attempted=count, completed=count):
            raise PhaseStopped('incomplete', 'Successful engine counts differ: ' + key)
    for key in ('task_passed', 'domain_passed', 'heldout_acceptance'):
        if type(worker.get(key)) is not bool:
            raise PhaseStopped('incomplete', 'Missing complete quality result: ' + key)
    if worker['heldout_acceptance'] != (worker['task_passed'] and worker['domain_passed']):
        raise PhaseStopped('incomplete', 'Heldout acceptance differs from the completed criteria')


def execute(args) -> dict:
    root, out = paths.RESULTS_DIR.resolve(), Path(args.out).absolute()
    export_run = Path(args.export_run).resolve()
    if out.resolve() == root or not out.resolve().is_relative_to(root):
        raise ValueError('--out must name a new child directory of RESULTS_DIR')
    if any(out.resolve().is_relative_to(p) for p in _protected_sources(export_run)):
        raise ValueError('--out must remain separate from every saved source run')
    if out.exists():
        raise ValueError('Existing run directories are preserved')
    out.mkdir(parents=True)
    work = out / 'worker'
    work.mkdir()
    (out / 'resources.jsonl').touch()
    recorder = StageRecorder(out / 'events.jsonl')
    processes, child_events, observations = [], [], []
    previous, source, worker, worker_read_error = None, {}, None, None
    status, error = 'ok', None
    started_ns = time.perf_counter_ns()
    try:
        with recorder.span('turnaround', export_run=str(export_run)):
            with recorder.span('preflight'):
                _observe_budget(out, 'preflight_start', observations)
                previous, source = resolve_source(export_run)
                interpreter = paths.INTAKE_PYTHON.absolute()
                required = list(source.values()) + [source['model_info'] / 'model_info.json',
                                                   source['records'] / 'dataset.json',
                                                   source['records'] / 'test-09000.records']
                missing = [str(p) for p in required if not p.exists()]
                if not interpreter.is_file() and not interpreter.is_symlink():
                    missing.append(str(interpreter))
                if missing:
                    raise DependencyMissing('Missing local prerequisites: ' + ', '.join(missing))
                if shutil.disk_usage(out).free < 10 * 1024**3:
                    raise DependencyMissing('At least 10 GiB free disk space is required')
                _observe_budget(out, 'preflight_end', observations)
            argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds',
                    str(INTERNAL_SECONDS), '--python', str(interpreter), '--', *_command(source, work)]
            process = None
            with recorder.span('fixed_rule_heldout_validation', wall_seconds=OUTER_SECONDS) as span:
                try:
                    process = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                                          log=out / 'worker.log', resources=out / 'resources.jsonl',
                                          timeout=OUTER_SECONDS, mem_gib=MEMORY_GIB, fsize_gib=8,
                                          env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                               'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
                    processes.append(dict(label='fixed_rule_heldout_validation', command=argv, **process))
                finally:
                    for event in read_events(work / 'events.jsonl'):
                        if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                            event['parent_id'] = span['event_id']
                        child_events.append(event)
                    result_path = work / 'worker_result.json'
                    if result_path.is_file():
                        try:
                            worker = _json(result_path)
                        except (OSError, ValueError, PhaseStopped) as exc:
                            worker_read_error = f'{type(exc).__name__}: {exc}'[:2000]
                _observe_budget(out, 'worker_return', observations)
                if process is None:
                    raise PhaseStopped('process_failed', 'Worker process observation is unavailable')
                if process['timed_out']:
                    raise PhaseStopped('timeout', 'The R15 outer process exceeded 1860 seconds')
                if process['signal'] or process['remaining_pids']:
                    raise PhaseStopped('process_failed', str({k: process[k] for k in
                                                            ('exit_code', 'signal', 'remaining_pids')}))
                if worker is None:
                    raise PhaseStopped('result_missing', worker_read_error or 'Worker result unavailable')
                if worker.get('status') != 'ok':
                    raise PhaseStopped(worker.get('status', 'error'), str(worker.get('error')))
                if process['exit_code'] != 0:
                    raise PhaseStopped('process_failed', 'Worker exit code: ' + str(process['exit_code']))
                _check_success(worker, previous['model'])
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
        result['source_export'] = dict(run=str(export_run), result=str(export_run / 'result.json'),
                                      artifacts={k: str(v) for k, v in source.items()},
                                      prior_timing=(previous or {}).get('timing'),
                                      included_in_current_turnaround=False)
        _write_terminal_result(out, result, observations)
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round='R15', model=result['model'],
                                         mode=MODE, git_rev=result['git_rev'], metrics=result['metrics'],
                                         artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()),
                                    ensure_ascii=False) + '\n')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = execute(args)
    print(json.dumps(dict(status=result['status'], out=str(args.out)), ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
