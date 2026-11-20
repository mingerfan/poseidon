"""R17 outer owner for three cached-margin candidates and heldout validation."""

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
from .r13_common_models import FACTORIES
from .r14_common_preflight import DependencyMissing, PhaseStopped, _case_size, _git_state

INTERNAL_SECONDS = 1800
OUTER_SECONDS = 1860
MEMORY_GIB = 4
CASE_BUDGET_BYTES = 4 * 1024**3
MODE = 'cached-margin-heldout-validation'
STAGE_FIELDS = ('source_checked', 'calibration_reused', 'plan_completed', 'typecheck_passed',
                'signature_passed', 'candidate_checked', 'count_completed',
                'samples_checked', 'math_probes_passed')
COUNT_FIELDS = ('plan_attempts', 'plan_completions', 'count_attempts', 'count_completions',
                'model_restore_attempts', 'model_restore_completions')
RESULT_FIELDS = ('evaluation_completed', 'completed_samples', 'heldout_samples', 'outputs_partial',
                 'task_passed', 'domain_passed', 'heldout_acceptance', 'all_finite',
                 'candidate_frozen', 'max_u', 'max_logit_absolute_difference', 'validation_metrics',
                 'source_count_stats', 'count_stats', 'domain_rows', 'engine_counts',
                 'cached_source_reads')
ZERO_FIELDS = ('llm_calls', 'real_ckks_attempts', 'real_ckks_completed', 'trace_runs',
               'compiler_runs', 'keygen_runs', 'model_imports', 'downloads', 'installations',
               'calibration_attempts', 'emit_attempts', 'training_samples', 'confirm_samples',
               'report_samples', 'freeze_attempts', 'agent_runs', 'tuner_runs', 'cost_estimates')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')
CACHE_NAMES = ('r14_result', 'r13_result', 'calibration', 'calibration_source', 'original_program')
RUNS = (
    ('r14-common-resnet20-silu-20261115', 'resnet20_silu', 19),
    ('r14-common-vgg16-silu-avgpool-20261115', 'vgg16_silu_avgpool', 15),
    ('r14-common-alexnet-silu-avgpool-20261115', 'alexnet_silu_avgpool', 7),
)
SOURCE_FIELDS = ('r14_result', 'r13_result', 'model_info', 'calibration',
                 'calibration_source', 'original_program', 'records')


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise PhaseStopped('source_failed', 'Expected a saved JSON object: ' + str(path))
    return value


def _saved_path(value, name):
    if not isinstance(value, str) or not value:
        raise PhaseStopped('source_failed', 'Missing saved path: ' + name)
    return Path(value).resolve()


def resolve_source(r14_run) -> tuple[dict, dict]:
    """Resolve only the three original R14 caches and their original R13 records."""
    run = Path(r14_run).resolve()
    expected = next((item for item in RUNS if item[0] == run.name), None)
    if expected is None or run.parent != paths.RESULTS_DIR.resolve():
        raise PhaseStopped('source_failed', 'R17 requires one of the three original R14 runs')
    result_path = run / 'result.json'
    if not result_path.is_file():
        raise DependencyMissing('Missing original R14 result: ' + str(result_path))
    previous = _json(result_path)
    worker = previous.get('worker') or {}
    model = 'pt_agent.experiments.r13_common_models:' + expected[1]
    stages = ('source_checked', 'reference_passed', 'calibration_completed', 'plan_completed',
              'typecheck_passed', 'signature_passed', 'count_completed', 'stats_equal')
    wanted_status = 'ok' if expected[1] == 'resnet20_silu' else 'artifact_budget_exceeded'
    if (previous.get('round') != 'R14' or previous.get('mode') != 'common-model-rule-preparation'
            or previous.get('model') != model or worker.get('round') != 'R14'
            or worker.get('model') != model or previous.get('status') != wanted_status
            or worker.get('status') != wanted_status
            or (wanted_status != 'ok' and worker.get('failed_phase') != 'emit_write')
            or any(worker.get(key) is not True for key in stages)):
        raise PhaseStopped('source_failed', 'Original R14 model, retained status or completed stages differ')
    selected = dict(r14_result=result_path)
    artifacts = worker.get('artifacts') or {}
    for target, original in (('calibration', 'calibration'), ('calibration_source', 'calibration_source'),
                             ('original_program', 'program')):
        selected[target] = _saved_path(artifacts.get(original), 'R14.' + original)
        if not selected[target].is_relative_to(run):
            raise PhaseStopped('source_failed', 'Original R14 artifact is outside its run: ' + original)
    prior = previous.get('source_r13') or {}
    r13_run = (paths.RESULTS_DIR / expected[0].replace('r14-', 'r13-', 1)).resolve()
    prior_artifacts = prior.get('artifacts') or {}
    selected.update(r13_result=_saved_path(prior.get('result'), 'source_r13.result'),
                    model_info=_saved_path(prior_artifacts.get('model_info'), 'source_r13.model_info'))
    if (_saved_path(prior.get('run'), 'source_r13.run') != r13_run
            or selected['r13_result'] != r13_run / 'result.json'
            or _saved_path(prior_artifacts.get('r13_result'), 'source_r13.r13_result') != selected['r13_result']
            or prior.get('model') != model or not selected['model_info'].is_relative_to(r13_run)):
        raise PhaseStopped('source_failed', 'Original R13 run or ModelInfo mapping differs')
    if not selected['r13_result'].is_file():
        raise DependencyMissing('Missing original R13 result: ' + str(selected['r13_result']))
    r13 = _json(selected['r13_result'])
    r13_worker = r13.get('worker') or {}
    r13_stages = ('import_completed', 'real_probes_passed', 'graph_math_passed',
                  'roundtrip_passed', 'static_coverage')
    if (r13.get('round') != 'R13' or r13.get('mode') != 'common-model-intake-preparation'
            or r13.get('status') != 'ok' or r13_worker.get('round') != 'R13'
            or r13_worker.get('status') != 'ok' or r13.get('model') != model
            or r13_worker.get('model') != model
            or any(r13_worker.get(key) is not True for key in r13_stages)):
        raise PhaseStopped('source_failed', 'Original R13 model or completed success stages differ')
    r13_artifacts = r13_worker.get('artifacts') or {}
    for key in ('model_info', 'training_probes', 'training_probe_source'):
        actual = _saved_path(r13_artifacts.get(key), 'R13.' + key)
        if (actual != _saved_path(prior_artifacts.get(key), 'source_r13.' + key)
                or not actual.is_relative_to(r13_run)):
            raise PhaseStopped('source_failed', 'Original R13 artifact mapping differs: ' + key)
    selected['records'] = (paths.WORK_RESULTS / 'agent-cifar-offline-records-r1804/cifar10-records').resolve()
    return previous, selected


def _protected_sources():
    protected = {(paths.WORK_RESULTS / 'agent-cifar-offline-records-r1804/cifar10-records').resolve()}
    for name, _, _ in RUNS:
        for source_name in (name, name.replace('r14-', 'r13-', 1)):
            run = (paths.RESULTS_DIR / source_name).resolve()
            protected.add(run)
            record_path = run / 'result.json'
            if not record_path.is_file():
                continue
            try:
                artifacts = (_json(record_path).get('worker') or {}).get('artifacts') or {}
                for key in ('model_info', 'training_probes', 'training_probe_source',
                            'calibration', 'calibration_source', 'program'):
                    value = artifacts.get(key)
                    if isinstance(value, str) and value:
                        protected.add(Path(value).resolve())
            except (OSError, ValueError, PhaseStopped):
                pass
        protected.add((paths.RESULTS_DIR / name.replace('r14-common-', 'r15-rule-heldout-', 1)).resolve())
    for record in FACTORIES.values():
        for key in ('source', 'checkpoint', 'inference_preprocessing_source'):
            protected.add(Path(record[key]).resolve())
    return protected


def _check_out(out):
    out = Path(out).absolute()
    resolved, root = out.resolve(), paths.RESULTS_DIR.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError('--out must be a new child directory of RESULTS_DIR')
    if any(resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved)
           for source in _protected_sources()):
        raise ValueError('--out must remain separate from all original source paths')
    if out.exists():
        raise ValueError('Existing output directories are preserved')
    return out


def finalize_result(*, worker, processes, timing, status, error, git_state, out) -> dict:
    record = worker or {}
    metrics = {key: record.get(key) for key in STAGE_FIELDS + COUNT_FIELDS + RESULT_FIELDS + ZERO_FIELDS}
    metrics.update({key: None for key in UNMEASURED_FIELDS})
    metrics.update(status=status, candidate_frozen=False, acc_drop_pp_unit='percentage points',
                   scope='fixed cached-margin candidate; internal heldout validation',
                   evaluation_split='heldout', reference_criterion='1e-12+1e-12*abs(reference)',
                   primary_engine='Tensor float64, cache=False', packed_mock_probe_target=3,
                   original_margin=1.25, candidate_margin=1.50, degree=63,
                   formal_end_to_end_time=None)
    metrics['phase_elapsed_ns'] = {
        span['event_id']: dict(stage_id=span['stage_id'], inclusive_ns=span.get('inclusive_ns'),
                              exclusive_ns=span.get('exclusive_ns'))
        for span in timing.get('spans', [])}
    return dict(run_id=Path(out).name, round='R17', mode=MODE, model=record.get('model'),
                status=status, error=error, git_rev=git_state, worker=record,
                processes=processes, timing=timing, metrics=metrics)


def _observe_budget(out, stage, observations):
    for label, directory in (('case', out), ('queue', out.parent)):
        observation = dict(stage=stage, scope=label, **_case_size(directory),
                           limit_bytes=CASE_BUDGET_BYTES)
        observation['within_budget'] = observation['bytes'] <= CASE_BUDGET_BYTES
        observations.append(observation)
        if not observation['within_budget']:
            raise PhaseStopped('artifact_budget_exceeded', 'The R17 ' + label + ' exceeds 4 GiB')


def _write_terminal_result(out, result, observations):
    destination = out / 'result.json'
    previous_size = destination.stat().st_size if destination.exists() else 0
    base_case = _case_size(out)['bytes'] - previous_size
    base_queue = _case_size(out.parent)['bytes'] - previous_size
    terminal = dict(stage='terminal', case_bytes=base_case, queue_bytes=base_queue,
                    limit_bytes=CASE_BUDGET_BYTES, within_budget=True)
    observations.append(terminal)
    result['budget_observations'] = observations
    for _ in range(16):
        length = len((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        total_case, total_queue = base_case + length, base_queue + length
        within = max(total_case, total_queue) <= CASE_BUDGET_BYTES
        before = (terminal.copy(), result.get('case_artifact_bytes'), result.get('queue_artifact_bytes'),
                  result['status'], result['metrics'].copy())
        terminal.update(case_bytes=total_case, queue_bytes=total_queue, within_budget=within)
        result.update(case_artifact_bytes=total_case, queue_artifact_bytes=total_queue)
        result['metrics'].update(case_artifact_bytes=total_case, queue_artifact_bytes=total_queue,
                                 case_artifact_budget_passed=total_case <= CASE_BUDGET_BYTES,
                                 queue_artifact_budget_passed=total_queue <= CASE_BUDGET_BYTES)
        if not within and result['status'] != 'artifact_budget_exceeded':
            result['status_before_artifact_budget'] = result['status']
            result.update(status='artifact_budget_exceeded', error='The complete R17 artifacts exceed 4 GiB')
        result['metrics']['status'] = result['status']
        after = (terminal, result['case_artifact_bytes'], result['queue_artifact_bytes'],
                 result['status'], result['metrics'])
        if before == after:
            write_json(destination, result)
            return
    raise RuntimeError('R17 terminal artifact accounting did not settle')


def _command(source, work):
    command = ['-m', 'pt_agent.experiments.r17_margin_validation_worker']
    for key in SOURCE_FIELDS:
        command.extend(['--' + key.replace('_', '-'), str(source[key])])
    return command + ['--work-dir', str(work)]


def _check_success(worker, model):
    if (worker.get('round') != 'R17' or worker.get('model') != model
            or any(worker.get(key) is not True for key in STAGE_FIELDS)
            or worker.get('evaluation_completed') is not True
            or any(type(worker.get(key)) is not int or worker.get(key) != 1000
                   for key in ('completed_samples', 'heldout_samples'))
            or worker.get('outputs_partial') is not False
            or worker.get('all_finite') is not True or worker.get('candidate_frozen') is not False
            or any(type(worker.get(key)) is not int or worker.get(key) != 1 for key in COUNT_FIELDS)
            or any(type(worker.get(key)) is not int or worker.get(key) != 0 for key in ZERO_FIELDS)
            or any(worker.get(key) is not None for key in UNMEASURED_FIELDS)):
        raise PhaseStopped('incomplete', 'Successful R17 worker lacks required observations')
    for key in CACHE_NAMES:
        if (worker.get('cached_source_reads') or {}).get(key) != dict(attempted=1, completed=1):
            raise PhaseStopped('incomplete', 'Successful cached source reads differ: ' + key)
    for key, count in dict(reference_forward=1000, tensor_program=1000,
                           tensor_exact=3, mock_exact=3, mock_program=3).items():
        if (worker.get('engine_counts') or {}).get(key) != dict(attempted=count, completed=count):
            raise PhaseStopped('incomplete', 'Successful engine counts differ: ' + key)
    for key in ('task_passed', 'domain_passed', 'heldout_acceptance'):
        if type(worker.get(key)) is not bool:
            raise PhaseStopped('incomplete', 'Missing complete quality result: ' + key)
    if worker['heldout_acceptance'] != (worker['task_passed'] and worker['domain_passed']):
        raise PhaseStopped('incomplete', 'Heldout acceptance differs from completed criteria')


def execute(args) -> dict:
    out = _check_out(args.out)
    r14_run = Path(args.r14_run).resolve()
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
        with recorder.span('turnaround', r14_run=str(r14_run)):
            with recorder.span('preflight'):
                _observe_budget(out, 'preflight_start', observations)
                previous, source = resolve_source(r14_run)
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
            with recorder.span('cached_margin_heldout_validation', wall_seconds=OUTER_SECONDS) as span:
                try:
                    process = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                                          log=out / 'worker.log', resources=out / 'resources.jsonl',
                                          timeout=OUTER_SECONDS, mem_gib=MEMORY_GIB, fsize_gib=8,
                                          env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                               'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
                    processes.append(dict(label='cached_margin_heldout_validation', command=argv, **process))
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
                    raise PhaseStopped('timeout', 'The R17 outer process exceeded 1860 seconds')
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
        result['source_r14'] = dict(run=str(r14_run), result=str(r14_run / 'result.json'),
                                      artifacts={k: str(v) for k, v in source.items()},
                                      prior_timing=(previous or {}).get('timing'),
                                      prior_r13_import_spans=((previous or {}).get('source_r13') or {}).get('prior_import_spans'),
                                      included_in_current_turnaround=False)
        _write_terminal_result(out, result, observations)
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round='R17', model=result['model'],
                                         mode=MODE, git_rev=result['git_rev'], metrics=result['metrics'],
                                         artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()),
                                    ensure_ascii=False) + '\n')
    return result


def _write_queue_result(out, result):
    target = out / 'queue_result.json'
    base = _case_size(out)['bytes'] - (target.stat().st_size if target.exists() else 0)
    for _ in range(16):
        total = base + len((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        within = total <= CASE_BUDGET_BYTES
        if result.get('artifact_bytes') == total and result.get('artifact_budget_passed') == within:
            write_json(target, result)
            return
        result.update(artifact_bytes=total, artifact_budget_passed=within)
        if not within:
            result.update(status='artifact_budget_exceeded', error='The complete R17 queue exceeds 4 GiB')
    raise RuntimeError('R17 queue artifact accounting did not settle')


def run_queue(args) -> dict:
    out = _check_out(args.out)
    out.mkdir(parents=True)
    plan = [dict(r14_run=str(paths.RESULTS_DIR / name), model=short,
                 out=str(out / short), status='not_started', result=None)
            for name, short, _ in RUNS]
    result = dict(round='R17', mode=MODE, status='running', error=None, cases=plan,
                  attempted_cases=0, completed_cases=0, maximum_cases=3, retry_count=0,
                  candidate_frozen=False, formal_matrix_complete=False)
    _write_queue_result(out, result)
    for item in plan:
        if result['status'] == 'artifact_budget_exceeded':
            break
        result['attempted_cases'] += 1
        try:
            current = execute(argparse.Namespace(r14_run=Path(item['r14_run']),
                                                out=Path(item['out'])))
        except BaseException as exc:
            item.update(status='outer_error', error=f'{type(exc).__name__}: {exc}'[:2000])
            result.update(status='outer_error', error=item['error'])
            _write_queue_result(out, result)
            break
        item.update(status=current['status'], result=str(Path(item['out']) / 'result.json'))
        if current['status'] != 'ok':
            result.update(status='stopped', error=current['error'])
            _write_queue_result(out, result)
            break
        result['completed_cases'] += 1
        _write_queue_result(out, result)
    else:
        result.update(status='ok', error=None)
    _write_queue_result(out, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_queue(args)
    print(json.dumps(dict(status=result['status'], out=str(args.out),
                          attempted_cases=result['attempted_cases'], completed_cases=result['completed_cases']),
                     ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
