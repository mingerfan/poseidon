"""R16 serial outer owner for six original-model plaintext compatibility checks."""

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

INTERNAL_SECONDS = 300
OUTER_SECONDS = 360
MEMORY_GIB = 4
CASE_BUDGET_BYTES = 4 * 1024**3
MODE = 'baseline-plaintext-compatibility'
STAGE_FIELDS = ('source_checked', 'checkpoint_checked', 'model_built',
                'strict_load_passed', 'structure_passed', 'probes_checked', 'math_probes_passed')
RESULT_FIELDS = ('compatibility_passed', 'forward_attempts', 'forward_completed',
                 'checkpoint_load_attempts', 'checkpoint_load_completed',
                 'probe_load_attempts', 'probe_load_completed',
                 'model_build_attempts', 'model_build_completed', 'all_finite', 'candidate_frozen')
ZERO_FIELDS = ('llm_calls', 'trace_runs', 'compiler_runs', 'solver_runs', 'keygen_runs',
               'real_ckks_attempts', 'real_ckks_completed', 'init_scheme_calls',
               'backend_initializations', 'crypto_contexts', 'fit_calls', 'downloads',
               'installations', 'calibration_attempts', 'plan_attempts', 'count_attempts',
               'emit_attempts', 'confirm_samples', 'report_samples', 'heldout_samples',
               'freeze_attempts', 'agent_runs', 'tuner_runs', 'cost_estimates', 'model_imports')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance')
RUNS = (
    ('r13-common-resnet20-silu-20261115', 'resnet20_silu', 'ResNetSiLU.py'),
    ('r13-common-vgg16-silu-avgpool-20261115', 'vgg16_silu_avgpool', 'VGG16SiLU.py'),
    ('r13-common-alexnet-silu-avgpool-20261115', 'alexnet_silu_avgpool', 'AlexNetSiLU.py'),
)
_ORION_ENV_PREFIXES = ('LD_', 'NIX_', '__NIX_', 'HECATE_')
_ORION_ENV_KEYS = frozenset((
    'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONUSERBASE',
    'PYTHONPLATLIBDIR', 'PYTHONSAFEPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX',
    'TORCH_USE_RTLD_GLOBAL', 'TORCH_COMPILE_STROBELIGHT',
))
_THREAD_ENV_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _remove_orion_env_key(name):
    return name.startswith(_ORION_ENV_PREFIXES) or name in _ORION_ENV_KEYS


def _orion_pythonpath():
    return [str(paths.ORION_ROOT.resolve()), str(paths.REPO / 'DSL/DSL_agent/scripts')]


def _native_orion_env(env=None):
    """Copy the supplied environment and select the existing host runtime."""
    child = dict(os.environ if env is None else env)
    for name in tuple(child):
        if _remove_orion_env_key(name):
            del child[name]
    child.update(PYTHONPATH=os.pathsep.join(_orion_pythonpath()),
                 PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    child.update(dict.fromkeys(_THREAD_ENV_KEYS, '1'))
    return child


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise PhaseStopped('source_failed', 'Expected a saved JSON object: ' + str(path))
    return value


def _saved_path(value, name):
    if not isinstance(value, str) or not value:
        raise PhaseStopped('source_failed', 'Missing saved path: ' + name)
    return Path(value).resolve()


def resolve_source(baseline, r13_run) -> tuple[dict, dict]:
    """Resolve approved records and paths without reading tensors or sample arrays."""
    if baseline not in ('orion', 'orbit'):
        raise PhaseStopped('source_failed', 'Unknown R16 baseline')
    run = Path(r13_run).resolve()
    expected = next((item for item in RUNS if item[0] == run.name), None)
    if expected is None or run.parent != paths.RESULTS_DIR.resolve():
        raise PhaseStopped('source_failed', 'R16 requires one of the three original R13 runs')
    result_path = run / 'result.json'
    if not result_path.is_file():
        raise DependencyMissing('Missing original R13 result: ' + str(result_path))
    previous = _json(result_path)
    worker = previous.get('worker') or {}
    model = 'pt_agent.experiments.r13_common_models:' + expected[1]
    stages = ('import_completed', 'real_probes_passed', 'graph_math_passed',
              'roundtrip_passed', 'static_coverage')
    if (previous.get('round') != 'R13' or previous.get('status') != 'ok'
            or previous.get('model') != model or worker.get('round') != 'R13'
            or worker.get('status') != 'ok' or worker.get('model') != model
            or any(worker.get(key) is not True for key in stages)):
        raise PhaseStopped('source_failed', 'Original R13 success, model or stages differ')
    artifacts = worker.get('artifacts') or {}
    info_dir = _saved_path(artifacts.get('model_info'), 'model_info')
    selected = dict(r13_result=result_path, model_info=info_dir / 'model_info.json',
                    graph_record=info_dir / 'graph/graph.json',
                    probes=_saved_path(artifacts.get('training_probes'), 'training_probes'),
                    probe_source=_saved_path(artifacts.get('training_probe_source'), 'training_probe_source'))
    for key, value in selected.items():
        if not value.resolve().is_relative_to(run):
            raise PhaseStopped('source_failed', 'Original artifact is outside R13 run: ' + key)
    original = FACTORIES[model]
    baseline_root = paths.ORION_ROOT if baseline == 'orion' else paths.ORBIT_ROOT
    selected.update(checkpoint=Path(original['checkpoint']).resolve(),
                    model_source=(Path(original['source']).resolve() if baseline == 'orion' else
                                  (baseline_root / 'frontend/dacapo/python/poly/poly/models' / expected[2]).resolve()),
                    baseline_root=baseline_root.resolve(),
                    interpreter=((paths.ORION_ROOT.parent / 'orion-venv/bin/python').absolute()
                                 if baseline == 'orion' else paths.INTAKE_PYTHON.absolute()))
    return previous, selected


def _protected_sources():
    protected = {paths.ORION_ROOT.resolve(), paths.ORBIT_ROOT.resolve()}
    for name, _, _ in RUNS:
        run = (paths.RESULTS_DIR / name).resolve()
        protected.add(run)
        record_path = run / 'result.json'
        if record_path.is_file():
            try:
                artifacts = (_json(record_path).get('worker') or {}).get('artifacts') or {}
                for key in ('model_info', 'training_probes', 'training_probe_source'):
                    value = artifacts.get(key)
                    if isinstance(value, str) and value:
                        protected.add(Path(value).resolve())
            except (OSError, ValueError, PhaseStopped):
                pass
    for record in FACTORIES.values():
        for name in ('source', 'checkpoint', 'inference_preprocessing_source'):
            protected.add(Path(record[name]).resolve())
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
    metrics = {key: record.get(key) for key in STAGE_FIELDS + RESULT_FIELDS + ZERO_FIELDS}
    metrics.update({key: None for key in UNMEASURED_FIELDS})
    metrics.update(status=status, baseline=record.get('baseline'), candidate_frozen=False,
                   scope='original weights and three saved training probes; plaintext only',
                   reference_criterion='1e-12+1e-12*abs(reference)')
    metrics['phase_elapsed_ns'] = {
        span['event_id']: dict(stage_id=span['stage_id'], inclusive_ns=span.get('inclusive_ns'),
                              exclusive_ns=span.get('exclusive_ns'))
        for span in timing.get('spans', [])}
    return dict(run_id=Path(out).name, round='R16', mode=MODE, model=record.get('model'),
                baseline=record.get('baseline'), status=status, error=error, git_rev=git_state,
                worker=record, processes=processes, timing=timing, metrics=metrics)


def _observe_budget(out, stage, observations):
    for label, directory in (('case', out), ('queue', out.parent)):
        observation = dict(stage=stage, scope=label, **_case_size(directory),
                           limit_bytes=CASE_BUDGET_BYTES)
        observation['within_budget'] = observation['bytes'] <= CASE_BUDGET_BYTES
        observations.append(observation)
        if not observation['within_budget']:
            raise PhaseStopped('artifact_budget_exceeded', 'The R16 ' + label + ' exceeds 4 GiB')


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
            result.update(status='artifact_budget_exceeded', error='The complete R16 artifacts exceed 4 GiB')
        result['metrics']['status'] = result['status']
        after = (terminal, result['case_artifact_bytes'], result['queue_artifact_bytes'],
                 result['status'], result['metrics'])
        if before == after:
            write_json(destination, result)
            return
    raise RuntimeError('R16 terminal artifact accounting did not settle')


def _command(source, work, baseline, model):
    command = ['-m', 'pt_agent.experiments.r16_baseline_validation_worker',
               '--baseline', baseline, '--model', model]
    for key in ('r13_result', 'model_info', 'graph_record', 'probes', 'probe_source',
                'checkpoint', 'model_source', 'baseline_root'):
        command.extend(['--' + key.replace('_', '-'), str(source[key])])
    return command + ['--work-dir', str(work)]


def _launch(source, work, baseline, model):
    """Select exactly the R16a entry point; return no environment values as metadata."""
    inherited = dict(os.environ)
    command = _command(source, work, baseline, model)
    metadata = dict(supplemental_spec='R16a', removed_environment_keys=[],
                    controlled_pythonpath=None)
    if baseline == 'orion':
        argv = [str(source['interpreter']), '-B', *command]
        child_env = _native_orion_env(inherited)
        metadata.update(
            launch_mode='orion-native-existing-venv',
            removed_environment_keys=sorted(name for name in inherited if _remove_orion_env_key(name)),
            controlled_pythonpath=_orion_pythonpath(),
        )
    elif baseline == 'orbit':
        argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds',
                str(INTERNAL_SECONDS), '--python', str(source['interpreter']), '--', *command]
        child_env = dict(inherited, PYTHONDONTWRITEBYTECODE='1')
        child_env.update(dict.fromkeys(_THREAD_ENV_KEYS, '1'))
        metadata['launch_mode'] = 'orbit-existing-nix'
    else:
        raise PhaseStopped('source_failed', 'Unknown R16 baseline')
    return argv, child_env, metadata


def _check_success(worker, model, baseline):
    if (worker.get('round') != 'R16' or worker.get('model') != model
            or worker.get('baseline') != baseline
            or any(worker.get(key) is not True for key in STAGE_FIELDS)
            or worker.get('compatibility_passed') is not True
            or worker.get('all_finite') is not True or worker.get('candidate_frozen') is not False
            or any(type(worker.get(key)) is not int or worker.get(key) != 0 for key in ZERO_FIELDS)
            or any(worker.get(key) is not None for key in UNMEASURED_FIELDS)):
        raise PhaseStopped('incomplete', 'Successful R16 worker lacks required observations')
    for name, expected in (('forward', 3), ('checkpoint_load', 1), ('probe_load', 1), ('model_build', 1)):
        for suffix in ('attempts', 'completed'):
            value = worker.get(name + '_' + suffix)
            if type(value) is not int or value != expected:
                raise PhaseStopped('incomplete', 'Successful R16 call counts differ: ' + name)


def execute(args) -> dict:
    out = _check_out(args.out)
    r13_run, baseline = Path(args.r13_run).resolve(), args.baseline
    out.mkdir(parents=True)
    work = out / 'worker'
    work.mkdir()
    (out / 'resources.jsonl').touch()
    recorder = StageRecorder(out / 'events.jsonl')
    processes, child_events, observations = [], [], []
    previous, source, worker, read_error = None, {}, None, None
    launch_metadata = dict(
        launch_mode={'orion': 'orion-native-existing-venv',
                     'orbit': 'orbit-existing-nix'}.get(baseline),
        supplemental_spec='R16a', removed_environment_keys=None, controlled_pythonpath=None,
    )
    status, error = 'ok', None
    started_ns = time.perf_counter_ns()
    try:
        with recorder.span('turnaround', baseline=baseline, r13_run=str(r13_run)):
            with recorder.span('preflight'):
                _observe_budget(out, 'preflight_start', observations)
                previous, source = resolve_source(baseline, r13_run)
                required = [path for key, path in source.items() if key != 'interpreter']
                missing = [str(path) for path in required if not path.exists()]
                interpreter = source['interpreter']
                if not interpreter.is_file() and not interpreter.is_symlink():
                    missing.append(str(interpreter))
                if missing:
                    raise DependencyMissing('Missing local prerequisites: ' + ', '.join(missing))
                if shutil.disk_usage(out).free < 10 * 1024**3:
                    raise DependencyMissing('At least 10 GiB free disk space is required')
                _observe_budget(out, 'preflight_end', observations)
            argv, child_env, launch_metadata = _launch(source, work, baseline, previous['model'])
            process = None
            with recorder.span('baseline_plaintext_validation', wall_seconds=OUTER_SECONDS,
                               launch_mode=launch_metadata['launch_mode'], supplemental_spec='R16a') as span:
                try:
                    process = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                                          log=out / 'worker.log', resources=out / 'resources.jsonl',
                                          timeout=OUTER_SECONDS, mem_gib=MEMORY_GIB, fsize_gib=8,
                                          env=child_env)
                    processes.append(dict(label='baseline_plaintext_validation', command=argv, **process))
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
                            read_error = f'{type(exc).__name__}: {exc}'[:2000]
                _observe_budget(out, 'worker_return', observations)
                if process is None:
                    raise PhaseStopped('process_failed', 'Worker process observation unavailable')
                if process['timed_out']:
                    raise PhaseStopped('timeout', 'R16 outer process exceeded 360 seconds')
                if process['signal'] or process['remaining_pids']:
                    raise PhaseStopped('process_failed', str({key: process[key] for key in
                                                            ('exit_code', 'signal', 'remaining_pids')}))
                if worker is None:
                    raise PhaseStopped('result_missing', read_error or 'Worker result unavailable')
                if worker.get('status') != 'ok':
                    raise PhaseStopped(worker.get('status', 'error'), str(worker.get('error')))
                if process['exit_code'] != 0:
                    raise PhaseStopped('process_failed', 'Worker exit code: ' + str(process['exit_code']))
                _check_success(worker, previous['model'], baseline)
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
                                        elapsed_ns=ended_ns-started_ns)
        if status == 'ok' and not timing.get('attribution_ready'):
            status, error = 'timing_invalid', str(timing.get('diagnostics'))
        with (out / 'stages.jsonl').open('w') as stream:
            for stage in timing['spans']:
                stream.write(json.dumps(stage, ensure_ascii=False) + '\n')
        result = finalize_result(worker=worker, processes=processes, timing=timing,
                                 status=status, error=error, git_state=_git_state(), out=out)
        fallback_model = next(('pt_agent.experiments.r13_common_models:' + short
                               for name, short, _ in RUNS if name == r13_run.name), None)
        result.update(model=(previous or {}).get('model', fallback_model), baseline=baseline,
                      worker_result_read_error=read_error)
        result.update(launch_metadata)
        result['metrics'].update(launch_metadata)
        result['metrics']['baseline'] = baseline
        result['source_r13'] = dict(run=str(r13_run), result=str(r13_run / 'result.json'),
                                   artifacts={key: str(value) for key, value in source.items()},
                                   prior_timing=(previous or {}).get('timing'),
                                   included_in_current_turnaround=False)
        _write_terminal_result(out, result, observations)
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round='R16', model=result['model'],
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
            result.update(status='artifact_budget_exceeded', error='The complete R16 queue exceeds 4 GiB')
    raise RuntimeError('R16 queue artifact accounting did not settle')


def run_queue(args) -> dict:
    out = _check_out(args.out)
    out.mkdir(parents=True)
    plan = [dict(baseline=baseline, r13_run=str(paths.RESULTS_DIR / name), model=short,
                 out=str(out / (baseline + '-' + short)), status='not_started', result=None)
            for name, short, _ in RUNS for baseline in ('orion', 'orbit')]
    result = dict(round='R16', mode=MODE, status='running', error=None, cases=plan,
                  attempted_cases=0, completed_cases=0, maximum_cases=6, retry_count=0,
                  candidate_frozen=False, formal_matrix_complete=False)
    _write_queue_result(out, result)
    for item in plan:
        if result['status'] == 'artifact_budget_exceeded':
            break
        result['attempted_cases'] += 1
        try:
            current = execute(argparse.Namespace(baseline=item['baseline'], r13_run=Path(item['r13_run']),
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
