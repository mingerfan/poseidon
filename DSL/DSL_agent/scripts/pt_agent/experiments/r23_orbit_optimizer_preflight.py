"""R23: one native Orbit optimizer attempt using the saved full R22 model."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import sys
import threading
import time

from .. import paths
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r22_orbit_model_preflight import case_size

MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-resnet20-native-optimizer-preflight'
OUT_NAME = 'orbit-resnet20-optimizer-r23-20261115'
SOURCE_NAME = 'orbit-resnet20-trace-tdag-r22-20261115'
PROFILE = Path('cost_models/profiled_LATTIGONEW_CPU64k_3_16.json')
MODULE = 'pt_agent.experiments.r23_orbit_optimizer_worker'
CATEGORIES = ('tests', 'optimize', 'validate')
OVERALL_SECONDS = 8400
CASE_BYTES, FREE_BYTES = 4 * 1024**3, 10 * 1024**3
FORMAL = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')


def _locations():
    root = paths.ORBIT_ROOT.resolve()
    python = root.parent / 'orbit-venv/bin/python'
    return dict(results=paths.RESULTS_DIR.resolve(), repo=paths.REPO,
                scripts=paths.REPO / 'DSL/DSL_agent/scripts', orbit=root,
                orbit_python=python, profile=root / PROFILE,
                license_file=python.parent.parent / 'lib/python3.11/site-packages/gurobipy/.libs/gurobi.lic',
                r22_result=paths.RESULTS_DIR / SOURCE_NAME / 'result.json')


def _source_preflight(locations):
    required = [locations[k] for k in ('orbit_python', 'profile', 'license_file', 'r22_result')]
    if any(not p.is_file() or not p.stat().st_size for p in required):
        raise ValueError('Missing approved R23 source or existing environment')
    expected_license = locations['orbit_python'].parent.parent / 'lib/python3.11/site-packages/gurobipy/.libs/gurobi.lic'
    if locations['license_file'].absolute() != expected_license.absolute():
        raise ValueError('R23 requires the fixed existing bundled restricted license')
    if locations['profile'].resolve() != (locations['orbit'] / PROFILE).resolve():
        raise ValueError('R23 requires the original 64k profile')
    r = json.loads(locations['r22_result'].read_text())
    if (r.get('round') != 'R22' or r.get('model') != MODEL
            or r.get('mode') != 'orbit-resnet20-trace-tdag-preflight' or r.get('status') != 'ok'
            or r.get('orbit_model_trace_passed') is not True or r.get('tdag_parse_passed') is not True
            or any(r.get(k, {}).get(p) != 1 for k in ('attempted', 'completed') for p in ('trace', 'parse'))):
        raise ValueError('R23 requires the saved successful full R22 model')
    trace, parse = (r.get('phase_results', {}).get(p, {}) for p in ('trace', 'parse'))
    if (trace.get('status') != 'ok' or parse.get('status') != 'ok'
            or trace.get('trace_runs') != 1 or parse.get('parse_runs') != 1
            or trace.get('activation_calls') != 19 or len(trace.get('activation_names', [])) != 19):
        raise ValueError('R22 full-model counts differ')
    gc, metadata = parse.get('graph_check', {}), parse.get('parser_metadata', {})
    if (gc.get('passed') is not True or gc.get('node_count') != 10113 or gc.get('edge_count') != 13342
            or metadata.get('poly_deg') != 131072 or metadata.get('max_slot') != 65536
            or Path(metadata.get('profile', '')).resolve() != locations['profile'].resolve()):
        raise ValueError('R22 graph/profile metadata differ')
    mlir_record = trace.get('mlir', {})
    mlir = Path(mlir_record.get('path', '')).resolve()
    expected_trace = locations['r22_result'].parent / 'work/trace-worker/trace'
    if (not mlir.is_relative_to(expected_trace.resolve()) or not mlir.is_file()
            or mlir.stat().st_size <= 0 or mlir.stat().st_size != mlir_record.get('size_bytes')):
        raise ValueError('R23 input must be the saved nonempty R22 trace file')
    return dict(r22_result=str(locations['r22_result']), input_mlir=str(mlir),
                model=MODEL, source_graph_nodes=10113, source_graph_edges=13342,
                source_activations=19, coefficient_source_equivalence_verified=False,
                training_preprocessing_verified=False, security_validated=False,
                inputs=[dict(path=str(p), bytes=p.stat().st_size) for p in required + [mlir]],
                license_scope='Existing fixed bundled restricted license; contents not read by preparation')


def phase_command(locations, work, phase, input_mlir):
    source = locations['r22_result'] if phase == 'optimize' else work / 'optimize-worker/phase_result.json'
    return [str(locations['orbit_python']), '-B', '-m', MODULE, '--phase', phase,
            '--work-dir', str(work / (phase + '-worker')), '--orbit-root', str(locations['orbit']),
            '--profile', str(locations['profile']), '--input-mlir', str(input_mlir),
            '--output-mlir', str(work / 'optimized.mlir'), '--license-file', str(locations['license_file']),
            '--source-record', str(source)]


def child_environment(work, locations):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('LD_', 'NIX_', '__NIX_', 'HECATE_', 'GRB_', 'GUROBI_'))
           and k not in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', MPLBACKEND='Agg',
               R23_TEST_WORK=str(work / 'test-worker'), R23_ORBIT_PYTHON=str(locations['orbit_python']),
               GRB_LICENSE_FILE=str(locations['license_file']))
    for key, dirname in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'),
                         ('XDG_CACHE_HOME', 'cache'), ('XDG_CONFIG_HOME', 'config'), ('MPLCONFIGDIR', 'mpl')):
        env[key] = str(work / dirname)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def check_budget(out):
    amount = case_size(out)
    if amount > CASE_BYTES or shutil.disk_usage(out).free < FREE_BYTES:
        raise RuntimeError('R23 artifact or disk reserve allowance exceeded')
    return amount


def write_terminal_result(out, result):
    for _ in range(20):
        result['metrics'].update(status=result['status'], case_artifact_bytes=result.get('case_artifact_bytes', 0))
        write_json(out / 'result.json', result)
        amount = case_size(out)
        if amount == result.get('case_artifact_bytes'):
            return
        result['case_artifact_bytes'] = result['artifact_bytes'] = amount
    raise RuntimeError('R23 terminal artifact-size fixed point did not converge')


def append_registry(out, result, registry):
    row = dict(run_id=out.name, round='R23', model=MODEL, mode=MODE, git_rev=result['git_rev'],
               metrics=result['metrics'], artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def _solver_observations(work, optimize_attempted):
    events = []
    path = work / 'optimize-worker/solve-ilp-events.jsonl'
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                return dict(attempts=None, completed=None, feasible=None, infeasible=None, errors=None,
                            model_optimize_calls=None, incomplete_event_line=True)
    # No solver-event file after a killed worker gives no evidence that no calls ran.
    if not events:
        worker = work / 'optimize-worker/phase_result.json'
        data = json.loads(worker.read_text()) if worker.is_file() else {}
        known = not optimize_attempted or ('elapsed_ns' in data and data.get('status') not in (None, 'running'))
        return dict(attempts=data.get('solve_ilp_attempts', 0) if known else None,
                    completed=data.get('solve_ilp_completed', 0) if known else None,
                    feasible=data.get('solve_ilp_feasible', 0) if known else None,
                    infeasible=data.get('solve_ilp_infeasible', 0) if known else None,
                    errors=data.get('solve_ilp_errors', 0) if known else None, model_optimize_calls=None)
    kinds = [e.get('type') for e in events]
    if any(k not in ('begin', 'end', 'error') for k in kinds):
        raise ValueError('Unknown R23 solver event type')
    ended = [e for e in events if e['type']=='end']
    started = [e for e in events if e['type']=='begin']
    closed = {e.get('call_index') for e in events if e['type'] in ('end','error')}
    return dict(events=events, event_count=len(events), attempts=len(started), completed=len(ended),
                feasible=sum(e.get('result')=='solution' for e in ended),
                infeasible=sum(e.get('result')=='infeasible' for e in ended), errors=kinds.count('error'),
                open_call_indices=[e.get('call_index') for e in started if e.get('call_index') not in closed],
                model_optimize_calls=None)


def _external_profile_estimates(log, profile):
    text = log.read_text() if log.is_file() else ''
    values = {}
    for key, expression in (('assignment_seconds', r'Final assignment latency: ([0-9.eE+-]+) sec\.'),
                             ('tdag_seconds', r'Final tdag latency: ([0-9.eE+-]+) sec\.')):
        found = re.findall(expression, text)
        values[key] = float(found[-1]) if found else None
    stages = re.findall(r"^\s+'([^']+)': ([0-9.eE+-]+) sec\.$", text, re.M)
    return dict(profile=str(profile), source='Original native optimizer log using external latencyTable',
                units='seconds printed by native code after dividing original estimate by 1e6',
                observed_on_this_machine=False, values=values,
                native_stage_seconds={name: float(value) for name, value in stages})


def execute(args):
    started = time.perf_counter_ns()
    deadline = time.monotonic() + OVERALL_SECONDS
    locations = _locations()
    out = Path(args.out).absolute()
    expected = locations['results'] / OUT_NAME
    if out.exists() or out.is_symlink() or out.resolve() != expected.resolve():
        raise ValueError('R23 requires the single new approved result directory')
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))
    saved_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    saved_timer = signal.getitimer(signal.ITIMER_REAL)
    received_signals = []
    def expired(signum, frame):
        received_signals.append(signum)
        raise RuntimeError('R23 outer signal ' + str(signum))
    for sig in saved_handlers:
        signal.signal(sig, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.monotonic()))
    out.mkdir()
    work = out / 'work'
    work.mkdir()
    for name in ('tmp', 'cache', 'config', 'mpl', 'test-worker'):
        (work / name).mkdir()
    recorder = StageRecorder(out / 'events.jsonl')
    result = dict(run_id=out.name, round='R23', model=MODEL, mode=MODE, status='running', error=None,
                  failed_stage=None, git_rev=None, git_branch=None, attempted=dict.fromkeys(CATEGORIES, 0),
                  completed=dict.fromkeys(CATEGORIES, 0), processes=[], phase_results={}, retry_count=0,
                  git_metadata_attempted=0, git_metadata_completed=0, worker_record_errors=[],
                  first_fatal=None, termination_origin=None, resource_monitor_error=None, outer_received_signals=received_signals,
                  artifact_bytes=0, case_artifact_bytes=0, source_records=None,
                  native_optimizer_completed=False, placement_artifact_checked=False,
                  placement_bootstrap_nodes=None, placement_bootstrap_weight_sum=None,
                  backend_ready=False, security_validated=False, accuracy_validated=False,
                  formal_matrix_complete=False, new_llm_calls=0, new_downloads=0, new_installations=0,
                  model_forward_runs=0, trace_runs=0, sample_reads=0, keygens=0, ckks_runs=0,
                  confirm_samples=0, report_samples=0, candidates_frozen=0, extra_integrity_checks=0,
                  limited_original_launcher_check_used=False, native_go_verification_exception_used=False,
                  model_optimize_calls=None)
    env = child_environment(work, locations)
    stop, child_events = threading.Event(), []
    def checkpoint():
        if result['resource_monitor_error']:
            raise RuntimeError(result['resource_monitor_error'])
        check_budget(out)
        if time.monotonic() >= deadline:
            raise TimeoutError('R23 entire-batch allowance exhausted')
    def monitor():
        while not stop.wait(1):
            try:
                amount, free = case_size(out), shutil.disk_usage(out).free
                with (out / 'artifact-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(), artifact_bytes=amount, disk_free_bytes=free)) + '\n')
                if amount > CASE_BYTES or free < FREE_BYTES:
                    raise RuntimeError('R23 periodic artifact/disk allowance exceeded')
            except BaseException as exc:
                result['resource_monitor_error'] = f'{type(exc).__name__}: {exc}'
                os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    def launch(name, argv, seconds, category=None):
        checkpoint()
        result['failed_stage'] = name
        if category:
            result['attempted'][category] += 1
        else:
            result['git_metadata_attempted'] += 1
        cwd = work if category in ('optimize', 'validate') else locations['scripts']
        # Explicit argv carries the repo module path; cwd can remain in the new work.
        if category in ('optimize', 'validate'):
            command_env = dict(env, PYTHONPATH=str(locations['scripts']))
        else:
            command_env = env
        command = dict(stage=name, argv=[str(x) for x in argv], cwd=str(cwd), observation=None, native_termination_signal=None)
        result['processes'].append(command)
        print(json.dumps(dict(stage=name, state='started')), flush=True)
        with recorder.span(name, command=command['argv']) as parent:
            try:
                obs = run_process(command['argv'], cwd=cwd, log=out / (name + '.log'),
                                  resources=out / (name + '-resources.jsonl'),
                                  timeout=min(seconds, max(.01, deadline-time.monotonic())),
                                  mem_gib=4, fsize_gib=8, env=command_env)
                command['observation'] = obs
                if obs.get('signal') is not None:
                    received_signals.append(obs['signal'])
                code = obs['exit_code']
                command['native_termination_signal'] = -code if isinstance(code, int) and code < 0 else None
            finally:
                if category in ('optimize', 'validate'):
                    worker_dir = work / (category + '-worker')
                    try:
                        for event in read_events(worker_dir / 'events.jsonl'):
                            if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                                event['parent_id'] = parent['event_id']
                            child_events.append(event)
                        record = worker_dir / 'phase_result.json'
                        if record.is_file():
                            result['phase_results'][category] = json.loads(record.read_text())
                        fatal = worker_dir / 'first-fatal.json'
                        if fatal.is_file():
                            result['first_fatal'] = json.loads(fatal.read_text())
                            result['termination_origin'] = 'worker_caught_exception'
                        for filename in ('budget-fatal.json', 'budget-monitor-error.json'):
                            budget_record = worker_dir / filename
                            if budget_record.is_file():
                                observed_budget = json.loads(budget_record.read_text())
                                result.setdefault('worker_budget_records', []).append(observed_budget)
                                if result['termination_origin'] is None:
                                    result['termination_origin'] = observed_budget.get('termination_origin')
                    except BaseException as exc:
                        result['worker_record_errors'].append(dict(phase=category, error=f'{type(exc).__name__}: {exc}'))
            if code != 0 or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']:
                raise RuntimeError('R23 supervised phase failed: ' + name)
            if category in ('optimize', 'validate'):
                worker = result['phase_results'].get(category, {})
                flag = 'native_optimizer_completed' if category == 'optimize' else 'placement_artifact_checked'
                if (any(x['phase']==category for x in result['worker_record_errors'])
                        or result['first_fatal'] is not None or worker.get('status') != 'ok'
                        or worker.get(flag) is not True):
                    raise RuntimeError('R23 worker did not pass: ' + name)
            if category:
                result['completed'][category] += 1
            else:
                result['git_metadata_completed'] += 1
        checkpoint()
        print(json.dumps(dict(stage=name, state='completed')), flush=True)
        return obs
    try:
        watcher.start()
        with recorder.span('r23-preflight'):
            result['failed_stage'] = 'source-preflight'
            with recorder.span('source-preflight'):
                checkpoint()
                result['source_records'] = _source_preflight(locations)
                write_json(out / 'source-metadata.json', result['source_records'])
                source_dir = Path(__file__).parent.parent
                snapshot = work / 'source'
                for relative in ('experiments/r23_orbit_optimizer_worker.py', 'experiments/r23_orbit_optimizer_preflight.py',
                                 'tests/_r23_orbit_optimizer_cases.py', 'tests/test_r23_orbit_optimizer.py'):
                    target = snapshot / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_dir / relative, target)
            launch('git-revision', ['git', 'rev-parse', '--short', 'HEAD'], 30)
            result['git_rev'] = (out / 'git-revision.log').read_text().strip()
            launch('git-branch', ['git', 'branch', '--show-current'], 30)
            result['git_branch'] = (out / 'git-branch.log').read_text().strip()
            launch('r23-tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'pt_agent/tests', '-p', 'test_r23_*.py', '-v'], 600, 'tests')
            log = (out / 'r23-tests.log').read_text()
            counts = re.findall(r'Ran (\d+) tests? in ', log)
            if not counts or int(counts[-1]) <= 0 or 'skipped=' in log or '... skipped ' in log:
                raise RuntimeError('R23 requires positive tests and zero skips')
            summary = json.loads((work / 'test-worker/inner-wrapper-result.json').read_text())
            if (summary.get('status') != 'ok' or summary.get('wrapper_runs') != 1
                    or not isinstance(summary.get('inner_tests'), int) or summary['inner_tests'] <= 0
                    or summary.get('inner_skips') != 0):
                raise RuntimeError('R23 unique inner fake suite did not pass')
            result['test_summary'] = dict(system_tests=int(counts[-1]), inner=summary)
            mlir = result['source_records']['input_mlir']
            launch('model-optimize', phase_command(locations, work, 'optimize', mlir), 7200, 'optimize')
            result['native_optimizer_completed'] = True
            if not (work / 'optimized.mlir').is_file() or not (work / 'optimized.mlir').stat().st_size:
                raise RuntimeError('R23 native optimizer returned no nonempty output')
            launch('model-validate', phase_command(locations, work, 'validate', mlir), 300, 'validate')
            result['placement_artifact_checked'] = True
            valid = result['phase_results']['validate']
            for key in ('placement_bootstrap_nodes', 'placement_bootstrap_weight_sum'):
                result[key] = valid.get(key)
            result.update(status='ok', failed_stage=None)
    except BaseException as exc:
        result.update(status='stopped_on_error', error=f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join()
        ended = time.perf_counter_ns()
        events = read_events(out / 'events.jsonl') + child_events
        with (out / 'combined-events.jsonl').open('w') as stream:
            for event in events:
                stream.write(json.dumps(event) + '\n')
        timing = summarize_events(events, termination_observed_ns=ended)
        timing['outer_interval'] = dict(start_ns=started, end_ns=ended, elapsed_ns=ended-started,
                                       boundary='Before terminal timing/result fixed-point writes and registry append')
        write_json(out / 'timing.json', timing)
        result['timing'] = dict(attribution_ready=timing['attribution_ready'], diagnostics=timing['diagnostics'],
                                span_count=len(timing['spans']), recorded_outer_seconds=(ended-started)/1e9,
                                complete_batch_elapsed_seconds=None, boundary=timing['outer_interval']['boundary'])
        if result['resource_monitor_error']:
            result.update(status='stopped_on_error', error=result['resource_monitor_error'])
        # A stopped worker can leave open spans. Their unknown durations remain null.
        if timing['diagnostics'] or (result['status']=='ok' and not timing['attribution_ready']):
            result.update(status='stopped_on_error', timing_record_error=True)
        try:
            result['solver_observations'] = _solver_observations(work, result['attempted']['optimize'])
        except BaseException as exc:
            result['solver_observations'] = dict(attempts=None, completed=None, model_optimize_calls=None, error=f'{type(exc).__name__}: {exc}')
            result['worker_record_errors'].append(dict(phase='solver_records', error=result['solver_observations']['error']))
            result['status'] = 'stopped_on_error'
            if result['error'] is None:
                result['error'] = result['solver_observations']['error']
        try:
            result['external_profile_estimate'] = _external_profile_estimates(out / 'model-optimize.log', locations['profile'])
            result['native_stage_times'] = dict(seconds=result['external_profile_estimate'].pop('native_stage_seconds'),
                                                clock='Original optimizer time.time intervals', observed_on_this_machine=True)
        except BaseException as exc:
            record_error = f'{type(exc).__name__}: {exc}'
            result['external_profile_estimate'] = dict(values=None, error=record_error, observed_on_this_machine=False)
            result['native_stage_times'] = dict(seconds=None, clock='Original optimizer time.time intervals', observed_on_this_machine=True)
            result['worker_record_errors'].append(dict(phase='native_log_records', error=record_error))
            result['status'] = 'stopped_on_error'
            if result['error'] is None:
                result['error'] = record_error
        result['root_raw_getrusage_hwm_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        opt = result['phase_results'].get('optimize', {})
        result['metrics'] = dict(status=result['status'], preflight_only=True,
                                 attempted=dict(result['attempted']), completed=dict(result['completed']),
                                 native_optimizer_completed=result['native_optimizer_completed'],
                                 placement_artifact_checked=result['placement_artifact_checked'],
                                 placement_bootstrap_nodes=result['placement_bootstrap_nodes'],
                                 placement_bootstrap_weight_sum=result['placement_bootstrap_weight_sum'],
                                 optimizer_attempts=opt.get('optimize_attempts', None if result['attempted']['optimize'] else 0),
                                 optimizer_completed=opt.get('optimize_completed', None if result['attempted']['optimize'] else 0),
                                 model_optimize_calls=None, solver_counts={k: result['solver_observations'].get(k) for k in ('attempts','completed','feasible','infeasible','errors')},
                                 formal_matrix_complete=False,
                                 model_forward_runs=0, sample_reads=0, new_llm_calls=0, trace_runs=0,
                                 keygens=0, ckks_runs=0, **dict.fromkeys(FORMAL, None))
        write_terminal_result(out, result)
        try:
            check_budget(out)
        except BaseException as exc:
            result.update(status='stopped_on_error', terminal_budget_error=f'{type(exc).__name__}: {exc}')
            if result['error'] is None:
                result['error'] = result['terminal_budget_error']
            write_terminal_result(out, result)
        append_registry(out, result, locations['results'] / 'registry.jsonl')
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in saved_handlers.items():
            signal.signal(sig, handler)
        if saved_timer[0] or saved_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, *saved_timer)
        print(json.dumps(dict(status=result['status'], error=result['error'], failed_stage=result['failed_stage'],
                              out=str(out), attempted=result['attempted'], completed=result['completed'], artifact_bytes=result['artifact_bytes'])), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    return 0 if execute(parser.parse_args(argv))['status']=='ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
