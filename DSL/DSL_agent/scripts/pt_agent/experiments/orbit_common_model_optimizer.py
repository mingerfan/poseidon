"""Bounded original Orbit common-model optimizer and placed-TDAG validation.

The official model functions and checkpoints are selected explicitly. This outer
process owns results/registry; each inner worker receives all paths as argv.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import threading
import time

from .. import paths
from . import orbit_common_model_optimizer_worker as worker
from . import orbit_common_model_frontend as frontend
from .r23_orbit_optimizer_preflight import _solver_observations
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'ORBIT-COMMON-OPTIMIZER'
MODE = 'original-orbit-common-model-optimizer'
MODULE = 'pt_agent.experiments.orbit_common_model_optimizer_worker'
GIB = 1024 ** 3
MODEL_RECORDS = {
    'resnet20': dict(factory='pt_agent.experiments.r13_common_models:resnet20_silu',
                    checkpoint='resnet20.silu.model', r13='r13-common-resnet20-silu-20261115',
                    r16='orbit-resnet20_silu'),
    'vgg16': dict(factory='pt_agent.experiments.r13_common_models:vgg16_silu_avgpool',
                 checkpoint='vgg16_silu_avgpool_model', r13='r13-common-vgg16-silu-avgpool-20261115',
                 r16='orbit-vgg16_silu_avgpool'),
    'alexnet': dict(factory='pt_agent.experiments.r13_common_models:alexnet_silu_avgpool',
                   checkpoint='alexNet_silu_avgpool_model', r13='r13-common-alexnet-silu-avgpool-20261115',
                   r16='orbit-alexnet_silu_avgpool'),
}


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected an object source record')
    return value


def selected_paths(args):
    root = Path(args.orbit_root or paths.ORBIT_ROOT).resolve()
    python = Path(args.orbit_python or root.parent / 'orbit-venv/bin/python').absolute()
    return dict(model=MODEL_RECORDS[args.model]['factory'], scripts=paths.REPO / 'DSL/DSL_agent/scripts',
                repo=paths.REPO, orbit_root=root, orbit_python=python, profile=root / worker.PROFILE,
                license_file=root.parent / 'orbit-venv/lib/python3.11/site-packages/gurobipy/.libs/gurobi.lic',
                source_record=Path(args.source_result).resolve(), input_mlir=None)




def source_preflight(locations, work):
    for key in ('orbit_python','profile','license_file','source_record'):
        path=locations[key]
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError('Missing existing optimizer source/environment: '+key)
    source=_read(locations['source_record'])
    mlir=(source.get('phase_results',{}).get('trace',{}).get('mlir') or {}).get('path')
    if not isinstance(mlir,str):
        raise ValueError('Source frontend has no actual saved trace output')
    locations['input_mlir']=Path(mlir).resolve()
    proof=work/'source-check';proof.mkdir()
    argv=argparse.Namespace(**locations, work_dir=proof, output_mlir=work/'optimized.mlir')
    # Validate paths/profile/license independently before importing native Orbit modules.
    worker._selected_paths(argv)
    summary=worker._check_frontend_source(argv)
    summary.update(license_metadata=dict(path=str(locations['license_file']),size_bytes=locations['license_file'].stat().st_size),
                   license_contents_read=False, original_model_reopened=False, source_times_included=False)
    return summary




def child_environment(work, locations, env=None):
    inherited,metadata=frontend.child_environment(work,locations,env)
    extra=sorted(key for key in inherited if key.startswith(('GRB_','GUROBI_')))
    for key in extra:inherited.pop(key,None)
    inherited['GRB_LICENSE_FILE']=str(locations['license_file'])
    metadata['removed_environment_keys']=sorted(set(metadata['removed_environment_keys']+extra))
    metadata['fixed_license_path']=str(locations['license_file'])
    return inherited,metadata




def phase_command(locations, work, phase, args, mlir=None):
    return [str(locations['orbit_python']),'-B','-m',MODULE,'--phase',phase,'--model',locations['model'],
            '--work-dir',str(work/(phase+'-worker')),'--orbit-root',str(locations['orbit_root']),
            '--profile',str(locations['profile']),'--input-mlir',str(locations['input_mlir']),
            '--output-mlir',str(work/'optimized.mlir'),'--license-file',str(locations['license_file']),
            '--source-record',str(locations['source_record'] if phase=='optimize' else work/'optimize-worker/phase_result.json')]




def validate_phase(record, phase, locations, work):
    if (record.get('round')!=ROUND or record.get('model')!=locations['model'] or record.get('mode')!=MODE
            or record.get('phase')!=phase or record.get('status')!='ok' or not isinstance(record.get('elapsed_ns'),int)):
        raise ValueError('Optimizer phase identity or terminal completion differs')
    if any(record.get(key)!=0 for key in worker.ZERO_FIELDS):
        raise ValueError('Optimizer phase performed an excluded action')
    if any(record.get(key) is not None for key in worker.UNMEASURED_FIELDS):
        raise ValueError('Unmeasured formal optimizer values must remain null')
    if any(record.get(key) is not False for key in ('backend_ready','formal_matrix_complete','security_validated','accuracy_validated')):
        raise ValueError('Unmeasured backend/security/quality capability claimed')
    if (record.get('source_checked') is not True or record.get('params_checked') is not True
            or record.get('environment_restored') is not True or record.get('budget_monitor_completed') is not True
            or record.get('native_optimizer_completed') is not True or record.get('params_calls')!=1
            or record.get('params_completed')!=1):
        raise ValueError('Optimizer source/parameter/environment stages incomplete')
    params=record.get('params') or {}
    required=dict(mode='compile' if phase=='optimize' else 'execute',poly_deg=131072,max_slot=65536,Sf=51,Sw=40,Csw=40,
                  lvl_lb=1,lvl_ub=16,bts_lb=3,bts_ub=16,bpsdepth=15,threads=1,comp=True,part=True,reqbp=False,
                  netname=worker.frontend_worker.MODEL_RECORDS[locations['model']]['benchmark'],backend='Lattigo',
                  trunc_val=1,dacapo_mlir_in=True,dacapo_mlir_out=True)
    if any(params.get(key)!=value for key,value in required.items()):
        raise ValueError('The model-specific original native parameters differ')
    artifact=record.get('output_mlir') or {};path=Path(artifact.get('path','')).resolve()
    if (path!=(work/'optimized.mlir').resolve() or not path.is_file() or path.stat().st_size<=0
            or path.stat().st_size!=artifact.get('size_bytes')):
        raise ValueError('Actual optimized output file differs')
    if (work/'optimize-worker/first-fatal.json').exists():
        raise ValueError('Caught optimizer exception prohibits validation')
    if phase=='optimize':
        if (record.get('optimize_attempts')!=1 or record.get('optimize_completed')!=1
                or record.get('seed_calls')!=1 or record.get('seed_completed')!=1
                or record.get('observer_restored') is not True or record.get('solve_ilp_errors')!=0):
            raise ValueError('Native optimization/observer stages incomplete')
        return path
    if any(record.get(key)!=1 for key in ('validate_attempts','validate_completed','build_from_mlir_calls',
                                         'build_from_mlir_completed','check_tdag_calls','check_tdag_completed')):
        raise ValueError('Native placement checker incomplete')
    check=record.get('graph_check') or {}
    if record.get('placement_artifact_checked') is not True or check.get('passed') is not True:
        raise ValueError('Placed graph acceptance is absent')
    graph_path=Path(record.get('tdag','')).resolve()
    if graph_path!=(work/'validate-worker/tdag.json').resolve():raise ValueError('Placed TDAG location differs')
    actual=worker._placement_record(_read(graph_path))
    for key in ('node_count','edge_count','op_counts','placement_bootstrap_nodes','placement_bootstrap_weight_sum','bootstraps'):
        if actual[key]!=check.get(key):raise ValueError('Placed TDAG observation differs: '+key)
    return graph_path




def _snapshot(locations, out):
    target=out/'source';target.mkdir()
    sources=[Path(__file__),Path(worker.__file__),Path(frontend.__file__)]
    snapshots=[]
    for index,source in enumerate(sources):
        destination=target/(str(index).zfill(2)+'-'+source.name);shutil.copyfile(source,destination)
        snapshots.append(dict(source=str(source),snapshot=str(destination),size_bytes=destination.stat().st_size))
    return snapshots




def _clean(observation):
    return (observation.get('exit_code') == 0 and not observation.get('timed_out')
            and observation.get('signal') is None and not observation.get('signals_sent')
            and not observation.get('remaining_pids'))


def _terminal(out, result):
    for _ in range(20):
        result['metrics']['status'] = result['status']
        result['metrics']['case_artifact_bytes'] = result.get('case_artifact_bytes', 0)
        write_json(out / 'result.json', result)
        amount = artifact_bytes(out)
        if amount == result.get('case_artifact_bytes'):
            return
        result['case_artifact_bytes'] = amount
    raise RuntimeError('Terminal result artifact size did not stabilize')


def observed_action_counts(result):
    """Missing or interrupted worker evidence leaves excluded action totals unknown."""
    totals = {}
    for key in worker.ZERO_FIELDS:
        amount = 0
        for phase in ('optimize', 'validate'):
            if not result['attempted'][phase]:
                continue
            record = result['phase_results'].get(phase) or {}
            if (not isinstance(record.get('elapsed_ns'), int)
                    or record.get('status') in (None, 'running')
                    or type(record.get(key)) is not int or record[key] < 0):
                amount = None
                break
            amount += record[key]
        totals[key] = amount
    return totals


def solver_observations(work, attempted):
    try:
        return _solver_observations(work, attempted)
    except BaseException as exc:
        return dict(attempts=None, completed=None, feasible=None, infeasible=None, errors=None,
                    model_optimize_calls=None, record_error=f'{type(exc).__name__}: {exc}')


def append_registry(registry, result, out):
    registry.parent.mkdir(parents=True, exist_ok=True)
    row = dict(run_id=out.name, round=ROUND, model=result['model'], mode=MODE,
               git_rev=result.get('git_rev'), metrics=result['metrics'], artifacts_dir=str(out),
               ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('Preserving an existing registry run_id')
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def execute(args):
    started = time.perf_counter_ns()
    locations = selected_paths(args)
    out = Path(args.out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError('A fresh output directory is required')
    out.mkdir(parents=True)
    work = out / 'work'
    (work / 'tmp').mkdir(parents=True)
    recorder = StageRecorder(out / 'events.jsonl')
    deadline = time.monotonic() + args.overall_seconds
    result = dict(round=ROUND, mode=MODE, model=locations['model'], model_key=args.model,
                  status='running', error=None, failed_stage=None, git_rev=None,
                  attempted=dict(optimize=0, validate=0), completed=dict(optimize=0, validate=0),
                  phase_results={}, processes=[], worker_record_errors=[], source_records=None,
                  source_snapshots=[], native_optimizer_completed=False, placement_artifact_checked=False,
                  optimizer_ready=False, backend_ready=False, formal_matrix_complete=False,
                  extra_integrity_checks=0, retry_count=0, case_artifact_bytes=0,
                  resource_monitor_error=None, completion=None)
    result['case_artifact_bytes_scope'] = 'Required artifacts through result persistence; completion.json records the final self-count including its own marker'
    result.update(dict.fromkeys(worker.ZERO_FIELDS, 0))
    result.update(dict.fromkeys(worker.UNMEASURED_FIELDS, None))
    env, result['environment'] = child_environment(work, locations)
    result['limits'] = dict(optimize_seconds=args.optimize_seconds,
                            optimize_supervisor_seconds=args.optimize_seconds+60, optimize_memory_gib=4,
                            validate_seconds=args.validate_seconds, validate_supervisor_seconds=args.validate_seconds+60,
                            validate_memory_gib=4, overall_seconds=args.overall_seconds, case_gib=32,
                            file_gib=8, free_reserve_gib=10, single_concurrency=True,
                            memory_limit_semantics='Per-process RLIMIT_AS; sampled RSS is reported separately')
    stop = threading.Event()
    child_events = []
    saved_handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def interrupted(signum, frame):
        raise RuntimeError('Outer signal ' + str(signum))

    def checkpoint():
        if result['resource_monitor_error']:
            raise RuntimeError(result['resource_monitor_error'])
        if artifact_bytes(out) > args.case_gib * GIB:
            raise RuntimeError('Case artifacts exceed explicit allowance')
        if shutil.disk_usage(out).free < 10 * GIB:
            raise RuntimeError('Remaining disk is below the 10 GiB reserve')
        if time.monotonic() >= deadline:
            raise TimeoutError('Outer overall allowance exhausted')

    def monitor():
        while not stop.wait(.5):
            try:
                amount, free = artifact_bytes(out), shutil.disk_usage(out).free
                with (out / 'artifact-resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(), artifact_bytes=amount, disk_free_bytes=free)) + '\n')
                if amount > args.case_gib * GIB or free < 10 * GIB:
                    raise RuntimeError('Periodic artifact or disk reserve allowance exceeded')
            except BaseException as exc:
                result['resource_monitor_error'] = f'{type(exc).__name__}: {exc}'
                os.kill(os.getpid(), signal.SIGTERM)
                return

    watcher = threading.Thread(target=monitor, daemon=True)

    def launch(phase, mlir=None):
        checkpoint()
        result['failed_stage'] = phase
        argv = phase_command(locations, work, phase, args, mlir)
        result['attempted'][phase] += 1
        process = dict(phase=phase, argv=argv, cwd=str(locations['scripts']), observation=None,
                       native_termination_signal=None)
        result['processes'].append(process)
        write_json(out / 'progress.json', result)
        with recorder.span('orbit.' + phase, argv=argv) as parent:
            try:
                observation = run_process(argv, cwd=locations['scripts'], log=out / (phase + '.log'),
                                          resources=out / (phase + '-resources.jsonl'),
                                          timeout=min(args.optimize_seconds + 60 if phase == 'optimize' else args.validate_seconds + 60,
                                                      max(.01, deadline - time.monotonic())),
                                          mem_gib=4, fsize_gib=8, env=env)
                process['observation'] = observation
                code = observation.get('exit_code')
                process['native_termination_signal'] = -code if isinstance(code, int) and code < 0 else None
            finally:
                phase_dir = work / (phase + '-worker')
                try:
                    worker_events = read_events(phase_dir / 'events.jsonl')
                    if not worker_events:
                        raise ValueError('Inner phase event evidence is missing')
                    for event in worker_events:
                        if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                            event['parent_id'] = parent['event_id']
                        child_events.append(event)
                    record_path = phase_dir / 'phase_result.json'
                    if record_path.is_file():
                        result['phase_results'][phase] = _read(record_path)
                except BaseException as exc:
                    result['worker_record_errors'].append(dict(phase=phase, error=f'{type(exc).__name__}: {exc}'))
            if not _clean(observation):
                raise RuntimeError('Supervised ' + phase + ' failed')
            if any(row['phase'] == phase for row in result['worker_record_errors']):
                raise ValueError('The inner phase result is invalid')
            artifact = validate_phase(result['phase_results'].get(phase, {}), phase, locations, work)
            result['completed'][phase] += 1
        checkpoint()
        return artifact

    try:
        for signum in saved_handlers:
            signal.signal(signum, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.overall_seconds)
        watcher.start()
        with recorder.span('orbit.common.optimizer'):
            result['failed_stage'] = 'source_preflight'
            with recorder.span('source_preflight'):
                checkpoint()
                result['source_records'] = source_preflight(locations, work)
                write_json(out / 'source-metadata.json', result['source_records'])
                result['source_snapshots'] = _snapshot(locations, out)
                write_json(out / 'source-snapshots.json', result['source_snapshots'])
            result['failed_stage'] = 'git_metadata'
            with recorder.span('git_metadata'):
                obs = run_process(['git', 'rev-parse', '--short', 'HEAD'], cwd=locations['repo'],
                                  log=out / 'git-revision.log', resources=out / 'git-resources.jsonl',
                                  timeout=min(15, max(.01, deadline-time.monotonic())), mem_gib=.5, fsize_gib=8, env=env)
                result['git_process'] = obs
                if not _clean(obs):
                    raise RuntimeError('Git source metadata failed')
                result['git_rev'] = (out / 'git-revision.log').read_text().strip()
            launch('optimize')
            launch('validate')
            result.update(status='ok', error=None, failed_stage=None,
                          native_optimizer_completed=True, placement_artifact_checked=True)
    except BaseException as exc:
        result.update(status='stopped_on_error', error=f'{type(exc).__name__}: {exc}'[:4000])
    finally:
        # Supervised children have already been reaped. Persist failures also in the gaps between children.
        stop.set()
        if watcher.is_alive():
            watcher.join()
        try:
            ended = time.perf_counter_ns()
            events = read_events(out / 'events.jsonl') + child_events
            with (out / 'combined-events.jsonl').open('w') as stream:
                for event in events:
                    stream.write(json.dumps(event) + '\n')
            timing = summarize_events(events, termination_observed_ns=ended)
            timing['outer_interval'] = dict(start_ns=started, end_ns=ended, elapsed_ns=ended-started,
                                            boundary='Pre-persistence observation; complete endpoint is in completion.json')
            write_json(out / 'timing.json', timing)
            if result['resource_monitor_error']:
                result.update(status='stopped_on_error', error=result['resource_monitor_error'])
            if not timing['attribution_ready'] or timing['diagnostics']:
                result.update(status='stopped_on_error', timing_record_error=True)
            optimized, validated = (result['phase_results'].get(k, {}) for k in ('optimize', 'validate'))
            actions = observed_action_counts(result)
            result.update(actions)
            result['metrics'] = dict(status=result['status'], attempted=dict(result['attempted']), completed=dict(result['completed']),
                                     native_optimizer_completed=result['native_optimizer_completed'], placement_artifact_checked=result['placement_artifact_checked'],
                                     optimizer_ready=False, backend_ready=False, formal_matrix_complete=False,
                                     optimize_worker_counts={k: optimized.get(k, None if result['attempted']['optimize'] else 0) for k in worker.COUNT_FIELDS},
                                     validate_worker_counts={k: validated.get(k, None if result['attempted']['validate'] else 0) for k in worker.COUNT_FIELDS},
                                     solve_ilp=solver_observations(work,result['attempted']['optimize']),
                                     external_profile_latency_kind='Original external microsecond table; no local calibration',
                                     recorded_pre_persistence_elapsed_seconds=(ended-started)/1e9,
                                     full_elapsed_artifact=str(out / 'completion.json'),
                                     timing_attribution_ready=timing['attribution_ready'],
                                     **actions, **dict.fromkeys(worker.UNMEASURED_FIELDS, None))
            result['outer_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['artifacts'] = dict(timing=str(out / 'timing.json'), events=str(out / 'combined-events.jsonl'),
                                       optimize_result=str(work / 'optimize-worker/phase_result.json'),
                                       validate_result=str(work / 'validate-worker/phase_result.json'), completion=str(out / 'completion.json'))
            result['completion_reserved_bytes'] = frontend.completion_reserve_bytes(timing)
            _terminal(out, result)
            # Include terminal result and registry persistence in the measured completion boundary.
            try:
                checkpoint()
                if artifact_bytes(out) + result['completion_reserved_bytes'] > args.case_gib * GIB:
                    raise RuntimeError('Case allowance must cover the completion marker before registry persistence')
            except BaseException as exc:
                result.update(status='stopped_on_error', terminal_budget_error=f'{type(exc).__name__}: {exc}')
                _terminal(out, result)
            append_registry(Path(args.registry or paths.RESULTS_DIR / 'registry.jsonl'), result, out)
            marker = persist_case_completion(out, round_label=ROUND, started_ns=started,
                                             status=result['status'], timing=timing,
                                             required_artifact_bytes=artifact_bytes(out))
            if marker['case_artifact_bytes'] > args.case_gib * GIB:
                raise RuntimeError('Completion marker exhausted the case artifact allowance')
            result['completion'] = marker
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum, handler in saved_handlers.items():
                signal.signal(signum, handler)
            if previous_timer[0]:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    return result


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=tuple(MODEL_RECORDS))
    parser.add_argument('--source-result',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    for name in ('registry','orbit-root','orbit-python'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--optimize-seconds',type=float,default=7200)
    parser.add_argument('--validate-seconds',type=float,default=300)
    parser.add_argument('--overall-seconds',type=float,default=8400)
    parser.add_argument('--case-gib',type=float,default=32)
    return parser




def main(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    if (not 0<args.optimize_seconds<=7200 or not 0<args.validate_seconds<=300
            or not 0<args.overall_seconds<=8400 or not 0<args.case_gib<=32):
        parser.error('Positive limits within 7200s optimize/300s validate/8400s overall/32GiB case are required')
    return 0 if execute(args)['status']=='ok' else 1




if __name__ == '__main__':
    raise SystemExit(main())
