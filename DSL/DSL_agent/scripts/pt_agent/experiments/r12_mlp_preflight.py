"""R12 outer owner: one serial official-MLP chain and observed CKKS evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .. import paths
from .r10_paper_metrics import (StageRecorder, UnsupportedSupervision, read_events,
                               run_process, summarize_events, write_json)


PHASES = ('prepare', 'trace', 'compile', 'build', 'verify-client', 'execute')
COMPILE_MODES = ('dacapo-optimized', 'eva')
# Wrapper budgets leave the specified internal operation budgets intact.
LIMITS = {
    'prepare': (3660, 4), 'trace': (3660, 6), 'compile': (7260, 6),
    'build': (3720, 4), 'verify-client': (3660, 4), 'execute': (3660, 4),
}


class DependencyMissing(RuntimeError):
    pass


class PhaseStopped(RuntimeError):
    def __init__(self, phase, status, reason):
        super().__init__(reason)
        self.phase, self.status = phase, status


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _git_state():
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=paths.REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=paths.REPO,
                           capture_output=True, text=True, check=True).stdout
    return dict(branch=branch, working_tree_changed=bool(dirty))


def finalize_result(*, phase, worker, processes, timing, status, error, git_state, out,
                    compile_mode='dacapo-optimized'):
    """Summarize observed R12 fields; incomplete execution stays unknown."""
    records = worker or {}
    compiled = records.get('compile') or {}
    executed = records.get('execute') or {}
    client = executed.get('client_result')
    started = 'execute' in records or any(p.get('label') == 'execute' for p in processes)
    metrics = dict(
        llm_calls=0, compile_mode=compile_mode,
        real_ckks_attempts=(client.get('encrypted_execution_attempts') if client else
                            (None if started else 0)),
        real_ckks_completed=(client.get('encrypted_execution_completed') if client else
                             (None if started else 0)),
        execution_skipped=not started,
        ckks_tolerance_passed=client.get('ckks_tolerance_passed') if client else None,
        compiled_bootstraps=(compiled.get('plan_stats') or {}).get('boots'),
        calibrated_predictions=client.get('predictions') if client else None,
        compiler_profile_prediction=compiled.get('compiler_profile_prediction'),
        phase_elapsed_ns={s['stage_id']: s.get('inclusive_ns')
                          for s in timing.get('spans', []) if s['stage_id'] in PHASES})
    return dict(run_id=Path(out).name, round='R12', spec_revision='R12b',
                compile_mode=compile_mode, phase=phase, status=status,
                error=error, git_rev=git_state, worker=records, processes=processes,
                timing=timing, metrics=metrics)


def _command(phase, work_dir, **values):
    argv = ['-m', 'pt_agent.experiments.r12_mlp_worker', '--phase', phase,
            '--work-dir', str(work_dir)]
    for key, value in values.items():
        argv.extend(['--' + key.replace('_', '-'), str(value)])
    return argv


def execute(args):
    compile_mode = getattr(args, 'compile_mode', 'dacapo-optimized')
    if compile_mode not in COMPILE_MODES:
        raise ValueError('Unknown R12 compile mode')
    root = paths.RESULTS_DIR.resolve()
    out = Path(args.out).absolute()
    if not out.resolve().is_relative_to(root) or out.resolve() == root:
        raise ValueError('--out must name a new child directory of RESULTS_DIR')
    if out.exists():
        raise ValueError('Existing run directories are preserved')
    out.mkdir(parents=True)
    work = out / 'worker'
    work.mkdir()
    recorder = StageRecorder(out / 'events.jsonl')
    records, processes, child_events = {}, [], []
    status, error, current = 'ok', None, 'preflight'
    started_ns = time.perf_counter_ns()
    source = paths.DACAPO_UPSTREAM / 'examples/benchmarks/MLP.py'
    official_model = paths.DACAPO_UPSTREAM / 'examples/data/mlp.model'
    native = paths.WORK_RESULTS / 'agent-normal-cpu-approved-overlay-staging-r2680/sdk/library-view/libposeidon_shared.so'
    generated = paths.WORK_RESULTS / 'agent-current-cpu-build-r1701/build/poseidon'
    profile = paths.PROFILES_DIR / 'poseidon-compiler-profile-v1.json'
    spec = paths.PROFILES_DIR / 'poseidon-operator-spec-v1.json'
    model, mnist = Path(args.model).absolute(), Path(args.mnist).absolute()

    def artifact(record, key):
        value = (record.get('artifacts') or {}).get(key)
        if not value:
            raise PhaseStopped(current, 'failed', 'Missing artifact field: ' + key)
        path = Path(value)
        if not path.exists():
            raise PhaseStopped(current, 'failed', 'Missing artifact: ' + str(path))
        return path

    def launch(phase, **values):
        nonlocal current
        current = phase
        if shutil.disk_usage(out).free < 10 * 1024**3:
            raise DependencyMissing('At least 10 GiB free disk space is required before each phase')
        directory = work / phase
        directory.mkdir()
        seconds, memory = LIMITS[phase]
        argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun',
                '--seconds', str(seconds - 20)]
        if phase in ('prepare', 'trace'):
            argv += ['--python', str(paths.INTAKE_PYTHON)]
        argv += ['--', *_command(phase, directory, **values)]
        result_path = directory / (phase.replace('-', '_') + '_result.json')
        process = None
        with recorder.span(phase) as span:
            try:
                process = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                    log=out / (phase + '.log'), resources=out / 'resources.jsonl',
                    timeout=seconds, mem_gib=memory,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                         'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                         'MKL_NUM_THREADS': '1'})
                processes.append(dict(label=phase, command=argv, **process))
            finally:
                for event in read_events(directory / 'events.jsonl'):
                    if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                        event['parent_id'] = span['event_id']
                    child_events.append(event)
                record = _load(result_path)
                if record is None:
                    record = dict(phase=phase, status='result_missing', error='Worker did not write its result', artifacts={})
                if phase == 'execute' and record.get('client_result') is None:
                    location = (record.get('artifacts') or {}).get('client_result')
                    if location:
                        record['client_result'] = _load(location)
                records[phase] = record
            if process is None:
                raise PhaseStopped(phase, 'failed', 'Phase was interrupted before process status')
            if record.get('status') != 'ok':
                raise PhaseStopped(phase, record.get('status', 'failed'), str(record.get('error')))
            if process['exit_code'] != 0 or process['timed_out'] or process['signal'] or process['remaining_pids']:
                raise PhaseStopped(phase, 'process_failed', str({k: process[k] for k in
                    ('exit_code', 'timed_out', 'signal', 'remaining_pids')}))
        return record

    try:
        with recorder.span('turnaround'):
            if model.resolve() != official_model.resolve():
                raise ValueError('R12 uses the specified official artifact checkpoint')
            required = [model, mnist, source, native, generated / 'poseidon/util/config.h',
                        paths.HECATE_BUILD / 'bin/hecate-opt', paths.INTAKE_PYTHON,
                        paths.REPO / 'third_party/ckks-runtime/CMakeLists.txt', profile, spec]
            missing = [str(p) for p in required if not (
                p.exists() or (p == paths.INTAKE_PYTHON and p.is_symlink()))]
            if missing:
                raise DependencyMissing('Missing local prerequisites: ' + ', '.join(missing))
            if shutil.disk_usage(out).free < 10 * 1024**3:
                raise DependencyMissing('At least 10 GiB free disk space is required')
            prepared = launch('prepare', model=model, mnist=mnist, source=source)
            if prepared.get('math', {}).get('passed') is not True:
                raise PhaseStopped(current, 'math_failed', 'Independent and packed mathematics did not pass')
            samples = artifact(prepared, 'samples')
            traced = launch('trace', model=model, source=source)
            mlir = artifact(traced, 'mlir')
            compiled = launch('compile', input=mlir, hecate_opt=paths.HECATE_BUILD / 'bin/hecate-opt',
                              compiler_profile=profile, operator_spec=spec, compile_mode=compile_mode)
            plan = artifact(compiled, 'plan')
            built = launch('build', repo_root=paths.REPO, runtime_root=paths.REPO / 'third_party/ckks-runtime',
                           client_source_dir=Path(__file__).resolve().parent / 'r12_native_cpu',
                           generated_include_dir=generated, native_library=native,
                           build_dir=work / 'build' / 'native')
            client = artifact(built, 'client')
            verified = launch('verify-client', client=client, plan=plan, spec=spec,
                              inputs=samples, native_library=native)
            verification = artifact(verified, 'verification')
            launch('execute', client=client, plan=plan, spec=spec, inputs=samples,
                   native_library=native, verification=verification)
    except DependencyMissing as exc:
        status, error = 'dependency_missing', str(exc)
    except PhaseStopped as exc:
        status, error = exc.status, str(exc)
    except UnsupportedSupervision as exc:
        status, error = 'unsupported_process_supervision', str(exc)
    except BaseException as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'[:2000]
    finally:
        ended_ns = time.perf_counter_ns()
        timing = summarize_events(read_events(out / 'events.jsonl') + child_events,
                                  termination_observed_ns=ended_ns)
        timing['outer_interval'] = dict(start_ns=started_ns, end_ns=ended_ns,
                                        elapsed_ns=ended_ns - started_ns)
        with (out / 'stages.jsonl').open('w') as stream:
            for span in timing['spans']:
                stream.write(json.dumps(span, ensure_ascii=False) + '\n')
        result = finalize_result(phase=current, worker=records, processes=processes,
                                timing=timing, status=status, error=error,
                                git_state=_git_state(), out=out, compile_mode=compile_mode)
        write_json(out / 'result.json', result)
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round='R12', model=str(model),
                mode='official-artifact-mlp-native-ckks' + ('-eva' if compile_mode == 'eva' else ''),
                git_rev=result['git_rev'],
                metrics=result['metrics'], artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()), ensure_ascii=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--mnist', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--compile-mode', choices=COMPILE_MODES, default='dacapo-optimized')
    args = parser.parse_args()
    result = execute(args)
    print(json.dumps(dict(status=result['status'], phase=result['phase'], out=str(args.out)), ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
