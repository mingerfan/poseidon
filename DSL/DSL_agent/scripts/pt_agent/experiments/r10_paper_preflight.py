"""Outer R10 preflight: serial execution, observed failures and result ownership."""

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


class DependencyMissing(RuntimeError):
    pass


def _present(path):
    path = Path(path)
    return path.exists() or path.is_symlink()


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def _git_state():
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=paths.REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    changed = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=paths.REPO,
                                  capture_output=True, text=True, check=True).stdout)
    return dict(branch=branch, working_tree_changed=changed)


def capabilities():
    """Read source/configuration availability; no imports from native baselines."""
    records = {}
    sources = {
        'dacapo': [paths.DACAPO_UPSTREAM / 'examples/benchmarks/MLP.py',
                   paths.DACAPO_UPSTREAM / 'examples/data/mlp.model',
                   paths.HECATE_BUILD / 'bin/hecate-opt'],
        'orion': [paths.ORION_ROOT / 'orion/nn/linear.py',
                  paths.ORION_ROOT / 'orion/nn/activation.py',
                  paths.ORION_ROOT / 'configs/mlp.yml'],
        'orbit': [paths.ORBIT_ROOT / 'scripts/optimizer/orbit/optimizer.py',
                  paths.ORBIT_ROOT / 'backend/lattigo/lowering/fhe_binary',
                  paths.ORBIT_ROOT / 'backend/lattigo_patch/lowering/fhe/scheme.go'],
    }
    for name, files in sources.items():
        records[name] = dict(files=[dict(path=str(path), exists=path.is_file()) for path in files])
    metadata = {
        'dacapo': [paths.DACAPO_UPSTREAM / 'versions.txt'],
        'orion': [paths.ORION_ROOT / 'pyproject.toml'],
        'orbit': [paths.ORBIT_ROOT / 'requirements.txt',
                  paths.ORBIT_ROOT / 'backend/lattigo/go.mod'],
    }
    for name, files in metadata.items():
        records[name]['version_sources'] = [
            dict(path=str(path), content=path.read_text() if path.is_file() else None)
            for path in files]
    profile = paths.PROFILES_DIR / 'qwen48/compiler-config.json'
    records['agent_compiler_profile'] = dict(path=str(profile), content=_load(profile))
    orion_config = paths.ORION_ROOT / 'configs/mlp.yml'
    records['orion']['native_config_text'] = orion_config.read_text() if orion_config.is_file() else None
    records['common_configuration'] = dict(status='unresolved', executable_verified=False,
        pending=['original MNIST preprocessing', 'same-weight wrappers', 'complete shared CKKS contract',
                 'single-input CPU execution client', 'native system execution'])
    records['measurements'] = dict(model_runs=0, real_ckks_runs=0, llm_calls=0)
    return records


def finalize_result(*, phase, worker, processes, timing, status, error, git_state, out, round_label='R10'):
    """Missing measurements remain null even after a killed or failed worker."""
    worker = worker or {}
    return dict(run_id=Path(out).name, round=round_label, phase=phase, status=status, error=error,
        git_rev=git_state, worker=worker, processes=processes, timing=timing,
        metrics=dict(llm_calls=0, real_ckks_runs=0,
            compiled_bootstraps=(worker.get('plan_stats') or {}).get('boots'),
            predicted_latency=None, real_ckks_latency=None,
            import_completed=worker.get('import_completed'),
            real_probes_passed=worker.get('real_probes_passed'),
            normalization_completed=worker.get('normalization_completed'),
            static_coverage_ready=worker.get('static_coverage_ready'),
            lowering_ready=worker.get('lowering_ready')))


def _command(phase, work_dir, **values):
    argv = ['-m', 'pt_agent.experiments.r10_paper_worker', '--phase', phase,
            '--work-dir', str(work_dir)]
    for key, value in values.items():
        argv.extend(['--' + key.replace('_', '-'), str(value)])
    return argv


def execute(args):
    root = paths.RESULTS_DIR.resolve()
    out = Path(args.out).absolute()
    if not out.resolve().is_relative_to(root) or out.resolve() == root:
        raise ValueError('--out must name a new child directory inside RESULTS_DIR')
    if out.exists():
        raise ValueError('Existing run directories are preserved')
    out.mkdir(parents=True)
    work = out / 'worker'
    work.mkdir()
    recorder = StageRecorder(out / 'events.jsonl')
    processes, child_events = [], []
    status, error, worker = 'ok', None, None
    started = time.perf_counter_ns()

    def launch(argv, *, label, directory, seconds, memory):
        with recorder.span(label) as span:
            try:
                result = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                    log=out / (label + '.log'), resources=out / 'resources.jsonl',
                    timeout=seconds, mem_gib=memory,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                         'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                         'MKL_NUM_THREADS': '1'})
                processes.append(dict(label=label, **result))
            finally:
                rows = read_events(directory / 'events.jsonl')
                for event in rows:
                    if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                        event['parent_id'] = span['event_id']
                child_events.extend(rows)
            if result['exit_code'] != 0 or result['timed_out'] or result['signal']:
                raise RuntimeError(f'{label}: worker exited {result["exit_code"]}; '
                                   f'timeout={result["timed_out"]}, signal={result["signal"]}')
            if result['remaining_pids']:
                raise RuntimeError(f'{label}: descendants remain {result["remaining_pids"]}')

    try:
        with recorder.span('turnaround', phase=args.phase):
            if shutil.disk_usage(out).free < 10 * 1024**3:
                raise DependencyMissing('At least 10 GiB free disk space is required')
            if args.phase == 'capabilities':
                with recorder.span('capabilities'):
                    worker = capabilities()
                    write_json(work / 'capabilities.json', worker)
            else:
                if args.phase == 'compiler-limit':
                    record = Path(args.export_record)
                    expected = paths.RESULTS_DIR / 'E5a-agent-placement-20261106/mlp/REPORT.json'
                    if record.resolve() != expected.resolve():
                        raise ValueError('R10 compiler check uses the specified retained E5a MLP export')
                    export = (_load(record) or {}).get('export', {})
                    if not (export and Path(export['package']).is_dir() and Path(export['weights']).is_file()
                            and (paths.HECATE_BUILD / 'bin/hecate-opt').is_file()):
                        raise DependencyMissing('Retained trusted export or compiler is missing')
                    command = [sys.executable, '-m', 'pt_agent.tools.nixrun', '--seconds', '10920', '--',
                               *_command('compiler-limit', work, export_record=record)]
                    launch(command, label='compiler_execution', directory=work, seconds=11100, memory=6)
                else:
                    if not _present(paths.INTAKE_PYTHON):
                        raise DependencyMissing('Configured intake interpreter is missing')
                    mem_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                       if line.startswith('MemTotal:')))
                    if len(os.sched_getaffinity(0)) < 8 or mem_kib < 15 * 1024**2:
                        raise DependencyMissing('Qwen import requires 8 available cores and at least 15 GiB RAM')
                    inputs = Path(args.inputs).absolute()
                    if inputs.suffix.lower() == '.parquet':
                        if paths.PAPER_DATA_PYTHON is None or not _present(paths.PAPER_DATA_PYTHON):
                            raise DependencyMissing('Independent Parquet reader is not configured')
                        text_work = work / 'text-input'
                        text_work.mkdir()
                        command = [str(paths.PAPER_DATA_PYTHON), '-B',
                                   *_command('prepare-text', text_work, inputs=inputs)]
                        launch(command, label='prepare_text', directory=text_work, seconds=3600, memory=4)
                        inputs = text_work / 'samples.jsonl'
                    command = [sys.executable, '-m', 'pt_agent.tools.nixrun',
                               '--python', str(paths.INTAKE_PYTHON), '--seconds', '3600', '--',
                               *_command('qwen-import', work, model=Path(args.model).absolute(), inputs=inputs)]
                    launch(command, label='qwen_execution', directory=work, seconds=3720, memory=12)
                worker = _load(work / 'worker_result.json')
                if worker is None:
                    raise RuntimeError('Worker ended without its local result')
                if worker.get('status') != 'ok':
                    raise RuntimeError('Worker result: ' + str(worker.get('status')))
    except DependencyMissing as exc:
        status, error = 'dependency_missing', str(exc)
    except UnsupportedSupervision as exc:
        status, error = 'unsupported_process_supervision', str(exc)
    except BaseException as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'[:2000]
    finally:
        worker = worker or _load(work / 'worker_result.json')
        if worker and worker.get('status') not in (None, 'ok'):
            status = worker['status']
        ended = time.perf_counter_ns()
        timing = summarize_events(read_events(out / 'events.jsonl') + child_events,
                                  termination_observed_ns=ended)
        timing['outer_interval'] = dict(start_ns=started, end_ns=ended, elapsed_ns=ended-started)
        with (out / 'stages.jsonl').open('w') as stream:
            for span in timing['spans']:
                stream.write(json.dumps(span, ensure_ascii=False) + '\n')
        result = finalize_result(phase=args.phase, worker=worker, processes=processes, timing=timing,
                                status=status, error=error, git_state=_git_state(), out=out,
                                round_label=getattr(args, 'round_label', 'R10'))
        write_json(out / 'result.json', result)
        registry = paths.RESULTS_DIR / 'registry.jsonl'
        with registry.open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round=result['round'], model=args.model or 'trusted MLP / capabilities',
                mode=args.phase, git_rev=result['git_rev'], metrics=result['metrics'],
                artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()), ensure_ascii=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('capabilities', 'compiler-limit', 'qwen-import'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--round-label', choices=('R10', 'R11'), default='R10')
    parser.add_argument('--model')
    parser.add_argument('--inputs')
    parser.add_argument('--export-record')
    args = parser.parse_args()
    if args.phase == 'qwen-import' and not (args.model and args.inputs):
        parser.error('qwen-import requires --model and --inputs')
    if args.phase == 'compiler-limit' and not args.export_record:
        parser.error('compiler-limit requires --export-record')
    result = execute(args)
    print(json.dumps(dict(status=result['status'], out=str(args.out)), ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
