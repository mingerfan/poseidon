"""R13 outer owner for one bounded original-model intake preparation."""

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
from .r13_common_models import FACTORIES


INTERNAL_SECONDS = 1800
OUTER_SECONDS = 1860
MEMORY_GIB = 4
MODE = 'common-model-intake-preparation'


class DependencyMissing(RuntimeError):
    pass


class PhaseStopped(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _git_state():
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=paths.REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=paths.REPO,
                           capture_output=True, text=True, check=True).stdout
    return dict(branch=branch, working_tree_changed=bool(dirty))


def finalize_result(*, model, worker, processes, timing, status, error, git_state, out):
    """Keep observed intake stages separate from unmeasured compiler/CKKS fields."""
    record = worker or {}
    metrics = {key: record.get(key) for key in
               ('import_completed', 'real_probes_passed', 'graph_math_passed',
                'roundtrip_passed', 'static_coverage')}
    metrics.update(llm_calls=0, real_ckks_attempts=0, real_ckks_completed=0,
                   lowering_ready=None, compiled_bootstraps=None,
                   predicted_latency=None, CKKS_latency=None,
                   real_probe_criterion='1e-9*(1+abs(reference))',
                   graph_math_criterion='1e-12+1e-12*abs(reference)')
    metrics['phase_elapsed_ns'] = {
        span['event_id']: dict(stage_id=span['stage_id'], inclusive_ns=span.get('inclusive_ns'),
                              exclusive_ns=span.get('exclusive_ns'))
        for span in timing.get('spans', [])}
    return dict(run_id=Path(out).name, round='R13', mode=MODE, model=str(model),
                status=status, error=error, git_rev=git_state, worker=record,
                processes=processes, timing=timing, metrics=metrics)


def _command(model, inputs, selections, work_dir):
    return ['-m', 'pt_agent.experiments.r13_common_worker', '--model', str(model),
            '--inputs', str(inputs), '--selections', str(selections),
            '--work-dir', str(work_dir)]


def execute(args):
    root = paths.RESULTS_DIR.resolve()
    out = Path(args.out).absolute()
    if out.resolve() == root or not out.resolve().is_relative_to(root):
        raise ValueError('--out must name a new child directory of RESULTS_DIR')
    if out.exists():
        raise ValueError('Existing run directories are preserved')
    out.mkdir(parents=True)
    work = out / 'worker'
    work.mkdir()
    (out / 'resources.jsonl').touch()
    recorder = StageRecorder(out / 'events.jsonl')
    processes, child_events = [], []
    worker = None
    status, error = 'ok', None
    model = str(args.model)
    inputs = Path(args.inputs).absolute()
    records = paths.WORK_RESULTS / 'agent-cifar-offline-records-r1804/cifar10-records'
    selections = paths.WORK_RESULTS / 'agent-resnet20-original-accuracy-r1805/frozen-selections.json'
    started_ns = time.perf_counter_ns()
    try:
        with recorder.span('turnaround', model=model):
            with recorder.span('preflight'):
                if model not in FACTORIES:
                    raise ValueError('R13 accepts only the three specified original-model factories')
                if inputs.resolve() != records.resolve():
                    raise ValueError('--inputs must identify the existing original CIFAR records')
                required = [inputs, inputs / 'dataset.json', selections,
                            *[Path(FACTORIES[model][key]) for key in
                              ('source', 'checkpoint', 'inference_preprocessing_source')]]
                missing = [str(path) for path in required if not path.exists()]
                interpreter = paths.INTAKE_PYTHON
                if not interpreter.is_file() and not interpreter.is_symlink():
                    missing.append(str(interpreter))
                if missing:
                    raise DependencyMissing('Missing local prerequisites: ' + ', '.join(missing))
                if shutil.disk_usage(out).free < 10 * 1024**3:
                    raise DependencyMissing('At least 10 GiB free disk space is required')
            argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun',
                    '--seconds', str(INTERNAL_SECONDS), '--python', str(interpreter),
                    '--', *_command(model, inputs, selections, work)]
            process = None
            with recorder.span('common_model_intake', wall_seconds=OUTER_SECONDS) as span:
                try:
                    process = run_process(
                        argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=out / 'worker.log',
                        resources=out / 'resources.jsonl', timeout=OUTER_SECONDS,
                        mem_gib=MEMORY_GIB, fsize_gib=8,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                             'MKL_NUM_THREADS': '1'})
                    processes.append(dict(label='common_model_intake', command=argv, **process))
                finally:
                    for event in read_events(work / 'events.jsonl'):
                        if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                            event['parent_id'] = span['event_id']
                        child_events.append(event)
                    result_path = work / 'worker_result.json'
                    if result_path.is_file():
                        worker = json.loads(result_path.read_text())
                if process is None:
                    raise PhaseStopped('process_failed', 'Worker process observation is unavailable')
                if process['timed_out']:
                    raise PhaseStopped('timeout', 'The outer R13 process exceeded 1860 seconds')
                if process['signal'] or process['remaining_pids']:
                    raise PhaseStopped('process_failed', str({key: process[key] for key in
                                                            ('exit_code', 'signal', 'remaining_pids')}))
                if worker is None:
                    raise PhaseStopped('result_missing', 'Worker did not write its result')
                if worker.get('status') != 'ok':
                    raise PhaseStopped(worker.get('status', 'error'), str(worker.get('error')))
                if process['exit_code'] != 0:
                    raise PhaseStopped('process_failed', 'Worker exit code: ' + str(process['exit_code']))
                needed = ('import_completed', 'real_probes_passed', 'graph_math_passed',
                          'roundtrip_passed', 'static_coverage')
                if any(worker.get(key) is not True for key in needed):
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
        with (out / 'stages.jsonl').open('w') as stream:
            for stage in timing['spans']:
                stream.write(json.dumps(stage, ensure_ascii=False) + '\n')
        result = finalize_result(model=model, worker=worker, processes=processes,
                                 timing=timing, status=status, error=error,
                                 git_state=_git_state(), out=out)
        write_json(out / 'result.json', result)
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round='R13', model=model, mode=MODE,
                                         git_rev=result['git_rev'], metrics=result['metrics'],
                                         artifacts_dir=str(out),
                                         ts=datetime.now(timezone.utc).isoformat()),
                                    ensure_ascii=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=tuple(FACTORIES))
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = execute(args)
    print(json.dumps(dict(status=result['status'], out=str(args.out)), ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
