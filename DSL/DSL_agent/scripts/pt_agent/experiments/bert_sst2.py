"""Serial outer driver for fresh original-model BERT/SST-2 experiments.

Use ``--phase prepare`` for import and calibration only, ``prompt`` to retain
the exact upcoming public request, and ``step`` for one resumable Agent tool.
The optional ``run`` command advances the internal session and frozen report.
Real compilation and CKKS remain separate post-freeze stages.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading

from .. import paths
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json


WORKER = 'pt_agent.experiments.bert_sst2_worker'
ROUND = 'BERT-SST2'
MODEL = 'bert-tiny-sst2'
CASE_BYTES, FREE_BYTES = 4 * 1024**3, 10 * 1024**3
FORMAL = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')
PHASE_SECONDS = dict(prepare=3600, prompt=1800, step=7200, report=7200)


def command(args, phase, work):
    inner = PHASE_SECONDS[phase]
    result = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(inner + 60),
              '--python', str(args.intake_python), '--', '-B', '-m', WORKER, '--phase', phase,
              '--work-dir', str(work), '--model-dir', str(args.out / 'model'),
              '--session-dir', str(args.out / 'session')]
    for key in ('model', 'inputs', 'ledger', 'allowance', 'data_python'):
        result += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    return result


def source_preflight(args):
    from ..data.sst2_source import resolve_files
    if not (args.model / 'config.json').is_file():
        raise ValueError('BERT requires the supplied original local HF directory')
    files = resolve_files(args.inputs)
    model = json.loads((args.model / 'config.json').read_text())
    if model.get('model_type') != 'bert' or not any(name.endswith('ForSequenceClassification')
                                                  for name in model.get('architectures', [])):
        raise ValueError('BERT/SST-2 requires the original classification model')
    if not (args.intake_python.is_file() or args.intake_python.is_symlink()):
        raise ValueError('The installed original-model intake interpreter is required')
    if not (args.data_python.is_file() or args.data_python.is_symlink()):
        raise ValueError('The installed independent Parquet interpreter is required')
    protected = [args.model, args.inputs]
    if not args.out.is_relative_to(paths.RESULTS_DIR.resolve()) or args.out == paths.RESULTS_DIR.resolve():
        raise ValueError('BERT experiment output must be a new directory under PT_AGENT_RESULTS_ROOT')
    if any(args.out == path or args.out.is_relative_to(path) or path.is_relative_to(args.out) for path in protected):
        raise ValueError('BERT output and original sources must occupy separate directories')
    return dict(model=str(args.model), original_model_config=model,
                inputs=str(args.inputs), original_splits={name: dict(path=str(path), bytes=path.stat().st_size)
                                                          for name, path in files.items()},
                native_test_opened=False, sample_contents_read=False,
                partition_policy=dict(seed=20261024, calibration=1000, heldout=1000, confirm=1000,
                                      report=872, report_source='native validation after freeze'))


def child_environment(args, work):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('LD_', 'NIX_', '__NIX_', 'HECATE_'))
           and key not in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(work / 'tmp'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def case_size(out):
    total = 0
    for parent, _, names in os.walk(out):
        for name in names:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def check_budget(out):
    size = case_size(out)
    if size > CASE_BYTES or shutil.disk_usage(out).free < FREE_BYTES:
        raise RuntimeError('BERT experiment artifact or disk reserve allowance exceeded')
    return size


@contextmanager
def budget_guard(out):
    stop, failure = threading.Event(), {}
    old_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise RuntimeError(failure.get('error', 'BERT experiment interrupted by SIGTERM'))
    signal.signal(signal.SIGTERM, interrupted)
    def observe():
        while not stop.wait(1):
            try:
                check_budget(out)
            except Exception as exc:
                failure['error'] = type(exc).__name__ + ': ' + str(exc)
                os.kill(os.getpid(), signal.SIGTERM)
                return
    thread = threading.Thread(target=observe, name='bert-sst2-budget', daemon=True)
    thread.start()
    try:
        yield failure
    finally:
        stop.set()
        thread.join()
        signal.signal(signal.SIGTERM, old_handler)


def terminal_result(out, work, result):
    for _ in range(20):
        result['metrics']['status'] = result['status']
        result['metrics']['case_artifact_bytes'] = result.get('case_artifact_bytes', 0)
        write_json(work / 'result.json', result)
        write_json(out / 'result.json', result)
        size = case_size(out)
        if size == result.get('case_artifact_bytes'):
            return
        result['case_artifact_bytes'] = size
    raise RuntimeError('BERT terminal artifact accounting did not converge')


def append_registry(args, work, result):
    row = dict(run_id=args.out.name + '-' + work.name, round=ROUND, model=MODEL,
               mode='sst2-' + result['phase'], git_rev=result['git_rev'],
               metrics=result['metrics'], artifacts_dir=str(work),
               ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()


def account_api_calls(ledger, registry, work, *, before, after, git_rev):
    """Register this phase's ledger interval using explicit outer paths once."""
    from ..autonomous_budget import call_records
    ledger, registry, work = Path(ledger), Path(registry), Path(work)
    calls = call_records(ledger)
    if (type(before) is not int or type(after) is not int
            or not 0 <= before <= after <= len(calls)):
        raise ValueError('BERT API accounting requires a valid observed ledger interval')
    registered, retained = [], []
    with registry.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        rows = [json.loads(line) for line in stream if line.strip()]
        existing = {row.get('metrics', {}).get('global_call') for row in rows
                    if row.get('mode') == 'llm-call'}
        stream.seek(0, os.SEEK_END)
        for number in range(before + 1, after + 1):
            if number in existing:
                retained.append(number)
                continue
            call = calls[number - 1]
            row = dict(run_id=work.parent.parent.name + '-' + work.name + '-api-' + str(number),
                round=ROUND, model=MODEL, mode='llm-call', git_rev=git_rev,
                metrics=dict(call, global_call=number, source_ledger=str(ledger.resolve()),
                             prompt_version='r8-v11'),
                artifacts_dir=str(work), ts=call['ts'])
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            existing.add(number)
            registered.append(number)
        stream.flush()
    return dict(ledger=str(ledger.resolve()), registry=str(registry.resolve()),
                api_before=before, api_after=after, registered_calls=registered,
                already_registered_calls=retained, api_calls_replayed=0)


def check_worker(phase, worker):
    if worker.get('status') != 'ok' or worker.get('phase') != phase or worker.get('model') != MODEL:
        raise ValueError('BERT worker identity or completion differs from the requested phase')
    if any(worker.get(name) != 0 for name in ('native_test_reads', 'compiler_runs', 'keygen_runs',
                                             'real_ckks_attempts', 'real_ckks_completed')):
        raise ValueError('BERT intake/session/report invoked an unrelated backend or native test read')
    if any(worker.get(name) is not None for name in FORMAL):
        raise ValueError('BERT formal runtime metrics require later measured stages')
    if phase == 'prepare':
        if (worker.get('import_completed') is not True
                or worker.get('split_counts') != dict(calibration=1000, heldout=1000, confirm=1000, report=872)
                or len(worker.get('reference_checks', [])) != 3
                or any(row.get('passed') is not True for row in worker['reference_checks'])):
            raise ValueError('BERT original import or fixed training partitions are incomplete')
    if phase in ('prepare', 'prompt', 'step') and worker.get('report_samples') != 0:
        raise ValueError('BERT pre-freeze stages must preserve the native validation report')
    if phase == 'step' and worker.get('next_step') not in ('evaluate', 'tune', 'confirm', 'done'):
        raise ValueError('BERT Agent returned an unknown checkpoint phase')
    if phase == 'report' and (worker.get('frozen') is not True or worker.get('report_completed') is not True
                              or worker.get('report_samples') != 872):
        raise ValueError('BERT frozen report is incomplete')


def execute_phase(args, phase):
    from ..autonomous_budget import call_records
    from ..registry import git_rev
    source = source_preflight(args)
    args.out.mkdir(parents=True, exist_ok=True)
    phases = args.out / 'phases'
    phases.mkdir(exist_ok=True)
    work = phases / (str(len(list(phases.iterdir())) + 1).zfill(3) + '-' + phase)
    work.mkdir()
    (work / 'tmp').mkdir()
    recorder = StageRecorder(work / 'outer-events.jsonl')
    before = len(call_records(args.ledger))
    write_json(work / 'api-accounting.json', dict(api_before=before))
    result = dict(round=ROUND, model=MODEL, phase=phase, status='running', source=source,
                  git_rev=git_rev(), worker=None, process=None, metrics={name: None for name in FORMAL})
    try:
        with recorder.span('bert_sst2_turnaround', phase=phase):
            check_budget(args.out)
            with budget_guard(args.out) as budget_failure:
                observation = run_process(command(args, phase, work), cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                    log=work / 'worker.log', resources=work / 'resources.jsonl',
                    timeout=PHASE_SECONDS[phase] + 120, mem_gib=args.mem_gib, fsize_gib=8,
                    env=child_environment(args, work))
            result['process'] = observation
            worker_path = work / 'worker_result.json'
            if worker_path.is_file():
                result['worker'] = json.loads(worker_path.read_text())
            worker = result['worker'] or {}
            if budget_failure:
                raise RuntimeError(budget_failure['error'])
            if (observation['exit_code'] != 0 or observation['timed_out'] or observation['signal'] is not None
                    or observation['remaining_pids'] or worker.get('status') != 'ok'):
                raise RuntimeError('BERT ' + phase + ' did not complete; retained worker and resource records describe the failure')
            check_worker(phase, worker)
            result['status'] = 'ok'
            check_budget(args.out)
    except Exception as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        worker = result['worker'] or {}
        after = len(call_records(args.ledger))
        result['api_before'], result['api_after'] = before, after
        result['metrics'].update(llm_calls=after-before, frozen=worker.get('frozen'),
            import_completed=worker.get('import_completed'), report_completed=worker.get('report_completed'),
            report_samples=worker.get('report_samples'), next_step=worker.get('next_step'),
            task_policy_passed=worker.get('task_policy_passed'), validation_metrics=worker.get('validation_metrics'),
            compiler_runs=worker.get('compiler_runs'), keygen_runs=worker.get('keygen_runs'),
            real_ckks_attempts=worker.get('real_ckks_attempts'), real_ckks_completed=worker.get('real_ckks_completed'))
        events = read_events(work / 'outer-events.jsonl') + read_events(work / 'events.jsonl')
        write_json(work / 'stages.json', summarize_events(events))
        if after > before:
            try:
                result['api_registration'] = account_api_calls(args.ledger,
                    paths.RESULTS_DIR / 'registry.jsonl', work, before=before,
                    after=after, git_rev=result['git_rev'])
            except Exception as exc:
                result.update(status='error', accounting_error=type(exc).__name__ + ': ' + str(exc))
        terminal_result(args.out, work, result)
        if result['case_artifact_bytes'] > CASE_BYTES:
            result.update(status='error', error='Terminal BERT metadata exceeded the case artifact allowance')
            terminal_result(args.out, work, result)
        append_registry(args, work, result)
    return result


def execute(args):
    if args.phase != 'run':
        return execute_phase(args, args.phase)
    result = None
    if not (args.out / 'model/model_info.json').is_file():
        result = execute_phase(args, 'prepare')
        if result['status'] != 'ok':
            return result
    for _ in range(25):
        state = args.out / 'session/SESSION.json'
        if state.is_file() and json.loads(state.read_text()).get('phase') == 'done':
            break
        result = execute_phase(args, 'step')
        if result['status'] != 'ok':
            return result
    else:
        raise RuntimeError('Retained BERT session exceeded the finite step orchestration allowance')
    if (args.out / 'session/FROZEN.json').is_file():
        access = args.out / 'session/REPORT_ACCESS.json'
        if access.is_file():
            retained = json.loads(access.read_text())
            if not retained.get('complete'):
                raise RuntimeError('Retained frozen BERT report is incomplete; keep its original evidence')
            return json.loads((Path(retained['result']).parent / 'result.json').read_text())
        return execute_phase(args, 'report')
    if result is None:
        return json.loads((args.out / 'result.json').read_text())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'prompt', 'step', 'report', 'run'), default='prepare')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--intake-python', type=Path, default=paths.INTAKE_PYTHON)
    parser.add_argument('--data-python', type=Path, default=paths.PAPER_DATA_PYTHON or paths.ORBIT_ROOT.parent / 'paper-data-venv/bin/python')
    parser.add_argument('--ledger', type=Path, default=paths.RESULTS_DIR / 'llm_calls.jsonl')
    parser.add_argument('--allowance', type=Path, default=paths.RESULTS_DIR / 'autonomous-api-allowance.json')
    parser.add_argument('--mem-gib', type=float, default=4)
    args = parser.parse_args(argv)
    if args.mem_gib <= 0:
        parser.error('--mem-gib must be positive')
    for name in ('model', 'inputs', 'out', 'intake_python', 'ledger', 'allowance', 'data_python'):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    for name in ('model', 'inputs', 'out', 'ledger', 'allowance'):
        setattr(args, name, getattr(args, name).resolve())
    result = execute(args)
    print(json.dumps(dict(status=result['status'], out=str(args.out), phase=result['phase'])), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
