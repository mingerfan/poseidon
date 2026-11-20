"""Durable serial continuation of S2 HEIR after the already running ResNet."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .. import paths
from .a_group_runs import append_record, empty_record
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import run_process, write_json


def read(path):
    return json.loads(Path(path).read_text())


def same_birth(pid, birth):
    try:
        text = Path('/proc', str(pid), 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return fields[19] == str(birth) and fields[0] != 'Z'
    except OSError:
        return False


def closed_case(case):
    if not all((case / name).is_file() for name in ('result.json', 'completion.json', 'paper-record.json')):
        return False
    completion = read(case / 'completion.json')
    result = read(case / 'result.json')
    observation = result.get('observation') or {}
    return bool(completion.get('registry_persisted') and completion.get('required_artifacts_persisted') and
                result.get('monitor_stopped') and observation.get('remaining_pids') == [])


def command(args, model):
    case = paths.RESULTS_DIR / ('s2-heir-' + model + '-qwen48-rule63-20261118')
    return case, [sys.executable, '-B', '-m', 'pt_agent.experiments.s2_heir_compile', '--model', model,
                 '--out', str(case), '--source-check', str(args.source_check),
                 '--parameter-probe', str(args.parameter_probe), '--base-pipeline', str(args.base_pipeline),
                 '--paper-runs', str(args.paper_runs)]


def record_registry(case, round_name, model, mode, status, metrics):
    from ..registry import git_rev
    row = dict(run_id=case.name, round=round_name, model=model, mode=mode, git_rev=git_rev(),
               metrics=dict(status=status, **metrics), artifacts_dir=str(case), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def bert_worker(args):
    from .heir_cnn_export import saved_activations, SUPPORTED
    from ..dsl.parser import parse
    text = args.bert_program.read_text()
    program = parse(text)
    detail = dict(source_program=str(args.bert_program), source_entry='heir_cnn_export.saved_activations/export_graph',
                  source_entry_module=str(Path(__file__).with_name('heir_cnn_export.py')),
                  program_kinds=sorted({op.kind for op in program.ops}),
                  kinds_outside_existing_CNN_export=sorted({op.kind for op in program.ops} - SUPPORTED),
                  activation_functions=sorted({op.attrs.get('fn', '') for op in program.ops if op.kind == 'act'}),
                  compiler_attempts=0, model_forwards=0, sample_reads=0, API=0, CKKS=0,
                  official_HEIR_model_unsupported_proven=False, official_same_release_GELU_softmax_paths_exist=True)
    try:
        saved_activations(text)
    except ValueError as exc:
        detail.update(status='unsupported', status_detail='Existing project CNN export has no complete BERT path: ' + str(exc))
    else:
        detail.update(status='driver_error', status_detail='Existing entry unexpectedly admits BERT; no unsupported claim or unreviewed new adapter')
    write_json(args.out / 'entry-check.json', detail)


def check_bert(args, env):
    started = time.perf_counter_ns()
    case = paths.RESULTS_DIR / 's2-heir-bert-tiny-existing-entry-20261118'
    case.mkdir(exist_ok=False)
    argv = [str(paths.WORKSPACE / 'baselines/orbit-venv/bin/python'), '-B', '-m', __spec__.name,
            '--bert-worker', '--out', str(case), '--bert-program', str(args.bert_program)]
    write_json(case / 'command.json', dict(argv=argv, time_limit_s=900, memory_gib=4))
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('r') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observation = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=case / 'worker.log',
                                  resources=case / 'resources.jsonl', timeout=900, mem_gib=4, fsize_gib=1, env=env)
    result = read(case / 'entry-check.json') if (case / 'entry-check.json').is_file() else dict(
        status='driver_error', status_detail='Existing BERT entry check failed; see retained log', compiler_attempts=0)
    if observation['exit_code'] != 0 or observation['timed_out'] or observation['remaining_pids']:
        result.update(status='driver_error', status_detail='Existing frontend observation did not close cleanly')
    result.update(round='S2-HEIR-BERT-ENTRY', model='bert_tiny', observation=observation)
    write_json(case / 'result.json', result)
    record_registry(case, 'S2-HEIR-BERT-ENTRY', 'bert_tiny', 'existing-BERT-frontend-admission', result['status'],
                    dict(compiler_attempts=0, API=0, CKKS=0))
    completion = persist_case_completion(case, round_label='S2-HEIR-BERT-ENTRY', started_ns=started,
                status=result['status'], timing=dict(status='ok', attribution_ready=False, spans=[], diagnostics=[]),
                required_artifact_bytes=artifact_bytes(case))
    row = empty_record(run_id=case.name, task='S2-HEIR', model='bert_tiny', system='heir',
          param_group='ours-N16-boot5', status=result['status'], status_detail=result['status_detail'], budget_s=900,
          compile_s=None, api_calls=0, source=str(case / 'result.json'), finished_utc=datetime.now(timezone.utc).isoformat(),
          compiler_attempts=0, own_frontend_path_unavailable=True, official_HEIR_model_unsupported_proven=False,
          admission_wall_s=(observation['ended_ns'] - observation['started_ns']) / 1e9,
          complete_admission_case_s=completion['complete_case_elapsed_seconds'])
    write_json(case / 'paper-record.json', row)
    append_record(args.paper_runs, row)
    return dict(model='bert_tiny', case=str(case), status=result['status'], compiler_attempts=0)


def main(args):
    if args.bert_worker:
        return bert_worker(args)
    started = time.perf_counter_ns()
    args.out.mkdir(exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'NIX_', '__NIX_'))
           and k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    state = dict(round='S2-HEIR-SERIAL', status='waiting_for_original_ResNet_closure', jobs=[],
                 predecessor=dict(case=str(args.resnet_case), pid=args.resnet_pid, birth=args.resnet_birth),
                 no_parallel_heavy_work=True, API=0, CKKS=0, new_installations=0, new_downloads=0)
    for model in ('vgg16', 'alexnet'):
        case, argv = command(args, model)
        if case.exists():
            raise ValueError('Preserve prior scientific case: ' + str(case))
        state['jobs'].append(dict(model=model, case=str(case), argv=argv, status='queued', native_budget_s=7200, RSS_gib=14))
    state['jobs'].append(dict(model='bert_tiny', status='queued', admission_budget_s=900, compiler_attempts=0))
    write_json(args.out / 'manifest.json', state)
    write_json(args.out / 'progress.json', state)
    try:
        wait_started = time.monotonic()
        while same_birth(args.resnet_pid, args.resnet_birth) or not closed_case(args.resnet_case):
            if time.monotonic() - wait_started > 7500:
                raise TimeoutError('Original ResNet closure not established; no overlapping successor launched')
            time.sleep(2)
        state['status'] = 'executing_serial_jobs'
        for job in state['jobs'][:2]:
            if datetime.now(timezone.utc) >= datetime(2026, 11, 18, 12, tzinfo=timezone.utc):
                job['status'] = 'not_started_after_submission_data_cutoff'
                continue
            job['status'] = 'running'
            write_json(args.out / 'progress.json', state)
            label = job['model']
            obs = run_process(job['argv'], cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (label + '.log'),
                              resources=args.out / (label + '-resources.jsonl'), timeout=7500, mem_gib=24, fsize_gib=16, env=env)
            job['supervisor_observation'] = obs
            case = Path(job['case'])
            if obs['exit_code'] != 0 or obs['timed_out'] or obs['remaining_pids'] or not closed_case(case):
                raise RuntimeError('Successor did not persist a cleanly closed scientific case: ' + label)
            result = read(case / 'result.json')
            job['status'] = 'complete'
            job['scientific_status'] = result['status']
            job['native_compile_s'] = result['compile_s']
            write_json(args.out / 'progress.json', state)
        state['jobs'][-1].update(check_bert(args, env), schedule_status='complete')
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
    write_json(args.out / 'result.json', state)
    record_registry(args.out, 'S2-HEIR-SERIAL', None, 'serial-HEIR-matrix-continuation', state['status'],
                    dict(jobs=[{k: v for k, v in x.items() if k in ('model', 'status', 'scientific_status')} for x in state['jobs']], API=0, CKKS=0))
    persist_case_completion(args.out, round_label='S2-HEIR-SERIAL', started_ns=started, status=state['status'],
          timing=dict(status='ok', attribution_ready=False, spans=[], diagnostics=[]), required_artifact_bytes=artifact_bytes(args.out))
    print(json.dumps(dict(case=str(args.out), status=state['status'], jobs=[{k: v for k, v in x.items()
          if k in ('model', 'status', 'scientific_status', 'native_compile_s')} for x in state['jobs']]), ensure_ascii=False), flush=True)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bert-worker', action='store_true')
    for name in ('source-check', 'parameter-probe', 'base-pipeline', 'paper-runs', 'resnet-case', 'bert-program'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--resnet-pid', type=int)
    p.add_argument('--resnet-birth')
    args = p.parse_args(argv)
    if not args.bert_program or not args.bert_worker and any(getattr(args, x) is None for x in
            ('source_check', 'parameter_probe', 'base_pipeline', 'paper_runs', 'resnet_case', 'resnet_pid', 'resnet_birth')):
        p.error('Explicit original source, parameter, predecessor and publication paths required')
    return args


if __name__ == '__main__':
    main(arguments())
