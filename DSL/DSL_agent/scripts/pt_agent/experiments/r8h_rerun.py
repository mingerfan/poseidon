"""R8h: serial, model-checkpointed reruns using the existing frozen evaluation protocol."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..paths import pt_result
from ..run import _write
from .r8g_mlp_rerun import _calls, _run_args, _stop, INNER_SECONDS, STAGES


ORDER = ('mlp', 'cnn_plain', 'resnet20_silu', 'resnet20')
API_BUDGET = 30
GLOBAL_LIMIT = 78


def read(path):
    return json.loads(Path(path).read_text())


def trace_summary(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] \
        if Path(path).exists() else []
    iterations = []
    seconds = 0.
    actual = reused = 0
    for row in rows:
        fb, tuned = row.get('fb') or {}, row.get('tuned')
        reuse = row.get('tuned_reused_from')
        if tuned is not None:
            actual += reuse is None
            reused += reuse is not None
            if reuse is None:
                seconds += tuned['seconds']
        iterations.append(dict(iteration=row['iteration'], stage=fb.get('stage'),
            feasible=fb.get('feasible', False), errors=fb.get('error') or fb.get('errors'),
            before_latency_us=(fb.get('cost') or {}).get('latency_pred_us'),
            before_boots=(fb.get('cost') or {}).get('boots_pred'),
            tuned_feasible=tuned['feasible'] if tuned else None,
            after_latency_us=tuned['cost_after']['latency_pred_us'] if tuned else None,
            after_boots=tuned['cost_after']['boots_pred'] if tuned else None,
            tuned_reused_from=reuse))
    return dict(iterations=iterations, tune_calls=actual, tune_reuses=reused, tuner_seconds=seconds)


def inside(args):
    from .r8e6_reconfirm import _limits
    _limits()
    import torch
    from ..planners.confirm import MAX_CONFIRM_EVALS
    from ..planners.prompts import PROMPT_VERSION
    from ..run import _inside as run_inside

    if PROMPT_VERSION != 'r8-v4' or MAX_CONFIRM_EVALS != 6:
        raise RuntimeError('R8h requires r8-v4 and MAX_CONFIRM_EVALS=6')
    torch.set_num_threads(1)
    run_args = _run_args(args)
    run_args.model = args.model
    if args.continuation == 'test':
        run_args.stages = ['test']
    elif args.continuation == 'frozen':
        run_args.stages = [stage for stage in STAGES if stage != 'plan']
    code = run_inside(run_args)
    path = args.out / 'REPORT.json'
    report = read(path)
    report.update(round='R8h', model=args.model, program_path=str(args.out / 'program.ptdsl'))
    _write(path, report)
    return code


def _attempt(root, batch, entry, args, recorded):
    from ..registry import append, new_run_id
    case_root = root / 'models' / args.model
    case_root.mkdir(parents=True, exist_ok=True)
    directory = case_root / f"attempt-{len(entry['attempts']) + 1:02d}"
    directory.mkdir()
    previous = read(entry['report_path']) if entry.get('report_path') else {}
    previous_dir = Path(entry['report_path']).parent if entry.get('report_path') else None
    continuation = 'agent'
    if previous_dir and (previous_dir / 'program.ptdsl').exists():
        continuation = 'test' if (previous.get('compile') or {}).get('exit') == 0 else 'frozen'
        for name in ('program.ptdsl', 'calibration.json', 'agent_trace.jsonl'):
            if (previous_dir / name).exists():
                shutil.copy2(previous_dir / name, directory / name)
        _write(directory / 'REPORT.json', previous)
    budget_path = pt_result('llm_calls.jsonl').resolve()
    attempt = dict(directory=str(directory), report_path=str(directory / 'REPORT.json'),
        continuation=continuation, api_before=len(_calls(budget_path)), api_calls=0,
        call_indices=[], complete=False, stage='running', seconds=0, exit_code=None, error=None,
        started_ts=datetime.now(timezone.utc).isoformat(), worker_pid=None)
    entry['attempts'].append(attempt)
    entry.update(report_path=attempt['report_path'], complete=False, stage='running')
    batch.update(stage='running', active_model=args.model)
    _write(root / 'REPORT.json', batch)

    def account(*, wait=False):
        calls = _calls(budget_path, wait=wait)
        if calls is None:
            return
        indices = list(range(attempt['api_before'] + 1, len(calls) + 1))
        for index in indices:
            if index not in recorded:
                call = calls[index - 1]
                append(dict(run_id=new_run_id('R8h', args.model, f'api-{index}'), round='R8h',
                    model=args.model, mode='llm-call', git_rev=batch['git_rev'],
                    metrics=dict(call, global_call=index), artifacts_dir=str(directory), ts=call['ts']))
                recorded.add(index)
        attempt.update(api_calls=len(indices), call_indices=indices)
        batch.update(api_calls=len(calls) - batch['api_before'], api_total=len(calls))

    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(INNER_SECONDS), '--',
        '-m', 'pt_agent.experiments.r8h_rerun', '--inside', '--out', str(directory),
        '--model', args.model, '--models-dir', batch['models_dir'], '--budget-path', str(budget_path),
        '--call-limit', str(batch['api_limit']), '--continuation', continuation]
    attempt['command'] = command
    started, process = time.monotonic(), None
    try:
        if continuation == 'agent' and attempt['api_before'] >= batch['api_limit']:
            raise RuntimeError('R8h has no API calls remaining')
        with (directory / 'stdout.txt').open('w') as stdout, (directory / 'stderr.txt').open('w') as stderr:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            attempt['worker_pid'] = process.pid
            _write(root / 'REPORT.json', batch)
            while True:
                try:
                    attempt['exit_code'] = process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    account()
                    attempt['seconds'] = time.monotonic() - started
                    _write(root / 'REPORT.json', batch)
                    if attempt['seconds'] > INNER_SECONDS + 120:
                        raise TimeoutError('R8h exceeded the existing per-model Nix allowance')
    except (Exception, KeyboardInterrupt) as exc:
        attempt['error'] = f'{type(exc).__name__}: {exc}'
        _stop(process)
        attempt['exit_code'] = process.returncode if process is not None else None
    finally:
        account(wait=True)
        attempt['seconds'] = time.monotonic() - started
        current = read(directory / 'REPORT.json') if (directory / 'REPORT.json').exists() else {}
        attempt.update(stage=current.get('stage', 'failed'), error=current.get('error') or attempt['error'])
        attempt['complete'] = bool(current.get('complete') and current.get('stage') == 'complete'
            and attempt['exit_code'] == 0 and attempt['error'] is None)
        summary = trace_summary(directory / 'agent_trace.jsonl')
        # A post-freeze continuation copies the trace as evidence, with no repeated tuning.
        actual_summary = summary if continuation == 'agent' else dict(tune_calls=0, tune_reuses=0, tuner_seconds=0.)
        attempt.update(actual_summary)
        entry.update(complete=attempt['complete'], stage=attempt['stage'], error=attempt['error'],
            api_calls=sum(row['api_calls'] for row in entry['attempts']),
            seconds=sum(row['seconds'] for row in entry['attempts']),
            tune_calls=sum(row.get('tune_calls', 0) for row in entry['attempts']),
            tune_reuses=sum(row.get('tune_reuses', 0) for row in entry['attempts']),
            tuner_seconds=sum(row.get('tuner_seconds', 0.) for row in entry['attempts']),
            iterations=summary['iterations'])
        _write(root / 'REPORT.json', batch)
        append(dict(run_id=new_run_id('R8h', args.model, 'model-rerun'), round='R8h', model=args.model,
            mode='model-rerun', git_rev=batch['git_rev'], metrics=entry,
            artifacts_dir=str(directory), ts=datetime.now(timezone.utc).isoformat()))
    return attempt['complete']


def recover_accounting(root, batch, recorded):
    """Retain calls completed after the last checkpoint if an outer process was interrupted."""
    from ..registry import append, new_run_id
    name = batch.get('active_model')
    if not name:
        return
    entry = batch['models'][name]
    if not entry['attempts']:
        raise ValueError('Active model has no attempt checkpoint')
    attempt = entry['attempts'][-1]
    if attempt.get('worker_pid'):
        try:
            os.kill(attempt['worker_pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('The prior R8h worker is still running; retain its batch')
    calls = _calls(pt_result('llm_calls.jsonl'))
    indices = list(range(attempt['api_before'] + 1, len(calls) + 1))
    for index in indices:
        if index not in recorded:
            call = calls[index - 1]
            append(dict(run_id=new_run_id('R8h', name, f'api-{index}'), round='R8h', model=name,
                mode='llm-call', git_rev=batch['git_rev'], metrics=dict(call, global_call=index),
                artifacts_dir=attempt['directory'], ts=call['ts']))
            recorded.add(index)
    attempt.update(api_calls=len(indices), call_indices=indices)
    batch.update(api_calls=len(calls)-batch['api_before'], api_total=len(calls))
    current = read(attempt['report_path']) if Path(attempt['report_path']).exists() else {}
    completed = current.get('complete') and current.get('stage') == 'complete'
    attempt.update(complete=bool(completed), exit_code=0 if completed else attempt['exit_code'],
                   stage=current.get('stage', 'interrupted'))
    if attempt.get('started_ts'):
        elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(attempt['started_ts'])).total_seconds()
        attempt['seconds'] = max(attempt['seconds'], elapsed)
        attempt['seconds_scope'] = 'Wall interval including interruption until explicit resume'
    summary = trace_summary(Path(attempt['directory'])/'agent_trace.jsonl')
    if attempt['continuation'] == 'agent':
        attempt.update(summary)
    entry.update(complete=bool(completed), stage=attempt['stage'],
        api_calls=sum(row['api_calls'] for row in entry['attempts']),
        seconds=sum(row['seconds'] for row in entry['attempts']),
        tune_calls=sum(row.get('tune_calls',0) for row in entry['attempts']),
        tune_reuses=sum(row.get('tune_reuses',0) for row in entry['attempts']),
        tuner_seconds=sum(row.get('tuner_seconds',0.) for row in entry['attempts']),
        iterations=summary['iterations'])
    batch['active_model'] = None
    _write(root/'REPORT.json', batch)


def outer(args):
    from ..registry import append, git_rev, load, new_run_id
    root = (args.resume or args.out or pt_result('r8h-v11-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['PT_RESULTS'] = str(pt_result())
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / 'REPORT.json'
        if args.resume:
            batch = read(path)
            if batch['round'] != 'R8h':
                raise ValueError('Resume requires an R8h batch')
        else:
            if path.exists():
                raise FileExistsError('Use --resume to retain an existing R8h batch')
            before = len(_calls(pt_result('llm_calls.jsonl')))
            batch = dict(round='R8h', complete=False, stage='ready', api_before=before,
                api_calls=0, api_total=before, api_budget=API_BUDGET,
                api_limit=min(before + API_BUDGET, GLOBAL_LIMIT, 300), git_rev=git_rev(),
                models_dir=str((args.models_dir or pt_result('models-v2')).resolve()),
                order=list(ORDER), prompt_version='r8-v4', agent_iters=6, tuner_budget_s=2400,
                max_sweeps=1, val_split='test[9000:10000]', confirm_split='test[8000:9000]',
                test_split='test[0:8000]', max_confirm_evals=6,
                models={name: dict(model=name, complete=False, stage='not_started', attempts=[]) for name in ORDER})
            _write(path, batch)
        (root / 'runner.pid').write_text(str(os.getpid()) + '\n')
        recorded = {row['metrics'].get('global_call') for row in load() if row['mode'] == 'llm-call'}
        if args.resume:
            recover_accounting(root, batch, recorded)
        names = (args.model,) if args.model else ORDER
        for name in names:
            entry = batch['models'][name]
            if entry['complete']:
                continue
            if any(not batch['models'][prior]['complete'] for prior in ORDER[:ORDER.index(name)]):
                raise ValueError('R8h model order requires earlier models to complete first')
            args.model = name
            if not _attempt(root, batch, entry, args, recorded):
                break
        batch['complete'] = all(entry['complete'] for entry in batch['models'].values())
        batch['stage'] = 'complete' if batch['complete'] else 'failed' if any(
            entry.get('error') or (entry['attempts'] and not entry['complete'])
            for entry in batch['models'].values()) else 'partial'
        batch['active_model'] = None
        _write(path, batch)
        append(dict(run_id=new_run_id('R8h', 'four-models', 'batch'), round='R8h', model='four-models',
            mode='batch', git_rev=batch['git_rev'], metrics=batch, artifacts_dir=str(root),
            ts=datetime.now(timezone.utc).isoformat()))
        print('REPORT=' + str(path), flush=True)
        return 0 if batch['stage'] in ('complete', 'partial') else 1


def _report_metrics(path, baseline_boots):
    from ..dsl.parser import parse
    report = read(path)
    program = parse((Path(path).parent / 'program.ptdsl').read_text())
    confirmation = report.get('confirm') or (report.get('agent') or {}).get('confirm') or {}
    selected = confirmation.get('selected') or {}
    passed = [row for row in confirmation.get('evals', []) if row.get('passed')
              and row['iteration'] == selected.get('iteration') and row['origin'] == selected.get('origin')]
    boots = (report.get('plan') or {}).get('boots')
    return dict(report_path=str(path), source_path=str(Path(path).parent / 'program.ptdsl'),
        activations=[dict(index=index, line=op.lineno, call_lines=list(op.call_lines),
            fn=op.attrs['fn'], method=op.attrs['method'], deg=op.attrs.get('deg'),
            degrees=op.attrs.get('degrees'), domain=op.attrs['domain'])
            for index, op in enumerate(op for op in program.ops if op.kind == 'act')],
        selected_iteration=selected.get('iteration'), selected_origin=selected.get('origin'),
        boots_pred=selected.get('boots_pred'), boots_compiled=boots,
        rule_boots=baseline_boots, rule_boot_reduction=(baseline_boots-boots)/baseline_boots if boots is not None else None,
        latency_pred_us=selected.get('latency_pred_us') or report.get('latency_pred_us'),
        latency_plan_us=(report.get('plan_latency') or {}).get('latency_us'),
        confirm_acc_drop_pp=passed[0]['acc_drop_pp'] if passed else None,
        top1_report=report.get('top1_test'),
        report_acc_drop_pp=report['torch_top1_report']-report['top1_test'] if report.get('top1_test') is not None else None,
        n_diverged=report.get('n_diverged'), m1_item3_pass=report.get('m1_item3_pass'))


def summarize(args):
    root = args.out
    batch = read(root / 'REPORT.json')
    rule = {row['model']: row['boots'] for row in read(args.rule_report)['models']}
    original = read(args.old_generation / 'SUMMARY.json')
    rows = []
    for name in ORDER:
        old_path = args.old_v1 / 'models' / name / 'REPORT.json'
        old_generation = args.old_generation / name
        old = _report_metrics(old_path, rule[name])
        old_trace = trace_summary(old_generation / 'agent_trace.jsonl')
        old_agent = read(old_generation / 'REPORT.json')['agent']
        old_attempts = next(row for row in original['models'] if row['model'] == name)['attempts']
        old.update(old_trace, api_calls=old_agent['llm_calls'], iteration_count=old_agent['iterations'],
            original_generation_seconds=sum(row['seconds'] for row in old_attempts),
            reconfirmation_seconds=next(row for row in read(args.old_v1 / 'REPORT.json')['models'] if row['model'] == name)['seconds'])
        entry = batch['models'][name]
        new = None
        if entry.get('report_path') and (Path(entry['report_path']).parent / 'program.ptdsl').exists():
            new = _report_metrics(entry['report_path'], rule[name])
            new.update({key: entry.get(key) for key in ('api_calls', 'seconds', 'tune_calls', 'tune_reuses', 'tuner_seconds', 'iterations')})
            new['iteration_count'] = len(entry['iterations'])
        rows.append(dict(model=name, v1=old, v11=new, complete=entry['complete']))
    mlp_r8g = _report_metrics(args.r8g / 'mlp' / 'REPORT.json', rule['mlp'])
    r8g = read(args.r8g / 'REPORT.json')
    mlp_r8g.update(trace_summary(args.r8g / 'mlp' / 'agent_trace.jsonl'),
                   api_calls=r8g['api_calls'], seconds=r8g['seconds'], iteration_count=len(r8g['iterations']))
    result = dict(round='R8h', complete=batch['complete'], models=rows, mlp_r8g=mlp_r8g,
        api_calls=batch['api_calls'], api_total=batch['api_total'], no_tag=True,
        note='Historical generation and R8e7 reconfirmation timings remain separate; prediction is not a CKKS measurement.')
    _write(root / 'SUMMARY.json', result)
    print('SUMMARY=' + str(root / 'SUMMARY.json'))
    return 0 if result['complete'] else 1


def prepare_review(args):
    """Materialize first requests without calling the credential loader or the HTTP worker."""
    import gc
    from .r8e6_reconfirm import _limits
    _limits()
    import torch
    from ..analyzer import calibrate
    from ..data.cifar import calibration_set
    from ..llm import Config
    from ..models.zoo import get
    from ..planners.agent_tools import build_context
    from ..planners.prompts import PROMPT_VERSION, build_system_prompt, initial_message

    torch.set_num_threads(1)
    if PROMPT_VERSION != 'r8-v4':
        raise RuntimeError('R8h review requires r8-v4')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(round='R8h', endpoint='https://api.deepseek.com/chat/completions',
        order=list(ORDER), prompt_version=PROMPT_VERSION, requests=[],
        api_budget=30, global_call_limit=78, max_agent_iters=6, retries=0,
        extra_connectivity_requests=0, max_tokens=32768, reasoning_effort='high',
        initial_payload='Model operators, shapes, parameter names, calibration/heldout-derived statistics and generic prompts',
        followup_payload='Generated DSL, internal parse/typecheck/exact/validation feedback, tuning changes and predicted costs',
        excluded_payload='Raw images, raw model weights, confirmation/report-set results, compiler and CKKS results',
        authentication='Existing approved credential loader, only during later authorized real calls',
        api_calls=0, credentials_read=False, network_calls=0,
        confirm_split_read=False, report_split_read=False)
    for name in ORDER:
        model = get(name, args.models_dir)
        graph = model.import_graph()
        images, _ = calibration_set(normalization=model.normalization)
        images = model.adapt(images)
        calibration = calibrate(graph, {graph.inputs[0]: images})
        ctx = build_context(graph, calibration, normalization=model.normalization)
        body = Config(model='deepseek-flash', reasoning_effort='high', max_tokens=32768,
                      timeout_seconds=600).settings()
        body['messages'] = [dict(role='system', content=build_system_prompt(ctx)),
                            dict(role='user', content=initial_message(ctx))]
        destination = args.out / (name + '.first-request.pretty.json')
        if destination.exists():
            raise FileExistsError(destination)
        _write(destination, body)
        manifest['requests'].append(dict(model=name, path=str(destination), bytes=destination.stat().st_size,
                                         val_split=ctx.val_split))
        _write(args.out / 'manifest.json', manifest)
        del ctx, calibration, graph, images, model, body
        gc.collect()
    manifest['complete'] = True
    _write(args.out / 'manifest.json', manifest)
    print('MANIFEST=' + str(args.out / 'manifest.json'))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=ORDER)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--budget-path', type=Path)
    parser.add_argument('--call-limit', type=int)
    parser.add_argument('--continuation', choices=('agent', 'frozen', 'test'), default='agent')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--prepare-review', action='store_true')
    parser.add_argument('--old-v1', type=Path)
    parser.add_argument('--old-generation', type=Path)
    parser.add_argument('--rule-report', type=Path)
    parser.add_argument('--r8g', type=Path)
    args = parser.parse_args()
    if args.prepare_review:
        if args.out is None or args.models_dir is None:
            parser.error('Review preparation requires explicit output and model paths')
        return prepare_review(args)
    if args.summary:
        if not all((args.out, args.old_v1, args.old_generation, args.rule_report, args.r8g)):
            parser.error('Summary requires explicit result paths')
        return summarize(args)
    if args.inside:
        if not all((args.model, args.out, args.models_dir, args.budget_path)) or args.call_limit is None:
            parser.error('--inside requires explicit model, output, models, budget and call limit')
        if not 0 <= args.call_limit <= GLOBAL_LIMIT:
            parser.error('--call-limit must remain within the cumulative R8h cap of 78')
        return inside(args)
    if args.budget_path or args.call_limit is not None or (args.resume and args.out):
        parser.error('Budget is managed by the outer process; choose --out or --resume')
    return outer(args)


if __name__ == '__main__':
    raise SystemExit(main())
