"""R8h2: reconfirm retained pools, with the preregistered ResNet20 fallback."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from ..paths import pt_result
from ..run import _write
from .r8e6_reconfirm import _limits, reconstruct_candidates
from .r8g_mlp_rerun import INNER_SECONDS, _calls, _run_args, _stop


ORDER = ('mlp', 'cnn_plain', 'resnet20_silu', 'resnet20')
STAGES = ('check', 'export', 'compile', 'test')


def read(path):
    return json.loads(Path(path).read_text())


def inside(args):
    _limits()
    import torch
    from ..models.zoo import get
    from ..planners.agent_tools import build_context
    from ..planners.confirm import MAX_CONFIRM_EVALS, build_confirm_context, confirm_select
    from ..planners.prompts import PROMPT_VERSION
    from ..run import _inside as run_inside

    if PROMPT_VERSION != 'r8-v4' or MAX_CONFIRM_EVALS != 6:
        raise RuntimeError('R8h2 requires r8-v4 and MAX_CONFIRM_EVALS=6')
    torch.set_num_threads(1)
    path = args.out / 'REPORT.json'
    if args.agent:
        run_args = _run_args(args)
        run_args.model = args.model
        code = run_inside(run_args)
        report = read(path)
        report.update(round='R8h2', model=args.model, attempt='attempt-02', generation_attempts=2)
        _write(path, report)
        return code

    old = read(args.source_case / 'REPORT.json')
    model = get(args.model, args.models_dir)
    graph = model.import_graph()
    calibration_text = (args.source_case / 'calibration.json').read_text()
    ctx = build_context(graph, json.loads(calibration_text), normalization=model.normalization)
    confirm_ctx = build_confirm_context(ctx)
    candidates, costs = reconstruct_candidates(args.source_case / 'agent_trace.jsonl', graph.weights,
                                                profile=ctx.profile)
    if len(candidates) != old['agent']['pool_size'] or not all(row['matches'] for row in costs):
        raise RuntimeError('Reconstructed R8h pool or recorded costs differ')
    report = dict(round='R8h2', model=args.model, attempt='attempt-01', complete=False,
        stage='confirm', api_calls=0, retuning=False, generation_attempts=1,
        source_case=str(args.source_case), trace_path=str(args.source_case / 'agent_trace.jsonl'),
        val_split=ctx.val_split, confirm_split=confirm_ctx.val_split, test_split='test[0:8000]',
        pool_size=len(candidates), candidates=[asdict(candidate) for candidate in candidates],
        cost_comparisons=costs)
    _write(path, report)
    confirmation = confirm_select(candidates, confirm_ctx)
    selected = confirmation['selected']
    report['confirm'] = dict(confirmation, selected=asdict(selected) if selected else None)
    report['selected'] = report['confirm']['selected']
    report['status'] = 'feasible' if selected else 'confirm_failed'
    if selected is None:
        if len(confirmation['evals']) != confirmation['deduplicated']['after']:
            raise RuntimeError('Fallback requires confirmation of every distinct retained program')
        report.update(complete=True, stage='confirm_failed', compile=None, top1_test=None)
        _write(path, report)
        return 0

    old_program = args.source_case / 'program.ptdsl'
    report.update(best_source=selected.origin, best_iteration=selected.iteration,
        latency_pred_us=selected.latency_pred_us, boots_pred=selected.boots_pred,
        same_source_as_r8h=(selected.source == old_program.read_text()) if old_program.exists() else None)
    (args.out / 'program.ptdsl').write_text(selected.source)
    (args.out / 'calibration.json').write_text(calibration_text)
    _write(path, report)
    del ctx, confirm_ctx, graph, model, candidates, confirmation, selected
    gc.collect()
    code = run_inside(SimpleNamespace(out=args.out, model=args.model, models_dir=args.models_dir,
        planner='rule', deg=None, margin=None, refresh='compiler', act_method='cheb',
        stages=list(STAGES), final=True))
    report = read(path)
    report.update(round='R8h2', api_calls=0, retuning=False)
    if report.get('complete') and (not report.get('plan') or (report.get('compile') or {}).get('exit') != 0):
        report.update(complete=False, error='P2 compilation did not produce a RuntimePlan')
        code = 1
    _write(path, report)
    return code


def summarize(report):
    confirm = report.get('confirm') or (report.get('agent') or {}).get('confirm') or {}
    selected = confirm.get('selected')
    exact_top1, top1 = report.get('torch_top1_report'), report.get('top1_test')
    return dict(stage=report.get('stage'), complete=report.get('complete'), confirm=confirm,
        selected=selected, same_source_as_r8h=report.get('same_source_as_r8h'),
        boots_pred=selected.get('boots_pred') if selected else None,
        boots_plan=(report.get('plan') or {}).get('boots'), latency_pred_us=report.get('latency_pred_us'),
        top1_test=top1, torch_top1_report=exact_top1,
        report_acc_drop_pp=exact_top1 - top1 if top1 is not None and exact_top1 is not None else None,
        n_diverged=report.get('n_diverged'), diverged_indices=report.get('diverged_indices'),
        m1_item3_pass=report.get('m1_item3_pass'), error=report.get('error'))


def outer(args):
    from ..registry import append, git_rev, load, new_run_id

    os.environ['PT_RESULTS'] = str(pt_result())
    out = (args.out or pt_result('r8h2-reconfirm-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.models_dir = (args.models_dir or pt_result('models-v2')).resolve()
    source = (args.source_run or pt_result('r8h-v11-20261023T174607')).resolve()
    old = read(source / 'REPORT.json')
    v1 = {row['model']: row['v1'] for row in read(source / 'FINAL_REPORT.json')['models']}
    budget_path = pt_result('llm_calls.jsonl').resolve()
    before = len(_calls(budget_path))
    result = dict(round='R8h2', complete=False, stage='reconfirm', git_rev=git_rev(),
        run_id=new_run_id('R8h2', 'four-models', 'reconfirm'), source_run=str(source),
        models_dir=str(args.models_dir), api_before=before, api_calls=0, api_total=before,
        api_budget=8, api_limit=min(before + 8, 71, 300), resnet20_rerun_triggered=False,
        val_split='test[9000:10000]', confirm_split='test[8000:9000]', test_split='test[0:8000]',
        max_confirm_evals=6, models={}, seconds=0, error=None)
    path = out / 'REPORT.json'
    recorded = {row['metrics'].get('global_call') for row in load() if row['mode'] == 'llm-call'}
    started = time.monotonic()

    def account(case, *, wait=False):
        calls = _calls(budget_path, wait=wait)
        if calls is None:
            return
        for index in range(before + 1, len(calls) + 1):
            if index not in recorded:
                call = calls[index - 1]
                append(dict(run_id=new_run_id('R8h2', 'resnet20', f'api-{index}'), round='R8h2',
                    model='resnet20', mode='llm-call', git_rev=result['git_rev'],
                    metrics=dict(call, global_call=index), artifacts_dir=str(case), ts=call['ts']))
                recorded.add(index)
        result.update(api_calls=len(calls) - before, api_total=len(calls))

    def attempt(name, number):
        case = out / 'models' / name / f'attempt-{number:02d}'
        case.mkdir(parents=True, exist_ok=False)
        entry = result['models'][name]
        row = dict(attempt=number, report_path=str(case / 'REPORT.json'), complete=False,
            generation='new-agent' if number == 2 else 'retained-r8h-pool', api_calls=0)
        entry['attempts'].append(row)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(INNER_SECONDS), '--',
            '-m', 'pt_agent.experiments.r8h2_reconfirm', '--inside', '--model', name,
            '--out', str(case), '--models-dir', str(args.models_dir)]
        if number == 1:
            command += ['--source-case', str(Path(old['models'][name]['report_path']).parent)]
        else:
            command += ['--agent', '--budget-path', str(budget_path), '--call-limit', str(result['api_limit'])]
        row['command'] = command
        result['active_model'] = name
        _write(path, result)
        process, begin = None, time.monotonic()
        api_begin = result['api_total']
        try:
            if number == 2 and api_begin >= result['api_limit']:
                raise RuntimeError('No authorized R8h2 API calls remain')
            with (case / 'stdout.txt').open('w') as stdout, (case / 'stderr.txt').open('w') as stderr:
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                    stdout=stdout, stderr=stderr, start_new_session=True)
                while True:
                    try:
                        row['exit_code'] = process.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        account(case)
                        row['seconds'] = time.monotonic() - begin
                        result['seconds'] = time.monotonic() - started
                        _write(path, result)
                        if row['seconds'] > INNER_SECONDS + 120:
                            raise TimeoutError('Existing per-model Nix allowance exhausted')
        except (Exception, KeyboardInterrupt) as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            _stop(process)
            row['exit_code'] = process.returncode if process else None
        account(case, wait=True)
        current = read(case / 'REPORT.json') if (case / 'REPORT.json').exists() else {}
        confirmation = current.get('confirm') or (current.get('agent') or {}).get('confirm') or {}
        failed_confirmation = confirmation.get('selected') is None and bool(confirmation.get('evals'))
        ordinary_complete = current.get('complete') and current.get('stage') == 'complete' and row['exit_code'] == 0
        expected_failure = failed_confirmation and (
            (number == 1 and current.get('stage') == 'confirm_failed' and row['exit_code'] == 0)
            or (number == 2 and (current.get('agent') or {}).get('status') == 'confirm_failed'))
        row.update(complete=bool(ordinary_complete or expected_failure), seconds=time.monotonic() - begin,
            api_calls=result['api_total'] - api_begin, metrics=summarize(current),
            error=current.get('error') or row.get('error'), confirm_failed=bool(expected_failure))
        entry.update(report_path=str(case / 'REPORT.json'), metrics=row['metrics'],
            complete=row['complete'], generation_attempts=number)
        _write(path, result)
        append(dict(run_id=new_run_id('R8h2', name, f'attempt-{number:02d}'), round='R8h2', model=name,
            mode='agent-rerun' if number == 2 else 'reconfirm', git_rev=result['git_rev'],
            metrics=row, artifacts_dir=str(case), ts=datetime.now(timezone.utc).isoformat()))
        print(f"CASE {name} attempt={number} stage={current.get('stage')} complete={row['complete']}", flush=True)
        return row

    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out / 'runner.pid').write_text(str(os.getpid()) + '\n')
        _write(path, result)
        print('RUN_DIR=' + str(out), flush=True)
        for name in ORDER:
            result['models'][name] = dict(model=name, attempts=[], v1=v1[name])
            row = attempt(name, 1)
            if not row['complete']:
                result['error'] = row['error'] or 'Reconfirmation or evaluation failed'
                break
            if row['confirm_failed']:
                if name != 'resnet20':
                    result['error'] = f'{name}: retained candidates failed confirmation'
                    break
                result['resnet20_rerun_triggered'] = True
                row = attempt(name, 2)
                if not row['complete']:
                    result['error'] = row['error'] or 'attempt-02 failed'
                    break
                if row['confirm_failed']:
                    result['models'][name]['status'] = '确认失败'
                    break
        result.update(complete=len(result['models']) == len(ORDER)
            and all(entry['complete'] for entry in result['models'].values()) and result['error'] is None,
            seconds=time.monotonic() - started, active_model=None)
        result['stage'] = 'complete' if result['complete'] else 'failed'
        _write(path, result)
        append(dict(run_id=result['run_id'], round='R8h2', model='four-models', mode='reconfirm-summary',
            git_rev=result['git_rev'], metrics=result, artifacts_dir=str(out),
            ts=datetime.now(timezone.utc).isoformat()))
        print('REPORT=' + str(path), flush=True)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--agent', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--source-case', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--model', choices=ORDER)
    parser.add_argument('--budget-path', type=Path)
    parser.add_argument('--call-limit', type=int)
    args = parser.parse_args()
    if args.inside:
        required = (args.out, args.model, args.models_dir)
        required += (args.budget_path, args.call_limit) if args.agent else (args.source_case,)
        if any(value is None for value in required):
            parser.error('--inside requires explicit paths and model; --agent also requires API limit')
        if args.agent and (args.model != 'resnet20' or not 0 <= args.call_limit <= 71):
            parser.error('Only the preregistered ResNet20 attempt-02 is authorized')
        return inside(args)
    if args.agent or args.budget_path is not None or args.call_limit is not None:
        parser.error('Agent fallback is selected by the outer process after retained-pool confirmation')
    return outer(args)


if __name__ == '__main__':
    sys.exit(main())
