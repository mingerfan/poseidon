"""Four development-model Agent runs with pre-freeze numeric tuning."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .r8d_methods import BASELINE_REPORT, NAMES, _write


STAGES = 'plan,check,mock,count,emit,compile,test'


def _inside(args):
    from ..cost import estimate, plan_latency
    from ..dsl.parser import parse
    from ..models.zoo import get
    from ..run import _inside as run_inside

    code = run_inside(SimpleNamespace(out=args.out, model=args.model, models_dir=args.models_dir,
                                     planner='agent', deg=None, margin=None, refresh='compiler', act_method='cheb',
                                     stages=['test'] if getattr(args, 'test_only', False) else STAGES.split(','),
                                     final=True, agent_iters=getattr(args, 'agent_iters', 6),
                                     llm_model='deepseek-flash', llm_reasoning='high', llm_max_tokens=32768,
                                     budget_path=args.budget_path, llm_call_limit=args.call_limit,
                                     tuner=args.tuner, tuner_budget_s=2400))
    report_path = args.out/'REPORT.json'
    report = json.loads(report_path.read_text())
    report.update(round=args.round, tuner=args.tuner)
    try:
        graph = get(args.model, args.models_dir).import_graph()
        old = json.loads(args.baseline_rule_report.read_text())
        rule = estimate(parse(Path(old['program_path']).read_text()), graph.weights)
        old_plan = Path(old['emit']['run_dir'])/'lowered._hecate_golden.runtime-plan.json'
        report['rule_baseline'] = dict(report_path=str(args.baseline_rule_report),
                                     agree=old.get('agree'), top1=old.get('top1'),
                                     predicted=rule, plan=old.get('plan'), plan_latency=plan_latency(old_plan))
        best = (report.get('agent') or {}).get('best')
        if best:
            plan = report.get('plan') or {}
            latency = (report.get('plan_latency') or {}).get('latency_us')
            predicted = best['latency_pred_us']
            report['predicted_vs_compiled'] = dict(boots_pred=best['boots_pred'], boots_plan=plan.get('boots'),
                                                  latency_pred_us=predicted, latency_plan_us=latency,
                                                  relative_error=abs(predicted-latency)/latency if latency else None)
            report['hotspots'] = best.get('hotspots', [])[:3]
            report['boot_input_max_abs'] = best.get('boot_input_max_abs')
        if report.get('complete') and (not report.get('plan') or (report.get('compile') or {}).get('exit') != 0):
            report.update(complete=False, error='Compilation did not produce a RuntimePlan')
            code = 1
        report['r7c_b_boots'] = 288 if args.model == 'resnet20' else None
    except Exception as exc:
        report['comparison_error'] = f'{type(exc).__name__}: {exc}'
        code = 1
    _write(report_path, report)
    return code


def _calls(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _prompt_version():
    # The outer Python has no NumPy; import the canonical constant in the pure environment.
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '600', '--', '-c',
               'from pt_agent.planners.prompts import PROMPT_VERSION; print(PROMPT_VERSION)']
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, check=True, timeout=720)
    return result.stdout.strip().splitlines()[-1]


def _outer(args):
    from ..registry import append, git_rev, load, new_run_id

    root = Path(os.environ['PT_RESULTS']).resolve()
    out = (args.resume or args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out/'REPORT.json'
    lock = (out/'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out/'runner.pid').write_text(str(os.getpid())+'\n')
    budget_path = root/'llm_calls.jsonl'
    if args.resume:
        result = json.loads(path.read_text())
        if result.get('val_split') != 'test[9000:10000]':
            raise ValueError('Resume requires the held-out split; retain earlier runs separately')
        args.round = result['round']
        args.tuner = result['tuner']
    else:
        if path.exists():
            raise FileExistsError('Existing experiment reports must be preserved')
        base = len(_calls(budget_path))
        result = dict(round=args.round, complete=False, stage='agent',
                      models_dir=str((args.models_dir or root/'models-v2').resolve()),
                      baseline_report=str(args.baseline_report), prompt_version=_prompt_version(), tuner=args.tuner,
                      val_split='test[9000:10000]', test_split='test[0:9000]',
                      api_before=base, api_budget=24, api_limit=min(base+24, 300), api_calls=0,
                      models=[dict(model=name, report_path=str(out/name/'REPORT.json'), complete=False,
                                   call_indices=[], attempts=[]) for name in NAMES], git_rev=git_rev())
    retained = {row['model']: row for row in json.loads(Path(result['baseline_report']).read_text())['models']}
    recorded = {row['metrics'].get('global_call') for row in load() if row['mode'] == 'llm-call'}

    def account(row):
        calls = _calls(budget_path)
        if row.get('active_start') is not None:
            indices = range(row['active_start']+1, min(len(calls), row['active_limit'])+1)
            row['call_indices'] = sorted(set(row.get('call_indices', [])) | set(indices))
        for index in row.get('call_indices', []):
            if index in recorded:
                continue
            call = calls[index-1]
            append(dict(run_id=new_run_id(args.round, row['model'], f'api-{index}'), round=args.round,
                        model=row['model'], mode='llm-call', git_rev=result['git_rev'],
                        metrics=dict(call, global_call=index), artifacts_dir=str(Path(row['report_path']).parent),
                        ts=call['ts']))
            recorded.add(index)
        row['api_calls'] = len(row.get('call_indices', []))
        result.update(api_calls=sum(len(item.get('call_indices', [])) for item in result['models']),
                      api_total=len(calls))

    result.update(complete=False, stage='agent')
    _write(path, result)
    started = time.monotonic()
    for row in result['models']:
        account(row)
        name = row['model']
        case = Path(row['report_path']).parent
        current = json.loads((case/'REPORT.json').read_text()) if (case/'REPORT.json').exists() else {}
        if current.get('complete') is True and current.get('stage') == 'complete':
            row.update(complete=True, stage='complete', skipped_on_resume=True)
            row['active_start'] = None
            _write(path, result)
            continue
        test_only = ((case/'program.ptdsl').exists() and current.get('stage') == 'test'
                     and (current.get('compile') or {}).get('exit') == 0)
        remaining = max(0, 6-row['api_calls'])
        if not test_only and (remaining == 0 or len(_calls(budget_path)) >= result['api_limit']):
            row.update(complete=False, error='Original model or batch API budget exhausted')
            row['active_start'] = None
            _write(path, result)
            continue
        if not test_only and current:
            number = len(row.get('attempts', []))+1
            case = out/f'{name}-attempt-{number:02d}'
            while case.exists():
                number += 1
                case = out/f'{name}-attempt-{number:02d}'
        case.mkdir(exist_ok=True)
        row['report_path'] = str(case/'REPORT.json')
        call_start = len(_calls(budget_path))
        call_limit = min(call_start+remaining, result['api_limit'])
        row.update(active_start=call_start, active_limit=call_limit, complete=False)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '21600', '--',
                   '-m', 'pt_agent.experiments.r8b_dev', '--inside', '--round', args.round, '--out', str(case),
                   '--model', name, '--models-dir', result['models_dir'], '--tuner', args.tuner,
                   '--budget-path', str(budget_path), '--call-limit', str(call_limit),
                   '--agent-iters', str(max(1, remaining)),
                   '--baseline-rule-report', str(retained[name]['rule_report_path'])]
        if test_only:
            command.append('--test-only')
        attempt = dict(command=command, report_path=row['report_path'], test_only=test_only,
                       call_start=call_start, call_limit=call_limit)
        row.setdefault('attempts', []).append(attempt)
        _write(path, result)
        case_started = time.monotonic()
        try:
            with (case/'stdout.txt').open('a') as stdout, (case/'stderr.txt').open('a') as stderr:
                completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                           stdout=stdout, stderr=stderr, timeout=21720)
            row['exit_code'] = completed.returncode
        except Exception as exc:
            row.update(exit_code=None, error=f'{type(exc).__name__}: {exc}')
        current = json.loads((case/'REPORT.json').read_text()) if (case/'REPORT.json').exists() else {}
        account(row)
        row.update(complete=bool(current.get('complete') and current.get('stage') == 'complete'
                                 and row['exit_code'] == 0),
                   seconds=time.monotonic()-case_started, error=current.get('error') or row.get('error'),
                   stage=current.get('stage'), agent=current.get('agent'), best_source=current.get('best_source'),
                   llm_source_lines=current.get('llm_source_lines'), frozen_source_lines=current.get('frozen_source_lines'),
                   predicted_vs_compiled=current.get('predicted_vs_compiled'), top1_test=current.get('top1_test'),
                   torch_top1_report=current.get('torch_top1_report'), n_diverged=current.get('n_diverged'))
        attempt.update(exit_code=row['exit_code'], seconds=row['seconds'], api_calls=len(_calls(budget_path))-call_start)
        row['active_start'] = None
        _write(path, result)
        print(f"MODEL {name} complete={row['complete']} calls={row['api_calls']} "
              f"top1_test={row['top1_test']} n_diverged={row['n_diverged']}", flush=True)
    result.update(complete=all(row['complete'] for row in result['models']), stage='complete',
                  seconds=result.get('seconds', 0)+time.monotonic()-started,
                  run_id=new_run_id(args.round, 'dev-models', 'agent-tuner'))
    _write(path, result)
    append(dict(run_id=result['run_id'], round=args.round, model='dev-models', mode='agent-tuner',
                git_rev=result['git_rev'], metrics=result, artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(path), flush=True)
    lock.close()
    return 0 if result['complete'] else 1


def _launch(args):
    root = Path(os.environ['PT_RESULTS']).resolve()
    out = (args.resume or args.out or root/('r8e-dev-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = ['setsid', 'nohup', sys.executable, '-B', '-m', 'pt_agent.experiments.r8b_dev',
               '--worker', '--round', args.round, '--tuner', args.tuner,
               '--baseline-report', str(args.baseline_report)]
    command += ['--resume' if args.resume else '--out', str(out)]
    if args.models_dir is not None:
        command += ['--models-dir', str(args.models_dir)]
    with (out/'runner.stdout.txt').open('a') as stdout, (out/'runner.stderr.txt').open('a') as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   cwd=Path(__file__).resolve().parents[2])
    (out/'runner.pid').write_text(str(process.pid)+'\n')
    print('RUN_DIR='+str(out), flush=True)
    print('RUNNER_PID='+str(process.pid), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', default='R8e')
    parser.add_argument('--tuner', choices=('on', 'off'), default='on')
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--agent-iters', type=int, default=6)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--baseline-report', type=Path, default=BASELINE_REPORT)
    parser.add_argument('--baseline-rule-report', type=Path)
    parser.add_argument('--model', choices=NAMES)
    parser.add_argument('--budget-path', type=Path)
    parser.add_argument('--call-limit', type=int)
    args = parser.parse_args()
    if args.inside:
        if any(value is None for value in (args.out, args.models_dir, args.model, args.baseline_rule_report,
                                            args.budget_path, args.call_limit)):
            parser.error('--inside requires explicit model, paths and call limit')
        return _inside(args)
    return _outer(args) if args.worker else _launch(args)


if __name__ == '__main__':
    sys.exit(main())
