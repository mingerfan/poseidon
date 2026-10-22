"""Confirm retained R8e5 candidates once, without an LLM or numeric tuning."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

from ..paths import pt_result
from ..run import _write


NAMES = ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp')
SOURCE_RUN = pt_result('r8e5-dev-20261022T041646')
BASELINES = {
    'R6c': 'run-resnet20-rule-20261020T165817',
    'R7c-B': 'run-resnet20-agent-20261021T033036',
}
STAGES = ('check', 'mock', 'count', 'emit', 'compile', 'test')
INNER_SECONDS = 21600


def estimate(program, weights, *, profile='qwen48'):
    """Keep the scientific dependency inside the pure environment."""
    from ..cost import estimate as estimate_cost

    return estimate_cost(program, weights, profile=profile)


def reconstruct_candidates(trace_path, weights, *, profile='qwen48'):
    """Recover every feasible original and tuned source, preserving its text."""
    from ..dsl.parser import parse
    from ..planners.confirm import Candidate

    candidates, comparisons = [], []
    for line in Path(trace_path).read_text().splitlines():
        row = json.loads(line)
        retained = []
        if (row.get('fb') or {}).get('feasible'):
            assistant = next(message for message in reversed(row['messages'])
                             if message.get('role') == 'assistant')
            source = json.loads(assistant['content'])['program']
            retained.append(('llm', source, row['fb']['cost']))
        tuned = row.get('tuned') or {}
        if tuned.get('feasible'):
            retained.append(('tuned', tuned['source'], tuned['cost_after']))
        for origin, source, recorded in retained:
            if not isinstance(source, str):
                raise TypeError('Retained candidate source must be a string')
            cost = estimate(parse(source), weights, profile=profile)
            candidate = Candidate(source=source, origin=origin, iteration=row['iteration'],
                                  latency_pred_us=cost['latency_pred_us'], boots_pred=cost['boots_pred'])
            candidates.append(candidate)
            previous = {key: recorded.get(key) for key in ('latency_pred_us', 'boots_pred')}
            current = {key: cost[key] for key in previous}
            comparisons.append(dict(iteration=candidate.iteration, origin=origin,
                                    recorded=previous, recomputed=current,
                                    matches=previous == current))
    return candidates, comparisons


def _summary(report, report_path):
    return dict(report_path=str(report_path), test_split=report.get('test_split'),
                best_source=report.get('best_source'), best_iteration=report.get('best_iteration'),
                boots_plan=(report.get('plan') or {}).get('boots'),
                latency_pred_us=report.get('latency_pred_us'), top1_test=report.get('top1_test'),
                torch_top1_report=report.get('torch_top1_report'), n_diverged=report.get('n_diverged'),
                m1_item3_pass=report.get('m1_item3_pass'))


def _inside(args):
    import gc

    from ..dsl.parser import parse
    from ..models.zoo import get
    from ..planners.agent_tools import build_context
    from ..planners.confirm import Candidate, build_confirm_context, confirm_select
    from ..run import _inside as run_inside

    baseline = args.baseline is not None
    name = 'resnet20' if baseline else args.model
    retained = args.source_root / (BASELINES[args.baseline] if baseline else name)
    old_path = retained/'REPORT.json'
    old = json.loads(old_path.read_text())
    model = get(name, args.models_dir)
    graph = model.import_graph()
    if baseline:
        from ..analyzer import calibrate
        from ..data.cifar import calibration_set

        images, _ = calibration_set(normalization=model.normalization)
        calibration = calibrate(graph, {graph.inputs[0]: model.adapt(images)})
        calibration_text = json.dumps(calibration, indent=2, allow_nan=False)+'\n'
        del images
    else:
        calibration_text = (retained/'calibration.json').read_text()
        calibration = json.loads(calibration_text)
    ctx = build_context(graph, calibration, normalization=model.normalization)
    confirm_ctx = build_confirm_context(ctx)
    report = dict(round=args.round, model=name, baseline=args.baseline, complete=False,
                  stage='confirm', api_calls=0, retuning=False,
                  val_split=ctx.val_split, confirm_split=confirm_ctx.val_split,
                  test_split='test[0:8000]', original_frozen=_summary(old, old_path))
    if baseline:
        source = (retained/'program.ptdsl').read_text()
        cost = estimate(parse(source), graph.weights, profile=ctx.profile)
        candidates = [Candidate(source, 'llm' if args.baseline == 'R7c-B' else 'rule',
                                old.get('best_iteration') or 0,
                                cost['latency_pred_us'], cost['boots_pred'])]
        comparisons = []
        report['baseline_compiled_plan'] = old.get('plan')
    else:
        report['trace_path'] = str(retained/'agent_trace.jsonl')
        candidates, comparisons = reconstruct_candidates(retained/'agent_trace.jsonl', graph.weights,
                                                          profile=ctx.profile)
    report.update(pool_size=len(candidates), candidates=[asdict(candidate) for candidate in candidates],
                  cost_comparisons=comparisons)
    _write(args.out/'REPORT.json', report)
    confirmation = confirm_select(candidates, confirm_ctx, max_evals=1 if baseline else 6)
    selected = confirmation['selected']
    report['confirm'] = dict(selected=asdict(selected) if selected else None, evals=confirmation['evals'])
    report['selected'] = asdict(selected) if selected else None
    report['status'] = 'feasible' if selected else 'confirm_failed'
    # A fixed baseline is evaluated for comparison even when its confirmation fails.
    if not baseline and selected is None:
        report.update(complete=True, stage='confirm_failed', compile=None, top1_test=None,
                      torch_top1_report=None, n_diverged=None, m1_item3_pass=False)
        _write(args.out/'REPORT.json', report)
        return 0
    if not baseline:
        source = selected.source
        report.update(best_source=selected.origin, best_iteration=selected.iteration,
                      latency_pred_us=selected.latency_pred_us, boots_pred=selected.boots_pred,
                      same_source_as_r8e5=source == (retained/'program.ptdsl').read_text())
    (args.out/'program.ptdsl').write_text(source)
    (args.out/'calibration.json').write_text(calibration_text)
    _write(args.out/'REPORT.json', report)
    del ctx, confirm_ctx, graph, model, candidates
    gc.collect()
    code = run_inside(SimpleNamespace(out=args.out, model=name, models_dir=args.models_dir,
                                     planner='rule', deg=None, margin=None, refresh='compiler',
                                     act_method='cheb', stages=['test'] if baseline else list(STAGES),
                                     final=True))
    current = json.loads((args.out/'REPORT.json').read_text())
    current.update(round=args.round, api_calls=0, retuning=False)
    if not baseline and current.get('complete') and (
            not current.get('plan') or (current.get('compile') or {}).get('exit') != 0):
        current.update(complete=False, error='Compilation did not produce a RuntimePlan')
        code = 1
    if current.get('top1_test') is not None and current.get('torch_top1_report') is not None:
        current['report_acc_drop_pp'] = current['torch_top1_report']-current['top1_test']
    current['boots_plan'] = ((current.get('baseline_compiled_plan') if baseline else current.get('plan'))
                             or {}).get('boots')
    _write(args.out/'REPORT.json', current)
    return code


def _limits():
    # Trusted toolchain children may use the separately approved 6 GiB limit.
    _, existing_hard = resource.getrlimit(resource.RLIMIT_AS)
    hard = 6*1024**3 if existing_hard == resource.RLIM_INFINITY else min(existing_hard, 6*1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (min(4*1024**3, hard), hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _calls(root):
    path = root/'llm_calls.jsonl'
    return len(path.read_text().splitlines()) if path.exists() else 0


def _outer(args):
    from ..registry import append, git_rev, new_run_id

    root = Path(os.environ['PT_RESULTS']).resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'REPORT.json').exists():
        raise FileExistsError('Existing experiment reports must be preserved')
    (out/'runner.pid').write_text(str(os.getpid())+'\n')
    models = (args.models_dir or root/'models-v2').resolve()
    source = args.source_run.resolve()
    result = dict(round=args.round, complete=False, stage='reconfirm', source_run=str(source),
                  models_dir=str(models), val_split='test[9000:10000]', confirm_split='test[8000:9000]',
                  test_split='test[0:8000]', max_confirm_evals=6, api_calls=0, api_before=_calls(root),
                  compiles=0, models=[], baselines=[], git_rev=git_rev())
    path = out/'REPORT.json'
    _write(path, result)
    started = time.monotonic()
    for name, baseline in [(name, None) for name in NAMES]+[(name, name) for name in BASELINES]:
        case = out/('baselines' if baseline else 'models')/name
        case.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(INNER_SECONDS), '--',
                   '-m', 'pt_agent.experiments.r8e6_reconfirm', '--inside', '--round', args.round, '--out', str(case),
                   '--source-root', str(root if baseline else source), '--models-dir', str(models)]
        command += ['--baseline', baseline] if baseline else ['--model', name]
        row = dict(model='resnet20' if baseline else name, baseline=baseline, report_path=str(case/'REPORT.json'),
                   command=command, complete=False)
        result['baselines' if baseline else 'models'].append(row)
        _write(path, result)
        case_started = time.monotonic()
        try:
            with (case/'stdout.txt').open('w') as stdout, (case/'stderr.txt').open('w') as stderr:
                completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                           stdout=stdout, stderr=stderr, timeout=INNER_SECONDS+120,
                                           preexec_fn=_limits)
            row['exit_code'] = completed.returncode
        except Exception as exc:
            row.update(exit_code=None, error=f'{type(exc).__name__}: {exc}')
        current = json.loads((case/'REPORT.json').read_text()) if (case/'REPORT.json').exists() else {}
        row.update(complete=bool(current.get('complete') and row['exit_code'] == 0),
                   stage=current.get('stage'), status=current.get('status'), seconds=time.monotonic()-case_started,
                   pool_size=current.get('pool_size'), confirm=current.get('confirm'), selected=current.get('selected'),
                   boots_plan=current.get('boots_plan'), top1_test=current.get('top1_test'),
                   torch_top1_report=current.get('torch_top1_report'), report_acc_drop_pp=current.get('report_acc_drop_pp'),
                   n_diverged=current.get('n_diverged'), m1_item3_pass=current.get('m1_item3_pass'),
                   original_frozen=current.get('original_frozen'), error=current.get('error') or row.get('error'))
        result['compiles'] += int(current.get('compile') is not None)
        _write(path, result)
        print(f"CASE {name} complete={row['complete']} status={row['status']} "
              f"top1_test={row['top1_test']}", flush=True)
        if not row['complete']:
            break
    rows = result['models']+result['baselines']
    result.update(complete=len(rows) == len(NAMES)+len(BASELINES) and all(row['complete'] for row in rows),
                  stage='complete' if len(rows) == len(NAMES)+len(BASELINES) and all(row['complete'] for row in rows)
                        else 'failed',
                  api_after=_calls(root), seconds=time.monotonic()-started,
                  run_id=new_run_id(args.round, 'dev-models', 'reconfirm'))
    _write(path, result)
    append(dict(run_id=result['run_id'], round=args.round, model='dev-models', mode='reconfirm',
                git_rev=result['git_rev'], metrics=result, artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(path), flush=True)
    lock.close()
    return 0 if result['complete'] else 1


def _launch(args):
    root = Path(os.environ['PT_RESULTS']).resolve()
    out = (args.out or root/(args.round.lower()+'-reconfirm-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = ['setsid', 'nohup', sys.executable, '-B', '-m', 'pt_agent.experiments.r8e6_reconfirm',
               '--worker', '--round', args.round, '--out', str(out), '--source-run', str(args.source_run.resolve())]
    if args.models_dir is not None:
        command += ['--models-dir', str(args.models_dir.resolve())]
    with (out/'runner.stdout.txt').open('w') as stdout, (out/'runner.stderr.txt').open('w') as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   cwd=Path(__file__).resolve().parents[2])
    (out/'runner.pid').write_text(str(process.pid)+'\n')
    print('RUN_DIR='+str(out), flush=True)
    print('RUNNER_PID='+str(process.pid), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', default='R8e6')
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source-run', type=Path, default=SOURCE_RUN)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--model', choices=NAMES)
    parser.add_argument('--baseline', choices=BASELINES)
    args = parser.parse_args()
    if args.inside:
        if any(value is None for value in (args.out, args.source_root, args.models_dir)):
            parser.error('--inside requires explicit out, source-root and models-dir')
        if (args.model is None) == (args.baseline is None):
            parser.error('--inside requires exactly one model or baseline')
        return _inside(args)
    if args.worker and args.out is None:
        parser.error('--worker requires an explicit output directory')
    return _outer(args) if args.worker else _launch(args)


if __name__ == '__main__':
    sys.exit(main())
