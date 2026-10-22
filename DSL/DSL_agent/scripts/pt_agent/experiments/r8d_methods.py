"""Compare minimax ReLU rule programs with the retained R8a Chebyshev results."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from ..paths import pt_result


NAMES = ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp')
BASELINE_REPORT = pt_result('r8a-dev-20261021T074003-v2/REPORT.json')
STAGES = 'plan,check,mock,count,emit,compile'


def _write(path, record):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _inside(args):
    from ..cost import estimate, plan_latency
    from ..dsl.parser import parse
    from ..models.zoo import get
    from ..run import _inside as run_inside

    code = run_inside(SimpleNamespace(out=args.out, model=args.model, models_dir=args.models_dir,
                                     planner='rule', deg=127, margin=1.5, refresh='compiler',
                                     act_method='minimax_comp', stages=STAGES.split(','), final=False))
    result = dict(complete=False, model=args.model, predicted={}, plan_latency={}, activation_methods={})
    try:
        baseline = json.loads(args.baseline_rule_report.read_text())
        current = json.loads((args.out/'REPORT.json').read_text())
        if code:
            raise RuntimeError(current.get('error', 'rule evaluation failed'))
        graph = get(args.model, args.models_dir).import_graph()
        for method, report in [('cheb', baseline), ('minimax_comp', current)]:
            program = parse(Path(report['program_path']).read_text())
            result['predicted'][method] = estimate(program, graph.weights)
            plan_path = Path(report['emit']['run_dir'])/'lowered._hecate_golden.runtime-plan.json'
            result['plan_latency'][method] = plan_latency(plan_path)
            result['activation_methods'][method] = dict(Counter(op.attrs['method'] for op in program.ops
                                                               if op.kind == 'act'))
        result['complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    _write(args.out/'METRICS.json', result)
    return 0 if result['complete'] else 1


def _row(name, method, report, derived):
    compiled, plan = report.get('compile') or {}, report.get('plan') or {}
    latency = (derived.get('plan_latency') or {}).get(method) or report.get('plan_latency') or {}
    predicted = (derived.get('predicted') or {}).get(method) or {}
    return dict(model=name, method=method, agree=report.get('agree'),
                images=report.get('mock_images_done', 0), max_depth=report.get('max_depth'),
                boots_plan=plan.get('boots'), latency_pred_us=predicted.get('latency_pred_us'),
                plan_latency_us=latency.get('latency_us'), compile_s=compiled.get('seconds'),
                peak_rss_mib=compiled.get('peak_rss_mib'), compile_exit=compiled.get('exit'),
                complete=bool(report.get('complete') and compiled.get('exit') == 0 and plan),
                activation_methods=(derived.get('activation_methods') or {}).get(method),
                error=report.get('error'))


def comparison_markdown(rows):
    lines = ['| 模型 | 方法 | agree/1000 | max_depth | boots（plan） | 预测延迟（μs） | plan 延迟（μs） | 编译耗时 / 峰值内存 |',
             '|---|---|---|---|---|---|---|---|']
    for row in rows:
        def value(key):
            return str(row[key]) if row.get(key) is not None else '—'
        compiled = (f"{row['compile_s']:.2f}s / {row['peak_rss_mib']:.2f} MiB"
                    if row.get('compile_s') is not None and row.get('peak_rss_mib') is not None else '—')
        lines.append('| '+' | '.join((row['model'], row['method'], value('agree'), value('max_depth'),
                                     value('boots_plan'), value('latency_pred_us'),
                                     value('plan_latency_us'), compiled))+' |')
    return '\n'.join(lines)


def _outer(args):
    root = Path(os.environ['PT_RESULTS']).resolve()
    out = (args.out or root/('r8d-methods-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out/'REPORT.json'
    if path.exists():
        raise FileExistsError('Existing experiment reports must be preserved')
    models_dir = (args.models_dir or root/'models-v2').resolve()
    baseline_path = args.baseline_report.resolve()
    baseline = json.loads(baseline_path.read_text())
    retained = {row['model']: row for row in baseline['models']}
    result = dict(round='R8d', complete=False, stage='methods', models_dir=str(models_dir),
                  baseline_report=str(baseline_path), act_method='minimax_comp', deg=127, margin=1.5,
                  api_calls=0, models=[], comparison=[])
    _write(path, result)
    started = time.monotonic()
    for name in NAMES:
        case = out/name
        case.mkdir()
        baseline_rule = Path(retained[name]['rule_report_path'])
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '10800', '--',
                   '-m', 'pt_agent.experiments.r8d_methods', '--inside', '--out', str(case),
                   '--model', name, '--models-dir', str(models_dir),
                   '--baseline-rule-report', str(baseline_rule)]
        row = dict(model=name, command=command, report_path=str(case/'REPORT.json'),
                   metrics_path=str(case/'METRICS.json'), complete=False)
        result['models'].append(row)
        case_started = time.monotonic()
        try:
            with (case/'stdout.txt').open('w') as stdout, (case/'stderr.txt').open('w') as stderr:
                completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                           stdout=stdout, stderr=stderr, timeout=10920)
            row['exit_code'] = completed.returncode
        except Exception as exc:
            row.update(exit_code=None, error=f'{type(exc).__name__}: {exc}')
        current = json.loads((case/'REPORT.json').read_text()) if (case/'REPORT.json').exists() else {}
        derived = json.loads((case/'METRICS.json').read_text()) if (case/'METRICS.json').exists() else {}
        old = json.loads(baseline_rule.read_text())
        before, after = _row(name, 'cheb', old, derived), _row(name, 'minimax_comp', current, derived)
        after['complete'] = bool(after['complete'] and derived.get('complete') and row['exit_code'] == 0)
        row.update(complete=after['complete'], seconds=time.monotonic()-case_started,
                   error=row.get('error') or current.get('error') or derived.get('error'))
        result['comparison'].extend((before, after))
        _write(path, result)
        print('MODEL '+json.dumps(row), flush=True)
    result.update(complete=all(row['complete'] for row in result['models']), stage='complete',
                  seconds=time.monotonic()-started)
    from ..registry import append, git_rev, new_run_id

    result['run_id'] = new_run_id('R8d', 'dev-models', 'method-comparison')
    result['git_rev'] = git_rev()
    _write(path, result)
    append(dict(run_id=result['run_id'], round='R8d', model='dev-models', mode='method-comparison',
                git_rev=result['git_rev'], metrics=result, artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()))
    print(comparison_markdown(result['comparison']), flush=True)
    print('REPORT='+str(path), flush=True)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--baseline-report', type=Path, default=BASELINE_REPORT)
    parser.add_argument('--baseline-rule-report', type=Path)
    parser.add_argument('--model', choices=NAMES)
    args = parser.parse_args()
    if args.inside:
        if any(value is None for value in (args.out, args.models_dir, args.model, args.baseline_rule_report)):
            parser.error('--inside requires --out, --models-dir, --model and --baseline-rule-report')
        return _inside(args)
    return _outer(args)


if __name__ == '__main__':
    sys.exit(main())
