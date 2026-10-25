"""U3b2 offline weight diagnostics, one missing rule baseline and three reruns."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from ..paths import RESULTS_DIR, PLATFORM_DIR, pt_result
from ..run import _write
from . import u3b_agent_cnn as base


ORDER = ('mobilenet_relu', 'vgg16_relu_maxpool', 'alexnet_relu_maxpool')
API_BUDGET = 20
GLOBAL_LIMIT = 133
PREVIOUS = pt_result('u3b-agent-20261024T170907')


def test_inside(args):
    import unittest
    packaging = {'test_p2_export.py', 'test_toolchain_decoupled.py'}
    tensor = {'test_r8f_tensor_validate.py'}
    files = sorted(Path('pt_agent/tests').glob('test_*.py'))
    if args.part == 'unit':
        files = [p for p in files if p.name == 'test_u3b2_weight_usage.py']
    elif args.part == 'intake':
        python = PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'
        if Path(sys.executable).absolute() != python.absolute():
            os.execv(str(python), [str(python), '-B', '-m', __spec__.name, *sys.argv[1:]])
        files = [p for p in files if p.name == 'test_u1_onnx.py']
    else:
        files = [p for p in files if not p.name.startswith('test_u3b2_')]
        files = ([p for p in files if p.name in packaging] if args.part == 'packaging' else
                 [p for p in files if p.name in tensor] if args.part == 'tensor' else
                 [p for p in files if p.name not in packaging | tensor])
    if args.part == 'cache':
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(
            'pt_agent.tests.test_r8f_tensor_validate.TensorTests.'+name) for name in (
                'test_11_prefix_hits_survive_last_degree_change_and_renaming',
                'test_12_lru_eviction_and_oversize', 'test_13_result_cache_isolation',
                'test_14_cost_cache_keeps_source_locations_and_copies'))
    else:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
    expected = dict(main=435, packaging=17, tensor=17, intake=4, cache=4)
    if args.part in expected and suite.countTestCases() != expected[args.part]:
        raise RuntimeError('469-test inventory differs')
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    if args.part == 'unit':
        from ..tests.test_u3b2_weight_usage import ARTIFACT_RESULTS
        _write(args.out/'WEIGHT_USAGE.json', ARTIFACT_RESULTS)
    return dict(complete=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures),
                errors=len(result.errors), skipped=len(result.skipped), llm_calls=0)


def baseline_inside(args):
    from ..dsl.parser import parse
    from ..planners.confirm import build_confirm_context
    from ..tensor_sim import cache_info, clear_caches, POLYNOMIAL_WORKSPACE_BYTES
    if args.model != 'alexnet_relu_maxpool':
        raise ValueError('Only the missing AlexNet ReLU baseline is evaluated')
    path = args.out/'REPORT.json'
    record = base.read(path) if path.exists() else dict(round='U3b2', model=args.model,
        mode='rule-baseline-completion', complete=False, stage='ready', spots={}, report_metrics={},
        source_case=str(args.source_case), llm_calls=0)
    if record['complete']:
        return record
    record.update(prior_seconds=args.prior_seconds,
                  polynomial_workspace_bytes=POLYNOMIAL_WORKSPACE_BYTES,
                  prefix_cache_limit_bytes=cache_info()['prefix']['max_bytes'])
    ctx = base.context(args.source_case)
    program = parse((args.source_case/'program.ptdsl').read_text())
    if 'rule_confirm_metrics' not in record:
        record['stage'] = 'rule-confirm'; _write(path, record)
        clear_caches(); gc.collect()
        metric = base.report_evaluation(program, build_confirm_context(ctx), engine_profile='tensor')
        record['rule_confirm_metrics'] = {k:v for k,v in metric.items() if k != 'rows'}
        record['rule_confirm_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        _write(path, record)
    clear_caches(); gc.collect()
    record['stage'] = 'rule-heldout-spot'; _write(path, record)
    record['spots']['rule-heldout'] = base.mock_spot(program, ctx, args.out, 'rule', 'heldout', 20)
    record['stage'] = 'report-reference'; _write(path, record)
    report_ctx = base.split_context(ctx, 'report', args.out)
    clear_caches(); gc.collect()
    record['stage'] = 'rule-report-spot'; _write(path, record)
    record['spots']['rule-report'] = base.mock_spot(program, report_ctx, args.out, 'rule', 'report', 100)
    _write(path, record)
    if 'rule' not in record['report_metrics']:
        record['stage'] = 'rule-report'; _write(path, record)
        clear_caches(); gc.collect()
        metric = base.report_evaluation(program, report_ctx, engine_profile='tensor')
        record['report_metrics']['rule'] = {k:v for k,v in metric.items() if k != 'rows'}
    record.update(complete=True, stage='complete', peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    _write(path, record)
    return record


def seed_baseline(case, original):
    """Reuse fixed rule evidence; none of these fields is passed into run_agent."""
    if (case/'REPORT.json').exists():
        return
    saved = base.read(original/'REPORT.json')
    if not saved['complete'] or 'rule' not in saved['report_metrics']:
        raise ValueError('The fixed rule baseline is incomplete')
    record = dict(round='U3b2', model=case.name, complete=False, stage='ready',
        prompt_version='r8-v5', engine_profile='tensor', tune_policy='new_best', max_iters=6,
        source_case=saved['source_case'],
        rule_evidence_source=str(original), rule_confirm_metrics=saved['rule_confirm_metrics'],
        report_metrics={'rule':saved['report_metrics']['rule']},
        spots={k:v for k,v in saved['spots'].items() if k.startswith('rule-')})
    for name in ('rule-heldout-spot.json', 'rule-heldout-spot.npz',
                 'rule-report-spot.json', 'rule-report-spot.npz', 'report-reference.npz'):
        source = original/name
        if source.exists() and not (case/name).exists():
            shutil.copyfile(source, case/name)
    _write(case/'REPORT.json', record)


def weight_events(case):
    from ..dsl.errors import PTDSLError
    from ..dsl.parser import parse
    from ..dsl.typecheck import referenced_weights
    trace = case/'agent_trace.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    events = []
    for index, row in enumerate(rows):
        errors = [e for e in row['fb'].get('errors', []) if e['code'] == 'E_WEIGHT_UNUSED']
        if not errors:
            continue
        missing = [json.loads(e['message'].split(' 未被使用：', 1)[0]) for e in errors]
        event = dict(iteration=row['iteration'], weights=missing, errors=errors,
                     next_iteration=None, next_stage=None, fixed_next=None, missing_next=None)
        if index+1 < len(rows):
            following = rows[index+1]
            event.update(next_iteration=following['iteration'], next_stage=following['fb']['stage'])
            try:
                content = next(m['content'] for m in following['messages'] if m['role']=='assistant')
                used = referenced_weights(parse(json.loads(content)['program']))
            except (StopIteration, ValueError, KeyError, TypeError, PTDSLError):
                pass
            else:
                event.update(fixed_next=set(missing).issubset(used),
                             missing_next=[name for name in missing if name not in used])
        events.append(event)
    return events


def prepare_inside(args):
    from ..llm import Config
    from ..planners.prompts import build_system_prompt, initial_message
    previous = base.read(args.previous/'REPORT.json')
    if not base.read(args.baseline_case/'REPORT.json')['complete']:
        raise ValueError('Complete the missing rule baseline first')
    manifest = dict(round='U3b2', complete=False, order=list(ORDER),
        source_cases=previous['source_cases'], previous=str(args.previous), baseline_case=str(args.baseline_case),
        rule_sources={name:str(args.baseline_case if name=='alexnet_relu_maxpool' else
            args.previous/'models'/name) for name in ORDER},
        endpoint='https://api.deepseek.com/chat/completions', requests=[],
        prompt_version='r8-v5', engine_profile='tensor', weight_usage_check=True,
        tune_policy='new_best', max_iters=6, max_confirm_evals=6, max_sweeps=1, tuner_budget_s=2400,
        llm_model='deepseek-flash', reasoning_effort='high', max_tokens=32768,
        api_budget=20, global_limit=133, retries=0, extra_connectivity_requests=0,
        payload='Same U3b model/shape/parameter-name, calibration/heldout statistics and frozen r8-v5 prompt; '
                'later generated DSL, internal checks (including E_WEIGHT_UNUSED), tuning and predicted costs',
        excluded_payload='Raw inputs, raw weights, confirmation/report, compiler or CKKS feedback',
        network_calls=0, credentials_read=False)
    for name in ORDER:
        ctx = base.context(Path(manifest['source_cases'][name]))
        body = Config(model='deepseek-flash', reasoning_effort='high', max_tokens=32768,
                      timeout_seconds=600).settings()
        body['messages'] = [dict(role='system', content=build_system_prompt(ctx)),
                            dict(role='user', content=initial_message(ctx))]
        path = args.out/(name+'.first-request.pretty.json')
        _write(path, body)
        manifest['requests'].append(dict(model=name, path=str(path), bytes=path.stat().st_size))
        del ctx, body
        gc.collect()
    manifest['complete'] = True
    _write(args.out/'manifest.json', manifest)
    return manifest


def combined_summary(root):
    from collections import Counter
    from statistics import median
    from ..dsl.parser import parse
    batch = base.read(root/'REPORT.json')
    previous = base.read(Path(batch['previous'])/'SUMMARY.json')
    rows = []
    for original in previous['models']:
        row = deepcopy(original)
        name = row['model']
        row['origin'] = 'U3b retained'
        if name in ORDER:
            case = root/'models'/name
            entry = batch['models'][name]
            current = base.read(case/'REPORT.json') if (case/'REPORT.json').exists() else {}
            result = base.read(case/'AGENT.json') if (case/'AGENT.json').exists() else {}
            rule = base.read(Path(batch['rule_sources'][name])/'REPORT.json')
            selected = (result.get('confirm') or {}).get('selected')
            cost = current.get('cost') or result.get('best') or {}
            trace = current.get('trace_summary') or {}
            compiled = current.get('compile_result') or {}
            metrics = current.get('report_metrics') or {}
            points = []
            if selected:
                for i,op in enumerate(op for op in parse(selected['source']).ops if op.kind in ('act','maxpool')):
                    points.append(dict(index=i, kind=op.kind, fn=op.attrs.get('fn'), method=op.attrs['method'],
                        deg=op.attrs.get('deg'), degrees=op.attrs.get('degrees'), domain=op.attrs['domain'],
                        line=op.lineno, call_lines=list(op.call_lines)))
            confirmed = [v for v in (result.get('confirm') or {}).get('evals', []) if v['passed']]
            boots = cost.get('boots_pred') if selected else None
            latency = cost.get('latency_pred_us') if selected else None
            events = weight_events(case)
            distribution = Counter(json.dumps({k:p[k] for k in
                ('kind','fn','method','deg','degrees','domain')}, sort_keys=True) for p in points)
            row.update(origin='U3b2 rerun (weight usage check)', previous_report_path=original['report_path'],
                report_path=str(case/'REPORT.json'), complete=entry['complete'], stage=entry['stage'],
                agent_status=result.get('status'), selected=selected, approximation_points=points,
                approximation_distribution=[dict(**json.loads(k), count=v) for k,v in distribution.items()],
                boots_pred=boots, boots_compiled=compiled.get('boots_compiled'),
                rotations=cost.get('counts', {}).get('rot') if selected else None, latency_pred_us=latency,
                boot_reduction_pct=100*(row['rule_boots']-boots)/row['rule_boots'] if boots is not None else None,
                latency_reduction_pct=100*(row['rule_latency_pred_us']-latency)/row['rule_latency_pred_us'] if latency is not None else None,
                confirm_acc_drop_pp=confirmed[-1]['acc_drop_pp'] if confirmed else None,
                confirm_evals=(result.get('confirm') or {}).get('evals', []),
                rule_confirm=rule['rule_confirm_metrics'], rule_report=rule['report_metrics']['rule'],
                agent_report=metrics.get('agent'), api_calls=entry.get('api_calls', 0),
                iterations=result.get('iterations', 0), tune_calls=trace.get('tune_calls', 0),
                tuner_seconds=trace.get('tuner_seconds', 0), seconds=entry.get('seconds', 0),
                native_compiled=compiled.get('native_compiled', False), server_job=compiled.get('server_job'),
                compile_result=compiled, spots=current.get('spots', {}), rule_evidence_source=batch['rule_sources'][name],
                weight_usage_events=events, weight_usage_trigger_count=len(events))
        rows.append(row)
    reductions = [r['boot_reduction_pct'] for r in rows if r['boot_reduction_pct'] is not None and r['boot_reduction_pct']>0]
    report = dict(round='U3b2', complete=batch['complete'], model_denominator=9, models=rows,
        agent_report_accuracy_pass=sum(r['agent_report'] is not None and r['agent_report']['acc_drop_pp']<=.5 for r in rows),
        rule_report_accuracy_pass=sum(r['rule_report'] is not None and r['rule_report']['acc_drop_pp']<=.5 for r in rows),
        fewer_boots_models=len(reductions), median_boot_reduction_pct=median(reductions) if reductions else None,
        api_calls=batch['api_calls'], api_total=batch['api_total'], api_limit=batch['api_limit'],
        retained_models=[r['model'] for r in rows if r['model'] not in ORDER], rerun_models=list(ORDER),
        prompt_version='r8-v5', engine_profile='tensor', weight_usage_check=True, tune_policy='new_best')
    _write(root/'SUMMARY.json', report)
    return report


def inside(args):
    base.limits()
    if args.part == 'summary':
        combined_summary(args.out)
        return 0
    import torch
    torch.set_num_threads(1)
    report = dict(round='U3b2', complete=False, part=args.part, llm_calls=0)
    try:
        if args.part == 'baseline':
            report = baseline_inside(args)
        elif args.part == 'prepare':
            report.update(prepare_inside(args))
        elif args.part == 'model':
            seed_baseline(args.out, args.rule_source)
            if Path(base.read(args.out/'REPORT.json')['source_case']).resolve() != args.source_case.resolve():
                raise ValueError('Fixed baseline and model source paths differ')
            report = base.model_inside(args)
            report['round'] = 'U3b2'
            events = weight_events(args.out)
            report.update(weight_usage_events=events, weight_usage_trigger_count=len(events))
            _write(args.out/'WEIGHT_EVENTS.json', events)
        else:
            report.update(test_inside(args))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if args.part in ('baseline','model') and (args.out/'REPORT.json').exists():
            report = base.read(args.out/'REPORT.json')
        report.update(complete=False, error=f'{type(exc).__name__}: {exc}',
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    _write(args.out/'REPORT.json', report)
    return 0 if report['complete'] else 1


def register(model, mode, report, directory):
    from ..registry import append, git_rev, new_run_id
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    append(dict(run_id=new_run_id('U3b2', model, mode), round='U3b2', model=model, mode=mode,
        git_rev=git_rev(), metrics=report, artifacts_dir=str(directory), ts=datetime.now(timezone.utc).isoformat()))


def outer_part(args):
    args.out = (args.out or RESULTS_DIR/('u3b2-'+args.part+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    seconds = base.INNER_SECONDS-math.ceil(args.prior_seconds) if args.part=='baseline' else 3600
    if seconds <= 0:
        raise RuntimeError('Prior time exhausted the baseline allowance')
    command = [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds),'--',
        '-m',__spec__.name,'--inside','--part',args.part,'--out',str(args.out),'--previous',str(args.previous)]
    if args.part == 'baseline':
        args.source_case = Path(base.read(args.previous/'REPORT.json')['source_cases']['alexnet_relu_maxpool'])
        command += ['--model','alexnet_relu_maxpool','--source-case',str(args.source_case),
                    '--prior-seconds',str(args.prior_seconds)]
    if args.part == 'prepare':
        if not args.baseline_case:
            raise ValueError('--baseline-case is required')
        command += ['--baseline-case',str(args.baseline_case.resolve())]
    _write(args.out/'argv.json', command)
    print('RUN='+str(args.out), flush=True)
    begin = time.monotonic()
    with (args.out/'run.log').open('w') as stream:
        status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=base.environment(), preexec_fn=base.limits).returncode
    report = base.read(args.out/'REPORT.json') if (args.out/'REPORT.json').exists() else dict(complete=False)
    elapsed = time.monotonic()-begin
    report.update(exit=status, attempt_seconds=elapsed, prior_seconds=args.prior_seconds,
                  seconds=args.prior_seconds+elapsed)
    _write(args.out/'REPORT.json', report)
    register('alexnet_relu_maxpool' if args.part=='baseline' else args.part, args.part, report, args.out)
    print('REPORT='+str(args.out/'REPORT.json'), flush=True)
    return status


def outer_run(args):
    from ..registry import append, git_rev, load, new_run_id
    root = (args.resume or args.out or RESULTS_DIR/('u3b2-agent-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    budget = RESULTS_DIR/'llm_calls.jsonl'
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = root/'REPORT.json'
        if args.resume:
            batch = base.read(path)
        else:
            if path.exists():
                raise FileExistsError('Use --resume')
            manifest = base.read(args.review/'manifest.json')
            if not manifest['complete'] or manifest['order'] != list(ORDER):
                raise ValueError('Incomplete review manifest')
            before = len(base._calls(budget))
            batch = dict(round='U3b2', complete=False, stage='ready', git_rev=git_rev(), previous=manifest['previous'],
                source_cases=manifest['source_cases'], rule_sources=manifest['rule_sources'],
                api_before=before, api_total=before, api_calls=0, api_budget=20, api_limit=min(before+20,133,300),
                review=str(args.review), models={name:dict(complete=False, stage='not_started') for name in ORDER})
            _write(path, batch)
        recorded = {r['metrics'].get('global_call') for r in load() if r['mode']=='llm-call'}
        for name in ORDER:
            entry = batch['models'][name]
            if entry['complete']:
                continue
            case = root/'models'/name
            case.mkdir(parents=True, exist_ok=True)
            first = entry.setdefault('api_before', len(base._calls(budget)))
            if len(base._calls(budget)) >= batch['api_limit'] and not (case/'AGENT.json').exists():
                batch['stage'] = 'budget_exhausted'
                break
            if entry.get('worker_pid'):
                try:
                    os.kill(entry['worker_pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Previous worker is still running')
            prior = entry.get('seconds', 0)
            remaining = base.INNER_SECONDS-int(prior)
            if remaining <= 0:
                raise RuntimeError('Per-model allowance exhausted')
            command = [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(remaining),'--',
                '-m',__spec__.name,'--inside','--part','model','--model',name,'--out',str(case),
                '--source-case',batch['source_cases'][name],'--rule-source',batch['rule_sources'][name],
                '--budget-path',str(budget),'--call-limit',str(batch['api_limit']),'--prior-seconds',str(prior)]
            entry.update(stage='running', command=command)
            batch.update(stage='running', active_model=name)
            _write(path, batch)
            def account(*, wait=False):
                calls = base._calls(budget, wait=wait)
                if calls is None:
                    return
                for index in range(first+1, len(calls)+1):
                    if index not in recorded:
                        call = calls[index-1]
                        append(dict(run_id=new_run_id('U3b2',name,f'api-{index}'), round='U3b2', model=name,
                            mode='llm-call', git_rev=batch['git_rev'],
                            metrics=dict(call,global_call=index), artifacts_dir=str(case), ts=call['ts']))
                        recorded.add(index)
                entry['api_calls'] = len(calls)-first
                batch.update(api_calls=len(calls)-batch['api_before'], api_total=len(calls))
            begin, process = time.monotonic(), None
            try:
                with (case/'stdout.txt').open('a') as stdout, (case/'stderr.txt').open('a') as stderr:
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                               env=base.environment(), start_new_session=True)
                    entry['worker_pid'] = process.pid
                    _write(path, batch)
                    while True:
                        try:
                            code = process.wait(timeout=2)
                            break
                        except subprocess.TimeoutExpired:
                            account()
                            entry['seconds'] = prior+time.monotonic()-begin
                            _write(path, batch)
                            if entry['seconds'] > base.INNER_SECONDS+120:
                                raise TimeoutError('Per-model allowance exhausted')
            except (Exception, KeyboardInterrupt) as exc:
                base._stop(process)
                code, entry['error'] = 1, f'{type(exc).__name__}: {exc}'
            account(wait=True)
            current = base.read(case/'REPORT.json') if (case/'REPORT.json').exists() else {}
            entry.update(stage=current.get('stage','error'), complete=current.get('complete',False) and code==0,
                seconds=prior+time.monotonic()-begin, error=current.get('error') or entry.get('error'),
                report_path=str(case/'REPORT.json'))
            entry.pop('worker_pid', None)
            _write(path, batch)
            register(name, 'agent-cnn', entry, case)
            if not entry['complete']:
                batch['stage'] = 'error'
                break
        batch.update(complete=all(v['complete'] for v in batch['models'].values()), active_model=None)
        if batch['complete']:
            batch['stage'] = 'complete'
        _write(path, batch)
        command = [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
                   '-m',__spec__.name,'--inside','--part','summary','--out',str(root)]
        with (root/'summary.log').open('a') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           env=base.environment(), preexec_fn=base.limits, check=True)
        register('nine-models', 'agent-cnn-summary', base.read(root/'SUMMARY.json'), root)
        print('REPORT='+str(path), flush=True)
        return 0 if batch['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('unit','main','packaging','tensor','intake','cache','baseline','prepare','model','summary','run'), default='run')
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--baseline-case', type=Path)
    parser.add_argument('--source-case', type=Path)
    parser.add_argument('--rule-source', type=Path)
    parser.add_argument('--model', choices=ORDER)
    parser.add_argument('--budget-path', type=Path)
    parser.add_argument('--call-limit', type=int)
    parser.add_argument('--prior-seconds', type=float, default=0.)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.inside:
        return inside(args)
    if args.part in ('model','summary'):
        parser.error('model and summary are internal stages')
    if args.part == 'run':
        if not args.review and not args.resume:
            parser.error('--review or --resume is required')
        return outer_run(args)
    return outer_part(args)


if __name__ == '__main__':
    raise SystemExit(main())
