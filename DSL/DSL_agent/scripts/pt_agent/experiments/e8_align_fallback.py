"""E8 serial alignment study (API 0) and two bounded Qwen sessions under tune rule v8-6.25."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from ..paths import RESULTS_DIR
from . import e6_llm_select_tuning as e6
from . import e6r2_resume_after_outage as e6r2
from . import e7_confirm_tail_check as e7

read, write, ledger_rows = e6.read, e6.write, e6.ledger_rows
API_START, API_CEILING = 389, 427
NEW_LABELS = ('r1', 'r2')
RULE, OLD_RULE = 'v8-6.25', 'v8-6.24'
REPLAYS = (('E6r2-b-qwen-p1', RESULTS_DIR/'E6r2-resume-20261107', 'p1', 8),
           ('E7b-qwen-q1', RESULTS_DIR/'E7-confirm-tail-20261108', 'q1', 4),
           ('E7b-qwen-q2', RESULTS_DIR/'E7-confirm-tail-20261108', 'q2', 3))


def register(model, mode, out, metrics):
    from ..registry import append, git_rev, new_run_id
    append(dict(run_id=new_run_id('E8', model, mode), round='E8', model=model, mode=mode,
        git_rev=git_rev(), metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))


def with_rule(command, rule=RULE):
    command = list(command)
    command[command.index('--tune-rule')+1] = rule
    return command


# ---------------------------------------------------------------- inventory

def sessions():
    found = [case for case in e7.inventory()['sessions'] if case['family'] in ('E5d', 'E6a', 'E6-s1', 'E6r2-a', 'E6r2-b')]
    model = e7.inventory()['models']['qwen']
    root = RESULTS_DIR/'E7-confirm-tail-20261108'
    for label in ('q1', 'q2'):
        found.append(dict(id='E7b-qwen-'+label, family='E7b', model='qwen', label=label, model_dir=model,
                          session=str(root/'e7b/qwen'/label/'session'),
                          acceptance=str(root/'e7b/qwen'/label/'acceptance'), root=str(root)))
    return found


def _domain_rows(entry):
    return [row for row in entry['history'] if row.get('phase') == 'domain']


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    (root/'inputs').mkdir(exist_ok=True)
    cases, inputs, seen = sessions(), [], {}
    for case in cases:
        state = read(Path(case['session'])/'SESSION.json')
        for key, value in state['tune_cache'].items():
            result, iteration, source = value['$tuple']
            rows = _domain_rows(result)
            if not rows:
                continue
            identity = (case['model'], source)
            if identity in seen:
                seen[identity]['also'].append(dict(session=case['id'], iteration=iteration))
                continue
            row = rows[0]
            item_id = f'{case["id"]}-it{iteration}'
            while any(i['id'] == item_id for i in inputs):
                item_id += 'b'
            path = root/'inputs'/(item_id+'.ptdsl')
            path.write_text(source)
            item = dict(id=item_id, session_id=case['id'], family=case['family'], model=case['model'],
                        model_dir=case['model_dir'], session=case['session'], iteration=iteration, program=str(path),
                        saved_row=dict(accepted=row['accepted'],
                                       refits=[dict(line=f['line'], status=f['status'], setting=f.get('setting'))
                                               for f in row.get('refits', [])]),
                        rejected=not row['accepted'], also=[])
            seen[identity] = item
            inputs.append(item)
    manifest = dict(sessions=cases, inputs=inputs, replays=[dict(id=r[0], root=str(r[1]), label=r[2], responses=r[3])
                                                           for r in REPLAYS])
    write(root/'manifest.json', manifest)
    review = root/'network-review'
    review.mkdir(exist_ok=True)
    write(review/'manifest.json', dict(destination='https://api.deepseek.com/chat/completions', api_before=API_START,
        api_ceiling=API_CEILING, max_calls=38, order=['qwen r1', 'qwen r2'], per_session=dict(main=10, tuning=9),
        retry=dict(wait_s=300, max_retries=2, circuit=3, cooldown_s=600), confirmation_results_returned_to_agent=False,
        confirm_rule='c2', tune_rule=RULE, prompt='r8-v11', tuning_prompt='t1', model='qwen',
        contents='Original model structure, calibration/heldout derived statistics, generated DSL, internal evaluation '
                 'and legal tuning menu (unchanged from E7b; only the tuner rule differs)',
        authorization='User approved E8b (up to 38 calls, ledger <= 427/450) in chat on 2026-11-08; retries count'))
    return manifest


# ---------------------------------------------------------------- inner workers

def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _align(source, ctx, rule, points=None):
    from ..dsl.parser import parse
    from ..dsl.printer import print_program
    from ..planners.transformer_tuner import tune, tune_rule_scope
    started = time.monotonic()
    with tune_rule_scope(rule):
        result = tune(parse(source), ctx, budget_s=0, max_sweeps=8, engine_profile='tensor', align_points=points)
    rows = [row for row in result.history if row.get('phase') == 'domain']
    original = print_program(parse(source))
    return dict(rule=rule, points=sorted(points) if points is not None else None, row=rows[0] if rows else None,
                changed=result.source != original, source=result.source, cost_before=result.cost_before,
                cost_after=result.cost_after, feasible=result.feasible, full_evals=result.full_evals,
                seconds=time.monotonic()-started)


def _saved_equal(saved, row):
    if row is None:
        return False
    refits = [dict(line=f['line'], status=f['status'], setting=f.get('setting')) for f in row.get('refits', [])]
    return saved['accepted'] == row['accepted'] and json.loads(json.dumps(saved['refits'])) == json.loads(json.dumps(refits))


def run_worker(args):
    _limits()
    from ..planners.confirm import Candidate, confirm_select
    from ..planners.transformer_session import json_safe
    from ..transformer_validation import points
    from ..dsl.parser import parse
    manifest = read(args.root/'manifest.json')
    item = next(i for i in manifest['inputs'] if i['id'] == args.input_id)
    report = dict(complete=False, part=args.worker, input_id=item['id'], api_calls=0, report_images=0)
    try:
        ctx, state = e7.context(dict(model_dir=item['model_dir'], session=item['session']))
        ctx.tune_search = 'sweep'
        source = Path(item['program']).read_text()
        if args.worker == 'align':
            old = _align(source, ctx, OLD_RULE)
            new = _align(source, ctx, RULE)
            report.update(old=old, new=new, saved_equal=_saved_equal(item['saved_row'], old['row']),
                          rules_equal=old['source'] == new['source'] and
                          (old['row'] or {}).get('accepted') == (new['row'] or {}).get('accepted'))
            (args.out/'v8-6.24.ptdsl').write_text(old['source'])
            (args.out/'v8-6.25.ptdsl').write_text(new['source'])
            if item['rejected'] and new['changed']:
                candidate = Candidate(new['source'], 'aligned', item['iteration'], new['cost_after']['latency_pred_us'],
                                      new['cost_after']['boots_pred'])
                check = confirm_select([candidate], ctx, engine_profile='tensor', confirm_rule='c2')
                check['selected'] = asdict(check['selected']) if check['selected'] is not None else None
                report['confirm_check'] = check
        else:
            program = parse(source)
            names = [i for i, p in enumerate(points(program)) if p.op.kind != 'softmax']
            report['points'] = [_align(source, ctx, OLD_RULE, {index}) for index in names]
        report['complete'] = True
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc), failure_class='resource' if isinstance(exc, MemoryError) else 'code')
        import traceback
        traceback.print_exc()
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write(args.out/'REPORT.json', json_safe(report))
    return 0 if report['complete'] else 1


# ---------------------------------------------------------------- outer steps

def serial_child(root, item, part, out, seconds=3600):
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds), '--',
               '-m', 'pt_agent.experiments.e8_align_fallback', '--worker', part,
               '--root', str(root), '--input-id', item['id'], '--out', str(out)]
    started = time.monotonic()
    print('E8_STEP='+item['id']+' '+part+' '+str(out), flush=True)
    with (out/'run.log').open('w') as stream:
        status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    report = read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False, error='Missing worker report')
    report.update(exit=status.returncode, seconds=time.monotonic()-started)
    write(out/'REPORT.json', report)
    register(item['model'], part, out, {k: report.get(k) for k in ('complete', 'error', 'seconds', 'input_id')})
    if status.returncode or not report.get('complete'):
        raise RuntimeError(item['id']+' '+part+': '+str(report.get('error')))
    return report


def replay(root, spec, model_dir, ledger):
    from ..planners.transformer_session import load_state
    label = spec['label']
    session = root/'e8a/replay'/spec['id']/'session'
    session.mkdir(parents=True, exist_ok=False)
    saved = sorted((Path(spec['root'])/'calls/qwen/r8-v11'/label).glob('call-*.response.json'))
    if len(saved) != spec['responses']:
        raise RuntimeError(spec['id']+': saved response count differs')
    started, steps = time.monotonic(), []
    while True:
        state = load_state(session/'SESSION.json') if (session/'SESSION.json').exists() else None
        if state and state['phase'] == 'done':
            break
        phase = state['phase'] if state else 'evaluate'
        iteration = state['iteration'] if state else 1
        out = session/'steps'/f'{len(steps)+1:03d}-{phase}'
        command = with_rule(e6.base_command('session-step', out, 'qwen', model_dir)) + [
            '--step', phase, '--root', str(session), '--call-dir', str(root/'calls-replay'), '--budget-path', str(ledger),
            '--tune-search', 'sweep', '--session-label', 'replay-'+label, '--session-iters', str(len(saved)),
            '--confirm-rule', 'c2', '--provider-state', str(root/'provider-state.json')]
        if phase == 'evaluate':
            command.extend(['--response-file', str(saved[iteration-1])])
        result = e6.child(command, out, model='qwen', mode='e8a-replay-'+label+'-'+phase, ledger=ledger, label='replay-'+label)
        steps.append(dict(phase=phase, iteration=iteration, report=str(out/'REPORT.json'), seconds=result.get('seconds')))
        write(session/'STEPS.json', dict(steps=steps))
        if result.get('exit') or not result.get('complete'):
            report = dict(complete=False, error=result.get('error'), failure_class=result.get('failure_class', 'code'),
                          steps=steps, seconds=time.monotonic()-started, main_api_calls=0)
            write(session/'REPORT.json', report)
            return report
    agent = read(session/'AGENT.json')
    result = dict(complete=agent['status'] != 'error', frozen=agent['status'] == 'feasible', agent=agent, main_api_calls=0,
                  saved_replies=len(saved), steps=steps, seconds=time.monotonic()-started,
                  label='Development replay: saved main responses had seen llm-select + v8-6.24 feedback')
    write(session/'REPORT.json', result)
    register('qwen', 'e8a-replay', session, {k: result[k] for k in ('complete', 'frozen', 'seconds')})
    return result


def acceptance(root, session, model, model_dir, mode):
    parent = session.parent/'acceptance'
    for part in ('evaluate', 'cost', 'pars'):
        out = parent/('report' if part == 'evaluate' else part)
        if part == 'pars':
            command = e6r2.native_command(root, model, model_dir, session/'program.ptdsl', out)
        else:
            command = with_rule(e6.base_command(part, out, model, model_dir)) + [
                '--root', str(parent), '--program', str(session/'program.ptdsl'), '--frozen', str(session/'FROZEN.json')]
        result = e6.child(command, out, model=model, mode=mode+'-'+part)
        e6r2.check_child(result, mode+' '+part)
    return dict(report100=e7.original_report(dict(acceptance=str(parent))), boots=read(parent/'pars/REPORT.json').get('boots'))


def run_a(root, manifest, stage):
    """stage 'first': rejected inputs (§3.1/§3.2) and replays (§3.4); stage 'rest': attribution and regression."""
    data = read(root/'E8a.json') if (root/'E8a.json').exists() else dict(align={}, attribution={}, replays={})
    def checkpoint():
        write(root/'E8a.json', data)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    before = len(ledger_rows(ledger))
    rejected = [i for i in manifest['inputs'] if i['rejected']]
    others = [i for i in manifest['inputs'] if not i['rejected']]
    if stage == 'first':
        for item in rejected:
            if item['id'] not in data['align']:
                data['align'][item['id']] = serial_child(root, item, 'align', root/'e8a/align'/item['id'])
                checkpoint()
        model_dir = Path(e7.inventory()['models']['qwen'])
        for spec in manifest['replays']:
            if spec['id'] in data['replays']:
                continue
            entry = replay(root, spec, model_dir, ledger)
            if entry.get('frozen'):
                session = root/'e8a/replay'/spec['id']/'session'
                entry['acceptance'] = acceptance(root, session, 'qwen', model_dir, 'e8a-replay-'+spec['label'])
            data['replays'][spec['id']] = entry
            checkpoint()
    else:
        for item in rejected:
            if item['id'] not in data['attribution']:
                data['attribution'][item['id']] = serial_child(root, item, 'attribute', root/'e8a/attribute'/item['id'], seconds=7200)
                checkpoint()
        for item in others:
            if item['id'] not in data['align']:
                data['align'][item['id']] = serial_child(root, item, 'align', root/'e8a/align'/item['id'])
                checkpoint()
        register('all', 'e8a', root/'e8a', dict(complete=True, api_calls=0, inputs=len(manifest['inputs'])))
    if len(ledger_rows(ledger)) != before:
        raise RuntimeError('E8a changed the API ledger')
    return data


def gate(data):
    passed = []
    for name, entry in data['replays'].items():
        report = (entry.get('acceptance') or {}).get('report100') or {}
        if entry.get('frozen') and report and all(engine.get('feasible') for engine in report.values()):
            passed.append(name)
    return passed


def session_command(root, model, label):
    return [sys.executable, '-B', '-m', 'pt_agent.experiments.a1_transformer_agent', '--part', 'session',
        '--prompt-version', 'r8-v11', '--tune-rule', RULE, '--placement', 'program-plan',
        '--tune-search', 'llm-select', '--confirm-rule', 'c2', '--session-label', label, '--tune-call-limit', '9',
        '--model', 'qwen', '--model-dir', str(model), '--out', str(root/'e8b/qwen'/label/'session'), '--call-dir', str(root/'calls'),
        '--budget-path', str(RESULTS_DIR/'llm_calls.jsonl'), '--provider-state', str(root/'provider-state.json'),
        '--ledger-cap', str(API_CEILING), '--experiment-round', 'E8']


def session_outcome(returncode, report):
    """Status of one E8b session. A ledger-cap refusal is recorded as such, without the provider cooldown."""
    if not returncode and report.get('complete'):
        return 'frozen' if report.get('frozen') else 'not_frozen'
    if report.get('error') == 'budget_exhausted':
        return 'budget_exhausted'
    if report.get('failure_class') == 'provider':
        return 'provider_failed'
    return None


def run_session(root, policy, model, label):
    e6r2.boundary(policy)
    session = root/'e8b/qwen'/label/'session'
    session.parent.mkdir(parents=True, exist_ok=True)
    print('E8_SESSION=qwen '+label, flush=True)
    with (session.parent/'session.log').open('w') as stream:
        status = subprocess.run(session_command(root, model, label), stdout=stream, stderr=subprocess.STDOUT)
    report = read(session/'REPORT.json') if (session/'REPORT.json').exists() else dict(complete=False, error='Missing session report')
    outcome = session_outcome(status.returncode, report)
    if outcome is None:
        raise RuntimeError('Qwen '+label+': '+str(report.get('error')))
    if outcome == 'provider_failed':
        e6r2.boundary(policy, failed_session=session)
    result = dict(status=outcome, label=label)
    if report.get('frozen'):
        result['acceptance'] = acceptance(root, session, 'qwen', model, 'e8b-'+label)
    if len(ledger_rows(RESULTS_DIR/'llm_calls.jsonl')) > API_CEILING:
        raise RuntimeError('E8b exceeded the approved ledger ceiling')
    return result


def run_b(root, manifest):
    from ..planners.provider_retry import ProviderState
    policy = ProviderState(root/'provider-state.json')
    model = Path(e7.inventory()['models']['qwen'])
    results = {}
    for label in NEW_LABELS:
        results[label] = run_session(root, policy, model, label)
    write(root/'E8b.json', results)
    return results


RERUNS = {'r2': 'r2b'}


def rerun_b(root, label):
    """Rerun one E8b session that stopped on the stale ledger cap, under a fresh label; the first attempt stays in place.

    The user approved this rerun on 2026-11-08 within the 38 E8b calls (main 10, tuning 9 for the new session).
    """
    from ..planners.provider_retry import ProviderState
    results = read(root/'E8b.json')
    first = results[label]
    session = root/'e8b/qwen'/label/'session'
    if first.get('status') not in ('provider_failed', 'budget_exhausted') or read(session/'REPORT.json').get('error') != 'budget_exhausted':
        raise RuntimeError('Only a session stopped by the ledger cap is rerun')
    new_label = RERUNS[label]
    if (root/'e8b/qwen'/new_label).exists():
        raise RuntimeError('Rerun directory exists: '+new_label)
    keep = root/('before-rerun-'+label)
    keep.mkdir(exist_ok=False)
    for name in ('E8.md', 'E8b.json', 'FINAL_CHECKS.json'):
        (keep/name).write_bytes((root/name).read_bytes())
    write(root/('RERUN_B-'+label+'.json'), dict(api_before=len(ledger_rows(RESULTS_DIR/'llm_calls.jsonl')), api_ceiling=API_CEILING,
                                               label=new_label, per_session=dict(main=10, tuning=9)))
    result = run_session(root, ProviderState(root/'provider-state.json'), Path(e7.inventory()['models']['qwen']), new_label)
    result['first_attempt'] = dict(first, label=label, status='budget_exhausted', recorded_status=first.get('status'),
                                   session=str(session), cause='session-step ledger cap fixed at 403 (E7 ceiling)')
    results[label] = result
    write(root/'E8b.json', results)
    return result


def session_labels(b):
    """Labels of every E8b session directory, the first attempts included."""
    labels = []
    for label in NEW_LABELS:
        labels.append(label)
        entry = b.get(label) or {}
        if entry.get('first_attempt'):
            labels.append(entry['label'])
    return labels


def namespace_calls(folder):
    """Attempts sent to the provider, and attempts the ledger cap refused before sending."""
    sent = refused = 0
    for record in (folder/'attempts').glob('*.record.json'):
        error = folder/'attempts'/record.name.replace('.record.json', '.error.json')
        if error.exists() and read(error).get('error') == 'budget_exhausted':
            refused += 1
        else:
            sent += 1
    return sent, refused


# ---------------------------------------------------------------- summary

def _fmt(value, digits=4):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.{digits}g}'
    return str(value)


def _refit_brief(row):
    return '; '.join(f'{f["line"]}:{f.get("stage", 0)}/{f.get("method", "")}/{f["status"]}' for f in (row or {}).get('refits', []))


def summarize(root, manifest, error=None):
    a = read(root/'E8a.json') if (root/'E8a.json').exists() else dict(align={}, attribution={}, replays={})
    b = read(root/'E8b.json') if (root/'E8b.json').exists() else {}
    rows = ledger_rows(RESULTS_DIR/'llm_calls.jsonl')
    namespaces = []
    for label in session_labels(b):
        for version in ('r8-v11', 't1'):
            sent, refused = namespace_calls(root/'calls/qwen'/version/label)
            namespaces.append(dict(namespace=['qwen', version, label], calls=sent, refused=refused,
                                   limit=10 if version == 'r8-v11' else 9))
    result = dict(round='E8', error=error, e8a=a, e8b=b, api_before=API_START, api_after=len(rows),
                  api_delta=len(rows)-API_START, namespaces=namespaces,
                  accounting_equal=len(rows)-API_START == sum(n['calls'] for n in namespaces),
                  within_limits=len(rows) <= API_CEILING and all(n['calls'] <= n['limit'] for n in namespaces))
    anomalies = [error] if error else []
    for label, entry in b.items():
        first = entry.get('first_attempt')
        if first:
            anomalies.append(f'{first["label"]} {first["status"]}（{first["cause"]}），以 {entry["label"]} 重跑')
        if entry.get('status') not in ('frozen', 'not_frozen'):
            anomalies.append(f'{entry.get("label", label)} {entry.get("status")}')
    refused = sum(n['refused'] for n in namespaces)
    if refused:
        anomalies.append(f'账本上限拒绝、未发出的请求 {refused} 个')
    lines = ['# E8 定义域对齐：refit 扩展与逐点回退', '',
             f'API {result["api_delta"]}/38，累计 {len(rows)}/450；异常：{"；".join(anomalies) or "无"}。', '',
             '## 5.1 被拒原因（v8-6.24）', '',
             '| 输入 | 原因 | 对齐后 logit / 上限 | 对齐后校准 u / 解析 u | 单独放宽时被拒的行（原因） | 重跑与原记录一致 |',
             '|---|---|---|---|---|---|']
    inputs = {i['id']: i for i in manifest['inputs']}
    for name, item in inputs.items():
        if not item['rejected'] or name not in a['align']:
            continue
        old = a['align'][name]['old']['row'] or {}
        m = old.get('metrics') or {}
        single = [f'{p["row"]["refits"][0]["line"] if p["row"] and p["row"].get("refits") else "?"}({",".join(p["row"]["reasons"])})'
                  for p in a['attribution'].get(name, {}).get('points', []) if p['row'] and not p['row']['accepted']]
        lines.append(f'| {name} | {",".join(old.get("reasons", []))} | {_fmt(m.get("logit_rel_err"))} / {_fmt(m.get("move_logit_limit"))} | '
                     f'{_fmt(m.get("max_u_calibrated"))} / {_fmt(m.get("max_u_analytic"))} | {"; ".join(single) or "—"} | '
                     f'{a["align"][name]["saved_equal"]} |')
    lines += ['', '## 5.2 v8-6.25 对齐', '',
              '| 输入 | refit（行:阶段/方法/状态） | 整体接受 | 回退接受的行 | 输入 / 对齐后自举 | 对齐程序：确认集校准 u、logit、尾部 |',
              '|---|---|---|---|---|---|']
    for name, item in inputs.items():
        if not item['rejected'] or name not in a['align']:
            continue
        new = a['align'][name]['new']
        row = new['row'] or {}
        check = (a['align'][name].get('confirm_check') or {}).get('evals') or [{}]
        c = check[0]
        tail = c.get('tail') or {}
        lines.append(f'| {name} | {_refit_brief(row)} | {row.get("accepted")} | {row.get("accepted_lines", "—")} | '
                     f'{new["cost_before"]["boots_pred"]} / {new["cost_after"]["boots_pred"]} | '
                     f'{_fmt(c.get("max_u_calibrated"))}、{_fmt(c.get("logit_rel_err"))}、'
                     f'{_fmt(tail.get("logit_rel_err"))}（{tail.get("passed", "—")}） |')
    same = [n for n, v in a['align'].items() if v.get('rules_equal')]
    differ = [n for n, v in a['align'].items() if not v.get('rules_equal')]
    saved_bad = [n for n, v in a['align'].items() if not v.get('saved_equal')]
    lines += ['', '## 5.3 回归', '', f'去重后输入 {len(manifest["inputs"])} 个，已运行 {len(a["align"])} 个；两种规则结果相同 {len(same)} 个，不同 {len(differ)} 个；'
              f'v8-6.24 重跑与原记录不一致 {len(saved_bad)} 个。', '',
              '| 不同的输入 | v8-6.24 接受 | v8-6.25 接受 / 回退 | 来源 |', '|---|---|---|---|']
    for name in differ:
        old, new = a['align'][name]['old']['row'] or {}, a['align'][name]['new']['row'] or {}
        stages = sorted({f.get('stage', 0) for f in new.get('refits', [])})
        source = 'fallback' if new.get('fallback_accepted') else 'stage '+','.join(map(str, stages))
        lines.append(f'| {name} | {old.get("accepted")} | {new.get("accepted")} / {new.get("fallback_accepted", "—")} | {source} |')
    if saved_bad:
        lines += ['', 'v8-6.24 重跑与原记录不一致：'+', '.join(saved_bad)]
    lines += ['', '## 5.4 端到端', '',
              '| 会话 | tune_rule | confirm_rule | 主/精调调用 | 冻结 | 预测/导出/编译自举 | report100 tensor/mock logit | 循环秒 |',
              '|---|---|---|---|---|---|---|---|']
    historical = read(RESULTS_DIR/'E7-confirm-tail-20261108/E7.json')['end_to_end']
    for name, row in historical.items():
        if name.startswith(('E5d', 'E6-s1', 'E6r2-b', 'E7b')):
            boots = row.get('boots') or {}
            report = row.get('report100') or {}
            lines.append(f'| {name} | v8-6.24 | {row.get("confirm_rule")} | {row.get("main_calls")}/{row.get("tuning_calls")} | '
                         f'{row.get("frozen_origin")}/{row.get("frozen_iteration")} | {boots.get("predicted")}/{boots.get("exported")}/{boots.get("compiled")} | '
                         f'{_fmt((report.get("tensor") or {}).get("logit_rel_err"))}/{_fmt((report.get("mock") or {}).get("logit_rel_err"))} | {_fmt(row.get("loop_seconds"), 6)} |')
    e2e = {}
    for spec in manifest['replays']:
        session = root/'e8a/replay'/spec['id']/'session'
        if (session/'AGENT.json').exists():
            e2e['E8a-replay-'+spec['label']] = session
    for label in session_labels(b):
        session = root/'e8b/qwen'/label/'session'
        if (session/'AGENT.json').exists():
            e2e['E8b-qwen-'+label] = session
    result['end_to_end'] = {}
    for name, session in e2e.items():
        row = e6r2.session_summary(session)
        row.update(tune_rule=RULE, confirm_rule='c2', search='sweep replay' if 'replay' in name else 'llm-select')
        result['end_to_end'][name] = row
        boots = row.get('boots') or {}
        report = row.get('report100') or {}
        lines.append(f'| {name} | {RULE} | c2 | {row.get("main_calls")}/{row.get("tuning_calls")} | '
                     f'{row.get("frozen_origin")}/{row.get("frozen_iteration")} | '
                     f'{boots.get("predicted")}/{boots.get("exported")}/{boots.get("compiled")} | '
                     f'{_fmt((report.get("tensor") or {}).get("logit_rel_err"))}/{_fmt((report.get("mock") or {}).get("logit_rel_err"))} | '
                     f'{_fmt(row.get("loop_seconds"), 6)} |')
    lines += ['', '## 5.5 用时', '', '| 输入 | v8-6.24 对齐秒 | v8-6.25 对齐秒 | refit 秒（各点合计） | 回退测量次数 / 秒 |', '|---|---:|---:|---:|---:|']
    for name, value in a['align'].items():
        new = value['new']['row'] or {}
        refit_s = sum(f.get('seconds', 0.) for f in new.get('refits', []))
        fb = new.get('fallback', [])
        lines.append(f'| {name} | {_fmt(value["old"]["seconds"], 5)} | {_fmt(value["new"]["seconds"], 5)} | {_fmt(refit_s, 5)} | '
                     f'{len(fb)} / {_fmt(sum(e["seconds"] for e in fb), 5)} |')
    lines += ['', '以上只做描述。对齐程序的确认检查不参与选择；report 结果在选择结束后采集。']
    (root/'E8.md').write_text('\n'.join(lines)+'\n')
    write(root/'E8.json', result)
    return result


class _Done(Exception):
    """A rerun finished; the summary follows."""


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    root = args.root
    if not read(root/'OFFLINE_TESTS.json').get('complete'):
        raise RuntimeError('Offline tests must pass first')
    manifest = read(root/'manifest.json')
    error = None
    if len(ledger_rows(RESULTS_DIR/'llm_calls.jsonl')) != API_START and not (root/'RUN_B.json').exists():
        raise RuntimeError('Ledger start changed')
    try:
        if args.rerun_b:
            rerun_b(root, args.rerun_b)
            raise _Done
        # Order (deviation from spec §3 order, recorded in STATUS): §3.1/§3.2 on the rejected inputs and the §3.4
        # replays first, then E8b when its start condition holds, then the §3.1 attribution and §3.3 regression.
        data = run_a(root, manifest, 'first')
        passed = gate(data)
        write(root/'E8a-FIRST.json', dict(complete=True, api_calls=0, gate=passed))
        if not args.a_only:
            if not passed:
                write(root/'E8b.json', dict(status='not_started', reason='No E8a replay froze a program that passed report100'))
            elif not (root/'E8b.json').exists():
                write(root/'RUN_B.json', dict(api_before=len(ledger_rows(RESULTS_DIR/'llm_calls.jsonl')), api_ceiling=API_CEILING,
                                              per_session=dict(main=10, tuning=9)))
                run_b(root, manifest)
        run_a(root, manifest, 'rest')
        write(root/'E8a-COMPLETE.json', dict(complete=True, api_calls=0, gate=passed))
    except _Done:
        pass
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        write(root/'STOP.json', dict(error=error))
        import traceback
        traceback.print_exc()
    result = summarize(root, manifest, error)
    write(root/'FINAL_CHECKS.json', {k: result[k] for k in ('api_before', 'api_after', 'api_delta', 'namespaces',
                                                            'accounting_equal', 'within_limits')})
    register('all', 'summary', root, {k: result[k] for k in ('error', 'api_before', 'api_after', 'api_delta',
                                                            'accounting_equal', 'within_limits')})
    print('E8_DONE='+str(root/'E8.json')+' ERROR='+str(error), flush=True)
    return int(bool(error or not result['accounting_equal'] or not result['within_limits']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--a-only', action='store_true')
    stages.add_argument('--rerun-b', choices=tuple(RERUNS))
    parser.add_argument('--worker', choices=('align', 'attribute'))
    parser.add_argument('--input-id')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)
    if args.prepare:
        prepare(args.root)
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
