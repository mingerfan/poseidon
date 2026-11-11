"""E9a (API 0): tuner-only sessions that start from the rule programs R0/R1 in the current Transformer pipeline.

For BERT/Qwen x R0/R1 the rule program is handed to the session as the saved reply of its single iteration. The
session evaluates it under program-plan placement, runs the tuner (sweep search, rule v8-6.25), confirms under c2
and freezes a program when a candidate passes. A frozen program gets the usual acceptance (report100, cost, pars
compile). The rule program itself is also compiled under program-plan. No LLM client is created and the ledger
must stay unchanged.

Outer:  python3 -B -m pt_agent.experiments.e9a_tuner_only --root <dir>
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from ..paths import RESULTS_DIR
from . import e6_llm_select_tuning as e6
from . import e6r2_resume_after_outage as e6r2
from . import e7_confirm_tail_check as e7

read, write, ledger_rows = e6.read, e6.write, e6.ledger_rows
ROUND, RULE, PLACEMENT = 'E9', 'v8-6.25', 'program-plan'
RULE_ROOTS = dict(bert=RESULTS_DIR/'A1-bert-prepare-v2-20261028T135014-01b9',
                  qwen=RESULTS_DIR/'A1-qwen-prepare-v2-20261028T135355-6529')
CASES = tuple((model, base) for model in ('bert', 'qwen') for base in ('R0', 'R1'))
KEPT = ('complete', 'error', 'exit', 'seconds', 'step', 'next_step', 'iteration', 'frozen', 'boots', 'three_way_equal')


def register(model, mode, out, metrics):
    from ..registry import append, git_rev, new_run_id
    append(dict(run_id=new_run_id(ROUND, model, mode), round=ROUND, model=model, mode=mode, git_rev=git_rev(),
                metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))


def saved_reply(source):
    """The reply format the session reads from --response-file."""
    return dict(content=json.dumps(dict(program=source)), finish_reason='stop', usage={}, id=None, seconds=0.0)


def step_command(part, out, model, model_dir):
    command = e6.base_command(part, out, model, model_dir)
    command[command.index('--tune-rule')+1] = RULE
    return command


def session_arguments(session, phase, label, call_dir, ledger, reply):
    extra = ['--step', phase, '--root', str(session), '--call-dir', str(call_dir), '--budget-path', str(ledger),
             '--tune-search', 'sweep', '--session-label', label, '--session-iters', '1', '--confirm-rule', 'c2']
    if phase == 'evaluate':
        extra += ['--response-file', str(reply)]
    return extra


def child(command, out, *, model, mode):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print('E9_STEP='+model+' '+mode+' '+str(out), flush=True)
    with (out/'run.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    result = read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False, error='No child REPORT')
    result.update(exit=code, seconds=time.monotonic()-started)
    write(out/'REPORT.json', result)
    register(model, mode, out, {key: result.get(key) for key in KEPT if key in result})
    return result


def checked(result, context):
    if result.get('exit') or not result.get('complete'):
        raise RuntimeError(context+': '+str(result.get('error')))
    return result


# ---------------------------------------------------------------- inner worker

def header_worker(args):
    """Rewrite the program header exactly as the session does and save the result."""
    from ..planners.agent_tools import apply_placement
    source = args.program.read_text()
    placed = apply_placement(source, SimpleNamespace(placement=PLACEMENT))
    first = lambda text: next(line for line in text.splitlines() if line.lstrip().startswith('@fhe.program('))
    (args.out/'program.ptdsl').write_text(placed)
    write(args.out/'REPORT.json', dict(complete=placed != source, source=str(args.program), old=first(source),
                                       new=first(placed), api_calls=0))
    return 0 if placed != source else 1


# ---------------------------------------------------------------- outer steps

def rule_compile(root, folder, model, model_dir, program):
    """Compiled bootstrap count of the unchanged rule program under program-plan."""
    name = folder.name
    out = folder/'rule'
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '600', '--',
               '-m', 'pt_agent.experiments.e9a_tuner_only', '--worker', 'header', '--program', str(program), '--out', str(out)]
    checked(child(command, out, model=model, mode='e9a-'+name+'-rule-header'), name+' header')
    pars = folder/'rule-pars'
    result = child(e6r2.native_command(root, model, model_dir, out/'program.ptdsl', pars), pars, model=model,
                   mode='e9a-'+name+'-rule-pars')
    return dict(complete=bool(result.get('complete')) and not result.get('exit'), boots=result.get('boots'),
                error=result.get('error'), seconds=result.get('seconds'))


def tuner_session(root, folder, model, model_dir, program, ledger):
    from ..planners.transformer_session import load_state
    name = folder.name
    session = folder/'session'
    session.mkdir(parents=True, exist_ok=False)
    reply = folder/'reply.json'
    write(reply, saved_reply(program.read_text()))
    started, steps = time.monotonic(), []
    while True:
        state = load_state(session/'SESSION.json') if (session/'SESSION.json').exists() else None
        if state and state['phase'] == 'done':
            break
        phase = state['phase'] if state else 'evaluate'
        out = session/'steps'/f'{len(steps)+1:03d}-{phase}'
        command = step_command('session-step', out, model, model_dir) + session_arguments(
            session, phase, 'tuner-'+name, root/'calls-a', ledger, reply)
        result = child(command, out, model=model, mode='e9a-'+name+'-'+phase)
        steps.append(dict(phase=phase, report=str(out/'REPORT.json'), seconds=result.get('seconds')))
        write(session/'STEPS.json', dict(steps=steps))
        if result.get('exit') or not result.get('complete'):
            report = dict(complete=False, error=result.get('error'), steps=steps, seconds=time.monotonic()-started,
                          main_api_calls=0)
            write(session/'REPORT.json', report)
            return report
        if len(steps) > 6:
            raise RuntimeError(name+': a one-iteration session took more than six steps')
    agent = read(session/'AGENT.json')
    result = dict(complete=agent['status'] != 'error', frozen=agent['status'] == 'feasible', status=agent['status'],
                  history=agent.get('history'), best_source=agent.get('best_source'), main_api_calls=0, steps=steps,
                  seconds=time.monotonic()-started, tune_rule=agent.get('tune_rule'), confirm_rule=agent.get('confirm_rule'),
                  confirm_evals=[{key: row.get(key) for key in ('origin', 'iteration', 'boots_pred', 'passed', 'agree', 'total',
                                                                 'logit_rel_err', 'max_u', 'max_u_calibrated')}
                                 for row in (agent.get('confirm') or {}).get('evals', [])])
    write(session/'REPORT.json', result)
    register(model, 'e9a-'+name+'-session', session, {key: result[key] for key in ('complete', 'frozen', 'status', 'seconds')})
    return result


def acceptance(root, folder, model, model_dir):
    session, parent, name = folder/'session', folder/'acceptance', folder.name
    for part in ('evaluate', 'cost', 'pars'):
        out = parent/('report' if part == 'evaluate' else part)
        if part == 'pars':
            command = e6r2.native_command(root, model, model_dir, session/'program.ptdsl', out)
        else:
            command = step_command(part, out, model, model_dir) + [
                '--root', str(parent), '--program', str(session/'program.ptdsl'), '--frozen', str(session/'FROZEN.json')]
        checked(child(command, out, model=model, mode='e9a-'+name+'-'+part), name+' '+part)
    return dict(report100=e7.original_report(dict(acceptance=str(parent))), boots=read(parent/'pars/REPORT.json').get('boots'))


def run_case(root, model, base, ledger):
    folder = root/'a'/(model+'-'+base)
    folder.mkdir(parents=True, exist_ok=False)
    model_dir = Path(e7.inventory()['models'][model])
    program = RULE_ROOTS[model]/base/'program.ptdsl'
    entry = dict(model=model, base=base, program=str(program), model_dir=str(model_dir))
    try:
        entry['rule'] = rule_compile(root, folder, model, model_dir, program)
    except Exception as exc:
        entry['rule'] = dict(complete=False, error=type(exc).__name__+': '+str(exc))
    try:
        entry['session'] = tuner_session(root, folder, model, model_dir, program, ledger)
        if entry['session'].get('frozen'):
            entry['acceptance'] = acceptance(root, folder, model, model_dir)
    except Exception as exc:
        entry['error'] = type(exc).__name__+': '+str(exc)
    return entry


def _number(value, digits=4):
    return '—' if value is None else (f'{value:.{digits}g}' if isinstance(value, float) else str(value))


def summarize(root, data, error=None):
    lines = ['# E9a：从规则程序出发只运行精调器（API 0）', '',
             f'精调规则 {RULE}，放置 {PLACEMENT}，确认规则 c2，sweep 搜索，会话 1 次迭代。异常：{error or "无"}。', '',
             '| 模型 | 起点 | 规则程序自举（预测/导出/编译） | 起点评估：可行 / 预测自举 / max_u | 精调后：可行 / 预测自举 / max_u / 秒 | '
             '会话结果 | 冻结程序自举（预测/导出/编译） | report100 tensor / mock |', '|---|---|---|---|---|---|---|---|']
    for name, entry in data['cases'].items():
        rule = (entry.get('rule') or {}).get('boots') or {}
        session = entry.get('session') or {}
        first = (session.get('history') or [{}])[0]
        tuned = first.get('tuned') or {}
        accepted = entry.get('acceptance') or {}
        boots = accepted.get('boots') or {}
        report = accepted.get('report100') or {}
        cell = lambda engine: (f'{(report.get(engine) or {}).get("agree")}/{(report.get(engine) or {}).get("total")}，'
                               f'logit {_number((report.get(engine) or {}).get("logit_rel_err"))}') if report.get(engine) else '—'
        lines.append(f'| {entry["model"]} | {entry["base"]} | {rule.get("predicted")}/{rule.get("exported")}/{rule.get("compiled")} | '
                     f'{first.get("feasible")} / {first.get("boots_pred")} / {_number(first.get("max_u"))} | '
                     f'{tuned.get("feasible")} / {tuned.get("boots_pred")} / {_number(tuned.get("max_u"))} / {_number(tuned.get("seconds"))} | '
                     f'{entry.get("error") or session.get("error") or session.get("status")} | '
                     f'{boots.get("predicted")}/{boots.get("exported")}/{boots.get("compiled")} | {cell("tensor")} / {cell("mock")} |')
    (root/'E9a.md').write_text('\n'.join(lines)+'\n')
    return lines


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    path = root/'E9a.json'
    data = read(path) if path.exists() else dict(api_before=len(ledger_rows(ledger)), cases={})
    error = None
    try:
        for model, base in CASES:
            name = model+'-'+base
            if name in data['cases'] or (args.cases and name not in args.cases):
                continue
            data['cases'][name] = run_case(root, model, base, ledger)
            data['api_after'] = len(ledger_rows(ledger))
            write(path, data)
            summarize(root, data)
            if data['api_after'] != data['api_before']:
                raise RuntimeError('E9a changed the API ledger')
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        write(root/'STOP-a.json', dict(error=error))
        import traceback
        traceback.print_exc()
    data.update(api_after=len(ledger_rows(ledger)), error=error)
    data['complete'] = error is None and all(name in data['cases'] for name in (m+'-'+b for m, b in CASES)
                                             if not args.cases or name in args.cases)
    write(path, data)
    summarize(root, data, error)
    register('all', 'e9a-summary', root, dict(complete=data['complete'], error=error, api_before=data['api_before'],
                                              api_after=data['api_after'], cases=sorted(data['cases'])))
    print('E9A_DONE='+str(path)+' ERROR='+str(error), flush=True)
    return int(bool(error) or data['api_after'] != data['api_before'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--cases', nargs='+', choices=tuple(m+'-'+b for m, b in CASES))
    parser.add_argument('--worker', choices=('header',))
    parser.add_argument('--program', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.worker:
        return header_worker(args)
    if args.root is None:
        parser.error('--root is required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
