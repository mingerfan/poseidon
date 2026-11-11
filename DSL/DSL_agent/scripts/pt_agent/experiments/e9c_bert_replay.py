"""E9c (API 0): replay the four BERT agent sessions under the final rules.

The saved main replies of E5d, E6 s1 and E6r2 p1/p2 are fed back in their original order. Every session runs under
program-plan placement with tune rule v8-6.25, sweep search and confirm rule c2. A frozen program gets the usual
acceptance (report100, cost, pars compile). No LLM client is created and the ledger must stay unchanged.

The saved replies were written after feedback produced by the original rules (v8-6.24, c1), so every result here is
a development replay of an existing session.

Outer:  python3 -B -m pt_agent.experiments.e9c_bert_replay --root <dir>
"""
import argparse
import os
from pathlib import Path
import time

from ..paths import RESULTS_DIR
from . import e6r2_resume_after_outage as e6r2
from . import e7_confirm_tail_check as e7
from . import e9a_tuner_only as e9a

read, write, ledger_rows, _number = e9a.read, e9a.write, e9a.ledger_rows, e9a._number
ROUND, RULE, PLACEMENT, MODEL = e9a.ROUND, e9a.RULE, e9a.PLACEMENT, 'bert'
SEARCH, CONFIRM = 'sweep', 'c2'
NOTE = 'Development replay: the saved main replies had seen feedback from the original rules'
SPECS = (dict(id='E5d', root=RESULTS_DIR/'E5d-agent-sessions-20261106', calls='calls/bert/r8-v11',
              session='bert/session', acceptance='bert/acceptance', replies=7),
         dict(id='E6-s1', root=RESULTS_DIR/'E6-llm-select-20261107', calls='calls/bert/r8-v11/s1',
              session='e6b/bert/s1/session', acceptance='e6b/bert/s1/acceptance', replies=5),
         dict(id='E6r2-p1', root=RESULTS_DIR/'E6r2-resume-20261107', calls='calls/bert/r8-v11/p1',
              session='e6r2-b/bert/p1/session', acceptance='e6r2-b/bert/p1/acceptance', replies=5),
         dict(id='E6r2-p2', root=RESULTS_DIR/'E6r2-resume-20261107', calls='calls/bert/r8-v11/p2',
              session='e6r2-b/bert/p2/session', acceptance='e6r2-b/bert/p2/acceptance', replies=7))
ROW_KEYS = ('iteration', 'feasible', 'boots_pred', 'max_u')
TUNED_KEYS = ('feasible', 'boots_pred', 'max_u', 'seconds')
EVAL_KEYS = ('origin', 'iteration', 'boots_pred', 'passed', 'agree', 'total', 'logit_rel_err', 'max_u', 'max_u_calibrated')


def saved_replies(spec):
    """The main replies of one original session, in call order."""
    saved = sorted((Path(spec['root'])/spec['calls']).glob('call-*.response.json'))
    if len(saved) != spec['replies']:
        raise RuntimeError(f'{spec["id"]}: found {len(saved)} saved replies, expected {spec["replies"]}')
    return saved


def step_limit(replies):
    """Each reply leads to at most one evaluate and one tune step; one confirm step ends the session."""
    return 2*replies+1


def step_arguments(session, phase, label, call_dir, ledger, saved, iteration):
    extra = ['--step', phase, '--root', str(session), '--call-dir', str(call_dir), '--budget-path', str(ledger),
             '--tune-search', SEARCH, '--session-label', label, '--session-iters', str(len(saved)),
             '--confirm-rule', CONFIRM]
    if phase == 'evaluate':
        if not 1 <= iteration <= len(saved):
            raise RuntimeError(f'{label}: iteration {iteration} has no saved reply')
        extra += ['--response-file', str(saved[iteration-1])]
    return extra


def brief(history):
    rows = []
    for row in history or []:
        item = {key: row.get(key) for key in ROW_KEYS}
        if isinstance(row.get('tuned'), dict):
            item['tuned'] = {key: row['tuned'].get(key) for key in TUNED_KEYS}
        rows.append(item)
    return rows


def agent_summary(agent):
    confirm = agent.get('confirm') or {}
    selected = confirm.get('selected') if isinstance(confirm.get('selected'), dict) else {}
    return dict(status=agent.get('status'), iterations=agent.get('iterations'), best_iteration=agent.get('best_iteration'),
                best_source=agent.get('best_source'), best_boots=(agent.get('best') or {}).get('boots_pred'),
                tune_rule=agent.get('tune_rule'), tune_search=agent.get('tune_search'),
                confirm_rule=agent.get('confirm_rule'), history=brief(agent.get('history')),
                selected={key: selected.get(key) for key in ('origin', 'iteration', 'boots_pred') if key in selected},
                confirm_evals=[{key: row.get(key) for key in EVAL_KEYS} for row in confirm.get('evals', [])])


def original(spec):
    """What the original session recorded: its rules, history and the accepted bootstrap counts."""
    base = Path(spec['root'])
    result = agent_summary(read(base/spec['session']/'AGENT.json'))
    pars = base/spec['acceptance']/'pars/REPORT.json'
    result.update(boots=read(pars).get('boots') if pars.exists() else None,
                  report100=e7.original_report(dict(acceptance=str(base/spec['acceptance']))))
    return result


def replay(root, folder, spec, model_dir, ledger):
    from ..planners.transformer_session import load_state
    name = spec['id']
    session = folder/'session'
    session.mkdir(parents=True, exist_ok=False)
    saved = saved_replies(spec)
    started, steps = time.monotonic(), []
    while True:
        state = load_state(session/'SESSION.json') if (session/'SESSION.json').exists() else None
        if state and state['phase'] == 'done':
            break
        phase = state['phase'] if state else 'evaluate'
        iteration = state['iteration'] if state else 1
        if len(steps) >= step_limit(len(saved)):
            raise RuntimeError(name+': the replay took more steps than its replies allow')
        out = session/'steps'/f'{len(steps)+1:03d}-{phase}'
        command = e9a.step_command('session-step', out, MODEL, model_dir) + step_arguments(
            session, phase, 'replay-'+name, root/'calls-c', ledger, saved, iteration)
        result = e9a.child(command, out, model=MODEL, mode='e9c-'+name+'-'+phase)
        steps.append(dict(phase=phase, iteration=iteration, report=str(out/'REPORT.json'), seconds=result.get('seconds')))
        write(session/'STEPS.json', dict(steps=steps))
        if result.get('exit') or not result.get('complete'):
            report = dict(complete=False, error=result.get('error'), steps=steps, seconds=time.monotonic()-started,
                          main_api_calls=0, saved_replies=len(saved))
            write(session/'REPORT.json', report)
            return report
    agent = read(session/'AGENT.json')
    result = dict(agent_summary(agent), complete=agent['status'] != 'error', frozen=agent['status'] == 'feasible',
                  main_api_calls=0, saved_replies=len(saved), steps=steps, seconds=time.monotonic()-started, label=NOTE)
    write(session/'REPORT.json', result)
    e9a.register(MODEL, 'e9c-'+name+'-session', session, {key: result[key] for key in ('complete', 'frozen', 'status', 'seconds')})
    return result


def acceptance(root, folder, model_dir):
    session, parent, name = folder/'session', folder/'acceptance', folder.name
    for part in ('evaluate', 'cost', 'pars'):
        out = parent/('report' if part == 'evaluate' else part)
        if part == 'pars':
            command = e6r2.native_command(root, MODEL, model_dir, session/'program.ptdsl', out)
        else:
            command = e9a.step_command(part, out, MODEL, model_dir) + [
                '--root', str(parent), '--program', str(session/'program.ptdsl'), '--frozen', str(session/'FROZEN.json')]
        e9a.checked(e9a.child(command, out, model=MODEL, mode='e9c-'+name+'-'+part), name+' '+part)
    return dict(report100=e7.original_report(dict(acceptance=str(parent))), boots=read(parent/'pars/REPORT.json').get('boots'))


def run_session(root, spec, model_dir, ledger):
    folder = root/'c'/spec['id']
    folder.mkdir(parents=True, exist_ok=False)
    entry = dict(id=spec['id'], model_dir=str(model_dir), original=original(spec))
    try:
        entry['session'] = replay(root, folder, spec, model_dir, ledger)
        if entry['session'].get('frozen'):
            entry['acceptance'] = acceptance(root, folder, model_dir)
    except Exception as exc:
        entry['error'] = type(exc).__name__+': '+str(exc)
    return entry


def _rows(history):
    """One cell: per iteration 'F' or the predicted count, with the tuned count after an arrow."""
    cells = []
    for row in history or []:
        text = str(row.get('boots_pred')) if row.get('feasible') else 'F'
        tuned = row.get('tuned') or {}
        if tuned.get('feasible') and tuned.get('boots_pred') is not None:
            text += '→'+str(tuned['boots_pred'])
        cells.append(text)
    return ' '.join(cells) or '—'


def _triple(boots):
    return '—' if not boots else f'{boots.get("predicted")}/{boots.get("exported")}/{boots.get("compiled")}'


def _report(report, engine):
    row = (report or {}).get(engine)
    return '—' if not row else f'{row.get("agree")}/{row.get("total")}，logit {_number(row.get("logit_rel_err"))}'


def change(old, new):
    """Relative change of the compiled bootstrap count, in percent."""
    old, new = (old or {}).get('compiled'), (new or {}).get('compiled')
    return None if not old or new is None else 100.0*(new-old)/old


def summarize(root, data, error=None):
    lines = ['# E9c：在最终规则下重放 BERT 会话（API 0）', '',
             f'精调规则 {RULE}，放置 {PLACEMENT}，确认规则 {CONFIRM}，{SEARCH} 搜索。各会话沿用原会话保存的主回复，'
             f'这些回复写于原规则的反馈之后，结果属于开发重放。异常：{error or "无"}。', '',
             '各轮一栏：F 表示该轮程序不可行，数字是预测自举次数，箭头后是精调后的可行结果。', '',
             '| 会话 | 回复数 | 原规则（精调/搜索/确认） | 原会话各轮 | 原冻结自举（预测/导出/编译） | 重放各轮 | 重放结果 | '
             '重放选中 | 重放冻结自举（预测/导出/编译） | 编译自举变化 | report100 tensor / mock |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, entry in data['sessions'].items():
        old, session, accepted = entry.get('original') or {}, entry.get('session') or {}, entry.get('acceptance') or {}
        selected = session.get('selected') or {}
        chosen = f'{selected.get("origin")} 第 {selected.get("iteration")} 轮' if selected else '—'
        delta = change(old.get('boots'), accepted.get('boots'))
        lines.append(f'| {name} | {session.get("saved_replies", "—")} | '
                     f'{old.get("tune_rule")}/{old.get("tune_search") or "sweep"}/{old.get("confirm_rule") or "c1"} | '
                     f'{_rows(old.get("history"))} | {_triple(old.get("boots"))} | {_rows(session.get("history"))} | '
                     f'{entry.get("error") or session.get("error") or session.get("status")} | {chosen} | '
                     f'{_triple(accepted.get("boots"))} | {"—" if delta is None else f"{delta:+.1f}%"} | '
                     f'{_report(accepted.get("report100"), "tensor")} / {_report(accepted.get("report100"), "mock")} |')
    (root/'E9c.md').write_text('\n'.join(lines)+'\n')
    return lines


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    path = root/'E9c.json'
    data = read(path) if path.exists() else dict(api_before=len(ledger_rows(ledger)), sessions={})
    wanted = [spec for spec in SPECS if not args.sessions or spec['id'] in args.sessions]
    error = None
    try:
        model_dir = Path(e7.inventory()['models'][MODEL])
        for spec in wanted:
            if spec['id'] in data['sessions']:
                continue
            before = len(ledger_rows(ledger))
            data['sessions'][spec['id']] = run_session(root, spec, model_dir, ledger)
            data['api_after'] = len(ledger_rows(ledger))
            write(path, data)
            summarize(root, data)
            if data['api_after'] != before:
                raise RuntimeError('E9c changed the API ledger')
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        write(root/'STOP-c.json', dict(error=error))
        import traceback
        traceback.print_exc()
    data.update(api_after=len(ledger_rows(ledger)), error=error)
    data['complete'] = error is None and all(spec['id'] in data['sessions'] for spec in wanted)
    write(path, data)
    summarize(root, data, error)
    e9a.register('all', 'e9c-summary', root, dict(complete=data['complete'], error=error, api_before=data['api_before'],
                                                  api_after=data['api_after'], sessions=sorted(data['sessions'])))
    print('E9C_DONE='+str(path)+' ERROR='+str(error), flush=True)
    return int(bool(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sessions', nargs='+', choices=tuple(spec['id'] for spec in SPECS))
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
