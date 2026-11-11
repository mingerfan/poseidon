"""E9d (API 0): the E9b loop on MLP/CNN with the LLM replaced by the tuner's own output.

For mlp, cnn_plain, resnet20_silu and resnet20 one session runs through the E9b path: the r8-v4 settings,
program-plan placement, tuner v1, confirmation c1, at most 6 iterations and the same stale rule. The first reply
is the E1 rule program. Each later reply is the program the tuner returned in the previous iteration; when the
tuner did not run or returned an infeasible program, the previous reply is sent again. The session shows what the
tuner reaches from the rule program when no LLM writes a program. No LLM client is created and the ledger must stay
unchanged. A frozen program is compiled once and evaluated on the 8,000 report images, as in E9b. The rule
program's compiled count and the agent's count come from the E9b summary in the same root.

Outer:  python3 -B -m pt_agent.experiments.e9d_cnn_tuner_only --root <dir> [--models ...]
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from ..paths import RESULTS_DIR
from . import e9a_tuner_only as e9a
from . import e9b_cnn_sessions as e9b

read, write, ledger_rows = e9a.read, e9a.write, e9a.ledger_rows
ORDER, PLACEMENT, ITERATIONS = e9b.ORDER, e9b.PLACEMENT, e9b.CALLS_PER_MODEL
ORIGIN = dict(llm='回复', tuned='精调器输出')  # the session's origin names; every reply here comes from the echo client


def next_source(previous, row):
    """The tuner's program of the last trace row when it is feasible, else the previous reply."""
    tuned = (row or {}).get('tuned') or {}
    return tuned['source'] if tuned.get('feasible') and isinstance(tuned.get('source'), str) else previous


class EchoClient:
    """Stands in for the LLM: the rule program first, then the tuner's last result. Makes no API call."""

    def __init__(self, first, trace):
        self.first, self.trace, self.sent = first, Path(trace), []

    def chat(self, messages, *, purpose):
        source = self.first
        if self.sent:
            rows = [json.loads(line) for line in self.trace.read_text().splitlines() if line.strip()]
            if len(rows) != len(self.sent):
                raise RuntimeError(f'The trace holds {len(rows)} rows after {len(self.sent)} replies')
            source = next_source(self.sent[-1], rows[-1])
        self.sent.append(source)
        return dict(content=json.dumps(dict(program=source)), finish_reason='stop', usage={}, id=None, seconds=0.0)


def session_command(model, program, out, models_dir):
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(e9b.SESSION_SECONDS), '--',
            '-m', 'pt_agent.experiments.e9d_cnn_tuner_only', '--worker', 'session', '--model', model,
            '--program', str(program), '--out', str(out), '--models-dir', str(models_dir)]


def sweeps(session):
    """Per iteration: bootstraps of the reply and of the tuner's result (None when the tuner did not run)."""
    return [dict(iteration=row.get('iteration'), feasible=row.get('feasible'),
                 reply=((row.get('feedback') or {}).get('cost') or {}).get('boots_pred'),
                 tuned=((row.get('tuned') or {}).get('cost_after') or {}).get('boots_pred'),
                 tuned_feasible=(row.get('tuned') or {}).get('feasible'))
            for row in session.get('trace') or []]


def reference(root):
    """Compiled bootstraps of the rule program and of the E9b agent program, and the agent's report top-1."""
    path = root/'E9b.json'
    models = read(path).get('models', {}) if path.exists() else {}
    count = lambda item: ((item or {}).get('boots') or {}).get('compiled')
    return {model: dict(rule=count(entry.get('rule')), agent=count(entry.get('compile')),
                        agent_top1=(entry.get('test') or {}).get('top1_test')) for model, entry in models.items()}


# ---------------------------------------------------------------- inner worker

def session_worker(args):
    from .r8e6_reconfirm import _limits
    _limits()
    import torch
    from ..run import _inside as run_inside
    torch.set_num_threads(1)
    namespace = e9b.run_namespace(args.out, args.model, args.models_dir, e9b.SESSION_STAGES, iters=ITERATIONS)
    client = EchoClient(args.program.read_text(), args.out/'agent_trace.jsonl')
    namespace.llm_client = client
    code = run_inside(namespace)
    report = read(args.out/'REPORT.json')
    sent = client.sent
    write(args.out/'ECHO.json', dict(replies=len(sent), first=str(args.program),
                                     repeated=[index+1 for index in range(1, len(sent)) if sent[index] == sent[index-1]]))
    problems = []
    if report.get('placement') != PLACEMENT:
        problems.append('run.py did not apply the placement')
    if not sent or report.get('llm_calls') != len(sent):
        problems.append('run.py did not take every reply from the echo client')
    if problems:
        report.update(complete=False, error='; '.join(problems))
        write(args.out/'REPORT.json', report)
        return 1
    return code


# ---------------------------------------------------------------- outer steps

def run_model(root, model, ledger, entry=None):
    """Session, compile and report test of one model; `entry` keeps what ran if the round stops."""
    folder = root/'d'/model
    folder.mkdir(parents=True, exist_ok=False)
    entry = dict(model=model) if entry is None else entry
    program = RESULTS_DIR/e9b.RULE_RUNS[model]/'program.ptdsl'
    entry['program'] = str(program)
    before = len(ledger_rows(ledger))
    session = folder/'session'
    result = entry['session'] = e9b.session_child(
        session_command(model, program, session, e9b.MODELS_DIR), session, model=model, mode='e9d-'+model+'-session',
        ledger=ledger, before=before, recorded=e9b.recorded_calls())
    entry['sweeps'] = sweeps(result)
    if result['api_calls']:
        raise RuntimeError(model+': the ledger changed during an API-0 session')
    if result['outcome'] in ('provider_error', 'failed') or result['driver_error']:
        raise RuntimeError(model+' session '+result['outcome']+': '+str(
            result['driver_error'] or result['agent_error'] or result['error']))
    if result['frozen']:
        if not e9b.placed(result.get('header') or ''):
            raise RuntimeError(model+': frozen program header is '+str(result.get('header')))
        entry['compile'] = e9b.compiled(model, session/'program.ptdsl', folder/'compile', 'e9d-'+model+'-compile')
        if entry['compile'].get('complete'):
            shutil.copyfile(session/'REPORT.json', session/'REPORT.session.json')
            test = e9a.child(e9b.test_command(model, session, folder/'test'), folder/'test', model=model,
                             mode='e9d-'+model+'-test')
            entry['test'] = {key: test.get(key) for key in (*e9b.TEST_KEYS, 'complete', 'error', 'exit', 'seconds')}
    return entry


def _change(new, base):
    return f'{100*(new-base)/base:+.1f}%' if base and new is not None else '—'


def summarize(root, data, error=None):
    show = e9b._show
    lines = ['# E9d：MLP/CNN 上从规则程序出发只运行精调器（API 0）', '',
             f'路径与 E9b 相同（r8-v4 设置，放置 {PLACEMENT}，精调器 v1，确认 c1，至多 {ITERATIONS} 轮，同一停滞规则）。'
             '第 1 条回复是 E1 规则程序，之后每条回复是上一轮精调器的输出。'
             f'账本 {data.get("api_before")} → {data.get("api_after")}。异常：{error or "无"}。', '',
             '| 模型 | 规则程序自举（编译） | 各轮：回复 → 精调后（预测） | 会话：结果 / 轮数 / 秒 | 冻结来源 | '
             '冻结程序自举（预测/导出/编译） | 相对规则程序 | E9b Agent 冻结自举 | Agent 相对只跑精调器 | '
             'report 8000：top-1 / 明文 top-1 / 发散 |', '|---|---|---|---|---|---|---|---|---|---|']
    for model, entry in (data.get('models') or {}).items():
        known = (data.get('reference') or {}).get(model) or {}
        session, built, test = entry.get('session') or {}, entry.get('compile') or {}, entry.get('test') or {}
        frozen = (built.get('boots') or {}).get('compiled')
        steps = '；'.join(f'{show(row["reply"])} → {show(row["tuned"])}' for row in entry.get('sweeps') or []) or '—'
        source = '—' if session.get('best_source') is None else (
            f'{ORIGIN.get(session["best_source"], session["best_source"])}，第 {session.get("best_iteration")} 轮')
        lines.append(f'| {model} | {show(known.get("rule"))} | {steps} | '
                     f'{session.get("outcome")} / {session.get("iterations")} / {show(session.get("seconds"))} | {source} | '
                     f'{e9b._three(built.get("boots"))} | {_change(frozen, known.get("rule"))} | {show(known.get("agent"))} | '
                     f'{_change(known.get("agent"), frozen)} | '
                     f'{show(test.get("top1_test"))} / {show(test.get("torch_top1_report"))} / {show(test.get("n_diverged"))} |')
    (root/'E9d.md').write_text('\n'.join(lines)+'\n')
    return lines


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    path = root/'E9d.json'
    data = read(path) if path.exists() else dict(api_before=len(ledger_rows(ledger)), models={})
    data['reference'] = reference(root)
    error = None
    try:
        for model in ORDER:
            if model in data['models'] or (args.models and model not in args.models):
                continue
            entry = dict(model=model)
            try:
                run_model(root, model, ledger, entry)
            except Exception:
                data.setdefault('stopped', {})[model] = entry
                raise
            data['models'][model] = entry
            data['api_after'] = len(ledger_rows(ledger))
            write(path, data)
            summarize(root, data)
            if data['api_after'] != data['api_before']:
                raise RuntimeError('E9d changed the API ledger')
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        write(root/'STOP-d.json', dict(error=error))
        import traceback
        traceback.print_exc()
    data.update(api_after=len(ledger_rows(ledger)), error=error)
    wanted = [model for model in ORDER if not args.models or model in args.models]
    data['complete'] = error is None and all(model in data['models'] for model in wanted)
    write(path, data)
    summarize(root, data, error)
    e9a.register('all', 'e9d-summary', root, dict(complete=data['complete'], error=error, api_before=data['api_before'],
                                                  api_after=data['api_after'], models=sorted(data['models'])))
    print('E9D_DONE='+str(path)+' ERROR='+str(error), flush=True)
    return int(bool(error) or data['api_after'] != data['api_before'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--models', nargs='+', choices=ORDER)
    parser.add_argument('--worker', choices=('session',))
    parser.add_argument('--model', choices=ORDER)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--program', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.worker:
        if any(getattr(args, name) is None for name in ('model', 'models_dir', 'program', 'out')):
            parser.error('--worker session requires --model, --models-dir, --program and --out')
        return session_worker(args)
    if args.root is None:
        parser.error('--root is required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
