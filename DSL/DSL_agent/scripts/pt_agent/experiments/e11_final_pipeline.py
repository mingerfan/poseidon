"""E11: the final pipeline on all six models: tune the rule program first, then let the LLM restructure it.

Every model uses one setting. The first reply is the system's rule program (the E1 rule program for the MLP and the
CNNs, the rule softmax search R1 for BERT-tiny and the Qwen2 decoder), so iteration 1 tunes it to convergence. At
most three LLM rewrites follow. Every eligible program is tuned (tune policy "all"); after the first tune only the
points on lines that differ from the best program are retuned (focus retune). The loop confirms after one iteration
without a 5% gain. MLP/CNN: prompt r8-v4, program-plan placement, tuner v1 with up to 8 sweeps, confirmation c1
with every sweep's accepted state in the confirmation pool (pool_states).
BERT/Qwen: prompt r8-v11, program-plan placement, tune rule v8-6.25 with the sweep search, confirmation c2.

--variant e12 (round E12) adds the numeric lock: on every later reply, the operations whose structure is unchanged
get the best program's tuned degrees, iteration counts and domains, and the feedback tells the LLM that the tuner owns
these settings; every model gets all three rewrites (stale limit 3). --variant e13 (round E13) keeps all three
rewrites without the lock: the E12 replay showed that the lock removes the LLM's valid numeric gains on BERT-tiny.

--mode replay (API 0) sends saved replies of earlier sessions as the LLM rewrites, to exercise and time the path;
--replay-from takes them from another results root. --mode live calls the provider; it needs the user's approval and
stays within --ledger-cap.

Outer:  python3 -B -m pt_agent.experiments.e11_final_pipeline --root <dir> --mode replay|live [--models ...]
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

from ..paths import RESULTS_DIR
from . import e9a_tuner_only as e9a
from . import e9b_cnn_sessions as e9b

CNN = e9b.ORDER
TRANSFORMERS = ('bert', 'qwen')
ITERATIONS, CNN_SWEEPS, LEDGER_CAP = 4, 8, 450
VARIANTS = dict(e11=dict(round='E11', label='E11', stale=1, lock=False),
                e12=dict(round='E12', label='E12', stale=3, lock=True),
                e13=dict(round='E13', label='E13', stale=3, lock=False))
# Saved replies for the API-0 replay: the earlier session of each model whose LLM programs we reuse.
REPLAY_TRACES = dict(
    mlp=RESULTS_DIR/'E9-same-pipeline-20261110/b/mlp/session/agent_trace.jsonl',
    cnn_plain=RESULTS_DIR/'E9-same-pipeline-20261110/b/cnn_plain/session/agent_trace.jsonl',
    resnet20_silu=RESULTS_DIR/'E9-same-pipeline-20261110/b/resnet20_silu/session/agent_trace.jsonl',
    resnet20=RESULTS_DIR/'E9-same-pipeline-20261110/b/resnet20/session/agent_trace.jsonl',
    bert=RESULTS_DIR/'E6r2-resume-20261107/e6r2-b/bert/p2/session/agent_trace.jsonl',
    qwen=RESULTS_DIR/'E8-align-fallback-20261108/e8b/qwen/r1/session/agent_trace.jsonl')
read, write, ledger_rows = e9a.read, e9a.write, e9a.ledger_rows


def rule_program(model):
    if model in TRANSFORMERS:
        return e9a.RULE_ROOTS[model]/'R1'/'program.ptdsl'
    return RESULTS_DIR/e9b.RULE_RUNS[model]/'program.ptdsl'


def saved_replies(model, count=ITERATIONS-1, source=None):
    """Up to `count` LLM program replies of an earlier session (`source`: a results root with <model>/session).

    Replies served as the rule program by an earlier E11/E12 run (the first row, with no usage) are skipped."""
    path = REPLAY_TRACES[model] if source is None else Path(source)/model/'session'/'agent_trace.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if source is not None:
        rows = rows[1:]
    replies = [row['messages'][0]['content'] for row in rows if row.get('messages')]
    if not replies:
        raise ValueError(f'{model}: no saved replies in {path}')
    return replies[:count]


def reply_record(content):
    return dict(content=content, finish_reason='stop', usage={}, id=None, seconds=0.0)


class FirstReplyClient:
    """Serves the rule program as the first reply (no API call), then the wrapped client's replies."""

    def __init__(self, first, inner):
        self.first, self.inner, self.served = first, inner, 0

    def chat(self, messages, *, purpose):
        self.served += 1
        if self.served == 1:
            return reply_record(json.dumps(dict(program=self.first)))
        return self.inner.chat(messages, purpose=purpose)


# ---------------------------------------------------------------- MLP and CNN

def cnn_command(model, out, args, replies, iterations=ITERATIONS):
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(e9b.SESSION_SECONDS), '--',
               '-m', 'pt_agent.experiments.e11_final_pipeline', '--worker', 'cnn-session', '--model', model,
               '--out', str(out), '--models-dir', str(e9b.MODELS_DIR), '--mode', args.mode,
               '--variant', args.variant, '--iterations', str(iterations),
               '--budget-path', str(RESULTS_DIR/'llm_calls.jsonl'), '--ledger-cap', str(args.ledger_cap)]
    return command + (['--replies', str(replies)] if replies is not None else [])


def cnn_worker(args):
    from .r8e6_reconfirm import _limits
    _limits()
    import torch
    from ..run import _inside as run_inside
    torch.set_num_threads(1)
    settings = VARIANTS[args.variant]
    namespace = e9b.run_namespace(args.out, args.model, args.models_dir, e9b.SESSION_STAGES,
                                  budget_path=args.budget_path, limit=args.ledger_cap, iters=args.iterations)
    namespace.tune_policy, namespace.stale_limit = 'all', settings['stale']
    namespace.tune_sweeps, namespace.focus_retune, namespace.pool_states = CNN_SWEEPS, 'on', 'on'
    namespace.numeric_lock = 'on' if settings['lock'] else 'off'
    if args.mode == 'replay':
        inner = e9b.ReplayClient(read(args.replies))
    else:
        from ..llm import LLMClient
        inner = LLMClient(args.budget_path, model=namespace.llm_model, reasoning_effort=namespace.llm_reasoning,
                          max_tokens=namespace.llm_max_tokens, max_total_calls=args.ledger_cap)
    client = FirstReplyClient(rule_program(args.model).read_text(), inner)
    namespace.llm_client = client
    code = run_inside(namespace)
    report = read(args.out/'REPORT.json')
    problems = []
    if report.get('placement') != e9b.PLACEMENT:
        problems.append('run.py did not apply the placement')
    if report.get('llm_calls') != client.served:
        problems.append(f'run.py made {report.get("llm_calls")} calls; the client served {client.served}')
    if problems:
        report.update(complete=False, error='; '.join(problems))
        write(args.out/'REPORT.json', report)
        return 1
    return code


def run_cnn(root, model, args, ledger, entry):
    folder = root/model
    folder.mkdir(parents=True, exist_ok=False)
    replies, iterations = None, ITERATIONS
    if args.mode == 'replay':
        saved = saved_replies(model, source=args.replay_from)
        replies, iterations = folder/'replies.json', 1+len(saved)
        write(replies, saved)
    session = folder/'session'
    result = entry['session'] = e9b.session_child(cnn_command(model, session, args, replies, iterations), session,
                                                  model=model, mode=f'{args.variant}-{model}-session', ledger=ledger,
                                                  before=len(ledger_rows(ledger)), recorded=e9b.recorded_calls())
    if args.mode == 'replay' and result['api_calls']:
        raise RuntimeError(model+': the ledger changed during an API-0 replay')
    if result['outcome'] in ('provider_error', 'failed') or result['driver_error']:
        raise RuntimeError(f'{model} session {result["outcome"]}: '
                           + str(result['driver_error'] or result['agent_error'] or result['error']))
    if result['frozen']:
        if not e9b.placed(result.get('header') or ''):
            raise RuntimeError(model+': frozen program header is '+str(result.get('header')))
        entry['compile'] = e9b.compiled(model, session/'program.ptdsl', folder/'compile', f'{args.variant}-{model}-compile')
        if entry['compile'].get('complete'):
            shutil.copyfile(session/'REPORT.json', session/'REPORT.session.json')
            test = e9a.child(e9b.test_command(model, session, folder/'test'), folder/'test', model=model,
                             mode=f'{args.variant}-{model}-test')
            entry['test'] = {key: test.get(key) for key in (*e9b.TEST_KEYS, 'complete', 'error', 'exit', 'seconds')}
    return entry


# ---------------------------------------------------------------- BERT and Qwen

def transformer_arguments(session, phase, iteration, folder, args, ledger, iterations=ITERATIONS):
    settings = VARIANTS[args.variant]
    extra = ['--step', phase, '--root', str(session), '--call-dir', str(folder.parent/'calls'),
             '--budget-path', str(ledger), '--tune-search', 'sweep', '--session-label', settings['label'],
             '--session-iters', str(iterations), '--confirm-rule', 'c2', '--tune-policy', 'all',
             '--stale-limit', str(settings['stale']), '--focus-retune', 'on',
             '--numeric-lock', 'on' if settings['lock'] else 'off', '--ledger-cap', str(args.ledger_cap)]
    if phase == 'evaluate' and (iteration == 1 or args.mode == 'replay'):
        reply = folder/f'reply-{iteration}.json'
        if not reply.exists():
            raise RuntimeError(f'{folder.name}: no saved reply for iteration {iteration}')
        extra += ['--response-file', str(reply)]
    return extra


def run_transformer(root, model, args, ledger, entry):
    from ..planners.transformer_session import load_state
    from .a1_transformer_agent import account_calls
    from . import e7_confirm_tail_check as e7
    folder = root/model
    folder.mkdir(parents=True, exist_ok=False)
    model_dir = Path(e7.inventory()['models'][model])
    split = read(model_dir/'model_info.json')['split'].get('version', 'split-v1')
    settings, iterations = VARIANTS[args.variant], ITERATIONS
    write(folder/'reply-1.json', e9a.saved_reply(rule_program(model).read_text()))
    if args.mode == 'replay':
        saved = saved_replies(model, source=args.replay_from)
        iterations = 1+len(saved)
        for number, content in enumerate(saved, 2):
            write(folder/f'reply-{number}.json', reply_record(content))
    session = folder/'session'
    session.mkdir()
    before, started, steps = len(ledger_rows(ledger)), time.monotonic(), []
    while True:
        state = load_state(session/'SESSION.json') if (session/'SESSION.json').exists() else None
        if state and state['phase'] == 'done':
            break
        phase, iteration = (state['phase'], state['iteration']) if state else ('evaluate', 1)
        out = session/'steps'/f'{len(steps)+1:03d}-{phase}'
        command = e9a.step_command('session-step', out, model, model_dir) + transformer_arguments(
            session, phase, iteration, folder, args, ledger, iterations)
        result = e9a.child(command, out, model=model, mode=f'{args.variant}-{model}-{phase}')
        if args.mode == 'live':
            account_calls(ledger, session, model=model, prompt_version='r8-v11', split_version=split,
                          label=settings['label'], experiment_round=settings['round'])
        steps.append(dict(phase=phase, iteration=iteration, report=str(out/'REPORT.json'), seconds=result.get('seconds')))
        write(session/'STEPS.json', dict(steps=steps))
        if result.get('exit') or not result.get('complete'):
            entry['session'] = dict(complete=False, error=result.get('error'), steps=steps,
                                    seconds=time.monotonic()-started)
            raise RuntimeError(f'{model} {phase} step failed: {result.get("error")}')
        if len(steps) > 3*iterations+2:
            raise RuntimeError(model+': the session took more steps than its iterations allow')
    api_calls = len(ledger_rows(ledger)) - before
    if args.mode == 'replay' and api_calls:
        raise RuntimeError(model+': the ledger changed during an API-0 replay')
    agent = read(session/'AGENT.json')
    entry['session'] = dict(complete=agent['status'] != 'error', frozen=agent['status'] == 'feasible',
                            status=agent['status'], best_source=agent.get('best_source'),
                            best_iteration=agent.get('best_iteration'), iterations=agent.get('iterations'),
                            history=agent.get('history'), api_calls=api_calls, steps=steps,
                            seconds=time.monotonic()-started)
    e9a.register(model, f'{args.variant}-{model}-session', session, {key: entry['session'][key] for key in (
        'complete', 'frozen', 'status', 'seconds', 'api_calls')})
    if entry['session']['frozen']:
        entry['acceptance'] = e9a.acceptance(root, folder, model, model_dir)
    return entry


# ---------------------------------------------------------------- summary

def trace_rows(session):
    path = Path(session)/'agent_trace.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def iteration_steps(session):
    """Per iteration: the reply's predicted bootstraps, the tuned result, and how many lines the retune focused on."""
    steps = []
    for row in trace_rows(session):
        cost = (row.get('fb') or row.get('feedback') or {}).get('cost') or {}
        tuned = row.get('tuned') or {}
        steps.append(dict(iteration=row.get('iteration'), feasible=(row.get('fb') or {}).get('feasible'),
                          reply=cost.get('boots_pred'), tuned=(tuned.get('cost_after') or {}).get('boots_pred')
                          or tuned.get('boots_pred'), focus=len(tuned['focus_lines']) if 'focus_lines' in tuned else None,
                          tune_s=tuned.get('seconds'), llm_s=row.get('seconds')))
    return steps


def summarize(root, data):
    show = lambda value: '—' if value is None else (f'{value:.0f}' if isinstance(value, float) else str(value))
    lines = [f'# {data.get("round", "E11")}：最终流程（{data.get("mode")}）', '',
             '第 1 轮是规则程序并精调到收敛，之后至多 3 次 LLM 改写，每次只重调改动的行；一轮没有 5% 改进即确认。'
             f'账本 {data.get("api_before")} → {data.get("api_after")}。异常：{data.get("error") or "无"}。', '',
             '| 模型 | 各轮：回复 → 精调后（重调行数） | 结果 / 冻结来源 | 编译自举 | report | 会话分钟 |',
             '|---|---|---|---|---|---|']
    for model, entry in (data.get('models') or {}).items():
        session = entry.get('session') or {}
        steps = '；'.join(f'{show(s["reply"])} → {show(s["tuned"])}' + (f'（{s["focus"]}）' if s['focus'] is not None else '')
                         for s in entry.get('steps') or []) or '—'
        built = (entry.get('compile') or {}).get('boots') or ((entry.get('acceptance') or {}).get('boots')) or {}
        test = entry.get('test') or {}
        top1 = lambda value: '—' if value is None else f'{value:.2f}'
        tensor = (((entry.get('acceptance') or {}).get('report100')) or {}).get('tensor') or {}
        report = (f'{top1(test.get("top1_test"))} / {top1(test.get("torch_top1_report"))}' if test else
                  f'{tensor.get("agree")}/{tensor.get("total")}，logit {tensor.get("logit_rel_err"):.2g}'
                  if tensor.get('logit_rel_err') is not None else '—')
        source = session.get('best_source') or ((session.get('best') or {}).get('source'))
        lines.append(f'| {model} | {steps} | {session.get("status") or session.get("outcome")} / {source}，'
                     f'第 {session.get("best_iteration")} 轮 | {show(built.get("compiled"))} | {report} | '
                     f'{show((session.get("seconds") or 0)/60)} |')
    (Path(root)/'E11.md').write_text('\n'.join(lines)+'\n')


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    settings = VARIANTS[args.variant]
    e9a.ROUND = settings['round']   # the reused E9 helpers write registry rows under this round
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    path = root/'E11.json'
    data = read(path) if path.exists() else dict(mode=args.mode, variant=args.variant, round=settings['round'],
                                                 api_before=len(ledger_rows(ledger)), models={})
    if data['mode'] != args.mode or data.get('variant', 'e11') != args.variant:
        raise ValueError('The root holds a run in another mode or variant')
    error = None
    try:
        for model in (*CNN, *TRANSFORMERS):
            if model in data['models'] or (args.models and model not in args.models):
                continue
            entry = dict(model=model)
            try:
                (run_transformer if model in TRANSFORMERS else run_cnn)(root, model, args, ledger, entry)
            except Exception:
                data.setdefault('stopped', {})[model] = entry
                raise
            finally:
                entry['steps'] = iteration_steps(root/model/'session')
            data['models'][model] = entry
            data['api_after'] = len(ledger_rows(ledger))
            write(path, data)
            summarize(root, data)
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        import traceback
        traceback.print_exc()
    data.update(api_after=len(ledger_rows(ledger)), error=error)
    if args.mode == 'replay' and data['api_after'] != data['api_before']:
        data['error'] = (error or '') + ' The API ledger changed during a replay.'
    wanted = [model for model in (*CNN, *TRANSFORMERS) if not args.models or model in args.models]
    data['complete'] = data['error'] is None and all(model in data['models'] for model in wanted)
    write(path, data)
    summarize(root, data)
    e9a.register('all', f'{args.variant}-summary', root, dict(mode=args.mode, complete=data['complete'], error=data['error'],
                                                  api_before=data['api_before'], api_after=data['api_after']))
    print('E11_DONE='+str(path)+' ERROR='+str(data['error']), flush=True)
    return 0 if data['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--mode', choices=('replay', 'live'), required=True)
    parser.add_argument('--models', nargs='+', choices=(*CNN, *TRANSFORMERS))
    parser.add_argument('--ledger-cap', type=int, default=LEDGER_CAP)
    parser.add_argument('--variant', choices=tuple(VARIANTS), default='e11')
    parser.add_argument('--replay-from', type=Path)
    parser.add_argument('--iterations', type=int, default=ITERATIONS)
    parser.add_argument('--worker', choices=('cnn-session',))
    parser.add_argument('--model', choices=CNN)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--budget-path', type=Path)
    parser.add_argument('--replies', type=Path)
    args = parser.parse_args()
    if args.worker:
        if any(getattr(args, name) is None for name in ('model', 'models_dir', 'out', 'budget_path')):
            parser.error('--worker cnn-session requires --model, --models-dir, --out and --budget-path')
        if args.mode == 'replay' and args.replies is None:
            parser.error('replay mode requires --replies')
        return cnn_worker(args)
    if args.root is None:
        parser.error('--root is required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
