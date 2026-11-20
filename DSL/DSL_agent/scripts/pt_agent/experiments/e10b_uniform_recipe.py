"""E10b (API 0): the best uniform approximation recipe for the MLP and the three CNNs.

The E1 rule programs give every activation the same Chebyshev degree (127 for the ReLU models, 63 for ResNet-20 SiLU)
with each model's calibrated-domain margin. This round lowers that one degree step by step along the tuner's ladder.
Each uniform program goes once through the E9b session path with the tuner off and one iteration, so it meets the same
static checks, held-out evaluation, confirmation c1 and program-plan placement as the agent's programs. The descent
stops at the first degree that is not frozen; the lowest frozen degree is the best uniform recipe. Its program is
compiled once and evaluated on the 8,000 report images, as in E9b and E9d. Registry rows carry round E10.

Outer:  python3 -B -m pt_agent.experiments.e10b_uniform_recipe --root <dir> [--models ...]
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys

from ..paths import RESULTS_DIR
from . import e9a_tuner_only as e9a
from . import e9b_cnn_sessions as e9b
from .e9d_cnn_tuner_only import EchoClient

ROUND = 'E10'
LADDER = (255, 127, 63, 31, 15, 7)
DEG = re.compile(r'\bdeg=(\d+)')
E9_ROOT = RESULTS_DIR/'E9-same-pipeline-20261110'
read, write, ledger_rows = e9a.read, e9a.write, e9a.ledger_rows


def rule_program(model):
    return RESULTS_DIR/e9b.RULE_RUNS[model]/'program.ptdsl'


def uniform(source, degree):
    """The rule program with every approximation degree set to `degree`."""
    found = set(DEG.findall(source))
    if len(found) != 1:
        raise ValueError(f'Expected one uniform degree in the rule program, found {sorted(found)}')
    return DEG.sub(f'deg={degree}', source)


def degrees_below(rule_degree):
    return [degree for degree in LADDER if degree < rule_degree]


def reply_boots(session):
    """Predicted bootstraps and feasibility of the one reply in a session result."""
    rows = session.get('trace') or []
    row = rows[0] if rows else {}
    cost = (row.get('feedback') or {}).get('cost') or {}
    return dict(feasible=row.get('feasible'), boots_pred=cost.get('boots_pred'))


def session_command(model, program, out, models_dir):
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(e9b.SESSION_SECONDS), '--',
            '-m', 'pt_agent.experiments.e10b_uniform_recipe', '--worker', 'session', '--model', model,
            '--program', str(program), '--out', str(out), '--models-dir', str(models_dir)]


# ---------------------------------------------------------------- inner worker

def session_worker(args):
    from .r8e6_reconfirm import _limits
    _limits()
    import torch
    from ..run import _inside as run_inside
    torch.set_num_threads(1)
    namespace = e9b.run_namespace(args.out, args.model, args.models_dir, e9b.SESSION_STAGES, iters=1)
    namespace.tuner = 'off'
    client = EchoClient(args.program.read_text(), args.out/'agent_trace.jsonl')
    namespace.llm_client = client
    code = run_inside(namespace)
    report = read(args.out/'REPORT.json')
    problems = []
    if report.get('placement') != e9b.PLACEMENT:
        problems.append('run.py did not apply the placement')
    if len(client.sent) != 1 or report.get('llm_calls') != 1:
        problems.append('run.py did not take exactly one reply from the echo client')
    if problems:
        report.update(complete=False, error='; '.join(problems))
        write(args.out/'REPORT.json', report)
        return 1
    return code


# ---------------------------------------------------------------- outer steps

def run_model(root, model, ledger, entry):
    folder = root/'b'/model
    folder.mkdir(parents=True, exist_ok=False)
    source = rule_program(model).read_text()
    rule_degree = int(DEG.search(source).group(1))
    entry.update(rule_program=str(rule_program(model)), rule_degree=rule_degree, steps=[], best_degree=rule_degree)
    best = None
    for degree in degrees_below(rule_degree):
        work = folder/f'deg{degree}'
        work.mkdir()
        program = work/'uniform.ptdsl'
        program.write_text(uniform(source, degree))
        result = e9b.session_child(session_command(model, program, work/'session', e9b.MODELS_DIR), work/'session',
                                   model=model, mode=f'e10b-{model}-deg{degree}-session', ledger=ledger,
                                   before=len(ledger_rows(ledger)), recorded=e9b.recorded_calls())
        entry['steps'].append(dict(degree=degree, outcome=result['outcome'], frozen=result['frozen'],
                                   seconds=result['seconds'], **reply_boots(result)))
        if result['api_calls']:
            raise RuntimeError(model+': the ledger changed during an API-0 session')
        if result['outcome'] in ('provider_error', 'failed') or result['driver_error']:
            raise RuntimeError(f'{model} deg {degree} session {result["outcome"]}: '
                               + str(result['driver_error'] or result['agent_error'] or result['error']))
        if not result['frozen']:
            break
        if not e9b.placed(result.get('header') or ''):
            raise RuntimeError(model+': frozen program header is '+str(result.get('header')))
        best = dict(degree=degree, session=work/'session')
    if best is None:
        return entry
    entry['best_degree'] = best['degree']
    session = best['session']
    entry['compile'] = e9b.compiled(model, session/'program.ptdsl', folder/'compile', f'e10b-{model}-compile')
    if entry['compile'].get('complete'):
        shutil.copyfile(session/'REPORT.json', session/'REPORT.session.json')
        test = e9a.child(e9b.test_command(model, session, folder/'test'), folder/'test', model=model,
                         mode=f'e10b-{model}-test')
        entry['test'] = {key: test.get(key) for key in (*e9b.TEST_KEYS, 'complete', 'error', 'exit', 'seconds')}
    return entry


def references():
    """Compiled bootstraps of the rule (E9b), agent (E9b) and tuner-only (E9d) programs, with report top-1."""
    count = lambda item: ((item or {}).get('boots') or {}).get('compiled')
    top1 = lambda item: (item or {}).get('top1_test')
    e9b_models = read(E9_ROOT/'E9b.json').get('models', {}) if (E9_ROOT/'E9b.json').exists() else {}
    e9d_models = read(E9_ROOT/'E9d.json').get('models', {}) if (E9_ROOT/'E9d.json').exists() else {}
    return {model: dict(rule=count(e9b_models.get(model, {}).get('rule')),
                        agent=count(e9b_models.get(model, {}).get('compile')),
                        agent_top1=top1(e9b_models.get(model, {}).get('test')),
                        tuner_only=count(e9d_models.get(model, {}).get('compile')),
                        tuner_only_top1=top1(e9d_models.get(model, {}).get('test')),
                        plaintext=(e9b_models.get(model, {}).get('test') or {}).get('torch_top1_report'))
            for model in e9b.ORDER}


def _change(new, base):
    return f'{100*(new-base)/base:+.1f}%' if base and new is not None else '—'


def summarize(root, data):
    show = e9b._show
    lines = ['# E10b：MLP/CNN 的最佳统一配方（API 0）', '',
             '每个激活用同一个切比雪夫阶数，沿精调器阶梯从规则阶数往下降；每个程序经 E9b 会话路径（精调器关闭，1 轮，确认 c1，'
             'program-plan 放置），第一个未冻结的阶数处停止。'
             f'账本 {data.get("api_before")} → {data.get("api_after")}。异常：{data.get("error") or "无"}。', '',
             '| 模型 | 规则阶数 / 规则程序自举 | 各阶数：结果（预测自举） | 最佳统一阶数 | 编译自举（预测/导出/编译） | '
             '相对规则程序 | 只跑精调器 / Agent | Agent 相对最佳统一配方 | report 8000：top-1 / 明文 / 发散 |',
             '|---|---|---|---|---|---|---|---|---|']
    for model, entry in (data.get('models') or {}).items():
        ref = (data.get('reference') or {}).get(model) or {}
        steps = '；'.join(f'{step["degree"]}：{step["outcome"]}（{show(step["boots_pred"])}）' for step in entry.get('steps') or [])
        built, test = entry.get('compile') or {}, entry.get('test') or {}
        boots = (built.get('boots') or {}).get('compiled') if built else ref.get('rule')
        lines.append(f'| {model} | {entry.get("rule_degree")} / {show(ref.get("rule"))} | {steps or "—"} | '
                     f'{entry.get("best_degree")} | {e9b._three(built.get("boots")) if built else "（规则程序）"} | '
                     f'{_change(boots, ref.get("rule"))} | {show(ref.get("tuner_only"))} / {show(ref.get("agent"))} | '
                     f'{_change(ref.get("agent"), boots)} | {show(test.get("top1_test"))} / '
                     f'{show(test.get("torch_top1_report") or ref.get("plaintext"))} / {show(test.get("n_diverged"))} |')
    (root/'E10b.md').write_text('\n'.join(lines)+'\n')


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    e9a.ROUND = ROUND   # the reused E9 helpers write registry rows under this round
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    path = root/'E10b.json'
    data = read(path) if path.exists() else dict(api_before=len(ledger_rows(ledger)), models={})
    data['reference'] = references()
    error = None
    try:
        for model in e9b.ORDER:
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
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        import traceback
        traceback.print_exc()
    data.update(api_after=len(ledger_rows(ledger)), error=error)
    if data['api_after'] != data['api_before']:
        data['error'] = (error or '') + ' The API ledger changed.'
    wanted = [model for model in e9b.ORDER if not args.models or model in args.models]
    data['complete'] = data['error'] is None and all(model in data['models'] for model in wanted)
    write(path, data)
    summarize(root, data)
    e9a.register('all', 'e10b-summary', root, dict(complete=data['complete'], error=data['error'],
                                                   api_before=data['api_before'], api_after=data['api_after']))
    print('E10B_DONE='+str(path)+' ERROR='+str(data['error']), flush=True)
    return 0 if data['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--models', nargs='+', choices=e9b.ORDER)
    parser.add_argument('--worker', choices=('session',))
    parser.add_argument('--model', choices=e9b.ORDER)
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
