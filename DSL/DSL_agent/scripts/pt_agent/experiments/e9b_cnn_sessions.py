"""E9b: new MLP/CNN agent sessions with program-level placement, as in the Transformer sessions.

For mlp, cnn_plain, resnet20_silu and resnet20 one new agent session runs (prompt r8-v4, at most 6 calls each).
The session rewrites every reply's header to refresh="agent", placement="program-plan", so its cost feedback,
its tuner and its confirmation all use the program-level planner. A frozen program is exported, traced and
compiled once, and then evaluated on the 8,000 report images. The E1 rule program of each model gets the same
header and the same compile, which gives the baseline under the same planner.

Step b0 replays the saved R8e5 MLP replies through the same path and compiles the result (API 0). It must pass
before the first API call. The ledger stays at or below API_CEILING.

Outer:  python3 -B -m pt_agent.experiments.e9b_cnn_sessions --root <dir> [--models ...] [--smoke-only]
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from ..paths import RESULTS_DIR
from . import e9a_tuner_only as e9a
from . import r8g_mlp_rerun as r8g

read, write, ledger_rows = e9a.read, e9a.write, e9a.ledger_rows
ROUND, PLACEMENT, PROMPT_VERSION = e9a.ROUND, e9a.PLACEMENT, 'r8-v4'
ORDER = ('mlp', 'cnn_plain', 'resnet20_silu', 'resnet20')
CALLS_PER_MODEL, API_CEILING = 6, 439
MODELS_DIR = RESULTS_DIR/'models-v2'
RULE_RUNS = dict(mlp='run-mlp-rule-20261103T111657', cnn_plain='run-cnn_plain-rule-20261103T112051',
                 resnet20_silu='run-resnet20_silu-rule-20261103T113209', resnet20='run-resnet20-rule-20261103T123122')
SMOKE_MODEL, SMOKE_TRACE = 'mlp', RESULTS_DIR/'r8e5-dev-20261022T041646/mlp/agent_trace.jsonl'
SESSION_STAGES = ('plan', 'check', 'mock')
SESSION_SECONDS, COMPILE_SECONDS, TEST_SECONDS = 21600, 9000, 9000
SCRIPTS = Path(__file__).resolve().parents[2]
COST_KEYS = ('boots_pred', 'latency_pred_us', 'barriers', 'max_depth', 'ops_total', 'const_bytes', 'placement_model')
TEST_KEYS = ('top1_test', 'torch_top1_report', 'correct', 'n_diverged', 'diverged_indices', 'test_images_done',
             'test_seconds', 'm1_item3_pass', 'test_split')
SESSION_KEYS = ('complete', 'stage', 'error', 'placement', 'prompt_version', 'tune_policy', 'llm_calls',
                'prompt_tokens', 'completion_tokens', 'best_iteration', 'best_source', 'source_lines',
                'llm_source_lines', 'frozen_source_lines', 'latency_pred_us', 'agree', 'top1', 'logit_rmse',
                'exact_max_abs_diff', 'mock_images_done')


def call_limit(before):
    """The absolute ledger length one session may reach."""
    return min(before+CALLS_PER_MODEL, API_CEILING)


def run_namespace(out, model, models_dir, stages, *, budget_path=None, limit=0, iters=CALLS_PER_MODEL):
    """Arguments of run._inside for one stage list; the agent settings equal the R8e5/R8g sessions."""
    return SimpleNamespace(
        out=out, model=model, models_dir=models_dir, planner='agent', deg=None, margin=None, refresh='compiler',
        act_method='cheb', stages=list(stages), final=True, agent_iters=iters, llm_model='deepseek-flash',
        llm_reasoning='high', llm_max_tokens=32768, budget_path=budget_path, llm_call_limit=limit, tuner='on',
        tuner_budget_s=2400, tune_policy='new_best', prompt_version=PROMPT_VERSION, placement=PLACEMENT)


class ReplayClient:
    """Returns the saved replies of an earlier session in order; makes no API call and writes no ledger row."""

    def __init__(self, replies):
        self.replies, self.used = list(replies), 0

    @classmethod
    def from_trace(cls, path):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return cls(row['messages'][0]['content'] for row in rows if row.get('messages'))

    def chat(self, messages, *, purpose):
        if self.used >= len(self.replies):
            from ..llm import LLMError
            raise LLMError('budget_exhausted')
        self.used += 1
        return dict(content=self.replies[self.used-1], finish_reason='stop', usage={}, id=None, seconds=0.0)


def nix(seconds, *arguments):
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds), '--',
            '-m', 'pt_agent.experiments.e9b_cnn_sessions', *(str(value) for value in arguments)]


def session_command(model, out, ledger, limit, replay=None):
    return nix(SESSION_SECONDS, '--worker', 'session', '--model', model, '--out', out, '--budget-path', ledger,
               '--call-limit', limit, *(('--replay', replay) if replay else ()))


def compile_command(model, program, out):
    return nix(COMPILE_SECONDS, '--worker', 'compile', '--model', model, '--program', program, '--out', out)


def test_command(model, session, out):
    return nix(TEST_SECONDS, '--worker', 'test', '--model', model, '--session', session, '--out', out)


def header_command(program, out):
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '600', '--', '-m',
            'pt_agent.experiments.e9a_tuner_only', '--worker', 'header', '--program', str(program), '--out', str(out)]


def placed(header):
    """True when a program header asks for agent refresh under the program-level planner."""
    return all(re.search(name+r"""\s*=\s*['"]"""+re.escape(value)+r"""['"]""", header)
               for name, value in (('refresh', 'agent'), ('placement', PLACEMENT)))


def session_outcome(report, frozen):
    """frozen, confirm_failed, provider_error (the round stops) or failed."""
    agent = report.get('agent') or {}
    if report.get('complete') and frozen and agent.get('status') == 'feasible':
        return 'frozen'
    if agent.get('status') == 'confirm_failed':
        return 'confirm_failed'
    if agent.get('status') == 'error':
        return 'provider_error'
    return 'failed'


def smoke_problems(report, rows, header, replies, before, after):
    """Reasons the API-0 replay does not show the path the new sessions will take."""
    agent = report.get('agent') or {}
    problems = []
    if not report.get('complete'):
        problems.append('session report incomplete: '+str(report.get('error')))
    if report.get('placement') != PLACEMENT or agent.get('placement') != PLACEMENT:
        problems.append('placement was not passed to the session')
    if len(rows) != replies or any(row.get('placement') != PLACEMENT for row in rows):
        problems.append('trace rows do not all record the placement')
    if any((row.get('fb') or {}).get('cost') and row['fb']['cost'].get('placement_model') != PLACEMENT for row in rows):
        problems.append('cost feedback did not come from the program-level planner')
    if not any(row.get('tuned') for row in rows):
        problems.append('the tuner did not run')
    if not placed(header):
        problems.append('frozen program header is '+header)
    if agent.get('llm_calls') != replies or report.get('llm_calls') != replies:
        problems.append('the session did not use every saved reply')
    if after != before:
        problems.append('the ledger changed during the replay')
    return problems


# ---------------------------------------------------------------- inner workers

def _header(path):
    return next((line.strip() for line in Path(path).read_text().splitlines()
                 if line.lstrip().startswith('@fhe.program(')), '')


def session_worker(args):
    from .r8e6_reconfirm import _limits
    _limits()
    import torch
    from ..run import _inside as run_inside
    torch.set_num_threads(1)
    namespace = run_namespace(args.out, args.model, args.models_dir, SESSION_STAGES, budget_path=args.budget_path,
                              limit=args.call_limit)
    client = None
    if args.replay:
        client = ReplayClient.from_trace(args.replay)
        namespace.llm_client, namespace.agent_iters = client, len(client.replies)
    code = run_inside(namespace)
    report = read(args.out/'REPORT.json')
    problems = []
    if report.get('placement') != PLACEMENT:
        problems.append('run.py did not apply the placement')
    if client is not None and client.used == 0:
        problems.append('run.py did not use the replay client')
    if problems:
        report.update(complete=False, error='; '.join(problems))
        write(args.out/'REPORT.json', report)
        return 1
    return code


def test_worker(args):
    from .r8e6_reconfirm import _limits
    _limits()
    import torch
    from ..run import _inside as run_inside
    torch.set_num_threads(1)
    code = run_inside(run_namespace(args.session, args.model, args.models_dir, ('test',)))
    report = read(args.session/'REPORT.json')
    write(args.out/'REPORT.json', dict({key: report.get(key) for key in TEST_KEYS}, session=str(args.session),
                                       complete=code == 0 and bool(report.get('complete')), error=report.get('error')))
    return code


def compile_worker(args):
    """Export, trace and compile one program-plan program once; record predicted, exported and compiled boots."""
    import gc
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (6*1024**3, 6*1024**3))
    from .. import toolchain
    from ..cost import estimate, release_estimate_caches
    from ..dsl.parser import parse
    from ..export import export_package
    from ..models.zoo import get
    from .u4_relative_acceptance import native_entry

    toolchain.setup_env()
    out, path = args.out, args.out/'REPORT.json'
    text = args.program.read_text()
    program = parse(text)
    result = dict(complete=False, model=args.model, source=str(args.program), header=_header(args.program),
                  hecate_opt=str(toolchain.host_paths()['hecate_opt']), limits=toolchain.TRUSTED_LIMITS,
                  api_calls=0, report_samples=0)
    write(path, result)
    if (program.header.refresh, program.header.placement) != ('agent', PLACEMENT):
        result['error'] = 'The program header is not refresh="agent", placement="'+PLACEMENT+'"'
        write(path, result)
        return 1
    (out/'program.ptdsl').write_text(text)
    weights = get(args.model, args.models_dir).import_graph().weights

    started = time.monotonic()
    predicted = estimate(program, weights)
    plan = predicted['refresh_plan']
    result.update(estimate=dict({key: predicted.get(key) for key in COST_KEYS}, seconds=time.monotonic()-started),
                  variant=plan.get('variant'), plan_calls=plan.get('calls'),
                  candidates={name: {key: entry.get(key) for key in ('boots', 'latency_us')}
                              for name, entry in (plan.get('candidates') or {}).items()})
    del predicted, plan
    release_estimate_caches()
    gc.collect()
    write(path, result)

    started = time.monotonic()
    directory = out/'export'
    package = export_package(text, program, weights, directory, run_info=dict(program_path=out/'program.ptdsl', run_dir=out))
    weights_path = directory/'weights'/(program.name+'.npz')
    saved = json.loads((package/'config/refresh_plan.json').read_text())
    result['export'] = dict(package=str(package), weights=str(weights_path), seconds=time.monotonic()-started,
                            boots=saved['boots'], calls=saved['calls'], variant=saved['placement'])
    del saved, weights
    gc.collect()
    write(path, result)

    try:
        result['trace'] = native_entry(package, weights_path, directory, 'compiler', 'trace', directory/'trace')
        write(path, result)
        result['compile'] = native_entry(package, weights_path, directory, 'compiler', 'compile', directory/'compiled',
                                         input_path=directory/'trace/trace_model.mlirbc')
    except RuntimeError as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
        for name in ('trace.log', 'compile.log'):
            if (directory/name).is_file():
                result[name.replace('.', '_')+'_tail'] = (directory/name).read_text(errors='replace')[-3000:]
        result['pruned'] = {name: toolchain.prune(directory/name) for name in ('trace', 'compiled') if (directory/name).is_dir()}
        write(path, result)
        return 1
    compiled = directory/'compiled'
    result['stats'] = toolchain.plan_stats(compiled, program.name)
    result['boots'] = dict(predicted=result['estimate']['boots_pred'], exported=result['export']['boots'],
                           compiled=result['stats']['boots'])
    result['three_way_equal'] = len(set(result['boots'].values())) == 1
    result['pruned'] = dict(trace=toolchain.prune(directory/'trace'), compiled=toolchain.prune(compiled))
    result['complete'] = True
    write(path, result)
    return 0


# ---------------------------------------------------------------- outer steps

def leftovers(out):
    """Processes outside this process's ancestry whose command line names this step's directory."""
    keep, pid = set(), os.getpid()
    while pid > 1:
        keep.add(pid)
        pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in keep:
            continue
        try:
            command = (entry/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if str(out) in command:
            found.append(dict(pid=int(entry.name), command=command[:300]))
    return found


def compiled(model, program, out, mode):
    """One compile child; a trusted child that outlives it is stopped and recorded."""
    result = e9a.child(compile_command(model, program, out), out, model=model, mode=mode)
    stopped = []
    for item in leftovers(out):
        try:
            os.kill(item['pid'], signal.SIGKILL)
            stopped.append(item)
        except ProcessLookupError:
            pass
    if stopped:
        result['leftover_processes_stopped'] = stopped
        write(out/'REPORT.json', result)
    return {key: result.get(key) for key in ('complete', 'error', 'exit', 'seconds', 'boots', 'three_way_equal',
                                              'variant', 'candidates', 'leftover_processes_stopped') if key in result}


def account(ledger, before, model, out, recorded, *, wait):
    """One registry row per new ledger row; returns the ledger length, or None while the ledger is locked."""
    from ..registry import append, git_rev, new_run_id
    calls = r8g._calls(ledger, wait=wait)
    if calls is None:
        return None
    for index in range(before+1, len(calls)+1):
        if index in recorded:
            continue
        call = calls[index-1]
        append(dict(run_id=new_run_id(ROUND, model, f'api-{index}'), round=ROUND, model=model, mode='llm-call',
                    git_rev=git_rev(), metrics=dict(call, global_call=index), artifacts_dir=str(out), ts=call['ts']))
        recorded.add(index)
    return len(calls)


def session_child(command, out, *, model, mode, ledger, before, recorded):
    """Run one session child, writing a registry row for each API call while it runs."""
    out.mkdir(parents=True, exist_ok=False)
    started, process, error, code = time.monotonic(), None, None, None
    print('E9_STEP='+model+' '+mode+' '+str(out), flush=True)
    try:
        with (out/'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                try:
                    code = process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    account(ledger, before, model, out, recorded, wait=False)
                    if time.monotonic()-started > SESSION_SECONDS+120:
                        raise TimeoutError('The session exceeded its Nix allowance')
    except (Exception, KeyboardInterrupt) as exc:
        error = type(exc).__name__+': '+str(exc)
        r8g._stop(process)
        code = process.returncode if process is not None else None
    after = account(ledger, before, model, out, recorded, wait=True)
    report = read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False, error='No child REPORT')
    frozen = (out/'program.ptdsl').exists() and bool(report.get('complete'))
    agent = report.get('agent') or {}
    iterations, tuner_seconds = r8g._trace_summary(out/'agent_trace.jsonl') if (out/'agent_trace.jsonl').exists() else ([], 0)
    result = dict({key: report.get(key) for key in SESSION_KEYS}, exit=code, seconds=time.monotonic()-started,
                  outcome=session_outcome(report, frozen), frozen=frozen, status=agent.get('status'),
                  agent_error=agent.get('error'), agent_seconds=agent.get('seconds'), iterations=agent.get('iterations'),
                  first_feasible_iteration=agent.get('first_feasible_iteration'), best=_plain(agent.get('best')),
                  history=agent.get('history'), confirm_evals=(agent.get('confirm') or {}).get('evals'),
                  trace=iterations, tuner_seconds=tuner_seconds, api_before=before, api_after=after,
                  api_calls=after-before, header=_header(out/'program.ptdsl') if frozen else None, driver_error=error)
    write(out/'SESSION.json', result)
    e9a.register(model, mode, out, {key: result[key] for key in (
        'complete', 'error', 'exit', 'seconds', 'outcome', 'status', 'api_calls', 'api_before', 'api_after',
        'iterations', 'best_iteration', 'frozen_source_lines', 'driver_error')})
    return result


def _plain(best):
    return None if best is None else {key: best.get(key) for key in (
        'boots_pred', 'latency_pred_us', 'max_depth', 'ops_total', 'iteration', 'source', 'agree', 'max_u')}


def recorded_calls():
    from ..registry import load
    return {row['metrics'].get('global_call') for row in load() if row['mode'] == 'llm-call'}


def smoke(root, ledger):
    """b0: the saved R8e5 MLP replies through the new path, then one compile. API 0."""
    folder = root/'b'/'smoke'
    folder.mkdir(parents=True, exist_ok=False)
    before = len(ledger_rows(ledger))
    replies = len(ReplayClient.from_trace(SMOKE_TRACE).replies)
    session = session_child(session_command(SMOKE_MODEL, folder/'session', ledger, before, replay=SMOKE_TRACE),
                            folder/'session', model=SMOKE_MODEL, mode='e9b-smoke-session', ledger=ledger,
                            before=before, recorded=recorded_calls())
    report = read(folder/'session/REPORT.json')
    trace = folder/'session/agent_trace.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()] if trace.exists() else []
    problems = smoke_problems(report, rows, session.get('header') or '', replies, before, len(ledger_rows(ledger)))
    result = dict(model=SMOKE_MODEL, trace=str(SMOKE_TRACE), replies=replies, session=session, problems=problems)
    if not problems:
        result['compile'] = compiled(SMOKE_MODEL, folder/'session/program.ptdsl', folder/'compile', 'e9b-smoke-compile')
        if not result['compile'].get('complete') or not result['compile'].get('three_way_equal'):
            problems.append('compile: '+str(result['compile'].get('error') or result['compile'].get('boots')))
    result.update(complete=not problems, api_after=len(ledger_rows(ledger)))
    return result


def rule_baseline(folder, model):
    """The E1 rule program with the same header rewrite and the same compile."""
    program = RESULTS_DIR/RULE_RUNS[model]/'program.ptdsl'
    header = e9a.child(header_command(program, folder/'rule'), folder/'rule', model=model, mode='e9b-'+model+'-rule-header')
    if header.get('exit') or not header.get('complete'):
        return dict(complete=False, program=str(program), error='header: '+str(header.get('error')))
    return dict(compiled(model, folder/'rule/program.ptdsl', folder/'rule-compile', 'e9b-'+model+'-rule-compile'),
                program=str(program))


def run_model(root, model, ledger, entry=None):
    """Rule baseline, session, compile and report test of one model; `entry` keeps what ran if the round stops."""
    folder = root/'b'/model
    folder.mkdir(parents=True, exist_ok=False)
    entry = dict(model=model) if entry is None else entry
    try:
        entry['rule'] = rule_baseline(folder, model)
    except Exception as exc:
        entry['rule'] = dict(complete=False, error=type(exc).__name__+': '+str(exc))
    before = len(ledger_rows(ledger))
    if before >= API_CEILING:
        raise RuntimeError('The ledger already holds '+str(before)+' rows; the approved ceiling is '+str(API_CEILING))
    session = folder/'session'
    entry['session'] = session_child(session_command(model, session, ledger, call_limit(before)), session, model=model,
                                     mode='e9b-'+model+'-session', ledger=ledger, before=before, recorded=recorded_calls())
    if entry['session']['api_after'] > API_CEILING or entry['session']['api_calls'] > CALLS_PER_MODEL:
        raise RuntimeError(model+': the session passed its call limit')
    if entry['session']['outcome'] in ('provider_error', 'failed') or entry['session']['driver_error']:
        raise RuntimeError(model+' session '+entry['session']['outcome']+': '+str(
            entry['session']['driver_error'] or entry['session']['agent_error'] or entry['session']['error']))
    if entry['session']['frozen']:
        entry['compile'] = compiled(model, session/'program.ptdsl', folder/'compile', 'e9b-'+model+'-compile')
        if entry['compile'].get('complete'):
            shutil.copyfile(session/'REPORT.json', session/'REPORT.session.json')
            test = e9a.child(test_command(model, session, folder/'test'), folder/'test', model=model, mode='e9b-'+model+'-test')
            entry['test'] = {key: test.get(key) for key in (*TEST_KEYS, 'complete', 'error', 'exit', 'seconds')}
    return entry


def _show(value, digits=4):
    return '—' if value is None else (f'{value:.{digits}g}' if isinstance(value, float) else str(value))


def _three(boots):
    boots = boots or {}
    return '/'.join(_show(boots.get(key)) for key in ('predicted', 'exported', 'compiled'))


def summarize(root, data, error=None):
    smoke_result = data.get('smoke') or {}
    lines = ['# E9b：MLP/CNN 在程序级放置下的新会话', '',
             f'提示词 {PROMPT_VERSION}，放置 {PLACEMENT}，每个模型至多 {CALLS_PER_MODEL} 次调用，账本上限 {API_CEILING}。'
             f'账本 {data.get("api_before")} → {data.get("api_after")}。异常：{error or "无"}。', '',
             f'b0 回放（API 0）：{"通过" if smoke_result.get("complete") else "未通过"}，'
             f'自举 {_three((smoke_result.get("compile") or {}).get("boots"))}，问题：{smoke_result.get("problems") or "无"}。', '',
             '| 模型 | 规则程序：depth-dp / program-plan（预测/导出/编译） | 会话：结果 / 调用 / 迭代 / 秒 | 冻结程序行数 | '
             '冻结程序：depth-dp / program-plan（预测/导出/编译） | 相对规则程序 | report 8000：top-1 / 明文 top-1 / 发散 |',
             '|---|---|---|---|---|---|---|']
    for model, entry in (data.get('models') or {}).items():
        rule, session = entry.get('rule') or {}, entry.get('session') or {}
        built, test = entry.get('compile') or {}, entry.get('test') or {}
        depth = lambda item: _show(((item.get('candidates') or {}).get('depth-dp') or {}).get('boots'))
        base, new = (rule.get('boots') or {}).get('compiled'), (built.get('boots') or {}).get('compiled')
        change = f'{100*(new-base)/base:+.1f}%' if base and new is not None else '—'
        lines.append(f'| {model} | {depth(rule)} / {_three(rule.get("boots"))} | '
                     f'{session.get("outcome")} / {session.get("api_calls")} / {session.get("iterations")} / {_show(session.get("seconds"))} | '
                     f'{_show(session.get("frozen_source_lines"))} | {depth(built)} / {_three(built.get("boots"))} | {change} | '
                     f'{_show(test.get("top1_test"))} / {_show(test.get("torch_top1_report"))} / {_show(test.get("n_diverged"))} |')
    (root/'E9b.md').write_text('\n'.join(lines)+'\n')
    return lines


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    path = root/'E9b.json'
    data = read(path) if path.exists() else dict(api_before=len(ledger_rows(ledger)), api_ceiling=API_CEILING, models={})
    error = None
    try:
        if 'smoke' not in data:
            data['smoke'] = smoke(root, ledger)
            data['api_after'] = len(ledger_rows(ledger))
            write(path, data)
        if not data['smoke'].get('complete'):
            raise RuntimeError('b0 replay: '+'; '.join(data['smoke'].get('problems') or ['incomplete']))
        for model in ORDER:
            if args.smoke_only or model in data['models'] or (args.models and model not in args.models):
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
        write(root/'STOP-b.json', dict(error=error))
        import traceback
        traceback.print_exc()
    data.update(api_after=len(ledger_rows(ledger)), error=error)
    wanted = [model for model in ORDER if not args.smoke_only and (not args.models or model in args.models)]
    data['complete'] = error is None and all(model in data['models'] for model in wanted)
    write(path, data)
    summarize(root, data, error)
    e9a.register('all', 'e9b-summary', root, dict(complete=data['complete'], error=error, api_before=data['api_before'],
                                                  api_after=data['api_after'], models=sorted(data['models'])))
    print('E9B_DONE='+str(path)+' ERROR='+str(error), flush=True)
    return int(bool(error) or data['api_after'] > API_CEILING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--models', nargs='+', choices=ORDER)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--worker', choices=('session', 'compile', 'test'))
    parser.add_argument('--model', choices=ORDER)
    parser.add_argument('--models-dir', type=Path, default=MODELS_DIR)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--program', type=Path)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--budget-path', type=Path)
    parser.add_argument('--call-limit', type=int)
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    if args.worker:
        needed = dict(session=('model', 'out', 'budget_path', 'call_limit'), compile=('model', 'out', 'program'),
                      test=('model', 'out', 'session'))[args.worker]
        if any(getattr(args, name) is None for name in needed):
            parser.error('--worker '+args.worker+' requires '+', '.join('--'+name.replace('_', '-') for name in needed))
        if args.worker == 'session' and not 0 <= args.call_limit <= API_CEILING:
            parser.error('--call-limit must stay within the approved ceiling')
        return dict(session=session_worker, compile=compile_worker, test=test_worker)[args.worker](args)
    if args.root is None:
        parser.error('--root is required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
