"""Faithful continuation of the retained, interrupted VGG Agent session.

The old case remains closed. A new case owns its clock, worker and provider
artifacts; old requests are copied as inherited evidence and never replayed.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .. import paths
from ..autonomous_budget import call_records, ensure_allowance
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'R30-CONTINUATION'
MODE = 'faithful-common-model-whole-session-continuation'
VARIANT = 'retained-hybrid-session-continuation'
MODEL = 'vgg16_silu_avgpool'
FACTORY = 'pt_agent.experiments.r13_common_models:' + MODEL
GIB = 1024**3
OPTIONS = dict(max_iters=6, tuner=True, tuner_budget_s=600, tune_policy='all',
    engine_profile='tensor', prompt_version='r8-v11', tune_rule='v8-6.25',
    tune_search='sweep', confirm_rule='c1', placement='program-plan',
    stale_limit=3, focus_retune=True)
PLANNER_FUNCTIONS = ('new_agent_state', '_evaluate_step', '_finish_iteration',
                     '_finish_tune', '_selected_record', 'agent_step')


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):raise ValueError('Expected an object source record')
    return value


def _same(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('A regular retained source file is required')
    return dict(path=str(path.resolve()), size_bytes=path.stat().st_size)


def validate_state(state, *, initial_messages=None):
    if (any(state.get(key) != value for key, value in OPTIONS.items())
            or state.get('numeric_lock', False) is not False
            or state.get('phase') != 'tune' or state.get('iteration') != 4
            or state.get('calls') != 4 or state.get('failed') is not None
            or state.get('confirmation') is not None):
        raise ValueError('Retained session must preserve the original four-call pending tune and all options')
    history, pending = state.get('history'), state.get('pending') or {}
    messages = state.get('messages')
    if (not isinstance(history, list) or [row.get('iteration') for row in history] != [1, 2, 3]
            or not isinstance(messages, list) or len(messages) != 8
            or [row.get('role') for row in messages] != ['system', 'user'] + ['assistant', 'user'] * 3
            or any(not isinstance(row.get('content'), str) for row in messages)
            or pending.get('iteration') != 4 or pending.get('tuned') is not None
            or not isinstance(pending.get('source'), str)
            or pending.get('row', {}).get('stage') != 'complete'
            or pending.get('row', {}).get('feasible') is not True
            or not isinstance(pending.get('reply'), dict)
            or len(pending.get('candidates') or []) != 1
            or pending['candidates'][0][3] != 'llm'):
        raise ValueError('Pending state, provider conversation or candidate provenance differs')
    if initial_messages is not None and messages[:2] != initial_messages:
        raise ValueError('Reconstructed original context differs from the retained initial messages')
    return dict(initial_calls=4, maximum_total_calls=6, remaining_calls=2,
                next_step='tune', next_step_requires_API=False, tuner=True)


def check_source(case, *, ledger=None):
    """Read public/session metadata only; no model, samples or weight archive."""
    from ..planners.transformer_session import load_state
    case = Path(case).resolve()
    original, completion = _read(case/'result.json'), _read(case/'completion.json')
    agent_dir = case/'work/agent'
    state = load_state(agent_dir/'SESSION.json')
    summary = validate_state(state)
    worker, phase = _read(agent_dir/'worker-result.json'), _read(agent_dir/'phase_result.json')
    if (original.get('round') != 'R30' or original.get('model') != MODEL
            or original.get('status') != 'error' or original.get('api_delta') != 4
            or original.get('variant') != 'llm-first-generation+generic-deterministic-tuning'
            or original.get('attempted') != dict(intake=1, calibration=1, agent=1)
            or original.get('completed') != dict(intake=1, calibration=1)
            or completion.get('status') != 'error'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True
            or worker.get('model') != MODEL or worker.get('status') != 'running'
            or worker.get('frozen') is not False or worker.get('strict_math_probes') != []
            or any(worker.get(key) != 0 for key in ('report_samples', 'compiler_runs', 'real_ckks_runs'))
            or phase.get('phase') != 'agent' or phase.get('status') != 'running'):
        raise ValueError('Source must be the closed interrupted whole R30 VGG case before confirmation/compilation')
    model_info = case/'work/intake/model_info/model_info.json'
    calibration = case/'work/calibration/calibration.json'
    if not _same(worker.get('source_model_info'), model_info) or not _same(worker.get('source_calibration'), calibration):
        raise ValueError('Retained original intake/calibration paths differ')
    info = _read(model_info)
    if (info.get('source', {}).get('factory') != FACTORY
            or info.get('threshold') != {'acc_drop_pp': .5}
            or info.get('split', {}).get('kind') != 'legacy_cifar'):
        raise ValueError('Original model, quality policy or fixed split differs')
    _regular(calibration)
    records = []
    requests = sorted((agent_dir/'api').glob('*.request.json'))
    responses = sorted((agent_dir/'api').glob('*.response.json'))
    if len(requests) != 4 or len(responses) != 4 or list((agent_dir/'api').glob('*.error.json')):
        raise ValueError('Exactly four completed original provider requests/responses are required')
    traces = [json.loads(row) for row in (agent_dir/'agent_trace.jsonl').read_text().splitlines() if row.strip()]
    if [row.get('iteration') for row in traces] != [1, 2, 3]:raise ValueError('Retained completed Agent trace differs')
    prompt_tokens = completion_tokens = 0
    for index, request_path in enumerate(requests, 1):
        response_path = request_path.with_name(request_path.name.removesuffix('.request.json')+'.response.json')
        request, response = _read(request_path), _read(response_path)
        if (request.get('purpose') != 'agent-iter-'+str(index)
                or request.get('messages') != state['messages'][:2+2*(index-1)]):
            raise ValueError('Provider request history differs from the retained conversation')
        if index < 4:
            expected = dict(role='assistant', content=response.get('content'))
            if (state['messages'][2+2*(index-1)] != expected
                    or traces[index-1].get('messages') != state['messages'][2*index:2*index+2]):
                raise ValueError('Completed provider replies differ from the retained history')
        elif response != state['pending']['reply']:
            raise ValueError('Fourth original response differs from the retained pending reply')
        raw = json.loads(response['content']).get('program')
        raw_path = agent_dir/'sources'/('iteration-'+str(index)+'.ptdsl')
        if not isinstance(raw, str) or raw_path.read_text() != raw:
            raise ValueError('Original response program differs from its retained source')
        usage = response.get('usage') or {}
        prompt_tokens += usage.get('prompt_tokens', 0)
        completion_tokens += usage.get('completion_tokens', 0)
        records.append(dict(iteration=index, request=_regular(request_path), response=_regular(response_path),
                            source=_regular(raw_path)))
    if prompt_tokens != state.get('prompt_tokens') or completion_tokens != state.get('completion_tokens'):
        raise ValueError('Retained provider token totals differ')
    if ledger is not None:
        rows = call_records(ledger)
        before, after = original.get('api_before'), original.get('api_after')
        if (type(before) is not int or type(after) is not int or after-before != 4 or len(rows) < after
                or any(row.get('status') != 200 or row.get('error') is not None
                       or row.get('purpose') != 'agent-iter-'+str(index)
                       for index, row in enumerate(rows[before:after], 1))):
            raise ValueError('Existing global ledger does not retain the original four successful calls')
    startup = case/'source/pt_agent/planners/agent.py'
    current = Path(__file__).parents[1]/'planners/agent.py'
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in PLANNER_FUNCTIONS}
    if functions(startup) != functions(current) or set(functions(startup)) != set(PLANNER_FUNCTIONS):
        raise ValueError('Trusted Agent continuation state semantics differ from the startup source')
    summary.update(original_case=str(case), original_session=str(agent_dir/'SESSION.json'),
        original_complete_seconds=completion['complete_case_elapsed_seconds'],
        model=MODEL, model_info=str(model_info), calibration=str(calibration),
        response_records=records, provider_history_verified=True, planner_state_function_AST_equal=True,
        original_phase='interrupted_agent', cold_start=False,
        original_time_scope='Closed failed case; preserved separately from the new continuation clock')
    return state, summary


def restore_for_worker(args):
    """The R28 worker calls this before loading the retained original context."""
    from ..planners.transformer_session import load_state
    state, summary = check_source(args.resume_case, ledger=args.ledger)
    copied = load_state(args.resume_session)
    if copied != state:
        raise ValueError('Copied continuation SESSION differs semantically from the retained original state')
    if (args.model != MODEL or not _same(args.model_info, summary['model_info'])
            or not _same(args.calibration, summary['calibration'])
            or args.calls != 6 or args.tuner_seconds != 600 or getattr(args, 'tuner', True) is not True
            or args.prepare_only or args.resume_source is not None):
        raise ValueError('Continuation worker must retain the original model/context/options')
    return copied, summary


def verify_context(state, ctx):
    from ..planners.prompts import build_system_prompt, initial_message
    return validate_state(state, initial_messages=[dict(role='system', content=build_system_prompt(ctx)),
                                                  dict(role='user', content=initial_message(ctx))])


def initialize_worker(args, work, result, *, save):
    """Restore through the public serializer before the original context loads."""
    from ..planners.transformer_session import save_state
    state, source = restore_for_worker(args)
    work, inherited = Path(work), Path(args.resume_session).parent
    if any((work/name).exists() for name in ('sources', 'agent_trace.jsonl', 'SESSION.json', 'api')):
        raise ValueError('Continuation worker requires fresh evidence paths')
    shutil.copytree(inherited/'sources', work/'sources')
    shutil.copy2(inherited/'agent_trace.jsonl', work/'agent_trace.jsonl')
    save_state(work/'SESSION.json', state)
    result.update(recovered_api_calls=4, actual_api_calls=0,
        recovery=dict(kind='retained_whole_session', source_case=source['original_case'],
            source_session=str(args.resume_session), provider_history_verified=True,
            inherited_provider_artifacts=str(inherited/'api'), initial_phase='tune',
            original_calls=4, remaining_calls=2, prior_elapsed_s=state['elapsed_s'],
            clock_scope='New continuation worker; inherited elapsed state is recorded separately'))
    save()
    return state


def preserve_session(case, destination, state):
    """New inherited-evidence directory; original provider records stay separate."""
    from ..planners.transformer_session import load_state, save_state
    source, destination = Path(case)/'work/agent', Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('api', 'sources'):
        shutil.copytree(source/name, destination/name)
    for name in ('agent_trace.jsonl', 'events.jsonl', 'worker-result.json', 'phase_result.json'):
        shutil.copy2(source/name, destination/name)
    save_state(destination/'SESSION.json', state)
    if load_state(destination/'SESSION.json') != state:
        raise ValueError('SESSION serialization did not preserve the full state')
    return destination/'SESSION.json'


def _clean(record):
    return (record.get('exit_code') == 0 and record.get('timed_out') is False and record.get('signal') is None
            and record.get('signals_sent') == [] and record.get('remaining_pids') == [])


def check_worker(worker, session, source, new_calls, *, agent_dir):
    if (worker.get('model') != MODEL or worker.get('tuner_enabled') is not True
            or worker.get('recovered_api_calls') != 4 or worker.get('actual_api_calls') != new_calls
            or not 0 <= new_calls <= 2 or session.get('phase') != 'done'
            or session.get('calls') != 4+new_calls or session.get('max_iters') != 6
            or any(session.get(key) != value for key, value in OPTIONS.items())
            or session.get('failed') is not None
            or not _same(worker.get('source_model_info'), source['model_info'])
            or not _same(worker.get('source_calibration'), source['calibration'])
            or any(worker.get(name) != 0 for name in ('report_samples', 'compiler_runs', 'real_ckks_runs'))):
        raise ValueError('Continuation accounting/context or terminal protocol differs')
    agent=worker.get('agent') or {}
    from .r30_continue_alex_session import same_state
    def confirmation_json(value):
        if isinstance(value,dict):return {key:confirmation_json(item)for key,item in value.items()}
        if isinstance(value,(list,tuple)):return [confirmation_json(item)for item in value]
        return value
    if (agent.get('llm_calls') != session['calls']
            or not same_state(confirmation_json(agent.get('confirm')),confirmation_json(session.get('confirmation')))):
        raise ValueError('Worker terminal result differs from the restored session')
    if worker.get('status') == 'quality_rejected' and worker.get('frozen') is False:
        return 'quality_rejected'
    if (worker.get('status') != 'ok' or worker.get('frozen') is not True
            or worker.get('quality_acceptance', {}).get('passed') is not True
            or len(worker.get('strict_math_probes', [])) != 3):
        raise ValueError('Continuation did not retain the original frozen quality and mathematics gates')
    selected=(agent.get('confirm') or {}).get('selected')
    if not selected:raise ValueError('Frozen continuation requires a confirmed selection')
    rows=[row for row in agent['confirm'].get('evals', [])
          if (row.get('iteration'), row.get('origin')) == (selected['iteration'], selected['origin'])]
    if len(rows) != 1:raise ValueError('Frozen selection requires one matching confirmation row')
    row=rows[0]
    if (row.get('passed') is not True or row.get('diverged') is not False
            or row.get('n_diverged') != 0 or row.get('total') != 1000
            or not all(math.isfinite(row[key]) for key in ('acc_drop_pp', 'max_u_calibrated', 'max_u_analytic'))
            or row['acc_drop_pp'] > .5 or row['max_u_calibrated'] > .90 or row['max_u_analytic'] > 1):
        raise ValueError('Confirmation differs from the original quality policy')
    for index, probe in enumerate(worker['strict_math_probes']):
        if probe.get('sample_id') != 'test:'+str(9000+index):raise ValueError('Strict probe identity differs')
        for role in ('exact', 'approximate'):
            check=probe.get(role) or {}
            if (any(check.get(key) is not True for key in ('passed', 'finite', 'dtype_ok', 'shape_ok'))
                    or check.get('criterion') != '1e-12+1e-12*abs(reference)'
                    or not math.isfinite(check['max_error_ratio']) or check['max_error_ratio'] > 1):
                raise ValueError('Strict independent arithmetic evidence differs')
    program=Path(worker['selected_program']).resolve()
    if not program.is_relative_to(Path(agent_dir).resolve()) or not program.is_file():
        raise ValueError('Frozen source must be saved in the new continuation worker')
    identity=lambda text:ast.dump(ast.parse(text), include_attributes=False)
    if (identity(program.read_text()) != identity(selected['source'])
            or identity(agent['best_program']) != identity(selected['source'])
            or identity(session['best_program']) != identity(selected['source'])):
        raise ValueError('Frozen source differs from the confirmed session selection')
    return 'ok'


def check_provider_history(initial, current, agent_dir, new_calls):
    """Check inherited conversation and every new successful provider artifact."""
    agent_dir=Path(agent_dir)
    requests=sorted((agent_dir/'api').glob('*.request.json'))
    responses=list((agent_dir/'api').glob('*.response.json'))
    if (len(requests) != new_calls or len(responses) != new_calls
            or list((agent_dir/'api').glob('*.error.json'))
            or current.get('history', [])[:3] != initial['history']
            or current.get('messages', [])[:8] != initial['messages']
            or len(current.get('messages', [])) != 10+2*new_calls
            or current['messages'][8] != dict(role='assistant', content=initial['pending']['reply']['content'])):
        raise ValueError('Continuation must retain the original history and all new provider artifacts')
    prompt, completion=initial['prompt_tokens'], initial['completion_tokens']
    for offset, request_path in enumerate(requests):
        response_path=request_path.with_name(request_path.name.removesuffix('.request.json')+'.response.json')
        request, response=_read(request_path), _read(response_path)
        if (request.get('purpose') != 'agent-iter-'+str(5+offset)
                or request.get('messages') != current['messages'][:10+2*offset]
                or current['messages'][10+2*offset] != dict(role='assistant', content=response.get('content'))):
            raise ValueError('New provider request/reply differs from the continued conversation')
        try:
            payload=json.loads(response['content'])
            raw=payload.get('program') if isinstance(payload, dict) else None
        except (TypeError, ValueError):raw=None
        raw_path=agent_dir/'sources'/('iteration-'+str(5+offset)+'.ptdsl')
        if isinstance(raw, str):
            if raw_path.read_text() != raw:raise ValueError('New response source differs from its retained program')
        else:
            history=current['history'][4+offset]
            if raw_path.exists() or history.get('stage') not in ('response', 'length') or history.get('feasible') is not False:
                raise ValueError('Malformed provider reply lacks its original Agent rejection record')
        usage=response.get('usage') or {}
        prompt+=usage.get('prompt_tokens', 0);completion+=usage.get('completion_tokens', 0)
    if current.get('prompt_tokens') != prompt or current.get('completion_tokens') != completion:
        raise ValueError('Continued provider token totals differ')
    traces=[json.loads(line) for line in (agent_dir/'agent_trace.jsonl').read_text().splitlines() if line.strip()]
    if [row.get('iteration') for row in traces] != list(range(1, 5+new_calls)):
        raise ValueError('Continued trace must preserve the full iteration sequence')
    return dict(inherited_successful_calls=4, actual_new_calls=new_calls,
                complete_provider_conversation=True, old_ledger_records_recreated=False)


def execute(args):
    started = time.perf_counter_ns()
    out = Path(args.out).resolve()
    if (not out.is_relative_to(paths.RESULTS_DIR.resolve()) or out == paths.RESULTS_DIR.resolve()
            or out.exists() or out.is_relative_to(Path(args.source_case).resolve())):
        raise ValueError('Continuation requires a new result directory separate from its original case')
    out.mkdir(parents=True);(out/'work').mkdir()
    result = dict(round=ROUND, model=MODEL, mode=MODE, variant=VARIANT, status='running', error=None,
        cold_start=False, tuner_enabled=True, worker_attempts=0, worker_completed=0, worker=None,
        original_api_calls=4, actual_new_api_calls=None, original_case=str(Path(args.source_case).resolve()),
        historical_elapsed_reused=False, compiler_runs=0, real_ckks_runs=0, report_samples=0,
        predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None, security_acceptance=None)
    recorder = StageRecorder(out/'events.jsonl');stop = threading.Event();failures = []
    before = len(call_records(args.ledger));result['api_before'] = before
    def checkpoint():
        if artifact_bytes(out) > args.case_gib*GIB or shutil.disk_usage(out).free < 10*GIB:
            raise RuntimeError('Continuation artifact allowance or 10GiB disk reserve exhausted')
        if failures:raise RuntimeError(failures[0])
    def monitor():
        while not stop.wait(1):
            try:checkpoint()
            except BaseException as exc:
                failures.append(type(exc).__name__+': '+str(exc));os.kill(os.getpid(), signal.SIGTERM);return
    watcher = threading.Thread(target=monitor, daemon=True)
    worker_parent=None
    try:
        with recorder.span('source.continuation_gate'):
            state, source = check_source(args.source_case, ledger=args.ledger)
            result['source'] = source
            checkpoint()
            retained = preserve_session(args.source_case, out/'inherited-agent', state)
            snapshots = out/'source';snapshots.mkdir()
            for name in ('r30_continue_common_session.py', 'r28_common_agent.py'):
                shutil.copy2(Path(__file__).with_name(name), snapshots/name)
        if args.prepare_only:
            result['status'] = 'prepared'
        else:
            ensure_allowance(args.allowance, args.ledger, required=2)
            command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(args.seconds+60),
                '--python', str(args.python), '--', '-B', '-m', 'pt_agent.experiments.r28_common_agent', '--worker',
                '--model', MODEL, '--work-dir', str(out/'work/agent'), '--model-info', source['model_info'],
                '--calibration', source['calibration'], '--ledger', str(args.ledger), '--allowance', str(args.allowance),
                '--calls', '6', '--tuner-seconds', '600', '--memory-gib', str(args.memory_gib),
                '--resume-session', str(retained), '--resume-case', str(Path(args.source_case).resolve())]
            result['worker_argv'] = command
            env = {k:v for k,v in os.environ.items() if not k.startswith(('LD_', 'NIX_', '__NIX_', 'HECATE_'))
                   and k not in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
            env['PYTHONDONTWRITEBYTECODE'] = '1';env['PYTHONNOUSERSITE'] = '1'
            watcher.start()
            with recorder.span('continuation.agent') as worker_parent:
                result['worker_attempts'] = 1;write_json(out/'result.json', result)
                observation = run_process(command, cwd=paths.REPO/'DSL/DSL_agent/scripts', log=out/'worker.log',
                    resources=out/'resources.jsonl', timeout=args.seconds+120, mem_gib=args.memory_gib, fsize_gib=8, env=env)
                result['process'] = observation
                target = out/'work/agent/worker-result.json'
                result['worker'] = _read(target) if target.is_file() else None
                if not _clean(observation):raise RuntimeError('Continuation worker interrupted; partial state retained')
                from ..planners.transformer_session import load_state
                current = load_state(out/'work/agent/SESSION.json')
                new_calls = len(call_records(args.ledger))-before
                result['provider_history'] = check_provider_history(state, current, out/'work/agent', new_calls)
                result['status'] = check_worker(result['worker'] or {}, current, source, new_calls, agent_dir=out/'work/agent')
                result['worker_completed'] = 1
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        target = out/'work/agent/worker-result.json'
        if result['worker'] is None and target.is_file():
            try:result['worker'] = _read(target)
            except (OSError, ValueError) as exc:
                result.update(status='error', error='Partial worker record unreadable: '+str(exc)[:2000])
        result['api_after'] = len(call_records(args.ledger))
        result['actual_new_api_calls'] = result['api_after']-before
        result['resource_monitor_error'] = failures[0] if failures else None
        if failures:result.update(status='error', error='Resource monitor: '+failures[0])
        result['original_time_seconds'] = (result.get('source') or {}).get('original_complete_seconds')
        result['elapsed_scope'] = 'New continuation only; prior failed case time and inherited state elapsed remain separate'
        child_events=read_events(out/'work/agent/events.jsonl')
        if worker_parent is not None:
            for event in child_events:
                if event.get('parent_id') is None:event['parent_id']=worker_parent['event_id']
        timing = summarize_events(read_events(out/'events.jsonl')+child_events, time.perf_counter_ns())
        result['timing'] = timing;write_json(out/'result.json', result)
        metrics = {key:result[key] for key in ('status', 'variant', 'cold_start', 'original_api_calls',
                    'actual_new_api_calls', 'worker_attempts', 'worker_completed', 'original_time_seconds')}
        registry = Path(args.registry or paths.RESULTS_DIR/'registry.jsonl');registry.parent.mkdir(parents=True, exist_ok=True)
        with registry.open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX);stream.seek(0)
            if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
                raise ValueError('Preserve existing continuation registry run')
            stream.seek(0, os.SEEK_END)
            stream.write(json.dumps(dict(run_id=out.name, round=ROUND, model=MODEL, mode=MODE,
                git_rev=None, metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))+'\n')
            stream.flush();os.fsync(stream.fileno())
        persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                                timing=timing, required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-case', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, default=paths.RESULTS_DIR/'llm_calls.jsonl')
    parser.add_argument('--allowance', type=Path, default=paths.RESULTS_DIR/'autonomous-api-allowance.json')
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--python', type=Path, default=paths.INTAKE_PYTHON)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--seconds', type=int, default=7200)
    parser.add_argument('--memory-gib', type=int, default=12)
    parser.add_argument('--case-gib', type=int, default=4)
    args = parser.parse_args(argv)
    if not 0 < args.seconds <= 10800 or not 4 <= args.memory_gib <= 12 or not 0 < args.case_gib <= 4:
        parser.error('Continuation limits are finite:10800seconds, memory4..12GiB, artifacts<=4GiB')
    result = execute(args)
    return 0 if result['status'] in ('prepared', 'ok', 'quality_rejected') else 1


if __name__ == '__main__':raise SystemExit(main())
