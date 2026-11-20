"""Continue one closed original Alex R30 SESSION, with its six-call policy.

The original case is retained. This entry stops after the original one-time
confirmation and strict packed mathematics; post-freeze compilation is separate.
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
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from .r31_lattigo_runtime_plan import physical_memory_observation
from .r30_continue_common_session import OPTIONS, PLANNER_FUNCTIONS, _read, _same, _regular

ROUND = 'R30-ALEX-CONTINUATION'
MODE = 'faithful-alex-whole-session-to-freeze-and-strict-math'
MODEL = 'alexnet_silu_avgpool'
FACTORY = 'pt_agent.experiments.r13_common_models:' + MODEL
MODULE = 'pt_agent.experiments.r30_continue_alex_session_worker'
GIB, RESERVE = 1024**3, 1024**2
COMMON_FUNCTIONS = ('worker', 'agent_options', 'image_inputs', 'single_image_input',
                    'final_quality_acceptance', 'observed_tools')


def same_state(left,right):
    """Compare saved state semantics, preserving nonfinite diagnostics exactly."""
    if type(left)is not type(right):return False
    if isinstance(left,dict):
        return left.keys()==right.keys()and all(same_state(value,right[key])for key,value in left.items())
    if isinstance(left,(list,tuple)):
        return len(left)==len(right)and all(same_state(a,b)for a,b in zip(left,right))
    if isinstance(left,float)and math.isnan(left):return math.isnan(right)
    return left==right


def _functions(path, names):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(Path(path).read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names}


def validate_state(state, *, initial_messages=None):
    """Accept the original unconfirmed evaluate/tune checkpoint, never reset it."""
    phase, calls, iteration = state.get('phase'), state.get('calls'), state.get('iteration')
    if (any(state.get(key) != value for key, value in OPTIONS.items())
            or state.get('numeric_lock', False) is not False
            or phase not in ('evaluate', 'tune') or type(calls) is not int
            or not 1 <= calls <= 6 or type(iteration) is not int
            or iteration != calls + (phase == 'evaluate')
            or (phase == 'evaluate' and calls == 6)
            or state.get('failed') is not None or state.get('confirmation') is not None):
        raise ValueError('Keep the original Alex six-call, tuner600, unconfirmed SESSION policy')
    completed = calls - (phase == 'tune')
    messages, history = state.get('messages'), state.get('history')
    if (not isinstance(history, list)
            or [row.get('iteration') for row in history] != list(range(1, completed+1))
            or not isinstance(messages, list) or len(messages) != 2+2*completed
            or [row.get('role') for row in messages] != ['system', 'user'] + ['assistant', 'user']*completed
            or any(not isinstance(row.get('content'), str) for row in messages)
            or not isinstance(state.get('tune_cache'), dict)
            or any(type(state.get(key)) is not int or state[key] < 0
                   for key in ('prompt_tokens', 'completion_tokens'))):
        raise ValueError('Retain the complete original conversation, history, cache and usage')
    pending = state.get('pending')
    if phase == 'tune':
        if (not isinstance(pending, dict) or pending.get('iteration') != calls
                or pending.get('tuned') is not None or not isinstance(pending.get('source'), str)
                or not isinstance(pending.get('reply'), dict)
                or pending.get('row', {}).get('stage') != 'complete'
                or pending.get('row', {}).get('feasible') is not True
                or len(pending.get('candidates') or []) != 1
                or pending['candidates'][0][3] != 'llm'):
            raise ValueError('The saved pending tune must retain its evaluated original reply')
    elif pending is not None:
        raise ValueError('An evaluate checkpoint must have no pending tune')
    if initial_messages is not None and messages[:2] != initial_messages:
        raise ValueError('Reconstructed original context differs from its saved initial messages')
    return dict(initial_calls=calls, completed_iterations=completed, maximum_total_calls=6,
                remaining_calls=6-calls, next_step=phase, next_step_requires_API=phase == 'evaluate',
                tuner=True, prior_state_elapsed_s=state.get('elapsed_s'))


def _reply_program(response):
    try:
        payload = json.loads(response['content'])
        return payload.get('program') if isinstance(payload, dict) else None
    except (KeyError, TypeError, ValueError):
        return None


def _usage(response, ledger_row=None):
    usage = response.get('usage') or {}
    result = []
    for key in ('prompt_tokens', 'completion_tokens'):
        value = usage.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError('Successful provider usage must retain nonnegative integer counts')
        if ledger_row is not None and ledger_row.get(key) != usage.get(key):
            raise ValueError('Response usage differs from its actual ledger record')
        result.append(value)
    seconds = response.get('seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Successful response duration must remain finite')
    if ledger_row is not None and (ledger_row.get('seconds') != seconds
            or ledger_row.get('finish_reason') != response.get('finish_reason')):
        raise ValueError('Response duration or finish reason differs from the retained ledger')
    return result


def _provider_records(directory, state, summary, ledger_rows=None):
    directory = Path(directory); calls = summary['initial_calls']; completed = summary['completed_iterations']
    requests = sorted((directory/'api').glob('*.request.json'))
    responses = sorted((directory/'api').glob('*.response.json'))
    if len(requests) != calls or len(responses) != calls or list((directory/'api').glob('*.error.json')):
        raise ValueError('Every original call must have one successful request and response')
    traces = [json.loads(line) for line in (directory/'agent_trace.jsonl').read_text().splitlines() if line.strip()]
    if [row.get('iteration') for row in traces] != list(range(1, completed+1)):
        raise ValueError('Completed original trace iterations differ from SESSION')
    tokens = [0, 0]; records = []
    for number, request_path in enumerate(requests, 1):
        response_path = request_path.with_name(request_path.name.removesuffix('.request.json')+'.response.json')
        request, response = _read(request_path), _read(response_path)
        if request.get('purpose') != 'agent-iter-'+str(number) or request.get('messages') != state['messages'][:2*number]:
            raise ValueError('Original provider request differs from the retained conversation')
        if number <= completed:
            if (state['messages'][2*number] != dict(role='assistant', content=response.get('content'))
                    or traces[number-1].get('messages') != state['messages'][2*number:2*number+2]):
                raise ValueError('Original reply or completed feedback differs from SESSION')
        elif response != state['pending']['reply']:
            raise ValueError('Pending original reply differs from its saved complete response')
        raw = _reply_program(response); raw_path = directory/'sources'/('iteration-'+str(number)+'.ptdsl')
        if isinstance(raw, str):
            if raw_path.read_text() != raw:raise ValueError('Retained raw program differs from its response')
        elif (number > completed or raw_path.exists()
              or state['history'][number-1].get('stage') not in ('response', 'length')):
            raise ValueError('Malformed reply lacks its original rejection history')
        ledger_row = ledger_rows[number-1] if ledger_rows is not None else None
        used = _usage(response, ledger_row)
        tokens = [a+b for a, b in zip(tokens, used)]
        records.append(dict(iteration=number, request=_regular(request_path), response=_regular(response_path)))
    if tokens != [state['prompt_tokens'], state['completion_tokens']]:
        raise ValueError('Original SESSION token totals differ from all retained successful replies')
    return records


def check_source(case, *, ledger):
    """Reject a live case before reading SESSION or any model/calibration payload."""
    case = Path(case).resolve()
    if not (case/'result.json').is_file():
        raise ValueError('Continuation requires an already closed result; live or missing cases are refused')
    original = _read(case/'result.json')
    if original.get('status') != 'error':
        raise ValueError('Continuation requires an already closed error case; running sources are refused')
    if not (case/'completion.json').is_file():
        raise ValueError('Continuation requires an already closed durable completion endpoint')
    completion = _read(case/'completion.json')
    if (completion.get('status') != 'error' or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True
            or original.get('round') != 'R30' or original.get('model') != MODEL
            or original.get('variant') != 'llm-first-generation+generic-deterministic-tuning'
            or original.get('tuner_enabled') is not True
            or original.get('attempted') != dict(intake=1, calibration=1, agent=1)
            or original.get('completed') != dict(intake=1, calibration=1)
            or any(original.get(key) != 0 for key in ('report_samples', 'keygen_runs', 'real_ckks_runs'))):
        raise ValueError('Closed original Alex R30 must stop before confirmation and post-freeze phases')
    processes = original.get('processes')
    if (not isinstance(processes, list) or [row.get('phase') for row in processes] != ['intake','calibration','agent']
            or any(not isinstance(row.get('observation'), dict)
                   or row['observation'].get('remaining_pids') != [] for row in processes)):
        raise ValueError('All original phase process trees must have a saved empty remaining_pids list')
    if any(row['observation'].get('exit_code')!=0 or row['observation'].get('timed_out')is not False
           or row['observation'].get('signal')is not None or row['observation'].get('signals_sent')!=[]
           for row in processes[:2]):
        raise ValueError('Original intake and calibration processes must have completed cleanly')
    from ..planners.transformer_session import load_state
    agent_dir = case/'work/agent'; state = load_state(agent_dir/'SESSION.json'); summary = validate_state(state)
    worker, phase = _read(agent_dir/'worker-result.json'), _read(agent_dir/'phase_result.json')
    if (worker.get('model') != MODEL or worker.get('status') not in ('running','error')
            or worker.get('frozen') is not False or worker.get('agent') is not None
            or worker.get('strict_math_probes') != []
            or any(worker.get(key) != 0 for key in ('report_samples','compiler_runs','real_ckks_runs'))
            or phase.get('phase') != 'agent' or phase.get('status') not in ('running','error')
            or (agent_dir/'program.ptdsl').exists() or (agent_dir/'FROZEN.json').exists()):
        raise ValueError('Original worker must be unconfirmed and unfrozen')
    info_path = case/'work/intake/model_info/model_info.json'; calibration = case/'work/calibration/calibration.json'
    if not _same(worker.get('source_model_info'), info_path) or not _same(worker.get('source_calibration'), calibration):
        raise ValueError('Original intake or calibration path differs')
    info = _read(info_path); calibration_phase = _read(case/'work/calibration/phase_result.json')
    calibration_source = _read(case/'work/calibration/calibration-source.json')
    if (info.get('source',{}).get('factory') != FACTORY or info.get('threshold') != {'acc_drop_pp':.5}
            or info.get('split',{}).get('kind') != 'legacy_cifar'
            or calibration_phase.get('calibration_completed') is not True
            or calibration_phase.get('calibration_samples') != 1000
            or calibration_phase.get('calibration_batch') != 1
            or calibration_source.get('count') != 1000 or calibration_source.get('split') != 'train'
            or len(calibration_source.get('indices',[])) != 1000
            or any(type(index)is not int or not 0<=index<50000 for index in calibration_source.get('indices',[]))
            or len(set(calibration_source.get('indices',[]))) != 1000
            or calibration_source.get('normalization') != info['source'].get('normalization')):
        raise ValueError('Retain the original model,1000-training calibration and fixed CIFAR policy')
    _regular(calibration)
    before, after = original.get('api_before'), original.get('api_after'); rows = call_records(ledger)
    if (type(before) is not int or type(after) is not int or before < 0 or after-before != summary['initial_calls']
            or original.get('api_delta') != summary['initial_calls'] or after > len(rows)
            or any(row.get('status') != 200 or row.get('error') is not None
                   or row.get('purpose') != 'agent-iter-'+str(index)
                   for index, row in enumerate(rows[before:after],1))):
        raise ValueError('Original calls must retain their actual successful ledger interval')
    records = _provider_records(agent_dir, state, summary, rows[before:after])
    package = Path(__file__).parents[1]
    for relative, names in (('planners/agent.py',PLANNER_FUNCTIONS),
                            ('experiments/r28_common_agent.py',COMMON_FUNCTIONS)):
        startup, current = case/'source/pt_agent'/relative, package/relative
        saved, actual = _functions(startup,names), _functions(current,names)
        if set(saved) != set(names) or saved != actual:
            raise ValueError('Original trusted worker/planner state semantics changed: '+relative)
    seconds = completion.get('complete_case_elapsed_seconds')
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Retain a finite original completion clock')
    summary.update(model=MODEL, original_case=str(case), original_session=str(agent_dir/'SESSION.json'),
        model_info=str(info_path), calibration=str(calibration), calibration_samples=1000,
        metadata_template=str(case/'source/compile-metadata-template.json'),
        original_complete_seconds=seconds, original_status=original['status'], original_error=original.get('error'),
        original_clock={key:completion.get(key)for key in ('started_ns','ended_ns','complete_case_elapsed_ns','complete_case_elapsed_seconds')},
        original_agent_termination={key:processes[-1]['observation'].get(key)
            for key in ('exit_code','timed_out','signal','signals_sent','remaining_pids')},
        original_phase_record=phase, original_processes=processes, original_monitor_error=original.get('monitor_error'),
        original_api_before=before, original_api_after=after, response_records=records,
        provider_history_verified=True, trusted_source_AST_equal=True, cold_start=False,
        original_time_scope='Closed failed original case; never included in the new continuation clock')
    _regular(summary['metadata_template'])
    return state, summary


def restore_for_worker(args):
    from ..planners.transformer_session import load_state
    state, source = check_source(args.resume_case, ledger=args.ledger)
    copied = load_state(args.resume_session)
    if not same_state(copied,state):raise ValueError('Inherited SESSION differs from the full original state')
    if (args.model != MODEL or not _same(args.model_info,source['model_info'])
            or not _same(args.calibration,source['calibration']) or args.calls != 6
            or args.tuner_seconds != 600 or args.tuner is not True or args.prepare_only
            or args.resume_source is not None):
        raise ValueError('Alex continuation must retain original model/context/options')
    return copied, source


def verify_context(state, ctx):
    from ..planners.prompts import build_system_prompt, initial_message
    return validate_state(state,initial_messages=[dict(role='system',content=build_system_prompt(ctx)),
                                                dict(role='user',content=initial_message(ctx))])


def initialize_worker(args, work, result, *, save):
    from ..planners.transformer_session import save_state
    state, source = restore_for_worker(args); work = Path(work); inherited = Path(args.resume_session).parent
    if any((work/name).exists() for name in ('sources','agent_trace.jsonl','SESSION.json','api')):
        raise ValueError('New worker evidence paths must be fresh')
    shutil.copytree(inherited/'sources',work/'sources')
    shutil.copy2(inherited/'agent_trace.jsonl',work/'agent_trace.jsonl')
    save_state(work/'SESSION.json',state)
    result.update(recovered_api_calls=source['initial_calls'],actual_api_calls=0,
        recovery=dict(kind='retained_whole_session',source_case=source['original_case'],
            initial_phase=state['phase'],original_calls=source['initial_calls'],
            remaining_calls=source['remaining_calls'],prior_elapsed_s=state.get('elapsed_s'),
            pending_tune_redone=state['phase']=='tune',old_requests_replayed=False,
            inherited_provider_artifacts=str(inherited/'api'),cold_start=False))
    save(); return state


def preserve_session(case,destination,state):
    from ..planners.transformer_session import load_state,save_state
    source,destination=Path(case)/'work/agent',Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('api','sources'):shutil.copytree(source/name,destination/name)
    for name in ('agent_trace.jsonl','events.jsonl','worker-result.json','phase_result.json'):
        shutil.copy2(source/name,destination/name)
    save_state(destination/'SESSION.json',state)
    if not same_state(load_state(destination/'SESSION.json'),state):
        raise ValueError('Public SESSION serialization did not preserve all state semantics')
    return destination/'SESSION.json'


def check_provider_history(initial,current,directory,new_calls,*,ledger_rows=None):
    """Verify the inherited prefix and all additional successful request/replies."""
    directory=Path(directory); calls=initial['calls']; completed=len(initial['history'])
    if (not same_state(current.get('history',[])[:completed],initial['history'])
            or not same_state(current.get('messages',[])[:len(initial['messages'])],initial['messages'])
            or current.get('calls') != calls+new_calls
            or len(current.get('messages',[])) != 2+2*(calls+new_calls)):
        raise ValueError('Continued SESSION lost its original complete history')
    requests=sorted((directory/'api').glob('*.request.json'))
    if (len(requests)!=new_calls or len(list((directory/'api').glob('*.response.json')))!=new_calls
            or list((directory/'api').glob('*.error.json'))):
        raise ValueError('Actual new provider artifacts differ from the continued call count')
    if ledger_rows is not None and len(ledger_rows)!=new_calls:
        raise ValueError('Actual new ledger interval differs from the continued call count')
    if initial['phase']=='tune' and current['messages'][2*calls] != dict(role='assistant',content=initial['pending']['reply']['content']):
        raise ValueError('The original pending reply was not retained through tune completion')
    tokens=[initial['prompt_tokens'],initial['completion_tokens']]
    for offset,request_path in enumerate(requests,1):
        number=calls+offset; request=_read(request_path)
        response=_read(request_path.with_name(request_path.name.removesuffix('.request.json')+'.response.json'))
        if (request.get('purpose')!='agent-iter-'+str(number)
                or request.get('messages')!=current['messages'][:2*number]
                or current['messages'][2*number]!=dict(role='assistant',content=response.get('content'))):
            raise ValueError('New provider history differs from the continued conversation')
        raw=_reply_program(response); raw_path=directory/'sources'/('iteration-'+str(number)+'.ptdsl')
        if isinstance(raw,str):
            if raw_path.read_text()!=raw:raise ValueError('New raw program differs from its saved reply')
        elif raw_path.exists() or current['history'][number-1].get('stage') not in ('response','length'):
            raise ValueError('New malformed reply lacks its original rejection record')
        ledger_row=ledger_rows[offset-1]if ledger_rows is not None else None
        if ledger_row is not None and (ledger_row.get('status')!=200 or ledger_row.get('error')is not None
                or ledger_row.get('purpose')!='agent-iter-'+str(number)):
            raise ValueError('New response lacks its actual successful ledger record')
        tokens=[a+b for a,b in zip(tokens,_usage(response,ledger_row))]
    if tokens != [current.get('prompt_tokens'),current.get('completion_tokens')]:
        raise ValueError('Continued token totals differ from actual successful replies')
    traces=[json.loads(line) for line in (directory/'agent_trace.jsonl').read_text().splitlines() if line.strip()]
    if [row.get('iteration') for row in traces] != list(range(1,calls+new_calls+1)):
        raise ValueError('The complete original plus continued trace sequence differs')
    return dict(inherited_successful_calls=calls,actual_new_calls=new_calls,
                complete_provider_conversation=True,old_ledger_records_recreated=False)


def check_worker(worker,state,source,new_calls,*,agent_dir):
    if (worker.get('model')!=MODEL or worker.get('tuner_enabled')is not True
            or worker.get('recovered_api_calls')!=source['initial_calls']
            or worker.get('actual_api_calls')!=new_calls or not 0<=new_calls<=source['remaining_calls']
            or state.get('phase')!='done' or state.get('calls')!=source['initial_calls']+new_calls
            or any(state.get(key)!=value for key,value in OPTIONS.items())
            or state.get('failed')is not None
            or not _same(worker.get('source_model_info'),source['model_info'])
            or not _same(worker.get('source_calibration'),source['calibration'])
            or any(worker.get(key)!=0 for key in ('report_samples','compiler_runs','real_ckks_runs'))):
        raise ValueError('Terminal worker accounting or restored original context differs')
    agent=worker.get('agent') or {}
    if agent.get('llm_calls')!=state['calls'] or not same_state(agent.get('confirm'),state.get('confirmation')):
        raise ValueError('Terminal worker differs from its preserved SESSION')
    if worker.get('status')=='quality_rejected' and worker.get('frozen')is False:
        return 'quality_rejected'
    probes=worker.get('strict_math_probes') or []
    if (worker.get('status')!='ok' or worker.get('frozen')is not True
            or worker.get('quality_acceptance',{}).get('passed')is not True or len(probes)!=3):
        raise ValueError('Original frozen quality and strict mathematics must complete')
    selected=(agent.get('confirm') or {}).get('selected')
    rows=[row for row in agent.get('confirm',{}).get('evals',[])
          if selected and (row.get('iteration'),row.get('origin'))==(selected['iteration'],selected['origin'])]
    if (len(rows)!=1 or rows[0].get('passed')is not True or rows[0].get('diverged')is not False
            or rows[0].get('n_diverged')!=0 or rows[0].get('total')!=1000
            or any(type(rows[0].get(key))not in (int,float) or not math.isfinite(rows[0][key])
                   for key in ('acc_drop_pp','max_u_calibrated','max_u_analytic'))
            or rows[0]['acc_drop_pp']>.5 or rows[0]['max_u_calibrated']>.90 or rows[0]['max_u_analytic']>1):
        raise ValueError('The original one-time confirmation policy differs')
    for index,probe in enumerate(probes):
        if probe.get('sample_id')!='test:'+str(9000+index):raise ValueError('Strict probe identity differs')
        for role in ('exact','approximate'):
            check=probe.get(role) or {}; ratio=check.get('max_error_ratio')
            if (any(check.get(key)is not True for key in ('passed','finite','dtype_ok','shape_ok'))
                    or check.get('criterion')!='1e-12+1e-12*abs(reference)'
                    or type(ratio)not in (int,float) or not math.isfinite(ratio) or ratio>1):
                raise ValueError('Strict mathematical evidence differs')
    program=Path(worker['selected_program']).resolve()
    if not program.is_relative_to(Path(agent_dir).resolve()) or not program.is_file():
        raise ValueError('Frozen source must be in the new worker directory')
    identity=lambda text:ast.dump(ast.parse(text),include_attributes=False)
    if any(identity(text)!=identity(selected['source']) for text in
           (program.read_text(),agent['best_program'],state['best_program'])):
        raise ValueError('Frozen source differs from its confirmed original selection')
    return 'ok'


def worker_command(args,source,retained):
    seconds=args.seconds
    return [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds-60),
        '--python',str(args.python),'--','-B','-m',MODULE,'--model',MODEL,
        '--work-dir',str(args.out/'work/agent'),'--model-info',source['model_info'],
        '--calibration',source['calibration'],'--ledger',str(args.ledger),'--allowance',str(args.allowance),
        '--calls','6','--tuner-seconds','600','--memory-gib',str(args.memory_gib),
        '--resume-session',str(retained),'--resume-case',source['original_case'],
        '--seconds',str(seconds-120),'--metadata-template',source['metadata_template']]


def _register(args,result,before,rows):
    registry=Path(args.registry or paths.RESULTS_DIR/'registry.jsonl');registry.parent.mkdir(parents=True,exist_ok=True)
    def row(run_id,mode,metrics):
        return dict(run_id=run_id,round=ROUND,model=MODEL,mode=mode,git_rev=None,metrics=metrics,
                    artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat())
    records=[row(args.out.name+'-api-'+str(index), 'llm-call',dict(call,global_call=index))
             for index,call in enumerate(rows[before:],before+1)]
    records.append(row(args.out.name,MODE,{key:result.get(key) for key in
        ('status','cold_start','original_api_calls','actual_new_api_calls','worker_attempts','worker_completed',
         'original_time_seconds','incoming_signal','guard_trigger','wall_timeout_observed')}))
    with registry.open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        existing={json.loads(line).get('run_id')for line in stream if line.strip()}
        if any(record['run_id']in existing for record in records):raise ValueError('Preserve prior registry identities')
        stream.seek(0,os.SEEK_END)
        for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def execute(args):
    # Admission is read-only and precedes creation or any allowance renewal.
    state,source=check_source(args.source_case,ledger=args.ledger)
    out=Path(args.out).resolve();args.out=out
    if (out.exists() or out==paths.RESULTS_DIR.resolve() or not out.is_relative_to(paths.RESULTS_DIR.resolve())
            or out.is_relative_to(Path(args.source_case).resolve())):
        raise ValueError('Choose a fresh continuation case under RESULTS, separate from the source')
    started=time.perf_counter_ns();out.mkdir(parents=True);(out/'work').mkdir()
    before=len(call_records(args.ledger)); result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,
        cold_start=False,tuner_enabled=True,source=source,original_api_calls=state['calls'],actual_new_api_calls=0,
        api_before=before,worker_attempts=0,worker_completed=0,worker=None,process=None,
        incoming_signal=None,guard_stop_signal=None,timeout_stop_signal=None,guard_trigger=None,
        wall_timeout_observed=False,worker_wall_timeout_observed=False,child_exit_signal=None,
        monitor_error=None,original_time_seconds=source['original_complete_seconds'],
        elapsed_scope='New continuation clock only; original failed clock and inherited SESSION elapsed remain separate',
        post_freeze_compile='Pending independent stage; this entry stops after original confirmation and strict math',
        compiler_runs=0,real_ckks_runs=0,keygen_runs=0,report_samples=0,
        compiled_bootstraps=None,CKKS_latency=None,accuracy_acceptance=None,security_acceptance=None,
        limits=dict(outer_process_seconds=args.seconds,worker_seconds=args.seconds-120,Nix_seconds=args.seconds-60,
            per_process_AS_gib=args.memory_gib,subtree_RSS_gib=14,minimum_available_gib=1.25,
            case_gib=args.case_gib,minimum_free_gib=10,completion_reserve_bytes=RESERVE,concurrency=1))
    recorder=StageRecorder(out/'events.jsonl'); stop=threading.Event();resource_path=out/'resources.jsonl'
    old={};watcher=None;worker_parent=None;lock=None
    def check():
        size,free=artifact_bytes(out),shutil.disk_usage(out).free
        physical=physical_memory_observation(_latest_rss_sample(resource_path),_available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
        if physical['stop'] or size+RESERVE>args.case_gib*GIB or free<10*GIB:
            result['guard_trigger']=dict(physical=physical,artifact_bytes=size,disk_free_bytes=free,
                case_allowance_bytes=args.case_gib*GIB)
            raise RuntimeError('Continuation physical/artifact/disk allowance exhausted')
    def monitor():
        while not stop.wait(.5):
            try:check()
            except BaseException as exc:
                result['monitor_error']=type(exc).__name__+': '+str(exc)
                try:write_json(out/'resource-guard.json',dict(guard=result['guard_trigger'],error=result['monitor_error']))
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
    def interrupted(signum,frame):
        field='guard_stop_signal'if result['guard_trigger']or result['monitor_error'] else 'incoming_signal'
        result[field]=signum;raise InterruptedError('Continuation received signal'+str(signum))
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):old[sig]=signal.signal(sig,interrupted)
        lock=(paths.RESULTS_DIR/'.r30-alex-continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with recorder.span('continuation'):
            with recorder.span('preserve_original_session'):
                check();retained=preserve_session(args.source_case,out/'inherited-agent',state)
                snapshot=out/'source';snapshot.mkdir()
                for name in ('r30_continue_alex_session.py','r30_continue_alex_session_worker.py','r28_common_agent.py'):
                    shutil.copy2(Path(__file__).with_name(name),snapshot/name)
            if args.prepare_only:result['status']='prepared'
            else:
                if source['remaining_calls']:ensure_allowance(args.allowance,args.ledger,required=source['remaining_calls'])
                command=worker_command(args,source,retained);result['worker_argv']=command
                env={k:v for k,v in os.environ.items()if not k.startswith(('LD_','NIX_','__NIX_','HECATE_'))
                     and k not in ('HECATE','IN_NIX_SHELL','PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')}
                env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
                for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
                watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
                with recorder.span('continue_original_agent')as worker_parent:
                    result['worker_attempts']=1;write_json(out/'result.json',result)
                    observed=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/'worker.log',
                        resources=resource_path,timeout=args.seconds,mem_gib=args.memory_gib,fsize_gib=8,env=env)
                    result['process']=observed;result['wall_timeout_observed']=observed.get('timed_out')is True
                    if type(observed.get('exit_code'))is int and observed['exit_code']<0:
                        result['child_exit_signal']=-observed['exit_code']
                    if observed.get('signal')is not None:
                        field='guard_stop_signal'if result['guard_trigger']or result['monitor_error']else'incoming_signal'
                        result[field]=observed['signal']
                    target=out/'work/agent/worker-result.json'
                    if target.is_file():result['worker']=_read(target)
                    result['worker_wall_timeout_observed']=bool((result['worker']or{}).get('worker_wall_timeout_observed'))
                    if (observed.get('exit_code')!=0 or observed.get('timed_out')is not False
                            or observed.get('signal')is not None or observed.get('remaining_pids')!=[]
                            or observed.get('signals_sent')!=[]):
                        raise RuntimeError('Continuation worker stopped; partial SESSION and original case retained')
                    from ..planners.transformer_session import load_state
                    current=load_state(out/'work/agent/SESSION.json');actual_rows=call_records(args.ledger);new_calls=len(actual_rows)-before
                    result['provider_history']=check_provider_history(state,current,out/'work/agent',new_calls,ledger_rows=actual_rows[before:])
                    result['status']=check_worker(result['worker']or{},current,source,new_calls,agent_dir=out/'work/agent')
                    result['worker_completed']=1;check()
    except BaseException as exc:
        result.update(status='resource_stopped'if result['guard_trigger']or result['monitor_error']else
            'timeout'if result['wall_timeout_observed']or result['worker_wall_timeout_observed']else
            'interrupted'if result['incoming_signal']else'error',
            error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        stop.set()
        if watcher is not None:watcher.join(timeout=2)
        for sig,handler in old.items():signal.signal(sig,handler)
        if lock is not None:lock.close()
        target=out/'work/agent/worker-result.json'
        if target.is_file():
            try:result['worker']=_read(target)
            except (OSError,ValueError)as exc:result.update(status='error',partial_error=str(exc)[:1000])
        rows=call_records(args.ledger);result.update(api_after=len(rows),actual_new_api_calls=len(rows)-before)
        children=read_events(out/'work/agent/events.jsonl')
        if worker_parent is not None:
            for event in children:
                if event.get('parent_id')is None:event['parent_id']=worker_parent['event_id']
        timing=summarize_events(read_events(out/'events.jsonl')+children,time.perf_counter_ns());result['timing']=timing
        size,free=artifact_bytes(out),shutil.disk_usage(out).free
        result['terminal_resources']=dict(observed_artifact_bytes=size,disk_free_bytes=free,
            case_allowance_bytes=args.case_gib*GIB,completion_reserve_bytes=RESERVE,
            within_case_with_reserve=size+RESERVE<=args.case_gib*GIB,minimum_disk_free_passed=free>=10*GIB)
        if size+RESERVE>args.case_gib*GIB or free<10*GIB:
            result.update(status='resource_stopped',error=result.get('error')or'Terminal artifact/disk allowance exhausted')
        write_json(out/'result.json',result);_register(args,result,before,rows)
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=timing,required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,default=paths.RESULTS_DIR/'llm_calls.jsonl')
    parser.add_argument('--allowance',type=Path,default=paths.RESULTS_DIR/'autonomous-api-allowance.json')
    parser.add_argument('--registry',type=Path);parser.add_argument('--python',type=Path,default=paths.INTAKE_PYTHON)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--seconds',type=int,default=18000)
    parser.add_argument('--memory-gib',type=int,default=12);parser.add_argument('--case-gib',type=int,default=4)
    args=parser.parse_args(argv)
    if not 300<=args.seconds<=18000 or not 4<=args.memory_gib<=12 or not 1<=args.case_gib<=4:
        parser.error('Use finite300..18000seconds, AS4..12GiB, case1..4GiB; physical RSS14/available1.25/free10 remain fixed')
    return 0 if execute(args)['status']in('prepared','ok','quality_rejected')else 1


if __name__=='__main__':raise SystemExit(main())
