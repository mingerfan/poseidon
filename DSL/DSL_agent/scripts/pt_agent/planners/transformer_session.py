"""Bounded token Agent sessions; report data stays outside this module."""

from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path

from ..llm import LLMError

OUTPUT_TOKENS = {'r8-v7': 32768, 'r8-v8': 65536}


def output_tokens(version):
    # r8-v9 aliases v8's unchanged provider settings.
    return OUTPUT_TOKENS['r8-v8' if version == 'r8-v9' else version]


def model_call_offset(model, version):
    return 34 if version == 'r8-v9' and model == 'bert' else 0


def model_calls(directory, model, version):
    return model_call_offset(model, version)+len(list((Path(directory)/model/version).glob('call-*.request.json')))



def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else 'inf' if value>0 else '-inf'
    if isinstance(value,dict):return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(item) for item in value]
    return value


class ModelBudgetClient:
    """Reserve each of at most 40 model calls before entering the provider.

    Reservations survive interruption. A response is saved before the Agent
    parses it; neither a failed request nor a new session resets the allowance.
    The global provider ledger is passed explicitly by the outer process.
    """
    def __init__(self,client,directory,*,limit=40,model=None,prompt_version='r8-v7'):
        if not 0<=limit<=40:raise ValueError('A1 model call limit must be in 0..40')
        if prompt_version not in (*OUTPUT_TOKENS, 'r8-v9'):raise ValueError('Unknown token prompt version')
        if prompt_version in ('r8-v8','r8-v9','r8-v10'):
            if not model:raise ValueError('r8-v8 budget requires the model identity')
            directory=Path(directory)/model/prompt_version
        self.offset = model_call_offset(model, prompt_version)
        self.client,self.directory,self.limit=client,Path(directory),limit
        self.directory.mkdir(parents=True,exist_ok=True)

    @property
    def calls(self):
        return self.offset+len(list(self.directory.glob('call-*.request.json')))

    def chat(self,messages,*,purpose):
        from ..run import _write
        with (self.directory/'client.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            if self.calls>=self.limit:raise LLMError('budget_exhausted')
            stem='call-'+str(self.calls+1).zfill(3)
            _write(self.directory/(stem+'.request.json'),dict(messages=messages,purpose=purpose))
            try:reply=self.client.chat(messages,purpose=purpose)
            except LLMError as exc:
                _write(self.directory/(stem+'.error.json'),dict(error=exc.code,diagnostics=exc.diagnostics))
                raise
            _write(self.directory/(stem+'.response.json'),reply)
            return reply


def run_session(info,calibration,samples,client,directory,*,max_iters=8,tuner=True,
                tuner_budget_s=2400,tune_policy='new_best',prompt_version='r8-v7',
                tune_rule=None):
    from ..planners.agent_tools import build_context
    from .agent import run_agent
    from ..run import _write
    if not 1<=max_iters<=8:raise ValueError('A1 session requires max_iters in 1..8')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    ctx=build_context(info,calibration,samples=samples)
    if prompt_version not in (*OUTPUT_TOKENS, 'r8-v9'):raise ValueError('Unknown token prompt version')
    ctx.prompt_version=prompt_version
    if prompt_version == 'r8-v9':
        ctx.a1_model_call_offset = getattr(client, 'offset', 0)
    result=run_agent(ctx,client,max_iters=max_iters,trace_path=directory/'agent_trace.jsonl',
                     tuner=tuner,tuner_budget_s=tuner_budget_s,tune_policy=tune_policy,
                     engine_profile='tensor',tune_rule=tune_rule)
    return write_session_result(result,directory,prompt_version=prompt_version,tune_policy=tune_policy)


def write_session_result(result,directory,*,prompt_version='r8-v8',tune_policy='new_best'):
    from ..run import _write
    directory=Path(directory)
    selected=result['confirm']['selected']
    record=dict(result,confirm=dict(result['confirm'],selected=asdict(selected) if selected is not None else None))
    if prompt_version in ('r8-v8','r8-v9','r8-v10'):record['prompt_version']=prompt_version
    _write(directory/'AGENT.json',json_safe(record))
    if selected is not None:
        path=directory/'program.ptdsl'
        if path.exists() and path.read_text()!=selected.source:raise RuntimeError('Selected frozen program changed')
        if not path.exists():path.write_text(selected.source)
        _write(directory/'FROZEN.json',dict(program=str(path),origin=selected.origin,
              iteration=selected.iteration,confirm=record['confirm'],prompt_version=prompt_version,
              engine_profile='tensor',tune_policy=tune_policy,
              **(dict(tune_rule=record.get('tune_rule','v8-pre-6.11')) if prompt_version in ('r8-v8','r8-v9','r8-v10') else {}),
              **(dict(split_version=record['split_version']) if 'split_version' in record else {})))
    return json_safe(record)


def _state_value(value,*,restore=False):
    """Preserve tuple keys and nonfinite diagnostics without lossy JSON coercion."""
    if restore:
        if isinstance(value,list):return [_state_value(item,restore=True) for item in value]
        if isinstance(value,dict):
            if set(value)=={'$tuple'}:return tuple(_state_value(item,restore=True) for item in value['$tuple'])
            if set(value)=={'$dict'}:return {_state_value(key,restore=True):_state_value(item,restore=True)
                                           for key,item in value['$dict']}
            if set(value)=={'$float'}:return float(value['$float'])
            return {key:_state_value(item,restore=True) for key,item in value.items()}
        return value
    if isinstance(value,float) and not math.isfinite(value):return {'$float':repr(value)}
    if isinstance(value,tuple):return {'$tuple':[_state_value(item) for item in value]}
    if isinstance(value,list):return [_state_value(item) for item in value]
    if isinstance(value,dict):
        if any(not isinstance(key,str) for key in value):
            return {'$dict':[[_state_value(key),_state_value(item)] for key,item in value.items()]}
        return {key:_state_value(item) for key,item in value.items()}
    return value


def save_state(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(_state_value(state),stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def load_state(path):
    state=_state_value(json.loads(Path(path).read_text()),restore=True)
    if state.get('prompt_version')=='r8-v8':state.setdefault('tune_rule','v8-pre-6.11')
    return state


def session_step(ctx,directory,*,client=None,reply=None,max_iters=8,tune_rule=None):
    """One process owns one step. The outer orchestrator reads only its status."""
    from .agent import new_agent_state,agent_step,agent_result
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'SESSION.json'
    from .transformer_tuner import validate_tune_rule
    tune_rule=validate_tune_rule(ctx.prompt_version,tune_rule)
    state=load_state(path) if path.exists() else new_agent_state(ctx,max_iters=max_iters,
        engine_profile='tensor',tune_rule=tune_rule)
    from ..data.samples import split_version
    if state.get('split_version','split-v1')!=split_version(getattr(ctx,'model_info',None)):
        raise ValueError('Session and model use different sample partitions')
    if ctx.prompt_version == 'r8-v9':
        state.setdefault('model_call_offset', getattr(ctx, 'a1_model_call_offset',
                            model_call_offset(ctx.model_name, ctx.prompt_version)))
    before=state['phase']
    if before=='done':raise RuntimeError('Session already finalized')
    agent_step(ctx,state,trace_path=directory/'agent_trace.jsonl',client=client,reply=reply)
    save_state(path,state)
    record=None
    if state['phase']=='done':
        record=write_session_result(agent_result(state),directory,prompt_version=state['prompt_version'],
                                    tune_policy=state['tune_policy'])
    return dict(complete=True,step=before,next_step=state['phase'],iteration=state['iteration'],
                session_calls=state['calls'],state=str(path),agent=record,tune_rule=state['tune_rule'])


def recover_session(ctx,original,directory):
    """Reconstruct seven committed rows, reusing their stored feedback and tune."""
    from ..run import _write
    from .agent import new_agent_state,agent_step
    from .agent_tools import evaluate
    from .tuner import TuneResult
    from ..dsl.parser import parse
    original,directory=Path(original),Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'SESSION.json').exists() or (directory/'agent_trace.jsonl').exists():
        raise FileExistsError('Recovery output already contains a session')
    rows=[json.loads(line) for line in (original/'agent_trace.jsonl').read_text().splitlines()]
    if len(rows)==5 and (original/'SESSION.json').exists():
        return recover_budget_session(ctx,original,directory,rows)
    if len(rows)==6 and (original/'SESSION.json').exists():
        return recover_second_session(ctx,original,directory,rows)
    if len(rows)!=7:raise ValueError('A1 §6.8 recovery requires seven completed rows')
    source=(original/'sources/iteration-7.ptdsl').read_text()
    checked=evaluate(source,ctx,engine='tensor',engine_profile='tensor')
    comparisons={key:json_safe(checked.get(key))==rows[-1]['fb'].get(key)
                 for key in ('cost','validation','feasible')}
    verification=dict(complete=all(comparisons.values()),original=str(original),
                      comparisons=comparisons,recorded=rows[-1]['fb'],reevaluated=json_safe(checked),
                      llm_calls=0,report_images=0)
    _write(directory/'ROUND7_REEVALUATION.json',verification)
    if not verification['complete']:raise RuntimeError('Recovered round 7 differs from the retained trace')
    state=new_agent_state(ctx,max_iters=8,engine_profile='tensor',tune_rule='v8-pre-6.11')
    for iteration,row in enumerate(rows,1):
        if row['iteration']!=iteration:raise ValueError('Recovery trace iteration differs')
        reply=dict(content=row['messages'][0]['content'],usage=row['usage'],seconds=row['seconds'],finish_reason='stop')
        if json.loads(reply['content'])['program']!=(original/'sources'/f'iteration-{iteration}.ptdsl').read_text():
            raise ValueError('Recovery response and retained source differ')
        agent_step(ctx,state,trace_path=directory/'agent_trace.jsonl',reply=reply,feedback=row['fb'])
        if state['phase']=='tune':
            saved=row['tuned']
            if saved is None:raise ValueError('Recovery requires an unrecorded tune')
            result=TuneResult(program=parse(saved['source']),**saved)
            agent_step(ctx,state,trace_path=directory/'agent_trace.jsonl',tune_result=result)
        recreated=json.loads((directory/'agent_trace.jsonl').read_text().splitlines()[-1])
        recreated.pop('tune_rule',None)
        if recreated!=row:raise RuntimeError(f'Recovered trace differs at round {iteration}')
        save_state(directory/'SESSION.json',state)
    state['recovery']=dict(original=str(original),reused_tune_rounds=[1],completed_rounds=7,
                           round7_verified=True)
    save_state(directory/'SESSION.json',state)
    return dict(complete=True,step='recover',next_step=state['phase'],iteration=state['iteration'],
                session_calls=state['calls'],state=str(directory/'SESSION.json'),llm_calls=0,report_images=0)


def recover_second_session(ctx,original,directory,rows):
    """Restore calls 9–13 from retained records, without rerunning any tune."""
    from .agent import new_agent_state,agent_step
    from .tuner import TuneResult
    from ..dsl.parser import parse
    from ..run import _write
    retained=load_state(original/'SESSION.json')
    if retained['calls']!=6 or rows[-1]['iteration']!=6:
        raise ValueError('A1 §6.9 requires the retained second-session six calls')
    state=new_agent_state(ctx,max_iters=retained['max_iters'],engine_profile='tensor',
                          tune_rule='v8-pre-6.11')
    reused=[]
    for iteration,row in enumerate(rows[:5],1):
        if row['iteration']!=iteration:raise ValueError('Recovery trace iteration differs')
        source=(original/'sources'/f'iteration-{iteration}.ptdsl').read_text()
        reply=dict(content=row['messages'][0]['content'],usage=row['usage'],seconds=row['seconds'],
                   finish_reason='stop')
        if json.loads(reply['content'])['program']!=source:
            raise ValueError('Recovery response and retained source differ')
        agent_step(ctx,state,trace_path=directory/'agent_trace.jsonl',reply=reply,feedback=row['fb'])
        if state['phase']=='tune':
            saved=row['tuned']
            if saved is None:raise ValueError('Recovery requires an unrecorded tune')
            agent_step(ctx,state,trace_path=directory/'agent_trace.jsonl',
                       tune_result=TuneResult(program=parse(saved['source']),**saved))
            reused.append(iteration)
        recreated=json.loads((directory/'agent_trace.jsonl').read_text().splitlines()[-1])
        recreated.pop('tune_rule',None)
        if recreated!=row:raise RuntimeError(f'Recovered trace differs at round {iteration}')
        if json_safe(state['history'][-1])!=json_safe(retained['history'][iteration-1]):
            raise RuntimeError(f'Recovered decision differs at round {iteration}')
        save_state(directory/'SESSION.json',state)
    if state['messages']!=retained['messages'][:12]:
        raise RuntimeError('Recovered messages differ before call 14')
    if state['phase']!='evaluate' or state['calls']!=5 or state['llm_min']!=3233700613:
        raise RuntimeError('Recovered call-13 boundary differs')
    state['recovery']=dict(original=str(original),section='6.9',completed_rounds=5,
        reused_tune_rounds=reused,model_call_offset=8,saved_response_call=14,
        saved_response_iteration=6,llm_min_after_call13=state['llm_min'],
        target_tune_rule='v8-6.11',switch_before_model_call=15)
    save_state(directory/'SESSION.json',state)
    report=dict(complete=True,step='recover',next_step=state['phase'],iteration=state['iteration'],
        session_calls=state['calls'],state=str(directory/'SESSION.json'),recovery=state['recovery'],
        tune_rule=state['tune_rule'],llm_calls=0,report_images=0)
    _write(directory/'RECOVERY.json',report)
    return report


def recover_pending_tune(original, directory, response):
    """Retain a stopped checkpoint and resume its saved response under 6.17."""
    import shutil
    from .transformer_tuner import ACCURACY_TUNE_RULE, CONVERGENT_TUNE_RULE
    original,directory,response=Path(original),Path(directory),Path(response)
    state=load_state(original/'SESSION.json')
    if (state['phase']!='tune' or state['prompt_version']!='r8-v9'
            or state['tune_rule']!=CONVERGENT_TUNE_RULE or not state.get('pending')):
        raise ValueError('6.17 recovery requires a retained r8-v9 pending tune')
    reply=json.loads(response.read_text())
    if reply!=json_safe(state['pending']['reply']):
        raise ValueError('Saved response differs from the pending checkpoint')
    if json.loads(reply['content'])['program']!=state['pending']['source']:
        raise ValueError('Pending source differs from the saved response')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'SESSION.json').exists():
        raise FileExistsError(directory/'SESSION.json')
    shutil.copytree(original/'sources',directory/'sources')
    shutil.copy2(original/'agent_trace.jsonl',directory/'agent_trace.jsonl')
    state.setdefault('tune_cache_archive',{})[state['tune_rule']]=state['tune_cache']
    state.update(tune_rule=ACCURACY_TUNE_RULE,tune_cache={},recovery=dict(
        section='6.17',original=str(original),saved_response=str(response),
        calls=state['calls'],iteration=state['iteration'],old_rule=CONVERGENT_TUNE_RULE,
        new_rule=ACCURACY_TUNE_RULE,llm_calls=0))
    save_state(directory/'SESSION.json',state)
    return dict(complete=True,step='recover',next_step='tune',state=str(directory/'SESSION.json'),
                recovery=state['recovery'],tune_rule=ACCURACY_TUNE_RULE,llm_calls=0,report_images=0)


def recover_budget_session(ctx,original,directory,rows):
    """Restore the four committed rounds preceding saved BERT call 21."""
    from .agent import new_agent_state,agent_step
    from .tuner import TuneResult
    from ..dsl.parser import parse
    from ..run import _write
    retained=load_state(original/'SESSION.json')
    if (retained['calls']!=5 or retained['failed']!='analytic_domain_violation'
            or retained['tune_rule']!='v8-6.11' or rows[-1]['iteration']!=5):
        raise ValueError('A1 §6.12 requires the retained third-session analytic stop')
    state=new_agent_state(ctx,max_iters=retained['max_iters'],engine_profile='tensor',tune_rule='v8-6.11')
    reused=[]
    for iteration,row in enumerate(rows[:4],1):
        if row['iteration']!=iteration:raise ValueError('Recovery trace iteration differs')
        source=(original/'sources'/f'iteration-{iteration}.ptdsl').read_text()
        reply=dict(content=row['messages'][0]['content'],usage=row['usage'],seconds=row['seconds'],finish_reason='stop')
        if json.loads(reply['content'])['program']!=source:
            raise ValueError('Recovery response and retained source differ')
        agent_step(ctx,state,trace_path=directory/'agent_trace.jsonl',reply=reply,feedback=row['fb'])
        if state['phase']=='tune':
            saved=row['tuned']
            if saved is None:raise ValueError('Recovery requires an unrecorded tune')
            agent_step(ctx,state,trace_path=directory/'agent_trace.jsonl',
                       tune_result=TuneResult(program=parse(saved['source']),**saved))
            reused.append(iteration)
        recreated=json.loads((directory/'agent_trace.jsonl').read_text().splitlines()[-1])
        if recreated!=row or json_safe(state['history'][-1])!=json_safe(retained['history'][iteration-1]):
            raise RuntimeError(f'Recovered third-session row differs at round {iteration}')
    def cache_record(cache):
        import copy
        normalized=copy.deepcopy(cache)
        for saved,_,_ in normalized.values():
            for change in saved['changes']:
                change.pop('line_text',None);change.pop('call_texts',None)
        return json_safe(normalized)
    checks=dict(messages=state['messages']==retained['messages'][:10],
        pool=state['pool']==retained['pool'],records=json_safe(state['records'])==json_safe(retained['records']),
        tune_cache=cache_record(state['tune_cache'])==cache_record(retained['tune_cache']),boundary=state['phase']=='evaluate'
        and state['iteration']==5 and state['calls']==4 and state['failed'] is None)
    if not all(checks.values()):raise RuntimeError('Recovered call-20 boundary differs: '+str(checks))
    # The presentation trace uses JSON lists and adds line-text annotations.
    # Preserve the checkpoint's original typed records and undecorated cache.
    state['records']=retained['records'];state['tune_cache']=retained['tune_cache']
    state['recovery']=dict(original=str(original),section='6.12',completed_rounds=4,
        reused_tune_rounds=reused,model_call_offset=16,saved_response_call=21,saved_response_iteration=5)
    save_state(directory/'SESSION.json',state)
    result=dict(complete=True,step='recover',next_step='evaluate',iteration=5,session_calls=4,
        recovery=state['recovery'],checks=checks,tune_rule=state['tune_rule'],llm_calls=0,report_images=0)
    _write(directory/'RECOVERY.json',result)
    return result


def migrate_tune_rule(directory,*,next_model_call,target_rule='v8-6.11'):
    """Persist the approved decision change at a completed iteration boundary."""
    directory=Path(directory);path=directory/'SESSION.json';state=load_state(path)
    if target_rule=='v8-6.14':
        return _migrate_convergent_rule(path,state,next_model_call)
    if target_rule!='v8-6.11':raise ValueError('Unknown migration target: '+str(target_rule))
    if state.get('tune_rule_migration'):
        if state['tune_rule_migration']['next_model_call']!=next_model_call:
            raise ValueError('Retained rule migration has another call boundary')
        return state['tune_rule_migration']
    if state['prompt_version']!='r8-v8' or state['tune_rule']!='v8-pre-6.11':
        raise ValueError('Rule migration requires a historical r8-v8 state')
    if state['phase']!='evaluate' or state.get('pending') is not None:
        raise ValueError('Rule migration requires a completed evaluate boundary')
    recovery=state.get('recovery',{})
    if recovery.get('section')=='6.9' and (next_model_call!=15 or state['calls']!=6):
        raise ValueError('Recovered BERT rule migration must precede call 15')
    costs=[row['latency_pred_us'] for row in state['history']
           if row.get('feasible') and row.get('latency_pred_us') is not None]
    record=dict(old_rule=state['tune_rule'],new_rule='v8-6.11',old_llm_min=state['llm_min'],
        new_llm_min=min(costs) if costs else None,next_model_call=next_model_call,
        next_iteration=state['iteration'],completed_session_calls=state['calls'])
    if 'llm_min_after_call13' in recovery:
        record['llm_min_after_call13']=recovery['llm_min_after_call13']
    state.update(tune_rule='v8-6.11',llm_min=record['new_llm_min'],tune_rule_migration=record)
    save_state(path,state)
    return record


def _migrate_convergent_rule(path,state,next_model_call):
    """Keep prior decisions and consumption; isolate the approved new cache."""
    retained=state.get('convergent_rule_migration')
    if retained:
        if retained['next_model_call']!=next_model_call:
            raise ValueError('Retained rule migration has another call boundary')
        return retained
    if state['prompt_version']!='r8-v8' or state['tune_rule']!='v8-6.11':
        raise ValueError('Convergent migration requires v8-6.11')
    if state['phase']!='evaluate' or state.get('pending') is not None:
        raise ValueError('Rule migration requires a completed evaluate boundary')
    record=dict(old_rule=state['tune_rule'],new_rule='v8-6.14',next_model_call=next_model_call,
                next_iteration=state['iteration'],completed_session_calls=state['calls'],
                archived_cache_entries=len(state['tune_cache']))
    state.setdefault('tune_cache_archive',{})[state['tune_rule']]=state['tune_cache']
    state.update(tune_rule='v8-6.14',tune_cache={},convergent_rule_migration=record)
    save_state(path,state)
    return record
