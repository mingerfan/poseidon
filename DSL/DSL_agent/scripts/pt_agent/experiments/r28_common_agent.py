"""Autonomous real-provider sessions on the original literature CNN checkpoints."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from .. import paths
from ..autonomous_budget import call_records, ensure_allowance
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .completion_metrics import artifact_bytes, persist_case_completion

MODELS={'resnet20_silu':'resnet20-silu','vgg16_silu_avgpool':'vgg16-silu-avgpool','alexnet_silu_avgpool':'alexnet-silu-avgpool'}
ROUND='R28'
FORMAL=('compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance')
PROMPT_VERSION='r8-v11'


def image_inputs(entries):
    """Infer the current DSL image role from an old rank-four feature entry."""
    result=[]
    for entry in entries:
        item=dict(entry)
        if item.get('kind')=='features':
            if len(item['shape'])!=4 or item['shape'][0]!=1:
                raise ValueError('Literature CNN features must have a batch-one image shape')
            item['kind']='image'
        result.append(item)
    return result


def agent_options(calls,tuner_seconds,*,tuner=True):
    if type(tuner) is not bool:raise ValueError('Tuner selection must be a bool')
    return dict(max_iters=calls,tuner=tuner,tuner_budget_s=tuner_seconds,tune_policy='all',
                engine_profile='tensor',stale_limit=3,tune_rule='v8-6.25',focus_retune=True,
                numeric_lock=False)


def single_image_input(values,names):
    if not isinstance(values,dict):return values
    if len(names)!=1 or set(values)!=set(names):
        raise ValueError('A CNN tensor call requires its single original named image')
    return values[names[0]]


def final_quality_acceptance(confirmation):
    """Keep the formal freeze decision independent of a legacy planner flag."""
    selected=confirmation.get('selected')
    if not selected:return dict(passed=False,reason='no_confirmed_candidate')
    identity=(selected['iteration'],selected['origin'])
    rows=[row for row in confirmation['evals'] if (row['iteration'],row['origin'])==identity]
    if len(rows)!=1:raise ValueError('Selected candidate must have one final confirmation record')
    row=rows[0]
    calibrated=row.get('max_u_calibrated',row['max_u'])
    analytic=row.get('max_u_analytic',0.0)
    import math
    finite=all(math.isfinite(value) for value in (calibrated,analytic,row['acc_drop_pp']))
    passed=(finite and row['passed'] and not row['diverged'] and calibrated<=0.90 and analytic<=1
            and row['acc_drop_pp']<=0.5)
    return dict(passed=bool(passed),calibrated_max_u=calibrated,analytic_max_u=analytic,
                acc_drop_pp=row['acc_drop_pp'],reason='accepted' if passed else 'quality_rejected')


def _recover_first_reply(args, result, *, save):
    """Checkpoint program-only recovery provenance before evaluation or tuning."""
    previous=json.loads(args.resume_result.read_text())
    if previous['api_delta']!=1 or previous['worker']['frozen'] or previous['worker'].get('agent'):
        raise ValueError('Recovery requires one completed online response before any confirmation')
    if Path(previous['worker']['source_model_info'])!=args.model_info:
        raise ValueError('Recovery model differs')
    last=call_records(args.ledger)[previous['api_after']-1]
    if last['status']!=200 or last['purpose']!='agent-iter-1' or last['error'] is not None:
        raise ValueError('Recovery requires the recorded successful first provider attempt')
    reply=dict(content=json.dumps(dict(program=args.resume_source.read_text())),
               usage={k:last[k] for k in ('prompt_tokens','completion_tokens')},
               seconds=last['seconds'],finish_reason=last['finish_reason'])
    result['recovery']=dict(kind='retained_actual_online_program_only',source=str(args.resume_source),
        source_result=str(args.resume_result),global_call=previous['api_after'],raw_reply_recovered=False,
        replay_kind='JSON reply reconstructed from the retained program only',
        provider_request_sent_during_replay=False,
        usage_and_duration_source='original successful provider ledger record')
    result['recovered_api_calls']=1
    save()
    return reply


def sources(model, *, model_info=None, calibration=None):
    if (model_info is None)!=(calibration is None):
        raise ValueError('Explicit intake requires both ModelInfo and calibration')
    if model_info is not None:
        info,calibration=Path(model_info).resolve(),Path(calibration).resolve()
        record=json.loads(info.read_text())
        if record.get('source',{}).get('factory')!='pt_agent.experiments.r13_common_models:'+model:
            raise ValueError('Explicit intake identifies a different original model')
        if not calibration.is_file():raise FileNotFoundError(calibration)
        return dict(model_info=info,calibration=calibration)
    stem=MODELS[model];r13=paths.RESULTS_DIR/('r13-common-'+stem+'-20261115');r14=paths.RESULTS_DIR/('r14-common-'+stem+'-20261115')
    a=json.loads((r13/'result.json').read_text());b=json.loads((r14/'result.json').read_text())
    if a['status']!='ok' or a['model']!='pt_agent.experiments.r13_common_models:'+model:
        raise ValueError('Original model intake does not match the selected literature model')
    return dict(model_info=Path(a['worker']['artifacts']['model_info'])/'model_info.json',
                calibration=Path(b['worker']['artifacts']['calibration']),r13_result=r13/'result.json',r14_result=r14/'result.json')


@contextmanager
def observed_tools(module,recorder):
    from .. import transformer_validation
    from ..transformer_lower import uses_transformer
    original_tensor=transformer_validation.run_tensor
    def tensor_adapter(program,weights,inputs,**kwargs):
        if not uses_transformer(program):inputs=single_image_input(inputs,[p.name for p in program.params])
        return original_tensor(program,weights,inputs,**kwargs)
    transformer_validation.run_tensor=tensor_adapter
    saved={}
    for name in ('evaluate','tune','confirm_select'):
        original=getattr(module,name);saved[name]=original
        def wrapper(*args,_name=name,_function=original,**kwargs):
            with recorder.span('agent_'+_name):
                if _name=='tune':
                    from ..planners.transformer_tuner import tune
                    return tune(*args,**kwargs)
                return _function(*args,**kwargs)
        setattr(module,name,wrapper)
    try:yield
    finally:
        for name,original in saved.items():setattr(module,name,original)
        transformer_validation.run_tensor=original_tensor


def worker(args):
    from ..autonomous_budget import RenewingLLMClient
    from ..models.info import ModelInfo
    from ..data.samples import LegacyCifar
    from ..planners.agent_tools import build_context, evaluate
    from ..planners import agent
    from ..tensor_sim import run_tensor, timing_snapshot
    from ..dsl.parser import parse
    from ..analyzer import reference_forward
    from .r15_rule_validation_worker import _packed_logits, check_math
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    resource.setrlimit(resource.RLIMIT_AS,(args.memory_gib*1024**3,args.memory_gib*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    work=args.work_dir;work.mkdir(parents=True,exist_ok=True)
    if (work/'worker-result.json').exists():raise ValueError('Preserve earlier session evidence')
    recorder=StageRecorder(work/'events.jsonl');started=time.perf_counter_ns()
    result=dict(round=ROUND,model=args.model,status='running',error=None,frozen=False,real_ckks_runs=0,
                report_samples=0,compiler_runs=0,source_model_info=str(args.model_info),source_calibration=str(args.calibration),
                tuner_enabled=getattr(args,'tuner',True),
                max_api_calls=args.calls,prompt_version=PROMPT_VERSION,placement='program-plan',engine_profile='tensor',strict_math_probes=[],
                **dict.fromkeys(FORMAL,None))
    def save():write_json(work/'worker-result.json',result)
    save()
    try:
        with recorder.span('agent_session'):
            resumed_state=None
            if getattr(args,'resume_session',None) is not None:
                from .r30_continue_common_session import initialize_worker
                with recorder.span('restore_retained_whole_session'):
                    resumed_state=initialize_worker(args,work,result,save=save)
            with recorder.span('load_original_graph_and_calibration'):
                info=ModelInfo.load(args.model_info);calibration=json.loads(args.calibration.read_text())
                expected='pt_agent.experiments.r13_common_models:'+args.model
                if info.source.get('factory')!=expected or info.threshold!={'acc_drop_pp':.5} or info.split.get('kind')!='legacy_cifar':
                    raise ValueError('Original model identity, task policy or CIFAR split differs')
                info=replace(info,inputs=image_inputs(info.inputs))
                result['input_role_adaptation']='rank-four features inferred as image; original names/shapes/dtypes retained'
                norm=info.source['normalization'];normalization=(norm['mean'],norm['std'])
                sample_source=LegacyCifar(info.graph,normalization)
            with recorder.span('prepare_agent_context'):
                ctx=build_context(info,calibration,n_val=1000,samples=sample_source,normalization=normalization)
                ctx.prompt_version=PROMPT_VERSION;ctx.placement='program-plan'
                if resumed_state is not None:
                    from .r30_continue_common_session import verify_context
                    verify_context(resumed_state,ctx)
            if args.prepare_only:
                from ..autonomous_budget import audit_public_messages
                from ..planners.prompts import build_system_prompt, initial_message
                messages=[dict(role='system',content=build_system_prompt(ctx)),dict(role='user',content=initial_message(ctx))]
                audit=audit_public_messages(messages)
                audit.update(model='DaCapo original trained '+args.model,prompt_version=PROMPT_VERSION,
                    model_source='https://github.com/corelab-src/dacapo',
                    dataset='CIFAR-10 public benchmark',dataset_source='https://www.cs.toronto.edu/~kriz/cifar.html',
                    payload_categories=['public model operator graph, shapes and public weight names','public CIFAR calibration min/max and heldout feasibility summary','system DSL grammar and source examples'],
                    user_authorization='2026-11-16: full autonomous plan execution and all subsequent real API calls')
                write_json(work/'provider-payload.json',dict(messages=messages))
                write_json(work/'public-payload-audit.json',audit)
                result.update(status='prepared',payload_audit=audit,actual_api_calls=0)
                return 0
            client=RenewingLLMClient(args.ledger,args.allowance,recorder=recorder,artifact_dir=work/'api',model='deepseek-flash',
                                    reasoning_effort='high',max_tokens=32768,timeout=600)
            with recorder.span('agent_generation_and_tools'):
                with observed_tools(agent,recorder):
                    from ..planners.transformer_session import save_state
                    state=(resumed_state if resumed_state is not None else
                        agent.new_agent_state(ctx,**agent_options(args.calls,args.tuner_seconds,tuner=getattr(args,'tuner',True))))
                    save_state(work/'SESSION.json',state)
                    if args.resume_source is not None:
                        reply=_recover_first_reply(args,result,save=save)
                        agent.agent_step(ctx,state,trace_path=work/'agent_trace.jsonl',reply=reply)
                        save_state(work/'SESSION.json',state)
                    while state['phase']!='done':
                        try:agent.agent_step(ctx,state,trace_path=work/'agent_trace.jsonl',client=client)
                        finally:
                            save_state(work/'SESSION.json',state)
                            if resumed_state is not None:
                                result['actual_api_calls']=client.calls;save()
                    selected=agent.agent_result(state)
            confirmation=selected['confirm']['selected']
            safe=dict(selected,confirm=dict(selected['confirm'],selected=asdict(confirmation) if confirmation else None))
            result.update(agent=safe,actual_api_calls=client.calls,status=selected['status'],internal_timing=timing_snapshot())
            result['quality_acceptance']=final_quality_acceptance(safe['confirm'])
            save()
            if selected['best_program'] is not None and result['quality_acceptance']['passed']:
                source=selected['best_program'];program=parse(source)
                (work/'program.ptdsl').write_text(source)
                # Frozen selection is followed by independent exact and packed arithmetic checks on heldout probes.
                with recorder.span('strict_packed_arithmetic'):
                    for i in range(3):
                        image=ctx.val_images[i:i+1]
                        reference=reference_forward(info.graph,{info.graph.inputs[0]:image})[info.graph.outputs[0]]
                        exact=_packed_logits(program,ctx.weights,image,act_mode='exact')
                        approximate=_packed_logits(program,ctx.weights,image,act_mode='program')
                        tensor=run_tensor(program,ctx.weights,image,cache=False)
                        row=dict(sample_id='test:'+str(9000+i),exact=check_math(reference,exact),approximate=check_math(tensor,approximate))
                        result['strict_math_probes'].append(row);save()
                        if not row['exact']['passed'] or not row['approximate']['passed']:raise ValueError('Strict packed arithmetic check failed')
                result.update(frozen=True,selected_program=str(work/'program.ptdsl'),boots_pred=selected['best']['boots_pred'],
                              latency_pred_us=selected['best']['latency_pred_us'])
            result['status']='ok' if result['frozen'] else 'quality_rejected'
    except BaseException as exc:
        result.update(status='error',error=f'{type(exc).__name__}: {exc}',frozen=False);traceback.print_exc()
    finally:
        result.update(elapsed_ns=time.perf_counter_ns()-started,worker_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save()
    return 0 if result['status'] in ('ok','confirm_failed','quality_rejected') else 1


def append_record(registry,row):
    with registry.open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def outer(args):
    begin=time.perf_counter_ns()
    out=args.out.resolve()
    if out.exists():raise ValueError('Preserve earlier autonomous experiments')
    out.mkdir(parents=True);work=out/'work';work.mkdir();(work/'tmp').mkdir()
    r=StageRecorder(out/'events.jsonl');source=sources(args.model,model_info=getattr(args,'model_info',None),calibration=getattr(args,'calibration',None))
    ledger=paths.RESULTS_DIR/'llm_calls.jsonl';allowance=paths.RESULTS_DIR/'autonomous-api-allowance.json'
    before=len(call_records(ledger));policy=dict(ceiling=None) if args.prepare_only else ensure_allowance(allowance,ledger,required=args.calls)
    result=dict(round=ROUND,model=args.model,status='running',api_before=before,api_ceiling=policy['ceiling'],processes=[],worker=None,
                memory_gib=args.memory_gib,tuner_enabled=getattr(args,'tuner',True),error=None,**dict.fromkeys(FORMAL,None))
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','NIX_','__NIX_','HECATE_')) and k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(work/'tmp'))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    def launch(stage,command,seconds):
        with r.span(stage,command=command):
            print(json.dumps(dict(stage=stage,state='started')),flush=True)
            obs=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/(stage+'.log'),resources=out/(stage+'-resources.jsonl'),
                            timeout=seconds,mem_gib=args.memory_gib,fsize_gib=8,env=env)
            result['processes'].append(dict(stage=stage,argv=command,observation=obs))
            if obs['exit_code'] or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']:raise RuntimeError('Supervised phase failed: '+stage)
    rev='unknown';child=[]
    try:
        with r.span('autonomous_common_agent'):
            launch('git-revision',['git','rev-parse','--short','HEAD'],30);rev=(out/'git-revision.log').read_text().strip()
            if args.preflight_evidence is None:
                launch('budget-tests',[sys.executable,'-B','-m','unittest','discover','-s','pt_agent/tests','-p','test_r28_*.py','-v'],120)
            else:
                evidence=json.loads((args.preflight_evidence/'result.json').read_text())
                audit=json.loads((args.preflight_evidence/'work/agent/public-payload-audit.json').read_text())
                if (evidence['status']!='ok' or evidence['api_delta']!=0 or audit['destination']!='https://api.deepseek.com/chat/completions'
                    or audit['model']!='DaCapo original trained '+args.model or audit.get('prompt_version')!=PROMPT_VERSION):
                    raise ValueError('Public-payload preflight evidence does not match this request')
                result['public_payload_preflight']=str(args.preflight_evidence)
                result['public_payload_categories']=audit['payload_categories']
                result['budget_tests_reused']=str(args.preflight_evidence/'budget-tests.log')
            command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(args.seconds),'--python',str(paths.INTAKE_PYTHON),'--',
                     '-B','-m','pt_agent.experiments.r28_common_agent','--worker','--work-dir',str(work/'agent'),
                     '--model',args.model,'--ledger',str(ledger),'--allowance',str(allowance),'--calls',str(args.calls),
                     '--tuner-seconds',str(args.tuner_seconds),'--memory-gib',str(args.memory_gib)]
            if not getattr(args,'tuner',True):command+=['--no-tuner']
            if args.prepare_only:command+=['--prepare-only']
            if args.resume_source is not None:command+=['--resume-source',str(args.resume_source),'--resume-result',str(args.resume_result)]
            for key in ('model_info','calibration'):command+=['--'+key.replace('_','-'),str(source[key])]
            launch('real-agent',command,args.seconds+60)
            result.update(status='ok')
    except BaseException as exc:result.update(status='error',error=f'{type(exc).__name__}: {exc}')
    finally:
        wp=work/'agent/worker-result.json'
        if wp.exists():result['worker']=json.loads(wp.read_text())
        events=read_events(out/'events.jsonl');parent=next((e['event_id'] for e in events if e.get('stage_id')=='real-agent' and e.get('type')=='begin'),None)
        for e in read_events(work/'agent/events.jsonl'):
            if e.get('type') in ('begin','end') and e.get('parent_id') is None:e['parent_id']=parent
            child.append(e)
        timing=summarize_events(events+child,termination_observed_ns=time.perf_counter_ns())
        write_json(out/'timing.json',timing)
        after=call_records(ledger);result.update(api_after=len(after),api_delta=len(after)-before,recorded_outer_seconds=(time.perf_counter_ns()-begin)/1e9,
                                              complete_batch_elapsed_seconds=None,source={k:str(v) for k,v in source.items()},
                                              completion_path=str(out/'completion.json'),artifact_bytes=0,
                                              artifact_bytes_scope='required artifacts before completion marker; final case total is completion.json:case_artifact_bytes')
        for _ in range(12):
            write_json(out/'result.json',result)
            total=artifact_bytes(out)
            if total==result['artifact_bytes']:break
            result['artifact_bytes']=total
        else:raise RuntimeError('Required artifact size did not stabilize')
        registry=paths.RESULTS_DIR/'registry.jsonl';now=datetime.now(timezone.utc).isoformat()
        existing={json.loads(line)['metrics'].get('global_call') for line in registry.read_text().splitlines() if line.strip() and json.loads(line).get('mode')=='llm-call'}
        for i,call in enumerate(after,1):
            if i>before and i not in existing:append_record(registry,dict(run_id=out.name+'-api-'+str(i),round=ROUND,model=args.model,mode='llm-call',git_rev=rev,metrics=dict(call,global_call=i),artifacts_dir=str(out),ts=call['ts']))
        append_record(registry,dict(run_id=out.name,round=ROUND,model=args.model,mode='public-payload-audit' if args.prepare_only else 'autonomous-common-agent',git_rev=rev,
            metrics=dict(status=result['status'],api_delta=result['api_delta'],frozen=(result['worker'] or {}).get('frozen'),
                         recorded_outer_seconds=result['recorded_outer_seconds'],completion_path=result['completion_path'],
                         **dict.fromkeys(FORMAL,None)),artifacts_dir=str(out),ts=now))
        completion=persist_case_completion(out,round_label=ROUND,started_ns=begin,status=result['status'],
                                           timing=timing,required_artifact_bytes=result['artifact_bytes'])
        print(json.dumps(dict(status=result['status'],api_delta=result['api_delta'],error=result['error'],out=str(out),
                             completion=result['completion_path'],complete_case_elapsed_seconds=completion['complete_case_elapsed_seconds'])),flush=True)
    return 0 if result['status']=='ok' else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--model',choices=MODELS,default='resnet20_silu')
    p.add_argument('--preflight-evidence',type=Path);p.add_argument('--out',type=Path);p.add_argument('--work-dir',type=Path);p.add_argument('--model-info',type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--resume-source',type=Path);p.add_argument('--resume-result',type=Path)
    p.add_argument('--resume-session',type=Path);p.add_argument('--resume-case',type=Path)
    p.add_argument('--ledger',type=Path);p.add_argument('--allowance',type=Path);p.add_argument('--calls',type=int,default=6);p.add_argument('--tuner-seconds',type=int,default=600)
    p.add_argument('--no-tuner',dest='tuner',action='store_false',default=True,
                   help='Disable numerical tuning for a new session; evaluation and final quality checks remain enabled')
    p.add_argument('--memory-gib',type=int,default=6);p.add_argument('--seconds',type=int,default=7200)
    a=p.parse_args(argv)
    if not a.worker and (a.model_info is None)!=(a.calibration is None):p.error('Explicit intake requires both --model-info and --calibration')
    if (a.resume_source is None)!=(a.resume_result is None):p.error('Recovery requires both the actual prior program and result')
    if (a.resume_session is None)!=(a.resume_case is None):p.error('Whole-session continuation requires its retained SESSION and original case')
    if a.resume_session is not None and (not a.worker or a.prepare_only or a.resume_source is not None or not a.tuner):
        p.error('Whole-session continuation is a worker operation retaining the original hybrid options')
    if not 1<=a.calls<=6 or not 4<=a.memory_gib<=12:p.error('Use 1..6 requests per session and 4..12 GiB within the 16 GiB host')
    return worker(a) if a.worker else outer(a)


if __name__=='__main__':raise SystemExit(main())
