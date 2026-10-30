"""A1 bounded Transformer Agent and separate frozen-program evaluation."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

from ..paths import RESULTS_DIR
from ..run import _write


def read(path):
    return json.loads(Path(path).read_text())


def samples_for(model_dir,info):
    from ..data.samples import StoredSplits
    from ..paths import PLATFORM_DIR
    def missing(split):
        # U1 prepares the original sample split in its already installed intake
        # environment. It uses the saved eager/local-only HF loader unchanged.
        command=[str(PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'),'-B','-m','pt_agent.cli',
                 '--_worker','split','--_split',split,'--model',info.source['local_path'],
                 '--inputs',info.source['inputs_path'],'--out',str(model_dir)]
        subprocess.run(command,check=True,timeout=600)
    return StoredSplits(Path(model_dir)/'samples',info,missing)


def split_context(ctx,split,*,n=None):
    import numpy as np
    from ..analyzer import reference_forward
    arrays,labels=ctx.samples.get(split,n)
    reference=[]
    for start in range(0,len(next(iter(arrays.values()))),20):
        reference.append(reference_forward(ctx.graph,{key:v[start:start+20] for key,v in arrays.items()})[ctx.graph.outputs[0]])
    return replace(ctx,val_images=arrays[ctx.graph.inputs[0]],val_inputs=arrays,val_labels=labels,
                   val_exact_logits=np.concatenate(reference),val_split=split,val_exact_top1=None)


def evaluate_frozen(info,calibration,samples,program_path,directory,*,frozen,prompt_version='r8-v7',mock_report=None):
    """One post-freeze report access, two independent engines, no selection."""
    import numpy as np
    from ..dsl.parser import parse
    from ..planners.agent_tools import build_context
    from ..transformer_validation import validate
    from ..tensor_sim import clear_caches
    from ..planners.transformer_session import json_safe
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'REPORT.json'
    if path.exists():
        saved=read(path)
        if saved.get('complete') and saved['program']==str(program_path):return saved
        raise RuntimeError('Retained report evaluation is incomplete; do not repeat it')
    marker=read(frozen)
    from ..data.samples import report_count,split_version
    count=report_count(info)
    if split_version(info)=='split-v2' and marker.get('split_version')!='split-v2':
        raise ValueError('Frozen program and report sample partition differ')
    if Path(marker['program']).read_text()!=Path(program_path).read_text():
        raise ValueError('Report requires the unchanged frozen program')
    program=parse(Path(program_path).read_text())
    if prompt_version in ('r8-v8','r8-v9','r8-v10'):
        from ..dsl.interface import require_interface
        require_interface(program,info)
    _write(path,dict(complete=False,program=str(program_path),frozen=str(frozen),stage='loading-report'))
    ctx=build_context(info,calibration,samples=samples);ctx.prompt_version=prompt_version
    from ..margin import require_margin
    require_margin(program,ctx)
    ctx=split_context(ctx,'report',n=count)
    if len(ctx.val_images)!=count:raise ValueError(f'A1 requires exactly {count} report samples')
    result=dict(complete=False,program=str(program_path),report_images=count,engines={})
    if split_version(info)=='split-v2':result['split_version']='split-v2'
    outputs={}
    saved_mock=None
    if mock_report is not None:
        saved_mock=read(mock_report)
        if (not saved_mock.get('complete') or saved_mock.get('program')!=str(program_path)
                or saved_mock.get('engine')!='mock' or saved_mock.get('report_images')!=count
                or not saved_mock.get('release_checks') or not all(saved_mock['release_checks'].values())):
            raise ValueError('Mock reuse requires the same completed batch-release report')
        result['reused_mock_report']=str(mock_report)
    for engine in ('tensor','mock'):
        clear_caches();captured=[]
        if engine=='mock' and saved_mock is not None:
            measured=saved_mock['validation']
            with np.load(Path(mock_report).parent/'outputs.npz') as data:
                captured.append(data['mock'])
        else:
            measured=validate(program,ctx,engine=engine,stop_on_divergence=False,
                              output_callback=captured.append)
        outputs[engine]=captured[0]
        reference=ctx.val_exact_logits
        denominator=float(np.max(np.abs(reference)))
        measured['logit_rel_err']=(float(np.max(np.abs(captured[0]-reference)))/denominator
                                   if denominator and np.isfinite(captured[0]).all() else None)
        measured.setdefault('agreement_pct',100*measured.get('agree',0)/measured['total'])
        result['engines'][engine]=measured
        _write(path,json_safe(result))
        if prompt_version in ('r8-v8','r8-v9','r8-v10'):
            from ..domain_criteria import require_analytic_coverage
            require_analytic_coverage(measured)
    left,right=outputs['tensor'],outputs['mock']
    finite=np.isfinite(left)&np.isfinite(right)
    divergence_equal=result['engines']['tensor']['diverged_indices']==result['engines']['mock']['diverged_indices']
    result['comparison']=dict(max_abs_diff=float(np.abs(left[finite]-right[finite]).max(initial=0)),
        equal=bool(divergence_equal and np.array_equal(np.isfinite(left),np.isfinite(right))
                   and np.all(np.abs(left[finite]-right[finite])<=1e-9*(1+np.abs(right[finite])))))
    np.savez_compressed(directory/'outputs.npz',**outputs)
    result['complete']=result['comparison']['equal'];_write(path,json_safe(result))
    if not result['complete']:raise RuntimeError('Frozen report Mock/tensor comparison differs')
    return result


def acceptance_args(model_dir,program,root,part):
    root=Path(root);out=root/part;out.mkdir(parents=True,exist_ok=True)
    return SimpleNamespace(model_dir=Path(model_dir),program=Path(program),out=out,
        evaluation=root/'report/REPORT.json',cost_report=root/'cost/REPORT.json',old_cost=None,
        compiler_report=root/'compiler/REPORT.json',size_report=root/'preflight/REPORT.json',
        diagnostic=False,shared=None,saved_plan=None,baseline_manager=None)


def acceptance_part(model_dir,program,root,part,*,prompt_version='r8-v7'):
    from . import u4_relative_acceptance as acceptance
    args=acceptance_args(model_dir,program,root,part)
    args.prompt_version=prompt_version
    path=args.out/'REPORT.json'
    if path.exists():
        result=read(path)
        if result.get('complete'):return result
        raise RuntimeError('Retained native step is incomplete; do not rerun it')
    result=getattr(acceptance,part)(args)
    _write(path,result)
    return result


def native_stage(model_dir,program,root,*,prompt_version='r8-v7'):
    result={}
    for part in ('cost','compiler','preflight'):
        result[part]=acceptance_part(model_dir,program,root,part,prompt_version=prompt_version)
        if not result[part]['complete']:raise RuntimeError('A1 native '+part+' incomplete')
    sizing=result['preflight']
    print('MANAGED_IR_BYTES='+str(sizing['uncompressed_bytes'])+' ROUTE='+sizing['route'],flush=True)
    result['managed']=acceptance_part(model_dir,program,root,'managed',prompt_version=prompt_version)
    return result


def account_calls(budget,directory,*,model,prompt_version='r8-v7',split_version=None):
    """Outer/system Python only: retain one registry row per completed call."""
    from ..registry import append,git_rev,load,new_run_id
    from .r8g_mlp_rerun import _calls
    budget=Path(budget);directory=Path(directory)
    markers=directory/'api-accounting.json'
    calls=_calls(budget,wait=False)
    if calls is None:return
    before=read(markers)['api_before'] if markers.exists() else len(calls)
    recorded={row['metrics'].get('global_call') for row in load() if row['mode']=='llm-call'}
    for number,call in enumerate(calls,1):
        if number<=before or number in recorded:continue
        append(dict(run_id=new_run_id('A1',model,'api-'+str(number)),round='A1',model=model,
                    mode='llm-call',git_rev=git_rev(),metrics=dict(call,global_call=number,
                        **(dict(split_version=split_version) if split_version is not None else {}),
                        **(dict(prompt_version=prompt_version,budget_namespace=[model,prompt_version])
                           if prompt_version in ('r8-v8','r8-v9','r8-v10') else {})),
                    artifacts_dir=str(directory),ts=call['ts']))


def limits():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def unit(args):
    import unittest
    files=sorted(Path('pt_agent/tests').glob('test_a1_*.py'))
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
    return dict(complete=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),skipped=len(result.skipped))


def prepare(args):
    import shutil
    from ..models.info import ModelInfo
    from ..planners import transformer_rule as rule
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..planners.agent_tools import build_context
    from ..planners.prompts import build_system_prompt,initial_message
    from ..planners.transformer_session import json_safe
    from ..llm import Config
    model_dir=args.out/'model'
    if model_dir.exists():raise FileExistsError(model_dir)
    model_dir.mkdir()
    shutil.copy2(args.model_dir/'model_info.json',model_dir/'model_info.json')
    shutil.copytree(args.model_dir/'graph',model_dir/'graph')
    shutil.copytree(args.model_dir/'samples',model_dir/'samples',
                    ignore=shutil.ignore_patterns('report*','confirm*'))
    shutil.copy2(args.model_dir/'calibration.json',model_dir/'calibration.json')
    info=ModelInfo.load(model_dir);calibration=read(model_dir/'calibration.json')
    r0=args.out/'R0';r1=args.out/'R1';r0.mkdir();r1.mkdir()
    original=args.r0_program.read_text();(r0/'program.ptdsl').write_text(original)
    _write(r0/'FROZEN.json',dict(program=str(r0/'program.ptdsl'),baseline='R0',original=str(args.r0_program)))
    selections={}
    source=rule.plan(info.graph,calibration,softmax_budget=True,softmax_reports=selections)
    _write(r1/'CHOICES.json',json_safe(selections))
    checked=check(parse(source),{key:v.shape for key,v in info.graph.weights.items()})
    if not checked.ok:raise RuntimeError('R1 typecheck failed')
    (r1/'program.ptdsl').write_text(source)
    _write(r1/'FROZEN.json',dict(program=str(r1/'program.ptdsl'),baseline='R1',rho=.25))
    ctx=build_context(info,calibration,samples=samples_for(model_dir,info));ctx.prompt_version='r8-v7'
    messages=[dict(role='system',content=build_system_prompt(ctx)),dict(role='user',content=initial_message(ctx))]
    body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=32768,timeout_seconds=600).settings()
    body['messages']=messages
    _write(args.out/'first-request.pretty.json',body)
    if args.r0_program.read_text()!=original:raise RuntimeError('R0 source changed')
    return dict(complete=True,model_dir=str(model_dir),r0=str(r0/'program.ptdsl'),r1=str(r1/'program.ptdsl'),
                r0_unchanged=True,request=str(args.out/'first-request.pretty.json'),endpoint='https://api.deepseek.com/chat/completions',
                per_model_calls=40,model=args.model,report_images=0,llm_calls=0)


def calibration_changes(graph,old,new):
    from ..analyzer import approximation_points
    rows=[]
    for _,name in approximation_points(graph):
        before,after=old[name],new[name]
        row=dict(point=name,old_min=before['min'],old_max=before['max'],
                 new_min=after['min'],new_max=after['max'])
        for field in ('min','max'):
            value=before[field]
            row[field+'_relative_change']=None if value==0 else (after[field]-value)/abs(value)
        width=before['max']-before['min']
        row['range_ratio']=None if width==0 else (after['max']-after['min'])/width
        rows.append(row)
    return dict(rows=rows,relative_change_formula='(new-old)/abs(old); null when old=0',
                range_ratio_formula='(new_max-new_min)/(old_max-old_min); null when old range=0')


def prepare_split_v2(args):
    """A1 §6.13: a new model directory and both baselines, without report access."""
    import shutil
    from ..models.info import ModelInfo
    from ..data.samples import calibration_split_v2
    from ..analyzer import calibrate
    from ..planners import transformer_rule as rule
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..dsl.interface import require_interface
    from ..planners.agent_tools import build_context
    from ..planners.prompts import build_system_prompt,initial_message
    from ..planners.transformer_session import json_safe
    from ..llm import Config
    model_dir=args.out/'model';model_dir.mkdir()
    record=read(args.model_dir/'model_info.json')
    record['split']=calibration_split_v2(record['split'])
    _write(model_dir/'model_info.json',record)
    shutil.copytree(args.model_dir/'graph',model_dir/'graph')
    (model_dir/'samples').mkdir()
    shutil.copy2(args.model_dir/'samples/heldout.npz',model_dir/'samples/heldout.npz')
    info=ModelInfo.load(model_dir);samples=samples_for(model_dir,info)
    arrays,_=samples.get('calibration')
    if len(arrays[info.graph.inputs[0]])!=83:raise ValueError('split-v2 requires 83 calibration inputs')
    calibration=calibrate(info.graph,arrays)
    _write(model_dir/'calibration.json',calibration)
    _write(args.out/'CALIBRATION_CHANGES.json',calibration_changes(
        info.graph,read(args.model_dir/'calibration.json'),calibration))
    programs={}
    for name,budget in (('R0',False),('R1',True)):
        directory=args.out/name;directory.mkdir();choices={}
        source=rule.plan(info.graph,calibration,softmax_budget=budget,softmax_reports=choices)
        program=parse(source);require_interface(program,info)
        checked=check(program,{key:value.shape for key,value in info.graph.weights.items()},
                      graph=info.graph,prompt_version='r8-v8')
        if not checked.ok:raise ValueError(name+' split-v2 typecheck failed')
        path=directory/'program.ptdsl';path.write_text(source)
        _write(directory/'CHOICES.json',json_safe(choices))
        _write(directory/'FROZEN.json',dict(program=str(path),baseline=name,split_version='split-v2'))
        programs[name.lower()]=str(path)
    ctx=build_context(info,calibration,samples=samples);ctx.prompt_version='r8-v8'
    body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=65536,timeout_seconds=600).settings()
    body['messages']=[dict(role='system',content=build_system_prompt(ctx)),dict(role='user',content=initial_message(ctx))]
    _write(args.out/'first-request.pretty.json',body)
    return dict(complete=True,model=args.model,split_version='split-v2',previous_split_version='split-v1',
        model_dir=str(model_dir),**programs,calibration_samples=83,heldout_samples=20,confirm_samples=20,
        future_report_samples=100,report_images=0,llm_calls=0,
        calibration_changes=str(args.out/'CALIBRATION_CHANGES.json'),request=str(args.out/'first-request.pretty.json'))


def session(args):
    from ..models.info import ModelInfo
    from ..llm import LLMClient
    from ..planners.transformer_session import ModelBudgetClient,run_session,output_tokens
    info=ModelInfo.load(args.model_dir)
    client=ModelBudgetClient(LLMClient(args.budget_path,model='deepseek-flash',reasoning_effort='high',
                max_tokens=output_tokens(args.prompt_version),max_total_calls=450),args.call_dir,
                model=args.model,prompt_version=args.prompt_version)
    result=run_session(info,read(args.model_dir/'calibration.json'),samples_for(args.model_dir,info),
                       client,args.out,max_iters=min(8,max(1,40-client.calls)),prompt_version=args.prompt_version,
                       tune_rule=args.tune_rule)
    return dict(complete=result['status']!='error' if args.prompt_version in ('r8-v8','r8-v9','r8-v10') else True,
                agent=result,llm_calls=client.calls,report_images=0,
                **(dict(prompt_version=args.prompt_version,budget_namespace=[args.model,args.prompt_version],
                        call_dir=str(client.directory),tune_rule=args.tune_rule) if args.prompt_version in ('r8-v8','r8-v9','r8-v10') else {}),
                frozen=result['status']=='feasible',remaining_calls=40-client.calls)


def session_step(args):
    """Inner process: compute one step and commit its state before exiting."""
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context
    from ..planners.transformer_session import (ModelBudgetClient,output_tokens,model_calls,model_call_offset,load_state,
        recover_session,migrate_tune_rule,json_safe,session_step as advance)
    info=ModelInfo.load(args.model_dir)
    ctx=build_context(info,read(args.model_dir/'calibration.json'),samples=samples_for(args.model_dir,info))
    ctx.prompt_version=args.prompt_version
    ctx.a1_model_call_offset=model_call_offset(args.model,args.prompt_version)
    if ctx.prompt_version not in ('r8-v8','r8-v9','r8-v10'):raise ValueError('Split orchestration requires r8-v8')
    calls_dir=args.call_dir/args.model/args.prompt_version
    used=model_calls(args.call_dir,args.model,args.prompt_version)
    if args.step=='recover':
        if args.tune_rule=='v8-6.17':
            from ..planners.transformer_session import recover_pending_tune
            response=calls_dir/f'call-{used:03d}.response.json'
            result=recover_pending_tune(args.resume_session,args.root,response)
        else:
            result=recover_session(ctx,args.resume_session,args.root)
    elif args.step=='migrate':
        target='v8-6.14' if args.tune_rule=='v8-6.14' else 'v8-6.11'
        record=migrate_tune_rule(args.root,next_model_call=used+1,target_rule=target)
        result=dict(complete=True,step='migrate',next_step='evaluate',migration=record,
                    tune_rule=record['new_rule'],state=str(args.root/'SESSION.json'))
    else:
        state_path=args.root/'SESSION.json'
        state=load_state(state_path) if state_path.exists() else None
        phase=state['phase'] if state is not None else 'evaluate'
        if phase!=args.step:raise ValueError('Requested step differs from checkpoint: '+phase)
        client=None
        saved=read(args.response_file) if args.response_file is not None else None
        replay14=bool(saved is not None and state is not None
                      and state.get('recovery',{}).get('section')=='6.9')
        if replay14:
            request=read(calls_dir/'call-014.request.json')
            if state['calls']!=5 or state['tune_rule']!='v8-pre-6.11' or request['messages']!=state['messages']:
                raise RuntimeError('Recovered call-14 request context differs')
        if saved is not None and state is not None and state.get('recovery',{}).get('section')=='6.12':
            request=read(calls_dir/'call-021.request.json')
            if state['calls']!=4 or state['tune_rule']!='v8-6.11' or request['messages']!=state['messages']:
                raise RuntimeError('Recovered call-21 request context differs')
        if phase=='evaluate' and saved is None:
            from ..llm import LLMClient
            client=ModelBudgetClient(LLMClient(args.budget_path,model='deepseek-flash',reasoning_effort='high',
                max_tokens=output_tokens(args.prompt_version),max_total_calls=450),args.call_dir,
                model=args.model,prompt_version=args.prompt_version)
        scan_marker=calls_dir/'first-v8-6.14-convergent-scan.json'
        first_convergent_tune=(args.model=='qwen' and phase=='tune' and args.tune_rule=='v8-6.14'
                              and first_convergent_scan(state,scan_marker))
        result=advance(ctx,args.root,client=client,reply=saved,max_iters=min(8,max(1,40-used)),
                       tune_rule=args.tune_rule)
        if args.model=='qwen' and phase=='tune' and args.tune_rule=='v8-6.14':
            result['first_convergent_tune']=first_convergent_tune
            if first_convergent_tune:
                finished=load_state(args.root/'SESSION.json')['history'][-1]['tuned']
                if finished['sweeps']>0:
                    _write(scan_marker,dict(state=str(args.root/'SESSION.json'),
                        iteration=result['iteration']-1,tune_rule='v8-6.14',sweeps=finished['sweeps'],
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        if replay14:
            rows=[json.loads(line) for line in (Path(state['recovery']['original'])/'agent_trace.jsonl').read_text().splitlines()]
            expected=rows[5]['fb'];expected.pop('stop',None)
            expected['validation']['analytic_domain_violation']=False
            actual=json.loads((args.root/'agent_trace.jsonl').read_text().splitlines()[-1])['fb']
            comparison=dict(complete=actual==expected,recorded=expected,reevaluated=actual,
                            llm_calls=0,request_context_equal=True)
            _write(args.root/'ROUND14_REEVALUATION.json',comparison)
            if not comparison['complete']:raise RuntimeError('Recovered call-14 feedback differs')
    if args.model=='qwen' and args.step=='tune' and args.tune_rule=='v8-6.17':
        result['qwen_s617_tune']=True
    result.update(llm_calls=model_calls(args.call_dir,args.model,args.prompt_version),report_images=0,
                  prompt_version=args.prompt_version,budget_namespace=[args.model,args.prompt_version])
    return result


def split_session(args):
    """Outer process only connects bounded processes and accounts for calls."""
    root=args.root or args.out
    root.mkdir(parents=True,exist_ok=True)
    state_path=root/'SESSION.json'
    steps=[];began=time.monotonic()
    version=read(args.model_dir/'model_info.json')['split'].get('version','split-v1')
    while True:
        state=read(state_path) if state_path.exists() else None
        if state is not None and state['phase']=='done':break
        phase=(state['phase'] if state is not None else 'recover' if args.resume_session else 'evaluate')
        recovery=state.get('recovery',{}) if state is not None else {}
        if (phase=='evaluate' and recovery.get('section')=='6.9' and state['calls']==6
                and state.get('tune_rule','v8-pre-6.11')=='v8-pre-6.11'):
            phase='migrate'
        number=len(list((root/'steps').glob('*')))+1
        step_dir=root/'steps'/f'{number:03d}-{phase}'
        step_dir.mkdir(parents=True)
        command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
            '-m','pt_agent.experiments.a1_transformer_agent','--inside','--part','session-step',
            '--step',phase,'--out',str(step_dir),'--root',str(root),'--model',args.model,
            '--prompt-version',args.prompt_version,'--model-dir',str(args.model_dir),
            '--call-dir',str(args.call_dir),'--budget-path',str(args.budget_path),
            '--tune-rule',state.get('tune_rule','v8-pre-6.11') if state is not None else args.tune_rule]
        if phase=='recover':command.extend(['--resume-session',str(args.resume_session)])
        saved_call=None
        if state is not None and recovery and phase=='evaluate':
            if recovery.get('section')=='6.9' and state['calls']==5:saved_call=14
            elif recovery.get('section')=='6.12' and state['calls']==4:saved_call=21
            elif recovery.get('section') not in ('6.9','6.12','6.17') and state['calls']==7:saved_call=8
        if saved_call is not None:
            saved=args.call_dir/args.model/args.prompt_version/f'call-{saved_call:03d}.response.json'
            if not saved.exists():raise FileNotFoundError(saved)
            command.extend(['--response-file',str(saved)])
        print('SESSION_STEP='+phase+' OUT='+str(step_dir),flush=True)
        started=time.monotonic()
        with (step_dir/'run.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            while True:
                try:code=process.wait(timeout=2);break
                except subprocess.TimeoutExpired:
                    account_calls(args.budget_path,args.out,model=args.model,prompt_version=args.prompt_version,split_version=version)
        account_calls(args.budget_path,args.out,model=args.model,prompt_version=args.prompt_version,split_version=version)
        path=step_dir/'REPORT.json'
        result=read(path) if path.exists() else dict(complete=False,error='Step did not finish')
        result.update(exit=code,seconds=time.monotonic()-started)
        _write(path,result)
        steps.append(dict(step=phase,report=str(path),exit=code,complete=result['complete']))
        _write(args.out/'STEPS.json',dict(steps=steps,state=str(state_path)))
        if code or not result['complete']:
            return dict(complete=False,steps=steps,state=str(state_path),error=result.get('error'),
                        report_images=0,seconds=time.monotonic()-began)
    agent=read(root/'AGENT.json')
    from ..planners.transformer_session import model_calls
    used=model_calls(args.call_dir,args.model,args.prompt_version)
    return dict(complete=agent['status']!='error',agent=agent,llm_calls=used,report_images=0,
        prompt_version=args.prompt_version,budget_namespace=[args.model,args.prompt_version],
        tune_rule=agent.get('tune_rule',args.tune_rule),tune_rule_migration=state.get('tune_rule_migration'),
        convergent_rule_migration=state.get('convergent_rule_migration'),
        call_dir=str(args.call_dir/args.model/args.prompt_version),frozen=agent['status']=='feasible',
        remaining_calls=40-used,steps=steps,state=str(state_path),seconds=time.monotonic()-began)


def evaluate(args):
    from ..models.info import ModelInfo
    info=ModelInfo.load(args.model_dir)
    return evaluate_frozen(info,read(args.model_dir/'calibration.json'),samples_for(args.model_dir,info),
                           args.program,args.out/'report',frozen=args.frozen,prompt_version=args.prompt_version,
                           mock_report=getattr(args,'mock_report',None))


def violation_replay(args):
    """Re-evaluate one retained response; compare only the approved stop change."""
    from .a1_v8_replay import context
    from ..planners.agent_tools import evaluate as evaluate_source
    from ..planners.transformer_session import json_safe
    from ..domain_criteria import capture_violations
    ctx=context(args)
    iteration=int(args.program.stem.rsplit('-',1)[1])
    retained={row['iteration']:row for row in map(json.loads,
        (args.resume_session/'agent_trace.jsonl').read_text().splitlines())}[iteration]['fb']
    expected=json.loads(json.dumps(retained))
    expected.pop('stop',None)
    expected['validation']['analytic_domain_violation']=False
    with capture_violations() as violations:
        actual=json_safe(evaluate_source(args.program.read_text(),ctx,engine='tensor',engine_profile='tensor'))
    unchanged=actual==expected
    complete=bool(unchanged and actual['stage']=='validation' and not actual['feasible']
                  and actual['validation']['diverged'] and not actual.get('stop'))
    result=dict(complete=complete,source=str(args.program),retained_session=str(args.resume_session),
                iteration=iteration,feedback=actual,retained_feedback=retained,
                only_stop_classification_changed=unchanged,domain_violations=json_safe(violations),
                llm_calls=0,report_images=0)
    _write(args.out/'REPLAY.json',result)
    return result


def decision_replay(args):
    """Read the retained feedback only; do not simulate new tuning feedback."""
    if len(args.legacy_sessions)!=2:raise ValueError('Decision replay requires the two retained sessions')
    table=[];infeasible_tunes=[];offset=0
    for session,directory in enumerate(args.legacy_sessions,1):
        rows=[json.loads(line) for line in (directory/'agent_trace.jsonl').read_text().splitlines()]
        if len(rows)!=(8 if session==1 else 6):raise ValueError('Decision replay requires 8 + 6 retained rows')
        minimum=None;has_tuned=False
        for row in rows:
            fb=row['fb'];cost=fb.get('cost') or {};validation=fb.get('validation') or {}
            new_best=bool(fb.get('feasible') and cost and
                          (minimum is None or cost['latency_pred_us']<minimum))
            if fb.get('feasible') and cost:
                minimum=cost['latency_pred_us'] if minimum is None else min(minimum,cost['latency_pred_us'])
            eligible=bool((fb.get('exact') or {}).get('ok') and validation and not validation['diverged']
                and fb['stage']=='complete' and (validation.get('boot_input_bound') is None
                or validation['boot_input_max_abs']<=validation['boot_input_bound']))
            triggered=bool(eligible and not fb.get('stop') and (new_best or not has_tuned))
            has_tuned=has_tuned or triggered
            call=offset+row['iteration'];old=row.get('tuned')
            table.append(dict(model_call=call,session=session,iteration=row['iteration'],
                stage=fb['stage'],feasible=bool(fb.get('feasible')),eligible=eligible,
                old_triggered=old is not None,new_triggered=triggered,new_llm_best=new_best,
                new_llm_min=minimum,
                return_position=('domain then finish if aligned infeasible; retained alignment feasibility unknown'
                    if triggered and not fb.get('feasible') else 'normal scan' if triggered else None)))
            if old is not None and not fb.get('feasible'):
                infeasible_tunes.append(dict(model_call=call,seconds=old['seconds'],
                    coarse_evals=old['coarse_evals'],full_evals=old['full_evals'],
                    changes=len(old['changes']),alignment_seconds=None,aligned_feasible=None))
        offset+=len(rows)
    report=dict(complete=True,rows=table,infeasible_tunes=infeasible_tunes,
        historical_infeasible_tune_seconds=sum(row['seconds'] for row in infeasible_tunes),
        net_saved_seconds=None,
        savings_reason='Alignment feasibility/time and newly triggered tune durations were not retained; no tune was rerun.',
        scope='Fixed original replies, feedback and session boundaries; new feedback/pool/early stopping not simulated.',
        llm_calls=0,report_images=0,tune_calls=0,tune_rule='v8-6.11')
    _write(args.out/'DECISIONS.json',report)
    return report


def budget_diagnostic(args):
    """A1 §6.12(1): reproduce the upstream budget with the retained inputs."""
    import numpy as np
    from ..dsl.parser import parse
    from ..transformer_tensor import run, polynomial
    from ..kernels import transformer as tr
    from .a1_v8_replay import context
    from ..planners.transformer_session import json_safe
    ctx=context(args);program=parse(args.program.read_text())
    op=next(op for op in program.ops if op.lineno==12 and op.kind=='softmax')
    captured={};roles=('inv','step_inv[0]','final_inv')
    def evaluate(point,role,value,attrs,mask):
        result=polynomial(value,attrs)
        if point.id==op.id:
            if role=='exp':captured['n_row']=mask.sum(axis=-1,keepdims=True)
            if role in roles:
                captured[role]=dict(x=value.copy(),c=(value*result).copy())
        return result
    run(program,ctx.weights,ctx.val_inputs,point_evaluator=evaluate)
    errors={}
    references={'inv':.9697875035498709,'step_inv[0]':.34512760125894737,
                'final_inv':3.5786682783811585e-6}
    for role in roles:
        prefix=role.split('[',1)[0];attrs=tr.approximation(op.attrs,prefix,'inv')
        grid=np.geomspace(*attrs['domain'],100001)
        relative=np.abs(grid*polynomial(grid,attrs)-1)
        maximum=float(relative.max());reference=references[role]
        errors[role]=dict(attrs=attrs,points=len(grid),max_relative_error=maximum,
            argmax=float(grid[int(relative.argmax())]),reference=reference,
            relative_difference=abs(maximum-reference)/reference)
    rows=[];rho=op.attrs['inv_budget'];domain=op.attrs['step_inv_domain']
    for sample in (7,10,11,14,15):
        x=captured['inv']['x'][sample].reshape(-1)
        c=captured['inv']['c'][sample].reshape(-1)
        n=captured['n_row'][sample].reshape(-1)
        squares=captured['step_inv[0]']['x'][sample].reshape(-1)
        for row,(xx,cc,nn,ss) in enumerate(zip(x,c,n,squares)):
            bound=float(cc*cc/nn);outside=bool(ss<domain[0] or ss>domain[1])
            rows.append(dict(sample_index=sample,row=row,x=float(xx),c=float(cc),n_row=int(nn),
                sum_p_squared=float(ss),c_squared_over_n=bound,
                cauchy_schwarz=bool(ss+1e-12>=bound),analytic_input_outside=outside,
                upstream_budget_outside=bool(cc<1-rho or cc>1+rho)))
    complete=all(r['cauchy_schwarz'] and (not r['analytic_input_outside'] or r['upstream_budget_outside'])
                 for r in rows) and all(e['relative_difference']<=1e-3 for e in errors.values())
    result=dict(complete=complete,source=str(args.program),split='heldout',samples=20,
        rows=rows,inverse_errors=errors,c_min=min(r['c'] for r in rows),
        violated_rows=sum(r['analytic_input_outside'] for r in rows),llm_calls=0,report_images=0)
    _write(args.out/'DIAGNOSTIC.json',json_safe(result))
    return json_safe(result)


def budget_replay(args):
    """Rejudge 21 retained evaluations; reuse their exact and static cost checks."""
    import copy
    from ..dsl.parser import parse
    from ..transformer_validation import validate
    from ..planners.transformer_session import json_safe
    from ..domain_criteria import capture_violations
    from .a1_v8_replay import context
    ctx=context(args);cases=[];offset=0
    for directory in args.legacy_sessions:
        trace=[json.loads(x) for x in (directory/'agent_trace.jsonl').read_text().splitlines()]
        observations={}
        for path in sorted((directory/'steps').glob('*evaluate/REPORT.json')):
            result=read(path)
            if result.get('domain_violations'):
                observations[result['session_calls']]=result['domain_violations'][0]
        for row in trace:
            cases.append((offset+row['iteration'],directory,row,observations.get(row['iteration'])))
        offset+=len(trace)
    if [x[0] for x in cases]!=list(range(1,22)):
        raise ValueError('A1 §6.12 requires the retained 8 + 8 + 5 evaluations')
    results=[]
    for number,directory,row,previous_violations in cases:
        before=row['fb'];after=copy.deepcopy(before);reports=[]
        source=directory/'sources'/f"iteration-{row['iteration']}.ptdsl"
        if before.get('validation') is not None:
            with capture_violations() as reports:
                measured=validate(parse(source.read_text()),ctx,engine='tensor')
            after['validation']={key:value for key,value in measured.items()
                                 if key not in ('rows','feasible','error')}
            after['feasible']=bool(after['exact']['ok'] and measured['feasible'])
            after.pop('stop',None)
            if measured['analytic_domain_violation']:after['stop']='analytic_domain_violation'
            if 'error' in measured:after['error']=measured['error']
            else:after.pop('error',None)
            after['stage']='validation' if measured['diverged'] else 'complete'
        after=json_safe(after)
        changed=[key for key in before.keys()|after.keys() if before.get(key)!=after.get(key)]
        old_first=(None if previous_violations is None else
            [sample['first_violation'] for sample in previous_violations['samples']])
        new_first=(json_safe([sample['first_violation'] for sample in reports[0]['samples']]) if reports else None)
        result=dict(model_call=number,source=str(source),changed_fields=changed,
            feedback_equal=before==after,old_feasible=before.get('feasible'),new_feasible=after.get('feasible'),
            old_stop=before.get('stop'),new_stop=after.get('stop'),old_first_available=previous_violations is not None,
            old_first=old_first,new_first=new_first,
            first_changed=None if old_first is None else old_first!=new_first,
            exact_and_cost_reused=True,feedback=after,recorded_feedback=before,
            domain_violations=json_safe(reports))
        _write(args.out/f'call-{number:03d}.json',result)
        results.append(result)
        _write(args.out/'REPLAY.json',dict(complete=False,rows=results,llm_calls=0,tune_calls=0,report_images=0))
        print(f'BUDGET_REPLAY call={number} changed={changed}',flush=True)
        if after.get('stop'):raise RuntimeError('A1 §6.7 '+after['stop'])
    result=dict(complete=True,rows=results,changed_calls=[r['model_call'] for r in results if not r['feedback_equal']],
        exact_and_cost_reused=True,llm_calls=0,tune_calls=0,report_images=0)
    _write(args.out/'REPLAY.json',result)
    return result


def pruning_comparison(record,original):
    """Compare the same saved representation, including dataclass DSL refs."""
    from ..planners.transformer_session import json_safe
    def accepted(history):
        return [{key:row[key] for key in ('phase','sweep','line','move','attrs') if key in row}
                for row in history if row.get('accepted') and row['phase'] in ('softmax','coarse')]
    def changes(items):
        return json_safe([{k:v for k,v in item.items() if k not in ('line_text','call_texts')}
                          for item in items])
    return dict(changes=changes(record['changes'])==changes(original['changes']),
        accepted_sequence=json_safe(accepted(record['history']))==json_safe(accepted(original['history'])),
        final_program=record['source']==original['source'],cost=record['cost_after']==original['cost_after'])


def first_convergent_scan(state,marker=None):
    # An infeasible start can return before any scan; keep watching the first
    # actual scan after such a return.
    if marker is not None and Path(marker).exists():return False
    return not any(row.get('tuned') and row['tuned'].get('sweeps',0)>0 for row in state['history'])


def tuning_peak_error(report):
    if report.get('qwen_s617_tune') and report['peak_rss_bytes']>3.8*(1<<30):
        return 'A1 §6.17 Qwen tuning peak exceeds 3.8 GiB'
    if (report.get('first_convergent_tune') and report.get('tune_rule')=='v8-6.14'
            and report['peak_rss_bytes']>3.8*(1<<30)):
        return 'A1 §6.14 Qwen tuning peak exceeds 3.8 GiB'
    return None


def pruning_recheck(args):
    """Reuse the completed first replay after fixing its comparison adapter."""
    record=read(args.root/'TUNER.json')
    comparison=read(args.root/'COMPARISON.json')
    original=next(row['tuned'] for row in map(json.loads,
        (args.resume_session/'agent_trace.jsonl').read_text().splitlines())
        if row['iteration']==int(Path(comparison['source']).stem.split('-')[-1]))
    checks=pruning_comparison(record,original)
    return dict(comparison,complete=all(checks.values()),comparison=checks,
        tune_seconds=comparison['seconds'],
        reused_result=str(args.root/'TUNER.json'),original_failure=str(args.root/'REPORT.json'),
        rerun_tune=False,llm_calls=0,report_images=0)


def tune_pruning(args):
    """Finite §6.14 replay against saved histories, with no provider or report read."""
    from dataclasses import asdict
    from .a1_v8_replay import context
    from ..dsl.parser import parse
    from ..dsl.interface import require_interface
    from ..dsl.typecheck import check
    from ..planners.transformer_tuner import tune, tune_rule_scope
    from ..planners.transformer_session import json_safe
    ctx=context(args)
    program=parse(args.program.read_text())
    require_interface(program,ctx.model_info)
    checked=check(program,ctx.weight_shapes,graph=ctx.graph,prompt_version='r8-v8')
    if not checked.ok:raise ValueError('Pruning replay typecheck failed: '+str(checked.diags))
    rows=[json.loads(line) for line in (args.resume_session/'agent_trace.jsonl').read_text().splitlines()]
    iteration=int(args.program.stem.split('-')[-1])
    original=next(row['tuned'] for row in rows if row['iteration']==iteration)
    began=time.monotonic()
    with tune_rule_scope('v8-6.14'):
        result=tune(program,ctx,budget_s=2400,max_sweeps=args.max_sweeps,
                    engine='tensor',engine_profile='tensor')
    wall=time.monotonic()-began
    (args.out/'program.ptdsl').write_text(result.source)
    record={key:value for key,value in asdict(result).items() if key!='program'}
    _write(args.out/'TUNER.json',json_safe(record))
    comparison=pruning_comparison(record,original)
    chain_final=next(row['tuned'] for row in rows if row['iteration']==4)
    report=dict(complete=args.max_sweeps!=1 or all(comparison.values()),source=str(args.program),
        original_session=str(args.resume_session),model_call=iteration+16,tune_rule='v8-6.14',
        max_sweeps=args.max_sweeps,sweeps=result.sweeps,converged=result.converged,seconds=wall,
        tune_seconds=wall,
        original_seconds=original['seconds'],pruned=sum(row.get('rejected')=='dominated' for row in result.history),
        changes=len(result.changes),coarse_evals=result.coarse_evals,full_evals=result.full_evals,
        cost_after=result.cost_after,comparison=comparison,original_chain_final_cost=chain_final['cost_after'],
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        feasible=result.feasible,llm_calls=0,report_images=0)
    _write(args.out/'COMPARISON.json',json_safe(report))
    if not report['complete']:raise RuntimeError('A1 §6.14 accepted sequence or final program differs')
    return json_safe(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('unit','prepare','prepare-v2','session','session-step','evaluate','cost','compiler','preflight','managed',
                                         'replay','tune-diagnostic','baselines','requests','violation-replay','decision-replay',
                                         'budget-diagnostic','budget-replay','tune-pruning','pruning-recheck',
                                         'mock-case20','mock-estimate20','mock-case10','mock-release20','alignment-diagnostic',
                                         'mock-memory','mock-memory-unit','mock-before',
                                         'mock-equivalence','mock-release-check','frozen-metrics','margin-audit','margin-requests',
                                         'log-requests','log-regrade','log-step','log-baseline-reuse','log-depths','log-observe-rejection'),required=True)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--model',choices=('bert','qwen'),default='bert')
    parser.add_argument('--prompt-version',choices=('r8-v7','r8-v8','r8-v9','r8-v10'),default='r8-v7')
    parser.add_argument('--legacy-sessions',type=Path,nargs='+',default=())
    parser.add_argument('--step',choices=('recover','evaluate','tune','confirm','migrate'))
    parser.add_argument('--tune-rule',choices=('v8-pre-6.11','v8-6.11','v8-6.14','v8-6.17'))
    parser.add_argument('--max-sweeps',type=int,choices=(1,8),default=1)
    for key in ('model_dir','r0_program','program','frozen','budget_path','call_dir','root','resume_session','response_file','mock_report'):
        parser.add_argument('--'+key.replace('_','-'),type=Path)
    args=parser.parse_args()
    args.tune_rule=args.tune_rule or ('v8-6.17' if args.prompt_version=='r8-v10' else
                                    'v8-6.14' if args.prompt_version in ('r8-v8','r8-v9') else 'v8-6.11')
    if args.inside:
        limits();report=dict(complete=False)
        from ..domain_criteria import capture_violations
        from ..planners.transformer_session import json_safe
        failed=False
        with capture_violations() as violations:
            try:
                if args.part in ('unit','prepare','prepare-v2','session','session-step','evaluate','violation-replay','decision-replay','budget-diagnostic','budget-replay','tune-pruning','pruning-recheck'):
                    report={'unit':unit,'prepare':prepare,'prepare-v2':prepare_split_v2,'session':session,'session-step':session_step,
                            'evaluate':evaluate,'violation-replay':violation_replay,
                            'decision-replay':decision_replay,'budget-diagnostic':budget_diagnostic,
                            'budget-replay':budget_replay,'tune-pruning':tune_pruning,
                            'pruning-recheck':pruning_recheck}[args.part](args)
                elif args.part in ('mock-case20','mock-estimate20','mock-case10','mock-release20','alignment-diagnostic'):
                    from .a1_mock_single_batch import diagnostic,alignment
                    report=(alignment if args.part=='alignment-diagnostic' else diagnostic)(args)
                elif args.part in ('mock-memory','mock-memory-unit','mock-before','mock-equivalence','mock-release-check','frozen-metrics'):
                    from .a1_mock_memory import diagnostic,heldout,frozen_metrics,unit as memory_unit
                    if args.part in ('mock-memory','mock-release-check'):
                        report.update(program=str(args.program),requested_split='report',
                                      report_images=100,llm_calls=0)
                    report=(frozen_metrics if args.part=='frozen-metrics' else memory_unit if args.part=='mock-memory-unit' else
                            heldout if args.part in ('mock-before','mock-equivalence') else diagnostic)(args)
                elif args.part in ('margin-audit','margin-requests'):
                    from . import a1_v9_margin
                    report=getattr(a1_v9_margin,'audit' if args.part=='margin-audit' else 'requests')(args)
                elif args.part in ('log-requests','log-regrade','log-step','log-baseline-reuse','log-depths','log-observe-rejection'):
                    from . import a1_v10_replay
                    report=getattr(a1_v10_replay,{'log-requests':'requests','log-regrade':'regrade',
                                                'log-step':'replay_step','log-baseline-reuse':'baseline_reuse',
                                                'log-depths':'depths','log-observe-rejection':'observe_rejection'}[args.part])(args)
                elif args.part in ('replay','tune-diagnostic','baselines','requests'):
                    from . import a1_v8_replay
                    report=getattr(a1_v8_replay,args.part.replace('-','_'))(args)
                else:
                    from ..manage.greedy import timing_log
                    with timing_log(args.out/'memory-stages.json'):
                        report=acceptance_part(args.model_dir,args.program,args.root,args.part,prompt_version=args.prompt_version)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                failed=True
                report.update(error=f'{type(exc).__name__}: {exc}',peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            if args.prompt_version in ('r8-v8','r8-v9','r8-v10'):
                report.setdefault('domain_violations',json_safe(violations))
        report.setdefault('peak_rss_bytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        peak_error=tuning_peak_error(report)
        if peak_error:
            failed=True
            report.update(complete=False,error=peak_error)
        if args.model_dir is not None:
            report.setdefault('split_version','split-v2' if args.part=='prepare-v2' else
                              read(args.model_dir/'model_info.json')['split'].get('version','split-v1'))
        if failed:_write(args.out/'STOP.json',report)
        _write(args.out/'REPORT.json',report)
        return 0 if report['complete'] else 1
    os.environ['PT_RESULTS']=str(RESULTS_DIR)
    from ..registry import append,git_rev,new_run_id
    run_id=new_run_id('A1',args.model,args.part)
    args.out=args.out or RESULTS_DIR/run_id
    args.out.mkdir(parents=True,exist_ok=False)
    print('RUN='+str(args.out),flush=True)
    if args.part=='session':
        from .r8g_mlp_rerun import _calls
        if args.budget_path is None:args.budget_path=RESULTS_DIR/'llm_calls.jsonl'
        _write(args.out/'api-accounting.json',dict(api_before=len(_calls(args.budget_path))))
    if args.part=='session' and args.prompt_version in ('r8-v8','r8-v9','r8-v10'):
        began=time.monotonic()
        report=split_session(args)
        report.setdefault('split_version',read(args.model_dir/'model_info.json')['split'].get('version','split-v1'))
        report.update(round='A1',exit=0 if report['complete'] else 1,seconds=time.monotonic()-began)
        _write(args.out/'REPORT.json',report)
        append(dict(run_id=run_id,round='A1',model=args.model,mode=args.part,git_rev=git_rev(),
                    metrics=report,artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat()))
        print('REPORT='+str(args.out/'REPORT.json'),flush=True)
        return report['exit']
    seconds=1200 if args.part=='unit' else 3600
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds),'--',
             '-m','pt_agent.experiments.a1_transformer_agent','--inside','--part',args.part,'--out',str(args.out),'--model',args.model,
             '--prompt-version',args.prompt_version,'--tune-rule',args.tune_rule]
    command.extend(['--max-sweeps',str(args.max_sweeps)])
    if args.step is not None:command.extend(['--step',args.step])
    for key in ('model_dir','r0_program','program','frozen','budget_path','call_dir','root','resume_session','response_file','mock_report'):
        if getattr(args,key) is not None:command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    if args.legacy_sessions:
        command.extend(['--legacy-sessions',*[str(path) for path in args.legacy_sessions]])
    began=time.monotonic()
    with (args.out/'run.log').open('w') as log:
        if args.part=='session':
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            while True:
                try:
                    code=process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    account_calls(args.budget_path,args.out,model=args.model,prompt_version=args.prompt_version)
            status=SimpleNamespace(returncode=code)
        else:status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    path=args.out/'REPORT.json'
    report=json.loads(path.read_text()) if path.exists() else dict(complete=False)
    report.update(round='A1',exit=status.returncode,seconds=time.monotonic()-began)
    report.setdefault('llm_calls',0);report.setdefault('report_images',0)
    _write(path,report)
    os.environ['PT_RESULTS']=str(RESULTS_DIR)
    if args.part=='session':account_calls(args.budget_path,args.out,model=args.model,prompt_version=args.prompt_version)
    append(dict(run_id=run_id,round='A1',model=args.model,mode=args.part,git_rev=git_rev(),
                metrics=report,artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(path),flush=True)
    return 0 if report['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
