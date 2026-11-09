"""E7 serial non-report confirmation study and two bounded Qwen sessions."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from ..paths import RESULTS_DIR
from . import e6_llm_select_tuning as e6
from . import e6r2_resume_after_outage as e6r2

read, write, ledger_rows = e6.read, e6.write, e6.ledger_rows
API_START, API_CEILING = 365, 403
NEW_LABELS = ('q1', 'q2')


def register(model, mode, out, metrics):
    from ..registry import append, git_rev, new_run_id
    append(dict(run_id=new_run_id('E7', model, mode), round='E7', model=model, mode=mode,
        git_rev=git_rev(), metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))


def inventory():
    models = {name: str(value['model']) for name, value in e6.configurations().items()}
    sessions = []
    roots = dict(E5c=RESULTS_DIR/'E5c-agent-loop-placement-20261106',
                 E5d=RESULTS_DIR/'E5d-agent-sessions-20261106',
                 E6=RESULTS_DIR/'E6-llm-select-20261107',
                 E6r2=RESULTS_DIR/'E6r2-resume-20261107')
    for model in ('bert', 'qwen'):
        for family, root, relative, acceptance in (
            ('E5c', roots['E5c'], f'{model}/replay-session-01', f'{model}/acceptance'),
            ('E5d', roots['E5d'], f'{model}/session', f'{model}/acceptance'),
            ('E6a', roots['E6'], f'e6a/{model}/E6a/session', f'e6a/{model}/E6a/acceptance'),
            ('E6-s1', roots['E6'], f'e6b/{model}/s1/session', f'e6b/{model}/s1/acceptance'),
            ('E6r2-a', roots['E6r2'], f'e6r2-a/{model}/E6r2-a/session', f'e6r2-a/{model}/E6r2-a/acceptance')):
            sessions.append(dict(id=family+'-'+model, family=family, model=model,
                model_dir=models[model], session=str(root/relative), acceptance=str(root/acceptance), root=str(root)))
    for model, label in e6r2.ORDER:
        root=roots['E6r2']; prefix=root/'e6r2-b'/model/label
        sessions.append(dict(id='E6r2-b-'+model+'-'+label, family='E6r2-b', model=model,
            label=label, model_dir=models[model], session=str(prefix/'session'),
            acceptance=str(prefix/'acceptance'), root=str(root)))
    frozen = []
    base=RESULTS_DIR/'E5b-program-placement-20261106'
    for model in ('bert','qwen'):
        original=Path(read(base/model/'REPORT.json')['source']).parent
        frozen.append(dict(id='E5b-'+model, family='E5b', model=model, model_dir=models[model],
            program=str(base/model/'program.ptdsl'), session=str(original),
            acceptance=str(original.parent/'acceptance-replay-session-01')))
    for case in sessions:
        marker=Path(case['session'])/'FROZEN.json'
        if marker.exists():
            frozen.append(dict(case, program=read(marker)['program']))
    return dict(sessions=sessions, frozen=frozen, models=models)


def context(case):
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context
    from .a1_transformer_agent import samples_for
    model=Path(case['model_dir']);info=ModelInfo.load(model)
    ctx=build_context(info, read(model/'calibration.json'), samples=samples_for(model,info))
    retained=Path(case['session'])/'SESSION.json'
    state=read(retained) if retained.exists() else {}
    ctx.prompt_version=state.get('prompt_version','r8-v11')
    ctx.placement=state.get('placement','program-plan')
    ctx.tune_search=state.get('tune_search','sweep')
    return ctx, state


def serial_child(root, case, part, out, *, extra=()):
    out.mkdir(parents=True, exist_ok=False)
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
        '-m','pt_agent.experiments.e7_confirm_tail_check','--worker',part,
        '--root',str(root),'--case-id',case['id'],'--out',str(out),*extra]
    started=time.monotonic()
    print('E7_STEP='+case['id']+' '+part+' '+str(out),flush=True)
    with (out/'run.log').open('w') as stream:
        status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    report=read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False,error='Missing worker report')
    report.update(exit=status.returncode,seconds=time.monotonic()-started)
    write(out/'REPORT.json',report);register(case['model'],part,out,report)
    if status.returncode or not report.get('complete'):
        raise RuntimeError(case['id']+' '+part+': '+str(report.get('error')))
    return report


def run_worker(args):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    from ..planners.confirm import Candidate, build_non_report_context, confirm_select, tail_check
    from ..planners.transformer_session import json_safe
    manifest=read(args.root/'manifest.json')
    collection=manifest['frozen'] if args.worker=='tail' else manifest['sessions']
    case=next(item for item in collection if item['id']==args.case_id)
    report=dict(complete=False,part=args.worker,case_id=args.case_id,api_calls=0,report_images=0)
    try:
        ctx,state=context(case)
        if args.worker=='tail':
            report.update(tail=tail_check(Path(case['program']).read_text(),build_non_report_context(ctx),engine_profile='tensor'))
        elif args.worker=='reconfirm':
            candidates=([Candidate(**item) for item in read(args.pool)] if args.pool else
                        [Candidate(**item) for item in state['pool']])
            result=confirm_select(candidates,ctx,engine_profile='tensor',confirm_rule='c2')
            selected=result['selected']
            result['selected']=asdict(selected) if selected is not None else None
            old_program=Path(case['session'])/'program.ptdsl'
            changed=(selected is not None and (not old_program.exists() or selected.source!=old_program.read_text()))
            report.update(confirmation=result,program_changed=changed,pool_size=len(candidates),
                          original_frozen=old_program.exists(),confirm_rule='c2')
            if selected is not None:
                destination=args.out/'program.ptdsl';destination.write_text(selected.source)
                write(args.out/'FROZEN.json',dict(program=str(destination),origin=selected.origin,
                    iteration=selected.iteration,confirm=result,prompt_version=ctx.prompt_version,
                    engine_profile='tensor',tune_policy='new_best',tune_rule=state.get('tune_rule','v8-6.24'),
                    confirm_rule='c2',split_version=ctx.model_info.split.get('version','split-v1')))
        else:
            from ..dsl.parser import parse
            from ..dsl.printer import print_program
            from ..planners.agent_tools import apply_placement
            from ..planners.transformer_tuner import tune_rule_scope,convergent_rule
            from ..planners.tuner import tune
            from ..planners.tune_replay import SavedTuneClient,ReplayMismatch
            rows=[json.loads(line) for line in (Path(case['session'])/'agent_trace.jsonl').read_text().splitlines()]
            row=next(r for r in rows if r['iteration']==args.iteration)
            original_row=next(r for r in rows if r['iteration']==(row.get('tuned_reused_from') or args.iteration))
            raw=json.loads(original_row['messages'][0]['content'])['program']
            source=apply_placement(raw,ctx)
            search=state.get('tune_search','sweep');ctx.tune_search=search
            rule=state.get('tune_rule','v8-6.24');client=None
            if search=='llm-select':
                folder=Path(case['root'])/'calls'/case['model']/'t1'/state['session_label']
                limit=(8 if case['model']=='bert' else 16) if case['family']=='E6r2-a' else 12 if case['family']=='E6r2-b' else 20
                client=SavedTuneClient(folder,args.out/'requests',iteration=original_row['iteration'],limit=limit)
            started=time.monotonic()
            try:
                with tune_rule_scope(rule):
                    result=tune(parse(source),ctx,budget_s=2400,max_sweeps=8 if convergent_rule(rule) else 1,
                                engine_profile='tensor',client=client,iteration=original_row['iteration'])
            except ReplayMismatch as exc:
                report.update(complete=True,status='replay_stopped',error=str(exc),code_error=False,
                              iteration=args.iteration,search=search,seconds=time.monotonic()-started)
            else:
                if case['model']=='qwen' and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>3.8*1024**3:
                    raise MemoryError('Qwen tuning exceeded 3.8 GiB')
                final=next(c for c in state['pool'] if c['iteration']==args.iteration and c['origin']=='tuned')['source']
                record={k:v for k,v in asdict(result).items() if k!='program'}
                write(args.out/'TUNE.json',json_safe(record))
                (args.out/'program.ptdsl').write_text(result.source)
                report.update(status='replayed',iteration=args.iteration,search=search,final_source_equal=result.source==final,
                    program=str(args.out/'program.ptdsl'),accepted_states=result.accepted_states,
                    final_candidate=asdict(Candidate(result.source,'tuned',args.iteration,result.cost_after['latency_pred_us'],result.cost_after['boots_pred'])) if result.feasible else None,
                    seconds=time.monotonic()-started)
                if result.source!=final:
                    from ..planners.tune_replay import differences
                    report['source_differences']=differences(final.splitlines(),result.source.splitlines())
        report['complete']=True
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc),failure_class='resource' if isinstance(exc,MemoryError) else 'code')
        import traceback
        traceback.print_exc()
    report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write(args.out/'REPORT.json',json_safe(report))
    return 0 if report['complete'] else 1


def original_report(case):
    base=Path(case['acceptance'])/'report'
    for path in (base/'REPORT.json',base/'report/REPORT.json'):
        if path.exists():
            report=read(path)
            if 'engines' in report:return {e:{k:r.get(k) for k in ('logit_rel_err','agree','total','feasible')} for e,r in report['engines'].items()}
    return {}


def original_selection(case):
    path=Path(case['session'])/'AGENT.json'
    if not path.exists():return None
    return read(path).get('confirm',{}).get('selected')


def freeze_acceptance(root,case,choice,stage):
    if not choice['confirmation']['selected']:return dict(status='not_frozen')
    if not choice['program_changed']:
        native_path=Path(case['acceptance'])/'pars/REPORT.json'
        boots=read(native_path).get('boots') if native_path.exists() else None
        if case['family']=='E6-s1':
            boots=read(RESULTS_DIR/'E6r2-resume-20261107/E6r2.json')['E6_s1'][case['model']]['boots']
        return dict(status='reused',artifacts_dir=case['acceptance'],report100=original_report(case),boots=boots)
    source=root/'e7a'/case['id']/stage
    parent=source/'acceptance'
    for part in ('evaluate','cost','pars'):
        out=parent/('report' if part=='evaluate' else part)
        if part=='pars':command=e6r2.native_command(root,case['model'],Path(case['model_dir']),source/'program.ptdsl',out)
        else:command=e6.base_command(part,out,case['model'],Path(case['model_dir']))+[
            '--root',str(parent),'--program',str(source/'program.ptdsl'),'--frozen',str(source/'FROZEN.json')]
        child_command(command,out,case['model'],case['id']+'-'+stage+'-'+part)
    return dict(status='evaluated',artifacts_dir=str(parent),report100=original_report(dict(acceptance=str(parent))),
                boots=read(parent/'pars/REPORT.json')['boots'])


def child_command(command,out,model,mode):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    print('E7_STEP='+model+' '+mode+' '+str(out),flush=True)
    with (out/'run.log').open('w') as stream:status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    report=read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False,error='Missing child report')
    report.update(exit=status.returncode,seconds=time.monotonic()-started)
    write(out/'REPORT.json',report);register(model,mode,out,report)
    if status.returncode or not report.get('complete'):raise RuntimeError(mode+': '+str(report.get('error')))
    return report


def run_a(root,manifest):
    data=dict(tails={},confirmations={},replays={},acceptance={})
    def checkpoint():write(root/'E7a.json',data)
    checkpoint()
    for case in manifest['frozen']:
        data['tails'][case['id']]=serial_child(root,case,'tail',root/'e7a'/case['id']/'tail')
        checkpoint()
    for case in manifest['sessions']:
        data['confirmations'][case['id']]=serial_child(root,case,'reconfirm',root/'e7a'/case['id']/'c1-c2')
        checkpoint()
    for case in manifest['sessions']:
        selected=data['confirmations'][case['id']]
        state=read(Path(case['session'])/'SESSION.json')
        iterations=sorted({c['iteration'] for c in state['pool'] if c['origin']=='tuned'})
        if not iterations or (not selected['program_changed'] and selected['confirmation']['selected'] is not None):
            data['replays'][case['id']]=dict(status='not_applicable',intermediate_states=0)
            checkpoint()
            continue
        copy_pool=list(state['pool']);reports=[];stopped=False
        for iteration in iterations:
            out=root/'e7a'/case['id']/'replay'/str(iteration)
            result=serial_child(root,case,'replay',out,extra=('--iteration',str(iteration)))
            reports.append(result)
            if result['status']=='replay_stopped':stopped=True;break
            copy_pool=[c for c in copy_pool if not(c['origin']=='tuned' and c['iteration']==iteration)]
            if result.get('final_candidate'):copy_pool.append(result['final_candidate'])
            for item in result['accepted_states']:
                copy_pool.append(dict(source=item['source'],origin='tuned-r'+str(item['round']),iteration=iteration,
                                      latency_pred_us=item['cost']['latency_pred_us'],boots_pred=item['cost']['boots_pred']))
        intermediate=sum(len(r.get('accepted_states',[])) for r in reports)
        entry=dict(status='replay_stopped' if stopped else 'replayed',runs=reports,intermediate_states=intermediate)
        if not stopped:
            path=root/'e7a'/case['id']/'pool-with-states.json';write(path,copy_pool)
            entry['confirmation']=serial_child(root,case,'reconfirm',root/'e7a'/case['id']/'full-c2',extra=('--pool',str(path)))
        data['replays'][case['id']]=entry
        checkpoint()
    for case in manifest['sessions']:
        choices=[('c1-c2',data['confirmations'][case['id']])]
        if 'confirmation' in data['replays'][case['id']]:choices.append(('full-c2',data['replays'][case['id']]['confirmation']))
        results={}
        for stage,choice in choices:results[stage]=freeze_acceptance(root,case,choice,stage)
        data['acceptance'][case['id']]=results
        checkpoint()
    write(root/'E7a.json',data);register('all','e7a',root/'e7a',dict(complete=True,api_calls=0,tails=len(data['tails']),sessions=len(data['confirmations'])))
    return data


def run_b(root,manifest):
    from ..planners.provider_retry import ProviderState
    policy=ProviderState(root/'provider-state.json');model=Path(manifest['models']['qwen'])
    for label in NEW_LABELS:
        e6r2.boundary(policy)
        session=root/'e7b/qwen'/label/'session';session.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-B','-m','pt_agent.experiments.a1_transformer_agent','--part','session',
            '--prompt-version','r8-v11','--tune-rule','v8-6.24','--placement','program-plan',
            '--tune-search','llm-select','--confirm-rule','c2','--session-label',label,'--tune-call-limit','9',
            '--model','qwen','--model-dir',str(model),'--out',str(session),'--call-dir',str(root/'calls'),
            '--budget-path',str(RESULTS_DIR/'llm_calls.jsonl'),'--provider-state',str(root/'provider-state.json')]
        print('E7_SESSION=qwen '+label,flush=True)
        with (session.parent/'session.log').open('w') as stream:status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        report=read(session/'REPORT.json') if (session/'REPORT.json').exists() else dict(complete=False,error='Missing session report')
        if status.returncode or not report.get('complete'):
            if report.get('failure_class')=='provider':e6r2.boundary(policy,failed_session=session);continue
            raise RuntimeError('Qwen '+label+': '+str(report.get('error')))
        if report['frozen']:
            case=dict(id='E7b-qwen-'+label,model='qwen',model_dir=str(model))
            for part in ('evaluate','cost','pars'):
                out=session.parent/'acceptance'/('report' if part=='evaluate' else part)
                command=(e6r2.native_command(root,'qwen',model,session/'program.ptdsl',out) if part=='pars' else
                    e6.base_command(part,out,'qwen',model)+['--root',str(session.parent/'acceptance'),
                        '--program',str(session/'program.ptdsl'),'--frozen',str(session/'FROZEN.json')])
                child_command(command,out,'qwen',case['id']+'-'+part)


def summarize(root,manifest,error=None):
    a=read(root/'E7a.json') if (root/'E7a.json').exists() else dict(tails={},confirmations={},replays={},acceptance={})
    b={}
    for label in NEW_LABELS:
        session=root/'e7b/qwen'/label/'session'
        row=e6r2.session_summary(session)
        row['confirm_rule']='c2';b[label]=row
    rows=ledger_rows(RESULTS_DIR/'llm_calls.jsonl')
    namespaces=[dict(namespace=['qwen',version,label],calls=len(list((root/'calls/qwen'/version/label/'attempts').glob('*.record.json'))),
                     limit=10 if version=='r8-v11' else 9) for label in NEW_LABELS for version in ('r8-v11','t1')]
    b_complete=all((root/'e7b/qwen'/label/'session/REPORT.json').exists() for label in NEW_LABELS)
    result=dict(round='E7',complete=error is None and b_complete,error=error,e7a=a,e7b=b,api_before=365,api_after=len(rows),api_delta=len(rows)-365,
        namespaces=namespaces,accounting_equal=len(rows)-365==sum(n['calls'] for n in namespaces),
        within_limits=len(rows)<=403 and all(n['calls']<=n['limit'] for n in namespaces))
    from ..planners.provider_retry import ProviderState
    result['provider_state']=ProviderState(root/'provider-state.json').read()
    result['end_to_end']={}
    historical=read(RESULTS_DIR/'E6r2-resume-20261107/E6r2.json')
    for case in manifest['sessions']:
        if case['family'] in ('E5d','E6-s1','E6r2-b'):
            previous=(historical['E5d_comparison'][case['model']] if case['family']=='E5d' else
                      historical['E6_s1'][case['model']] if case['family']=='E6-s1' else
                      historical['e6rb'][case['model']+' '+case['label']])
            row=dict(previous,confirm_rule='c1')
            result['end_to_end'][case['id']]=row
    result['end_to_end'].update({'E7b-qwen-'+label:row for label,row in b.items()})
    write(root/'E7.json',result)
    lines=['# E7 确认尾部检查','',f'API {result["api_delta"]}/38，累计{len(rows)}/450；异常：{error or "无"}。',
        '', '## 尾部检查','', '| 程序 | 确认集logit | 非report logit / agree | 通过 | 最坏5样本 | 已有report100 |', '|---|---|---|---|---|---|']
    for case in manifest['frozen']:
        value=a['tails'].get(case['id'],{}).get('tail',{});old=original_selection(case)
        confirms=read(Path(case['session'])/'AGENT.json').get('confirm',{}).get('evals',[])
        old_metric=next((r.get('logit_rel_err') for r in confirms if old and r['origin']==old['origin'] and r['iteration']==old['iteration']),None)
        lines.append(f'| {case["id"]} | {old_metric} | {value.get("logit_rel_err")} / {value.get("agree")} | {value.get("passed")} | {value.get("worst_samples")} | {original_report(case)} |')
    lines+=['','## 重新确认','', '| 会话 | c1来源/自举 | c2来源/自举 | 尾部拒绝 | 中间状态数 | 重放 / 验收 |','|---|---|---|---|---|---|']
    timings=[]
    for case in manifest['sessions']:
        initial=a['confirmations'].get(case['id'],{});replayed=a['replays'].get(case['id'],{})
        choice=replayed.get('confirmation',initial);confirmation=choice.get('confirmation',{})
        old=original_selection(case);new=confirmation.get('selected')
        def brief(c):return '—' if c is None else f'{c["origin"]}({c["iteration"]})/{c["boots_pred"]}'
        rejected=[dict(iteration=r['iteration'],origin=r['origin'],tail=r['tail']) for r in confirmation.get('evals',[]) if 'tail' in r and not r['tail']['passed']]
        lines.append(f'| {case["id"]} | {brief(old)} | {brief(new)} | {rejected} | {replayed.get("intermediate_states",0)} | {replayed.get("status")} / {a["acceptance"].get(case["id"])} |')
        for stage,record in (('C1',initial),('C1+C2',replayed.get('confirmation',{}))):
            if record:
                q=record['confirmation'];timings.append(dict(case=case['id'],stage=stage,evals=len(q['evals']),tail_evals=q.get('tail_evals',0),tail_seconds=q.get('tail_seconds',0)))
    lines+=['','## 端到端','', '| 会话 | confirm_rule | 主/精调调用 | 冻结 | 预测/导出/编译自举 | 预测延迟(us) | report100 | tokens主/精调 | 循环秒 | 验收 |','|---|---|---|---|---|---|---|---|---|---|']
    for name,row in result['end_to_end'].items():
        lines.append(f'| {name} | {row["confirm_rule"]} | {row["main_calls"]}/{row["tuning_calls"]} | {row["frozen_origin"]}/{row["frozen_iteration"]} | {row["boots"]} | {row["latency_pred_us"]} | {row["report100"]} | {row["tokens"]["main"]}/{row["tokens"]["tuning"]} | {row["loop_seconds"]} | {row["acceptance_status"]} |')
    for label in NEW_LABELS:
        path=root/'e7b/qwen'/label/'session/AGENT.json'
        if path.exists():
            q=read(path)['confirm'];timings.append(dict(case='E7b-'+label,stage='C2',evals=len(q['evals']),tail_evals=q.get('tail_evals',0),tail_seconds=q.get('tail_seconds',0)))
    result['confirmation_timings']=timings
    lines+=['','## 确认用时','', '| 会话 / 阶段 | 评估数 | 尾部评估数 | 尾部秒 |','|---|---:|---:|---:|']
    for v in timings:lines.append(f'| {v["case"]}/{v["stage"]} | {v["evals"]} | {v["tail_evals"]} | {v["tail_seconds"]} |')
    lines+=['','仅做描述。尾部检查不读取report；所有报告结果在选择结束后采集，真实CKKS0。']
    (root/'E7.md').write_text('\n'.join(lines)+'\n');write(root/'E7.json',result)
    return result


def prepare(root):
    manifest=inventory()
    if len(manifest['sessions'])!=14 or len(manifest['frozen'])!=15:raise RuntimeError('Historical coverage inventory differs')
    write(root/'manifest.json',manifest)
    review=root/'network-review';review.mkdir(exist_ok=True)
    previous=RESULTS_DIR/'E6r2-resume-20261107/network-review'
    payloads=[]
    for path in previous.glob('*qwen*.json'):
        if path.name=='manifest.json':continue
        shutil.copyfile(path,review/path.name);payloads.append(str(review/path.name))
    write(review/'manifest.json',dict(destination='https://api.deepseek.com/chat/completions',api_before=365,api_ceiling=403,
        max_calls=38,order=['qwen q1','qwen q2'],per_session=dict(main=10,tuning=9),retry=dict(wait_s=300,max_retries=2,circuit=3,cooldown_s=600),
        inherited_payloads=payloads,confirmation_results_returned_to_agent=False,confirm_rule='c2',prompt='r8-v11',tuning_prompt='t1',
        model='qwen',contents='Original model structure, calibration/heldout derived statistics, generated DSL, internal evaluation and legal tuning menu',
        authorization='User explicitly approved E7b with the remaining 38 E6 calls; retries count'))
    return manifest


def run(args):
    os.environ['PT_RESULTS']=str(RESULTS_DIR);root=args.root
    if not read(root/'OFFLINE_TESTS.json').get('complete'):raise RuntimeError('Offline tests must pass first')
    continuation=getattr(args,'continue_b',False)
    if continuation:
        if not read(root/'E7a-COMPLETE.json').get('complete'):raise RuntimeError('E7a must complete before E7b')
        if (root/'RUN_B.json').exists():raise RuntimeError('E7b already started')
    elif (root/'RUN.json').exists():raise RuntimeError('E7 already started')
    if len(ledger_rows(RESULTS_DIR/'llm_calls.jsonl'))!=365:raise RuntimeError('Approved ledger start changed')
    manifest=read(root/'manifest.json')
    if not continuation:write(root/'RUN.json',dict(api_before=365,api_ceiling=403,confirm_rule='c2'))
    error=None
    try:
        if not continuation:
            run_a(root,manifest)
            write(root/'E7a-COMPLETE.json',dict(complete=True,api_calls=0))
        if len(ledger_rows(RESULTS_DIR/'llm_calls.jsonl'))!=365:raise RuntimeError('E7a API count changed')
        if not getattr(args,'a_only',False):
            write(root/'RUN_B.json',dict(api_before=365,api_ceiling=403,per_session=dict(main=10,tuning=9)))
            run_b(root,manifest)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);write(root/'STOP.json',dict(error=error))
    result=summarize(root,manifest,error)
    write(root/'FINAL_CHECKS.json',{k:result[k] for k in ('api_before','api_after','api_delta','namespaces','accounting_equal','within_limits')})
    register('all','summary',root,{k:result[k] for k in ('complete','error','api_before','api_after','api_delta','accounting_equal','within_limits')})
    print('E7_DONE='+str(root/'E7.json')+' ERROR='+str(error),flush=True)
    return int(bool(error or not result['accounting_equal'] or not result['within_limits']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    stages=parser.add_mutually_exclusive_group()
    stages.add_argument('--a-only',action='store_true')
    stages.add_argument('--continue-b',action='store_true')
    parser.add_argument('--worker',choices=('tail','reconfirm','replay'))
    parser.add_argument('--case-id');parser.add_argument('--out',type=Path)
    parser.add_argument('--iteration',type=int);parser.add_argument('--pool',type=Path)
    args=parser.parse_args()
    if args.worker:return run_worker(args)
    if args.prepare:prepare(args.root);return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
