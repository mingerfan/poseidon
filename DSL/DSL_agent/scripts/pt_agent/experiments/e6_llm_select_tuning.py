"""E6 serial replay/new sessions and post-freeze acceptance; no report feedback."""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from ..paths import RESULTS_DIR,REPO


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')
def ledger_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def configurations():
    from .e5d_agent_sessions_v11 import configurations as existing
    return existing()
def register(model,mode,path,result):
    from ..registry import append,git_rev,new_run_id
    append(dict(run_id=new_run_id('E6',model,mode),round='E6',model=model,mode=mode,
        git_rev=git_rev(),metrics=result,artifacts_dir=str(path),ts=datetime.now(timezone.utc).isoformat()))
def session_path(root,phase,model,label):return root/phase/model/label/'session'


def failure_class(code,report):
    """Report resource admission separately from implementation failures."""
    if report.get('failure_class')=='provider':return 'provider'
    if code in (124,137) or 'requires at least 10 GiB free disk space' in str(report.get('error')):
        return 'resource'
    return report.get('failure_class','code')


def prepare(root,old,models):
    """Capture the exact first selector request after the unchanged alignment."""
    from ..llm import Config
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..planners.agent import new_agent_state
    from ..planners.agent_tools import build_context,apply_placement
    from ..planners.transformer_tuner import tune,tune_rule_scope
    from .a1_transformer_agent import samples_for,limits
    limits()
    manifest=dict(destination='https://api.deepseek.com/chat/completions',api_before=259,
        api_ceiling=403,global_limit=450,max_calls=144,retries=0,models={},
        order=['E6a bert','E6a qwen','E6b bert s1','E6b qwen s1','E6b bert s2','E6b qwen s2'],
        e6a=dict(main_calls=0,tuning_calls_per_model=12,rounds_per_tune=3),
        e6b=dict(sessions=4,main_calls_per_session=10,tuning_calls_per_session=20))
    class Captured(Exception):pass
    for model,directory in models.items():
        info=ModelInfo.load(directory)
        ctx=build_context(info,read(directory/'calibration.json'),samples=samples_for(directory,info))
        ctx.prompt_version='r8-v11';ctx.placement='program-plan';ctx.tune_search='llm-select'
        state=new_agent_state(ctx,max_iters=10,engine_profile='tensor',tune_rule='v8-6.24')
        main_path=root/'network-review'/f'{model}-main-first-request.pretty.json'
        body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=65536,timeout_seconds=600).settings()
        body['messages']=state['messages'];write(main_path,body)
        tune_path=root/'network-review'/f'{model}-t1-first-request.pretty.json'
        class Capture:
            calls=0;limit=12
            def chat(self,messages,*,purpose):
                body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=16384,timeout_seconds=600).settings()
                body['messages']=messages;write(tune_path,body)
                write(tune_path.with_name(model+'-t1-context.json'),dict(purpose=purpose,program_call=4,api_calls=0))
                raise Captured()
        reply=read(old/'calls'/model/'r8-v11/call-004.response.json')
        source=apply_placement(json.loads(reply['content'])['program'],ctx)
        with tune_rule_scope('v8-6.24'):
            try:tune(parse(source),ctx,client=Capture(),iteration=4,engine_profile='tensor')
            except Captured:pass
        if not tune_path.exists():raise RuntimeError('First saved program did not reach the selector')
        manifest['models'][model]=dict(model_dir=str(directory),main_request=str(main_path),
            main_bytes=main_path.stat().st_size,tuning_request=str(tune_path),tuning_bytes=tune_path.stat().st_size,
            main_max_tokens=65536,tuning_max_tokens=16384,main_namespaces=[[model,'r8-v11',s] for s in ('s1','s2')],
            tuning_namespaces=[[model,'t1',s] for s in ('E6a','s1','s2')],split_version='split-v2',
            main_prompt='r8-v11',tuning_prompt='t1',tune_rule='v8-6.24',placement='program-plan')
        from ..cost import release_estimate_caches
        from ..tensor_sim import clear_caches
        del ctx,state,info
        release_estimate_caches();clear_caches();gc.collect()
    manifest['payload_scope']='Existing public model structure, shapes, parameter names, calibration/heldout derived statistics, generated DSL, internal feedback, legal tuning menus and candidate results; report/confirm/compiler results never sent.'
    write(root/'network-review/manifest.json',manifest)
    print('PREPARED_NO_API='+str(root/'network-review/manifest.json'),flush=True)


def child(command,out,*,model,mode,ledger=None,label=None):
    from .a1_transformer_agent import account_calls
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if ledger is not None:write(out/'api-accounting.json',dict(api_before=len(ledger_rows(ledger))))
    print('E6_STEP='+model+' '+mode+' '+str(out),flush=True)
    with (out/'run.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while True:
            try:code=process.wait(timeout=2);break
            except subprocess.TimeoutExpired:
                if ledger is not None:account_calls(ledger,out,model=model,prompt_version='r8-v11',label=label,split_version='split-v2')
    if ledger is not None:account_calls(ledger,out,model=model,prompt_version='r8-v11',label=label,split_version='split-v2')
    result=read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False,error='No child REPORT')
    result.update(exit=code,seconds=time.monotonic()-started)
    if ledger is not None:result['api_after']=len(ledger_rows(ledger))
    if code:result['failure_class']=failure_class(code,result)
    write(out/'REPORT.json',result);register(model,mode,out,result)
    return result


def base_command(part,out,model,model_dir):
    return [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
        '-m','pt_agent.experiments.a1_transformer_agent','--inside','--part',part,
        '--out',str(out),'--model',model,'--model-dir',str(model_dir),
        '--prompt-version','r8-v11','--tune-rule','v8-6.24','--placement','program-plan']


def replay(root,old,model,model_dir,ledger):
    """All main replies are files; only the t1 client can contact the provider."""
    from ..planners.transformer_session import load_state
    session=session_path(root,'e6a',model,'E6a');session.mkdir(parents=True,exist_ok=False)
    saved=sorted((old/'calls'/model/'r8-v11').glob('call-*.response.json'))
    expected=7 if model=='bert' else 8
    if len(saved)!=expected:raise RuntimeError('E5d response count differs')
    started=time.monotonic();steps=[]
    while True:
        state=load_state(session/'SESSION.json') if (session/'SESSION.json').exists() else None
        if state and state['phase']=='done':break
        phase=state['phase'] if state else 'evaluate';iteration=state['iteration'] if state else 1
        out=session/'steps'/f'{len(steps)+1:03d}-{phase}'
        command=base_command('session-step',out,model,model_dir)+[
            '--step',phase,'--root',str(session),'--call-dir',str(root/'calls'),'--budget-path',str(ledger),
            '--tune-search','llm-select','--session-label','E6a','--tune-call-limit','12','--session-iters',str(expected)]
        if phase=='evaluate':command.extend(['--response-file',str(saved[iteration-1])])
        result=child(command,out,model=model,mode='e6a-'+phase,ledger=ledger,label='E6a')
        steps.append(dict(phase=phase,iteration=iteration,report=str(out/'REPORT.json')))
        write(session/'STEPS.json',dict(steps=steps))
        if result.get('exit') or not result.get('complete'):
            report=dict(complete=False,error=result.get('error'),failure_class=result.get('failure_class','code'),
                steps=steps,seconds=time.monotonic()-started,main_api_calls=0)
            write(session/'REPORT.json',report);return report
    agent=read(session/'AGENT.json')
    result=dict(complete=agent['status']!='error',frozen=agent['status']=='feasible',agent=agent,
        main_api_calls=0,saved_replies=len(saved),steps=steps,seconds=time.monotonic()-started,
        label='Development replay: saved later replies had seen E5d sweep feedback')
    write(session/'REPORT.json',result);register(model,'e6a-session',session,result)
    return result


def acceptance(root,phase,model,label,model_dir):
    session=session_path(root,phase,model,label);parent=session.parent/'acceptance'
    for part in ('evaluate','cost','pars'):
        out=parent/('report' if part=='evaluate' else part)
        if part=='pars':
            from . import e5d_agent_sessions_v11 as e5d
            command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',str(Path(e5d.__file__)),
                '--scripts-dir',str(Path(__file__).resolve().parents[2]),'--root',str(root),'--native',
                '--model',model,'--model-dir',str(model_dir),'--program',str(session/'program.ptdsl'),'--out',str(out)]
        else:
            command=base_command(part,out,model,model_dir)+['--root',str(parent),
                '--program',str(session/'program.ptdsl'),'--frozen',str(session/'FROZEN.json')]
        result=child(command,out,model=model,mode=phase+'-'+label+'-'+part)
        if result.get('exit') or not result.get('complete'):raise RuntimeError(model+' '+part+' failed: '+str(result.get('error')))


def _trace(session):
    path=session/'agent_trace.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def tune_rows(session):
    result=[]
    for row in _trace(session):
        tune=row.get('tuned')
        if not tune:continue
        old_visible=2+sum('latency_pred_us' in h for h in tune['history'] if h['phase']!='result')
        result.append(dict(call=row['iteration'],raw_boots=(row.get('fb',{}).get('cost') or {}).get('boots_pred'),
            seconds=tune['seconds'],llm_s=tune.get('llm_s',0),llm_calls=tune.get('llm_calls',0),
            estimates=tune.get('estimates'),legacy_recorded_cost_entries=old_visible,
            estimates_note=None if 'estimates' in tune else 'E5d retained cost-entry records; cache misses were not recorded.',
            accepted_steps=len(tune['changes']),after_boots=tune['cost_after']['boots_pred'],
            latency_pred_us=tune['cost_after']['latency_pred_us'],changes=tune['changes'],
            menus=[h['menu_size'] for h in tune['history'] if h['phase']=='llm_round']))
    return result


def _signature(change):
    old,new=change.get('old') or {},change.get('new') or {}
    fields={k:v for k,v in new.items() if v!=old.get(k)}
    return json.dumps([change.get('line'),fields],sort_keys=True,default=str)


def session_summary(session):
    report=read(session/'REPORT.json') if (session/'REPORT.json').exists() else {}
    agent=read(session/'AGENT.json') if (session/'AGENT.json').exists() else {}
    parent=session.parent/'acceptance'
    native=read(parent/'pars/REPORT.json') if (parent/'pars/REPORT.json').exists() else {}
    evaluated=read(parent/'report/REPORT.json') if (parent/'report/REPORT.json').exists() else {}
    engines=evaluated.get('engines',{})
    rows=_trace(session);timings=dict(main_llm_s=0.,tuning_llm_s=0.,
        estimate_s=0.,validation_s=0.,tune_total_s=0.,confirm_s=0.,
        pars_s=native.get('compile',{}).get('seconds'),pars_acceptance_s=native.get('seconds'))
    for step in (session/'steps').glob('*/REPORT.json'):
        item=read(step);phase=item.get('step')
        for key in ('estimate_s','validation_s'):timings[key]+=item.get('timing',{}).get(key,0)
        if phase=='tune':timings['tune_total_s']+=item.get('seconds',0)
        if phase=='confirm':timings['confirm_s']+=item.get('seconds',0)
    model=session.parents[1].name;label=session.parent.name
    call_root=session.parents[3]/'calls'/model
    main_requests=list((call_root/'r8-v11'/label).glob('call-*.request.json'))
    tune_requests=list((call_root/'t1'/label).glob('call-*.request.json'))
    replies={key:[read(p) for p in (call_root/version/label).glob('call-*.response.json')]
             for key,version in (('main','r8-v11'),('tuning','t1'))}
    tokens={key:sum(sum((reply.get('usage') or {}).get(k,0) for k in ('prompt_tokens','completion_tokens'))
                    for reply in values) for key,values in replies.items()}
    timings['main_llm_s']=sum(r.get('seconds',0) for r in replies['main'])
    timings['tuning_llm_s']=sum(r.get('seconds',0) for r in replies['tuning'])
    if label=='E6a':timings['replayed_main_llm_s']=sum(r.get('seconds',0) for r in rows)
    native_failure=failure_class(native.get('exit'),native) if native.get('error') else None
    acceptance_status=('not_run' if not session.exists() else 'blocked_resource' if native_failure=='resource'
        else 'failed' if native.get('error') else 'complete' if native.get('complete')
        else 'not_frozen' if not (session/'FROZEN.json').exists() else 'pending')
    boots=native.get('boots') or dict(predicted=(agent.get('best') or {}).get('boots_pred'),
        exported=native.get('export',{}).get('boots'),compiled=native.get('stats',{}).get('boots'))
    cost_report=read(parent/'cost/REPORT.json') if (parent/'cost/REPORT.json').exists() else {}
    post_freeze_passed=bool(native.get('complete') and cost_report.get('complete')
                            and evaluated.get('complete') and engines
                            and all(r.get('feasible') for r in engines.values()))
    return dict(status=agent.get('status',report.get('error','not_run')),frozen_origin=agent.get('best_source'),
        frozen_iteration=agent.get('best_iteration'),first_feasible_iteration=agent.get('first_feasible_iteration'),
        acceptance_status=acceptance_status,native_error=native.get('error'),native_failure_class=native_failure,
        post_freeze_acceptance_passed=post_freeze_passed,
        boots=boots,latency_pred_us=(agent.get('best') or {}).get('latency_pred_us'),
        report100={e:{k:r.get(k) for k in ('agree','total','feasible','logit_rel_err')} for e,r in engines.items()},
        main_calls=len(main_requests),
        replayed_responses=agent.get('llm_calls',0) if session.parent.name=='E6a' else 0,
        tuning_calls=len(tune_requests),
        tokens=tokens,
        loop_seconds=report.get('seconds'),timings=timings,session=str(session),tunes=tune_rows(session))


def summarize(root,old,error=None):
    data=dict(round='E6',complete=error is None,error=error,api_before=259,
        api_after=len(ledger_rows(RESULTS_DIR/'llm_calls.jsonl')),e6a={},e6b={},selection_quality=[],namespaces=[])
    previous=read(old/'E5d.json');data['E5d']=previous.get('models',{})
    data['E5d_comparison']={}
    for model,v in data['E5d'].items():
        agent=read(old/model/'session/AGENT.json');timing=v['timing']
        data['E5d_comparison'][model]=dict(status=v['status'],acceptance_status='complete',
            frozen_origin=v['frozen']['origin'],frozen_iteration=v['frozen']['iteration'],
            boots=v['pars']['boots'],latency_pred_us=agent['best']['latency_pred_us'],
            report100={e:{k:r.get(k) for k in ('agree','total','feasible','logit_rel_err')}
                       for e,r in v['report']['engines'].items()},main_calls=v['calls'],tuning_calls=0,
            tokens=dict(main=agent['prompt_tokens']+agent['completion_tokens'],tuning=0),
            loop_seconds=v['session']['seconds'],timings=dict(main_llm_s=timing['llm_s'],tuning_llm_s=0,
                estimate_s=timing['estimate_s'],validation_s=timing['validation_s'],
                tune_total_s=timing['tune_s'],confirm_s=timing['confirm_s'],
                pars_s=timing['pars_s'],pars_acceptance_s=v['pars']['seconds']))
    for model in ('bert','qwen'):
        old_tunes=tune_rows(old/model/'session')
        current=session_summary(session_path(root,'e6a',model,'E6a'))
        current['E5d_tunes']=old_tunes;data['e6a'][model]=current
        old_signatures={_signature(c) for r in old_tunes for c in r['changes']}
        for phase,label in (('e6a','E6a'),('e6b','s1'),('e6b','s2')):
            session=session_path(root,phase,model,label)
            if phase=='e6b':
                data['e6b'][model+' '+label]=session_summary(session)
                if not session.exists():data['e6b'][model+' '+label]['not_started_reason']=error
            for r in _trace(session):
                if r.get('tuned_reused_from') is not None:continue
                tune=r.get('tuned') or {}
                for h in tune.get('history',[]):
                    if h['phase']!='llm_round':continue
                    candidates=h['candidates'];accepted=[c for c in candidates if c.get('accepted')]
                    moves={m['id']:m for m in h['menu']}
                    matching=sum(_signature(dict(line=m['line'],old={k:v['old'] for k,v in m['changes'].items()},
                        new={k:v['new'] for k,v in m['changes'].items()})) in old_signatures
                        for c in candidates if c.get('moves') for mid in c['moves'] if mid in moves for m in [moves[mid]])
                    data['selection_quality'].append(dict(phase=phase,model=model,label=label,iteration=r['iteration'],
                        round=h['round'],menu_size=h['menu_size'],proposed=len(candidates),
                        legal=sum(c.get('rejected') not in ('invalid_moves','duplicate_point','unknown_id') for c in candidates),
                        evaluated=sum(c.get('evaluated',False) for c in candidates),accepted=len(accepted),
                        stale=sum(c.get('rejected')=='stale' for c in candidates),
                        accept_rate=len(accepted)/len(candidates) if candidates else 0,
                        multi_move_accepted=sum(len(c['moves'])>=2 for c in accepted),overlap_e5d_selected_moves=matching))
        for version,labels in (('r8-v11',('s1','s2')),('t1',('E6a','s1','s2'))):
            for label in labels:
                count=len(list((root/'calls'/model/version/label).glob('call-*.request.json')))
                limit=12 if label=='E6a' else 10 if version=='r8-v11' else 20
                data['namespaces'].append(dict(namespace=[model,version,label],calls=count,limit=limit))
    data['api_delta']=data['api_after']-259
    data['accounting_equal']=data['api_delta']==sum(n['calls'] for n in data['namespaces'])
    data['within_limits']=(data['api_after']<=403 and data['api_delta']<=144
                           and all(n['calls']<=n['limit'] for n in data['namespaces']))
    data['stop_failure_class']=failure_class(1,dict(error=error)) if error else None
    data['live_sessions']=dict(planned=4,started=sum(v['status']!='not_run' for v in data['e6b'].values()),
        post_freeze_passed=sum(v['post_freeze_acceptance_passed'] for v in data['e6b'].values()),
        resource_blocked=sum(v['acceptance_status']=='blocked_resource' for v in data['e6b'].values()))
    write(root/'E6.json',data)
    lines=['# E6 LLM 选步精调','',f'累计 {data["api_after"]}/450；本轮 {data["api_delta"]}/144；错误：{error or "无"}。',
        '','E6a 为开发回放；E6b 为新会话。报告、确认和编译结果均未回传 Agent。真实 CKKS 运行0。',
        '','## E6a 逐次精调','',
        '| 模型/调用 | 原文自举 | E5d 秒/记录估计入口/接受步/后自举 | E6a 秒(LLM秒)/调用/菜单/估计miss/接受候选/后自举 |',
        '|---|---:|---|---|']
    for model,item in data['e6a'].items():
        by_call={t['call']:t for t in item['tunes']}
        for prior in item['E5d_tunes']:
            now=by_call.get(prior['call']);accepted=sum(q['accepted'] for q in data['selection_quality']
                if q['phase']=='e6a' and q['model']==model and q['iteration']==prior['call'])
            detail=(f'{now["seconds"]:.2f} ({now["llm_s"]:.2f}) / {now["llm_calls"]} / {now["menus"]} / {now["estimates"]} / {accepted} / {now["after_boots"]}' if now else '未执行')
            lines.append(f'| {model}/{prior["call"]} | {prior["raw_boots"]} | {prior["seconds"]:.2f} / {prior["legacy_recorded_cost_entries"]} / {prior["accepted_steps"]} / {prior["after_boots"]} | {detail} |')
    lines+=['','E5d 未保存 cache miss；历史列为代价入口次数，E6列为实际未命中次数，历史文件保持。',
        '','## 冻结对照与 E6b','',
        '| 会话 | 预测/导出/编译自举 | 预测延迟(us) | report100一致数(tensor/Mock) | 来源/轮 | 主/精调调用 | tokens(主/精调) | 循环秒 | 验收状态 |',
        '|---|---|---:|---|---|---|---|---:|---|']
    sessions=([(f'E5d {m}',v) for m,v in data['E5d_comparison'].items()]+
              [(f'E6a {m}',v) for m,v in data['e6a'].items()]+list(data['e6b'].items()))
    def number(value):return '—' if value is None else f'{value:.3f}' if isinstance(value,float) else str(value)
    for name,item in sessions:
        boots='/'.join(number((item['boots'] or {}).get(k)) for k in ('predicted','exported','compiled'))
        agreement='/'.join(str(item['report100'].get(e,{}).get('agree','—')) for e in ('tensor','mock'))
        lines.append(f'| {name} | {boots} | {number(item["latency_pred_us"])} | {agreement} | '
            f'{item["frozen_origin"] or "—"}/{number(item.get("frozen_iteration"))} | '
            f'{item["main_calls"]}/{item["tuning_calls"]} | {item["tokens"]["main"]}/{item["tokens"]["tuning"]} | '
            f'{number(item["loop_seconds"])} | {item["acceptance_status"]} |')
    lines+=['','report100分母均为100；未执行的单元格保留。逐引擎feasible与logit_rel_err见E6.json。',
        '','## 耗时拆分（秒）','',
        '| 会话 | 主LLM | 精调LLM | estimate | validation | tune合计 | confirm | pars编译 | pars整套验收 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys=('main_llm_s','tuning_llm_s','estimate_s','validation_s','tune_total_s','confirm_s','pars_s','pars_acceptance_s')
    for name,item in sessions:lines.append('| '+name+' | '+' | '.join(number(item['timings'].get(k)) for k in keys)+' |')
    lines+=['','tune合计含其LLM、estimate和validation；这些项有包含关系。pars编译与E5d使用同一口径；整套验收另含导出、trace等。E6a主LLM实际0，保存响应的历史耗时另记replayed_main_llm_s。',
        '','## 每轮选步质量','',
        '| 阶段/模型/会话/调用/轮 | 菜单 | 提出 | 合法 | 估计 | 接受 | stale | 接受/提出 | 多步接受 | 与E5d接受步重合出现次数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for q in data['selection_quality']:
        label='/'.join(str(q[k]) for k in ('phase','model','label','iteration','round'))
        fields=('menu_size','proposed','legal','evaluated','accepted','stale','accept_rate','multi_move_accepted','overlap_e5d_selected_moves')
        lines.append('| '+label+' | '+' | '.join(number(q[k]) for k in fields)+' |')
    lines+=['','接受率分母为提出数；重合列按各候选所列菜单编号的出现次数计数，候选间重复编号重复计。缓存复用记录不重复计为新选步。仅作描述。',
        '','## 停止与未完成项','',error or '已按预先登记顺序收尾。',
        '','若因资源停止，未开始的会话保留在表内；冻结结果只按已完成阶段汇报，编译缺口保持。原响应、源码、程序包及原报告均保留；不重试、不扩大资源。']
    (root/'E6.md').write_text('\n'.join(lines)+'\n');return data


def run(args):
    root=args.root;old=args.old;ledger=RESULTS_DIR/'llm_calls.jsonl';os.environ['PT_RESULTS']=str(RESULTS_DIR)
    if not (root/'OFFLINE_TESTS.log').read_text().rstrip().endswith('OK'):raise RuntimeError('Offline tests not passed')
    if (root/'RUN.json').exists():raise RuntimeError('Run already started; failed calls must not be retried')
    if len(ledger_rows(ledger))!=259:raise RuntimeError('Approved ledger start259 changed')
    write(root/'RUN.json',dict(api_before=259,api_ceiling=403,max_calls=144,retries=0,order=read(root/'network-review/manifest.json')['order']))
    error=None
    try:
        for model,config in configurations().items():
            result=replay(root,old,model,config['model'],ledger)
            if not result['complete']:
                if result.get('failure_class')=='provider':continue
                raise RuntimeError(model+' E6a stop: '+str(result.get('error')))
            if result['frozen']:acceptance(root,'e6a',model,'E6a',config['model'])
        for model,label in (('bert','s1'),('qwen','s1'),('bert','s2'),('qwen','s2')):
            model_dir=configurations()[model]['model'];session=session_path(root,'e6b',model,label)
            command=[sys.executable,'-B','-m','pt_agent.experiments.a1_transformer_agent','--part','session',
                '--prompt-version','r8-v11','--tune-rule','v8-6.24','--placement','program-plan',
                '--tune-search','llm-select','--session-label',label,'--tune-call-limit','20','--model',model,
                '--model-dir',str(model_dir),'--out',str(session),'--call-dir',str(root/'calls'),'--budget-path',str(ledger)]
            session.parent.mkdir(parents=True,exist_ok=True)
            print('E6_SESSION='+model+' '+label,flush=True)
            with (session.parent/'session.log').open('w') as log:status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            report=read(session/'REPORT.json') if (session/'REPORT.json').exists() else dict(complete=False,failure_class='code',error='Missing session report')
            if status.returncode or not report.get('complete'):
                if report.get('failure_class')=='provider':continue
                raise RuntimeError(model+' '+label+' E6b stop: '+str(report.get('error')))
            if report['frozen']:acceptance(root,'e6b',model,label,model_dir)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';write(root/'STOP.json',dict(error=error,api_after=len(ledger_rows(ledger))))
    result=summarize(root,old,error)
    processes=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
    result['final_processes']=[line for line in processes.splitlines() if len(line.split())>=2 and line.split()[1] in ('nix','nix-shell','hecate-opt')]
    write(root/'FINAL_CHECKS.json',{k:result[k] for k in ('api_before','api_after','api_delta','namespaces','accounting_equal','within_limits','final_processes')})
    write(root/'E6.json',result);register('all','summary',root,result)
    print('E6_DONE='+str(root/'E6.json')+' ERROR='+str(error),flush=True)
    return int(bool(error or result['final_processes'] or not result['accounting_equal'] or not result['within_limits']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--old',type=Path,default=RESULTS_DIR/'E5d-agent-sessions-20261106')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--bert-model-dir',type=Path)
    parser.add_argument('--qwen-model-dir',type=Path)
    args=parser.parse_args()
    if args.prepare:
        if args.bert_model_dir is None or args.qwen_model_dir is None:parser.error('Prepare needs both model directories')
        prepare(args.root,args.old,dict(bert=args.bert_model_dir,qwen=args.qwen_model_dir));return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
