"""E6r serial continuation and fixed-response/new-session comparison."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

from ..paths import RESULTS_DIR
from . import e6_llm_select_tuning as e6

read,write,ledger_rows=e6.read,e6.write,e6.ledger_rows
ORDER=(('bert','r1'),('qwen','r1'),('bert','r2'),('qwen','r2'))
REPLAY_LIMITS={'bert':8,'qwen':16}


def register(model,mode,path,result):
    from ..registry import append,git_rev,new_run_id
    append(dict(run_id=new_run_id('E6r',model,mode),round='E6r',model=model,mode=mode,
        git_rev=git_rev(),metrics=result,artifacts_dir=str(path),ts=datetime.now(timezone.utc).isoformat()))


def child(command,out,*,model,mode,ledger=None,label=None):
    from .a1_transformer_agent import account_calls
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if ledger is not None:write(out/'api-accounting.json',dict(api_before=len(ledger_rows(ledger))))
    print('E6r_STEP='+model+' '+mode+' '+str(out),flush=True)
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
    if code:result['failure_class']=e6.failure_class(code,result)
    write(out/'REPORT.json',result);register(model,mode,out,result)
    return result


def session_path(root,phase,model,label):return root/phase/model/label/'session'


def replay(root,old,model,model_dir,ledger):
    from ..planners.transformer_session import load_state
    label='E6r-a';session=session_path(root,'e6r-a',model,label)
    session.mkdir(parents=True,exist_ok=False)
    saved=sorted((old/'calls'/model/'r8-v11').glob('call-*.response.json'))
    expected=7 if model=='bert' else 8
    if len(saved)!=expected:raise RuntimeError('E5d saved response count differs')
    started=time.monotonic();steps=[]
    while True:
        state=load_state(session/'SESSION.json') if (session/'SESSION.json').exists() else None
        if state and state['phase']=='done':break
        phase=state['phase'] if state else 'evaluate';iteration=state['iteration'] if state else 1
        out=session/'steps'/f'{len(steps)+1:03d}-{phase}'
        command=e6.base_command('session-step',out,model,model_dir)+[
            '--step',phase,'--root',str(session),'--call-dir',str(root/'calls'),'--budget-path',str(ledger),
            '--tune-search','llm-select','--session-label',label,'--tune-call-limit',str(REPLAY_LIMITS[model]),
            '--session-iters',str(expected)]
        if phase=='evaluate':command.extend(['--response-file',str(saved[iteration-1])])
        result=child(command,out,model=model,mode='e6r-a-'+phase,ledger=ledger,label=label)
        steps.append(dict(phase=phase,iteration=iteration,report=str(out/'REPORT.json')))
        write(session/'STEPS.json',dict(steps=steps))
        if result.get('exit') or not result.get('complete'):
            report=dict(complete=False,error=result.get('error'),failure_class=result.get('failure_class','code'),
                steps=steps,seconds=time.monotonic()-started,main_api_calls=0)
            write(session/'REPORT.json',report);return report
    agent=read(session/'AGENT.json')
    result=dict(complete=agent['status']!='error',frozen=agent['status']=='feasible',agent=agent,
        main_api_calls=0,saved_replies=len(saved),steps=steps,seconds=time.monotonic()-started,
        label='Development replay: saved main responses had seen E5d sweep feedback')
    write(session/'REPORT.json',result);register(model,'e6r-a-session',session,result)
    return result


def native_command(root,model,model_dir,program,out):
    from . import e5d_agent_sessions_v11 as e5d
    return [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',str(Path(e5d.__file__)),
            '--scripts-dir',str(Path(__file__).resolve().parents[2]),'--root',str(root),'--native',
            '--model',model,'--model-dir',str(model_dir),'--program',str(program),'--out',str(out)]


def check_child(result,context):
    if result.get('exit') or not result.get('complete'):
        raise RuntimeError(context+': '+str(result.get('error')))


def supplement(root,previous,model_dir):
    program=session_path(previous,'e6b','qwen','s1')/'program.ptdsl'
    out=root/'e6-qwen-s1-compile'
    if (out/'REPORT.json').exists():
        result=read(out/'REPORT.json')
        check_child(result,'Retained E6 qwen s1 pars; failed steps are not retried')
    else:
        result=child(native_command(root,'qwen',model_dir,program,out),out,model='qwen',mode='e6-qwen-s1-compile')
        check_child(result,'E6 qwen s1 pars')
    updated=read(previous/'E6.json')
    row=updated['e6b']['qwen s1'];row.update(boots=result['boots'],acceptance_status='complete',
        native_error=None,native_failure_class=None,post_freeze_acceptance_passed=True,
        supplement_report=str(out/'REPORT.json'))
    row['timings'].update(pars_s=result['compile']['seconds'],pars_acceptance_s=result['seconds'])
    updated['continuation']=dict(round='E6r',source=str(previous/'E6.json'),api_calls=0,
        scope='Qwen s1 native only; E6 s2 unstarted; original E6 records retained')
    # The original E6 stopping status and its accounting remain historical.
    write(root/'E6-updated.json',updated)
    (root/'E6-updated.md').write_text('# E6 Qwen s1 编译接续\n\n'+
        '原 E6.md、E6.json 保留；仅补充 Qwen s1 的冻结程序编译。API0。\n\n'+
        '预测/导出/编译自举：'+str(result['boots'])+'。\n\n原 E6 两个 s2 未运行。\n')
    return result


def acceptance(root,phase,model,label,model_dir):
    session=session_path(root,phase,model,label);parent=session.parent/'acceptance'
    for part in ('evaluate','cost','pars'):
        out=parent/('report' if part=='evaluate' else part)
        if part=='pars':command=native_command(root,model,model_dir,session/'program.ptdsl',out)
        else:command=e6.base_command(part,out,model,model_dir)+['--root',str(parent),
                '--program',str(session/'program.ptdsl'),'--frozen',str(session/'FROZEN.json')]
        result=child(command,out,model=model,mode=phase+'-'+label+'-'+part)
        check_child(result,model+' '+part)


def prepare(root,old,models):
    from ..llm import Config
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..planners.agent import new_agent_state
    from ..planners.agent_tools import build_context,apply_placement
    from ..planners.transformer_tuner import tune,tune_rule_scope
    from .a1_transformer_agent import samples_for,limits
    limits()
    manifest=dict(destination='https://api.deepseek.com/chat/completions',api_before=283,
        api_ceiling=403,global_limit=450,max_calls=120,retries=0,models={},
        order=['supplement E6 qwen s1','E6r-a bert','E6r-a qwen']+[f'E6r-b {m} {s}' for m,s in ORDER],
        e6ra=dict(main_calls=0,tuning_limits=REPLAY_LIMITS,rounds_per_tune=4),
        e6rb=dict(sessions=4,main_calls_per_session=10,tuning_calls_per_session=14))
    class Captured(Exception):pass
    for model,directory in models.items():
        info=ModelInfo.load(directory)
        ctx=build_context(info,read(directory/'calibration.json'),samples=samples_for(directory,info))
        ctx.prompt_version='r8-v11';ctx.placement='program-plan';ctx.tune_search='llm-select'
        state=new_agent_state(ctx,max_iters=10,engine_profile='tensor',tune_rule='v8-6.24')
        body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=65536,timeout_seconds=600).settings()
        body['messages']=state['messages']
        main_path=root/'network-review'/f'{model}-main-first-request.pretty.json';write(main_path,body)
        tune_path=root/'network-review'/f'{model}-t1-first-request.pretty.json'
        class Capture:
            calls=0;limit=REPLAY_LIMITS[model]
            def chat(self,messages,*,purpose):
                body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=65536,timeout_seconds=600).settings()
                body['messages']=messages;write(tune_path,body);raise Captured()
        reply=read(old/'calls'/model/'r8-v11/call-004.response.json')
        source=apply_placement(json.loads(reply['content'])['program'],ctx)
        with tune_rule_scope('v8-6.24'):
            try:tune(parse(source),ctx,client=Capture(),iteration=4,engine_profile='tensor')
            except Captured:pass
        if not tune_path.exists():raise RuntimeError('Saved program did not reach the selector')
        manifest['models'][model]=dict(model_dir=str(directory),main_request=str(main_path),
            tuning_request=str(tune_path),main_max_tokens=65536,tuning_max_tokens=65536,
            main_namespaces=[[model,'r8-v11',s] for s in ('r1','r2')],
            tuning_namespaces=[[model,'t1',s] for s in ('E6r-a','r1','r2')],
            split_version='split-v2',main_prompt='r8-v11',tuning_prompt='t1',tune_rule='v8-6.24',placement='program-plan')
        from ..cost import release_estimate_caches
        from ..tensor_sim import clear_caches
        del ctx,state,info;release_estimate_caches();clear_caches();gc.collect()
    manifest['payload_scope']='Existing BERT/Qwen structure, shapes, parameter names, calibration/heldout derived statistics, generated DSL, internal feedback, legal tuning menus and candidate results. Confirm/report/compiler results never sent.'
    write(root/'network-review/manifest.json',manifest)
    print('PREPARED_NO_API='+str(root/'network-review/manifest.json'),flush=True)


def selector_rounds(session):
    """Retain completed rounds carried by saved requests after provider failure.

    This reads records only; it neither evaluates nor resumes any candidate.
    """
    rounds={};requests={}
    for row in e6._trace(session):
        if row.get('tuned_reused_from') is not None:continue
        for h in (row.get('tuned') or {}).get('history',[]):
            if h['phase']=='llm_round':rounds[(row['iteration'],h['round'])]=(dict(h),'trace')
    folder=session.parents[3]/'calls'/session.parents[1].name/'t1'/session.parent.name
    for path in sorted(folder.glob('*.request.json')):
        request=read(path);match=re.fullmatch(r'tune-select-iter(\d+)-round(\d+)',request.get('purpose',''))
        if not match:continue
        iteration,number=map(int,match.groups())
        body=json.loads(request['messages'][1]['content'])
        requests[(iteration,number)]=(path,body)
        for h in body.get('previous_rounds',[]):
            rounds.setdefault((iteration,h['round']),(dict(h),'saved_next_request'))
    for key,(path,body) in requests.items():
        error_path=path.with_name(path.name.replace('.request.json','.error.json'))
        if key not in rounds and error_path.exists():
            rounds[key]=(dict(round=key[1],candidates=[],finish_reason=None,
                              error=read(error_path)['error']),'provider_error')
        if key in rounds:
            h,_=rounds[key];h.setdefault('menu',body['menu']);h.setdefault('menu_size',len(body['menu']))
    return [dict(iteration=key[0],history=h,record_source=source)
            for key,(h,source) in sorted(rounds.items())]


def session_summary(session):
    row=e6.session_summary(session)
    report=read(session/'REPORT.json') if (session/'REPORT.json').exists() else {}
    row.update(session_complete=report.get('complete',False),session_error=report.get('error'),
               session_failure_class=report.get('failure_class'))
    if report.get('failure_class')=='provider':row['acceptance_status']='provider_failed'
    folder=session.parents[3]/'calls'/session.parents[1].name
    row['tokens_unreported_calls']={};failures=[]
    for key,version in (('main','r8-v11'),('tuning','t1')):
        paths=list((folder/version/session.parent.name).glob('*.error.json'))
        elapsed=sum((read(p).get('diagnostics') or {}).get('elapsed_seconds',0) for p in paths)
        row['timings'][key+'_provider_error_s']=elapsed
        row['timings'][key+'_llm_s']+=elapsed
        row['tokens_unreported_calls'][key]=len(paths)
        failures.extend(dict(path=str(p),**read(p)) for p in paths)
    row['provider_errors']=failures;row['failed_tune_attempts']=[]
    row['timings']['tune_total_s']=0.
    for path in sorted((session/'steps').glob('*/REPORT.json')):
        step=read(path);phase=step.get('step') or path.parent.name.split('-',1)[-1]
        if phase!='tune':continue
        row['timings']['tune_total_s']+=step.get('seconds',0)
        if step.get('complete'):continue
        matches=[x for x in selector_rounds(session) if x['record_source']!='trace']
        iteration=matches[-1]['iteration'] if matches else None
        row['failed_tune_attempts'].append(dict(call=iteration,seconds=step.get('seconds'),
            failure_class=step.get('failure_class'),error=step.get('error'),
            llm_calls=sum(x['iteration']==iteration for x in matches),
            llm_s=row['timings']['tuning_llm_s'],estimates=None,after_boots=None,
            partial_accepted=sum(bool(c.get('accepted')) for x in matches
                for c in x['history']['candidates']),source=str(path)))
    if session.parent.name=='E6r-a':
        row['replayed_responses']=(7 if session.parents[1].name=='bert' else 8) if session.exists() else 0
        row['timings']['replayed_main_llm_s']=sum(r.get('seconds',0) for r in e6._trace(session))
    return row


def summarize(root,old,previous,error=None):
    old6=read(previous/'E6.json')
    data=dict(round='E6r',complete=error is None,error=error,api_before=283,
        api_after=len(ledger_rows(RESULTS_DIR/'llm_calls.jsonl')),e6ra={},e6rb={},selection_quality=[],namespaces=[],
        E5d_comparison=old6['E5d_comparison'],E6_s1={},three_way_tuning=[])
    supplement_path=root/'E6-updated.json'
    prior=read(supplement_path) if supplement_path.exists() else old6
    data['E6_s1']={m:prior['e6b'][m+' s1'] for m in ('bert','qwen')}
    for model in ('bert','qwen'):
        item=session_summary(session_path(root,'e6r-a',model,'E6r-a'));data['e6ra'][model]=item
        old_tunes={r['call']:r for r in e6.tune_rows(old/model/'session')}
        e6_tunes={r['call']:r for r in old6['e6a'][model]['tunes']}
        new_tunes={r['call']:r for r in item['tunes']}
        for iteration,row in old_tunes.items():
            data['three_way_tuning'].append(dict(model=model,call=iteration,raw_boots=row['raw_boots'],
                sweep=row,e6=e6_tunes.get(iteration),e6r=new_tunes.get(iteration),e6r_attempt=next((a for a in item['failed_tune_attempts'] if a['call']==iteration),None)))
        for phase,label in (('e6r-a','E6r-a'),('e6r-b','r1'),('e6r-b','r2')):
            session=session_path(root,phase,model,label)
            if phase=='e6r-b':
                data['e6rb'][model+' '+label]=session_summary(session)
                if not session.exists():data['e6rb'][model+' '+label]['not_started_reason']=error
            for record in selector_rounds(session):
                h=record['history'];candidates=h['candidates']
                counts=Counter(c.get('rejected') for c in candidates if c.get('rejected'))
                merge_counts=Counter(c['merge'].get('rejected') for c in candidates if c.get('merge',{}).get('rejected'))
                data['selection_quality'].append(dict(phase=phase,model=model,label=label,iteration=record['iteration'],
                    round=h['round'],menu_size=h['menu_size'],proposed=len(candidates),
                    legal=sum(c.get('rejected') not in ('invalid_moves','conflict','unknown_id','not_legal_after_structural') for c in candidates),
                    estimated=sum(bool(c.get('evaluated')) for c in candidates),
                    validated=sum(bool(c.get('accepted')) for c in candidates),
                    accepted=sum(bool(c.get('accepted')) for c in candidates),
                    merge_attempts=h.get('merge_attempts',sum('merge' in c for c in candidates)),
                    merge_accepted=h.get('merge_accepted',sum(bool(c.get('merge',{}).get('accepted')) for c in candidates)),
                    not_reached=sum(c.get('rejected')=='not_reached' for c in candidates),
                    rejection_reasons=dict(counts),merge_rejection_reasons=dict(merge_counts),
                    truncated=int(h.get('finish_reason')=='length'),provider_error=h.get('error'),
                    record_source=record['record_source']))
        for version,labels in (('r8-v11',('r1','r2')),('t1',('E6r-a','r1','r2'))):
            for label in labels:
                count=len(list((root/'calls'/model/version/label).glob('call-*.request.json')))
                limit=REPLAY_LIMITS[model] if label=='E6r-a' else 10 if version=='r8-v11' else 14
                data['namespaces'].append(dict(namespace=[model,version,label],calls=count,limit=limit))
    data['api_delta']=data['api_after']-283
    data['accounting_equal']=data['api_delta']==sum(n['calls'] for n in data['namespaces'])
    data['within_limits']=data['api_after']<=403 and data['api_delta']<=120 and all(n['calls']<=n['limit'] for n in data['namespaces'])
    data['protocol_complete']=data['complete']
    data['provider_errors']=[dict(model=m,label=label,**read(path)) for m in ('bert','qwen')
        for version,label in (('t1','E6r-a'),('r8-v11','r1'),('t1','r1'),('r8-v11','r2'),('t1','r2'))
        for path in (root/'calls'/m/version/label).glob('*.error.json')]
    data['experiment_outcome']='provider_blocked' if data['provider_errors'] else 'stopped' if error else 'finished'
    data['new_frozen_count']=sum(bool((session_path(root,'e6r-b',m,label)/'FROZEN.json').exists()) for m,label in ORDER)
    data['new_post_freeze_passed']=sum(v['post_freeze_acceptance_passed'] for v in data['e6rb'].values())
    write(root/'E6r.json',data)
    lines=['# E6r LLM 选步修正','',f'本轮 {data["api_delta"]}/120，累计 {data["api_after"]}/450。编排异常：{error or "无"}。',
        '', 'E6r-a 为保存响应的开发回放；E6r-b 为新会话。报告与编译结果不回传 Agent，真实 CKKS 0。',
        '',f'流程按序收尾={data["protocol_complete"]}；实验结论={data["experiment_outcome"]}；provider失败{len(data["provider_errors"])}次；新冻结{data["new_frozen_count"]}/4，冻结后验收通过{data["new_post_freeze_passed"]}/4。complete仅表示流程收尾，模型成果按各行单独判定。',
        '', '## 同一输入三种精调','',
        '| 模型/调用 | 原文自举 | E5d sweep 秒/记录入口/接受步/后自举 | E6a 秒/实际估计/接受步/后自举 | E6r-a 秒(LLM秒)/调用/估计/接受步/后自举 |','|---|---:|---|---|---|']
    def detail(r,new=False,attempt=None):
        if not r and attempt:
            return f'provider失败；{attempt["seconds"]:.2f} ({attempt["llm_s"]:.2f}) / {attempt["llm_calls"]} / 未记录 / 局部候选{attempt["partial_accepted"]} / —'
        if not r:return '未执行'
        if new:return f'{r["seconds"]:.2f} ({r["llm_s"]:.2f}) / {r["llm_calls"]} / {r["estimates"]} / {r["accepted_steps"]} / {r["after_boots"]}'
        return f'{r["seconds"]:.2f} / {r["estimates"] if r["estimates"] is not None else r["legacy_recorded_cost_entries"]} / {r["accepted_steps"]} / {r["after_boots"]}'
    for r in data['three_way_tuning']:
        lines.append(f'| {r["model"]}/{r["call"]} | {r["raw_boots"]} | {detail(r["sweep"])} | {detail(r["e6"])} | {detail(r["e6r"],True,r.get("e6r_attempt"))} |')
    lines+=['','E5d只保留代价入口记录，未记录cache miss；E6/E6r为实际未命中次数。历史未补写。',
        '', '## 端到端','', '| 会话 | 预测/导出/编译自举 | 预测延迟(us) | report100 tensor/Mock | 冻结来源/轮 | 主/精调调用 | tokens主/精调 | 循环秒 | 验收 |','|---|---|---:|---|---|---|---|---:|---|']
    sessions=([(f'E5d {m}',r) for m,r in data['E5d_comparison'].items()]+
              [(f'E6 {m} s1',r) for m,r in data['E6_s1'].items()]+
              [(f'E6r-a {m}',r) for m,r in data['e6ra'].items()]+[(f'{m} {label}',data['e6rb'][f'{m} {label}']) for m,label in ORDER])
    def num(x):return '—' if x is None else f'{x:.3f}' if isinstance(x,float) else str(x)
    for name,r in sessions:
        boots='/'.join(num((r['boots'] or {}).get(k)) for k in ('predicted','exported','compiled'))
        agree='/'.join(str(r['report100'].get(e,{}).get('agree','—')) for e in ('tensor','mock'))
        lines.append(f'| {name} | {boots} | {num(r["latency_pred_us"])} | {agree} | {r["frozen_origin"] or "—"}/{num(r.get("frozen_iteration"))} | {r["main_calls"]}/{r["tuning_calls"]} | {r["tokens"]["main"]}/{r["tokens"]["tuning"]} | {num(r["loop_seconds"])} | {r["acceptance_status"]}{": "+r["session_error"] if r.get("session_error") else ""} |')
    lines+=['','tokens列仅累计收到的usage，provider错误缺usage的调用另列tokens_unreported_calls，不据此认定实际token消费为0。未完成精调的最终自举和实际cache miss保持缺失；局部接受不计正式精调完成。','', '## 耗时拆分（秒）','', '| 会话 | 主LLM | 精调LLM | estimate | validation | tune合计 | confirm | pars编译 | pars整套 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in sessions:
        lines.append('| '+name+' | '+' | '.join(num(r['timings'].get(k)) for k in ('main_llm_s','tuning_llm_s','estimate_s','validation_s','tune_total_s','confirm_s','pars_s','pars_acceptance_s'))+' |')
    lines+=['','tune合计包含LLM、估计和验证；失败步骤按外层保留的整步墙钟计入。主/精调LLM秒含已记录的provider错误时长，JSON另列其拆分。失败精调内部estimate/validation分项未保存，不补算；保存响应主LLM实际0，历史秒数另存JSON。',
        '', '## 每轮选步质量','', '| 阶段/模型/会话/调用/轮 | 提出 | 合法 | 估计 | 全验证可行 | 接受 | 合并接受 | not_reached | 截断 | 拒绝原因/合并拒绝/provider错误 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for q in data['selection_quality']:
        label='/'.join(str(q[k]) for k in ('phase','model','label','iteration','round'))
        values=' | '.join(str(q[k]) for k in ('proposed','legal','estimated','validated','accepted','merge_accepted','not_reached','truncated'))
        lines.append(f'| {label} | {values} | {q["rejection_reasons"]} / {q["merge_rejection_reasons"]} / {q["provider_error"] or "—"} |')
    lines+=['','各会话与未执行项均保留。已完成精调见trace；中断前的已完成轮次从下一份保存请求的previous_rounds恢复，record_source标明来源。仅重整已保存记录，不重新评估、恢复精调或发请求。完整汇总见E6r.json。', '', error or '按预登记顺序收尾。']
    (root/'E6r.md').write_text('\n'.join(lines)+'\n');return data


def run(args):
    root,old,previous=args.root,args.old,args.previous
    ledger=RESULTS_DIR/'llm_calls.jsonl';os.environ['PT_RESULTS']=str(RESULTS_DIR)
    if not read(root/'OFFLINE_TESTS.json').get('complete'):raise RuntimeError('Offline tests not passed')
    if (root/'RUN.json').exists():raise RuntimeError('Already started; no retries')
    if len(ledger_rows(ledger))!=283:raise RuntimeError('Approved ledger start283 changed')
    write(root/'RUN.json',dict(api_before=283,api_ceiling=403,max_calls=120,retries=0,
        order=read(root/'network-review/manifest.json')['order']))
    error=None
    try:
        supplement(root,previous,e6.configurations()['qwen']['model'])
        for model,config in e6.configurations().items():
            result=replay(root,old,model,config['model'],ledger)
            if not result['complete']:
                if result.get('failure_class')=='provider':continue
                raise RuntimeError(model+' E6r-a stop: '+str(result.get('error')))
            if result['frozen']:acceptance(root,'e6r-a',model,'E6r-a',config['model'])
        for model,label in ORDER:
            model_dir=e6.configurations()[model]['model'];session=session_path(root,'e6r-b',model,label)
            command=[sys.executable,'-B','-m','pt_agent.experiments.a1_transformer_agent','--part','session',
                '--prompt-version','r8-v11','--tune-rule','v8-6.24','--placement','program-plan',
                '--tune-search','llm-select','--session-label',label,'--tune-call-limit','14',
                '--model',model,'--model-dir',str(model_dir),'--out',str(session),
                '--call-dir',str(root/'calls'),'--budget-path',str(ledger)]
            session.parent.mkdir(parents=True,exist_ok=True)
            print('E6r_SESSION='+model+' '+label,flush=True)
            with (session.parent/'session.log').open('w') as log:status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            report=read(session/'REPORT.json') if (session/'REPORT.json').exists() else dict(complete=False,failure_class='code',error='Missing session report')
            if status.returncode or not report.get('complete'):
                if report.get('failure_class')=='provider':continue
                raise RuntimeError(model+' '+label+' E6r-b stop: '+str(report.get('error')))
            if report['frozen']:acceptance(root,'e6r-b',model,label,model_dir)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';write(root/'STOP.json',dict(error=error,api_after=len(ledger_rows(ledger))))
    result=summarize(root,old,previous,error)
    processes=subprocess.check_output(['ps','-eo','pid,comm'],text=True)
    result['final_processes']=[line for line in processes.splitlines() if len(line.split())>=2 and line.split()[1] in ('nix','nix-shell','hecate-opt')]
    registry=ledger_rows(RESULTS_DIR/'registry.jsonl')
    recorded=[r for r in registry if r.get('round')=='E6r' and r.get('mode')=='llm-call']
    result['registry_calls']=len(recorded)
    result['registry_accounting_equal']=len(recorded)==result['api_delta']
    result['registry_namespaces_equal']=all(sum(r['metrics'].get('budget_namespace')==n['namespace'] for r in recorded)==n['calls'] for n in result['namespaces'])
    result['accounting_equal']=result['accounting_equal'] and result['registry_accounting_equal'] and result['registry_namespaces_equal']
    write(root/'FINAL_CHECKS.json',{k:result[k] for k in ('api_before','api_after','api_delta','namespaces','accounting_equal','within_limits','final_processes','registry_calls','registry_namespaces_equal')})
    write(root/'E6r.json',result);register('all','summary',root,result)
    print('E6r_DONE='+str(root/'E6r.json')+' ERROR='+str(error),flush=True)
    return int(bool(error or result['final_processes'] or not result['accounting_equal'] or not result['within_limits']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--old',type=Path,default=RESULTS_DIR/'E5d-agent-sessions-20261106')
    parser.add_argument('--previous',type=Path,default=RESULTS_DIR/'E6-llm-select-20261107')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--bert-model-dir',type=Path)
    parser.add_argument('--qwen-model-dir',type=Path)
    args=parser.parse_args()
    if args.prepare:
        if args.bert_model_dir is None or args.qwen_model_dir is None:parser.error('Prepare requires both model dirs')
        prepare(args.root,args.old,dict(bert=args.bert_model_dir,qwen=args.qwen_model_dir));return 0
    return run(args)

if __name__=='__main__':raise SystemExit(main())
