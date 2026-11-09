"""E5d outer accounting, sequential v11 sessions and post-freeze pars acceptance."""
import argparse
import ast
from datetime import datetime,timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace


def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def calls(p):return [json.loads(t) for t in Path(p).read_text().splitlines() if t.strip()]
def configurations():
    from pt_agent.experiments.e5c_agent_loop_placement import sources
    return sources()
def register(model,mode,out,metrics):
    from pt_agent.registry import append,git_rev,new_run_id
    append(dict(run_id=new_run_id('E5d',model,mode),round='E5d',model=model,mode=mode,git_rev=git_rev(),
        metrics=metrics,artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat()))


def prepare(root):
    from pt_agent.experiments.a1_transformer_agent import samples_for
    from pt_agent.llm import Config
    from pt_agent.models.info import ModelInfo
    from pt_agent.planners.agent import new_agent_state
    from pt_agent.planners.agent_tools import build_context
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    manifest=dict(destination='https://api.deepseek.com/chat/completions',models={},max_calls=20,
        model_limit=10,global_limit=450,api_before=244,retries=0,order=['bert','qwen'],api_calls=0)
    for model,c in configurations().items():
        directory=c['model'];info=ModelInfo.load(directory)
        ctx=build_context(info,read(directory/'calibration.json'),samples=samples_for(directory,info))
        ctx.prompt_version='r8-v11';ctx.placement='program-plan'
        state=new_agent_state(ctx,max_iters=10,engine_profile='tensor',tune_rule='v8-6.24')
        body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=65536,timeout_seconds=600).settings()
        body['messages']=state['messages']
        path=root/'network-review'/f'{model}-first-request.pretty.json'
        write(path,body)
        manifest['models'][model]=dict(model_dir=str(directory),body=str(path),bytes=path.stat().st_size,
            prompt_version='r8-v11',placement='program-plan',tune_rule='v8-6.24',criterion='a1-6.20',split='split-v2')
        del ctx,state,info;gc.collect()
    write(root/'network-review/manifest.json',manifest)
    print('PREPARED_NO_API',flush=True)


def native(args):
    from pt_agent import toolchain
    from pt_agent.cost import estimate,release_estimate_caches,plan_latency
    from pt_agent.dsl.parser import parse
    from pt_agent.export import export_package
    from pt_agent.models.info import ModelInfo
    from pt_agent.experiments.u4_relative_acceptance import native_entry
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    toolchain.setup_env();info=ModelInfo.load(args.model_dir)
    source=args.program.read_text();program=parse(source)
    if (program.header.refresh,program.header.placement)!=('agent','program-plan'):
        raise ValueError('Frozen v11 program must use program-plan')
    result=dict(complete=False,model=args.model,api_calls=0,program=str(args.program),
        limits=toolchain.TRUSTED_LIMITS,flow='E5b export/bytecode trace/pars compile',
        prompt_version='r8-v11',preflight=dict(skipped=True,reason='compiler placement only'),
        managed=dict(skipped=True,reason='compiler placement only'))
    try:
        predicted=estimate(program,info.graph.weights)
        result['predicted']={k:predicted[k] for k in ('boots_pred','latency_pred_us','refresh_plan')}
        release_estimate_caches()
        package=export_package(source,program,info.graph.weights,args.out/'export',model_info=info,
            prompt_version='r8-v11',run_info=dict(program_path=args.program,run_dir=args.out))
        weights=args.out/'export/weights'/(program.name+'.npz')
        saved=read(package/'config/refresh_plan.json')
        result['export']=dict(package=str(package),weights=str(weights),boots=saved['boots'],calls=saved['calls'],
            timing=saved['timing'],replay=saved['replay'],variant=saved['placement'])
        del predicted,saved,info;release_estimate_caches();gc.collect()
        directory=args.out/'export';write(args.out/'REPORT.json',result)
        result['trace']=native_entry(package,weights,directory,'compiler','trace',directory/'trace')
        bytecode=directory/'trace/trace_model.mlirbc';result['bytecode_bytes']=bytecode.stat().st_size
        write(args.out/'REPORT.json',result)
        result['compile']=native_entry(package,weights,directory,'compiler','compile',directory/'compiled',input_path=bytecode)
        result['stats']=toolchain.plan_stats(directory/'compiled',program.name)
        result['plan_latency']=plan_latency(directory/'compiled'/f'lowered._hecate_{program.name}.runtime-plan.json')
        result['boots']=dict(predicted=result['predicted']['boots_pred'],exported=result['export']['boots'],compiled=result['stats']['boots'])
        result['three_way_equal']=len(set(result['boots'].values()))==1
        result['complete']=result['three_way_equal']
        if not result['complete']:raise RuntimeError('Predicted/exported/compiled bootstrap mismatch')
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['pruned']={n:toolchain.prune(args.out/'export'/n) for n in ('trace','compiled') if (args.out/'export'/n).exists()}
        write(args.out/'REPORT.json',result)


def header_values(source):
    try:
        for node in ast.parse(source).body:
            if isinstance(node,ast.FunctionDef):
                for d in node.decorator_list:
                    if isinstance(d,ast.Call) and ast.unparse(d.func)=='fhe.program':
                        return {k.arg:ast.literal_eval(k.value) for k in d.keywords}
    except (SyntaxError,ValueError,TypeError):pass
    return None


def summarize(root,error=None):
    from pt_agent.experiments.e5c_agent_loop_placement import statement_changes
    from pt_agent.paths import RESULTS_DIR
    old=read(RESULTS_DIR/'E5c-agent-loop-placement-20261106/E5c.json')
    result=dict(round='E5d',complete=error is None,error=error,models={},api_before=244,
        api_after=len(calls(RESULTS_DIR/'llm_calls.jsonl')),label='r8-v11 new online sessions; post-freeze report/compiler results never returned to Agent')
    for model,c in configurations().items():
        session=root/model/'session';path=session/'REPORT.json'
        state=read(session/'SESSION.json') if (session/'SESSION.json').exists() else {}
        report=read(path) if path.exists() else {}
        call_dir=root/'calls'/model/'r8-v11'
        response_files=sorted(call_dir.glob('*.response.json'))
        responses=[read(p) for p in response_files]
        requests=list(call_dir.glob('*.request.json'))
        previous=old['models'][model]
        row=dict(status=state.get('phase','not_started'),session=report,calls=len(requests),
            prior=dict(a624=(previous['a624_compiler'].get('stats') or {}).get('boots'),e5b=previous['e5b']['boots']['compiled'],
                       e5c=previous['pars']['boots']['compiled']),
            tokens=dict(input=sum((r.get('usage') or {}).get('prompt_tokens',0) for r in responses),
                        output=sum((r.get('usage') or {}).get('completion_tokens',0) for r in responses)))
        if state.get('pending'):row['last_feedback']=state['pending']['fb']
        traces=[json.loads(t) for t in (session/'agent_trace.jsonl').read_text().splitlines()] if (session/'agent_trace.jsonl').exists() else []
        if traces:row['last_feedback']=traces[-1]['fb']
        phases=[read(p) for p in sorted((session/'steps').glob('*/REPORT.json'))]
        timing=dict(llm_s=sum(r.get('seconds',0) or 0 for r in responses),estimate_s=0.,validation_s=0.,plan_s=0.,
            feedback_estimate_s=0.,feedback_validation_s=0.,tune_s=0.,tune_estimate_s=0.,tune_validation_s=0.,confirm_s=0.)
        for phase in phases:
            t=phase.get('timing',{})
            for k in ('estimate_s','validation_s','plan_s'):timing[k]+=t.get(k,0.)
            if phase.get('step')=='evaluate':
                timing['feedback_estimate_s']+=t.get('estimate_s',0.)
                timing['feedback_validation_s']+=t.get('validation_s',0.)
            if phase.get('step')=='tune':
                timing['tune_s']+=phase.get('seconds',0.)
                timing['tune_estimate_s']+=t.get('estimate_s',0.)
                timing['tune_validation_s']+=t.get('validation_s',0.)
            if phase.get('step')=='confirm':timing['confirm_s']+=phase.get('seconds',0.)
        row['tunes']=[dict(call=t['iteration'],**{k:t['tuned'].get(k) for k in ('sweeps','converged','coarse_evals','full_evals','seconds','changes')},
                         reused_from=t.get('tuned_reused_from')) for t in traces if t.get('tuned')]
        semantics=0;textual=0;usage=[]
        for source_path in sorted((session/'sources').glob('iteration-*.ptdsl')):
            source=source_path.read_text();values=header_values(source)
            expected=dict(refresh='agent',placement='program-plan',l_max=5,boot_out_level=5)
            semantics+=int(values is not None and any(values.get(k)!=v for k,v in expected.items()))
            header=next((line for line in source.splitlines() if line.lstrip().startswith('@fhe.program(')),None)
            if header is not None:
                try:textual+=int(header.strip()!='@'+ast.unparse(ast.parse(header.strip()[1:],mode='eval').body))
                except SyntaxError:pass
        for t in traces:
            if t['fb'].get('stage')!='complete':continue
            i=t['iteration'];before=session/'sources'/f'iteration-{i}.ptdsl';after=session/'sources'/f'iteration-{i+1}.ptdsl'
            if not before.exists() or not after.exists():continue
            try:changes=statement_changes(before.read_text(),after.read_text())
            except SyntaxError as exc:
                usage.append(dict(after_call=i,next_call=i+1,parse_error=str(exc)));continue
            hotspots=t['fb']['cost']['hotspots']
            for change in changes:
                change['previous_hotspots']=[{k:h.get(k) for k in ('line','call_lines','latency_share','levels_left_at_refresh')}
                    for h in hotspots if change['old_line']==h['line'] or change['old_line'] in h.get('call_lines',[])]
                change['was_hotspot']=bool(change['previous_hotspots'])
            usage.append(dict(after_call=i,next_call=i+1,changes=changes))
        row.update(header_semantic_rewrites=semantics,header_text_rewrites_including_formatting=textual,placement_feedback_usage=usage)
        for part in ('report','cost','pars'):
            p=root/model/'acceptance'/part/'REPORT.json'
            if p.exists():row[part]=read(p)
        frozen=session/'FROZEN.json'
        if frozen.exists():
            row['frozen']=read(frozen)
            row['statement_changes']=statement_changes((RESULTS_DIR/'A1-a624-20261102'/model/'replay-session-01/program.ptdsl').read_text(),(session/'program.ptdsl').read_text())
        if 'pars' in row:
            timing['pars_s']=row['pars']['compile']['seconds']
            timing['export_plan_s']=sum(row['pars']['export']['timing'].values())
        row['timing']=timing;result['models'][model]=row
    result['api_delta']=result['api_after']-result['api_before']
    result['namespace_calls']=sum(r['calls'] for r in result['models'].values())
    result['accounting_equal']=result['api_delta']==result['namespace_calls']
    result['within_limits']=result['api_delta']<=20 and all(r['calls']<=10 for r in result['models'].values())
    write(root/'E5d.json',result)
    lines=['# E5d r8-v11 真实会话','',f"API {result['api_before']}→{result['api_after']}，新增{result['api_delta']}；账本与命名空间一致={result['accounting_equal']}。",'',
        '| 模型 | A1-a624 depth-dp | E5b | E5c | E5d预测/导出/编译 | report100 tensor/Mock | 冻结来源 | 调用 | tokens输入/输出 |',
        '|---|---:|---:|---:|---|---|---|---:|---|']
    for model,row in result['models'].items():
        b=row.get('pars',{}).get('boots',{});report=row.get('report',{}).get('engines',{});f=row.get('frozen',{})
        engines=' / '.join(str(report.get(e,{}).get('agree'))+'/100 err='+str(report.get(e,{}).get('logit_rel_err')) for e in ('tensor','mock'))
        lines.append(f"| {model} | {row['prior']['a624']} | {row['prior']['e5b']} | {row['prior']['e5c']} | {b.get('predicted')}/{b.get('exported')}/{b.get('compiled')} | {engines} | {f.get('origin')} #{f.get('iteration')} | {row['calls']} | {row['tokens']['input']}/{row['tokens']['output']} |")
    for model,row in result['models'].items():
        lines+=['',f'## {model} 耗时与精调','', '```json',json.dumps(dict(timing=row['timing'],tunes=row['tunes']),indent=2,ensure_ascii=False),'```',
            '',f"预测延迟（us）：{row.get('cost',{}).get('cost',{}).get('latency_pred_us')}；header语义改写{row['header_semantic_rewrites']}次，含格式文本变化{row['header_text_rewrites_including_formatting']}次。",'',
            '冻结程序相对A1-a624的语句字段变化：','```json',json.dumps(row.get('statement_changes',[]),indent=2,ensure_ascii=False),'```','',
            '放置反馈使用（描述性，不参与选择）：','```json',json.dumps(row['placement_feedback_usage'],indent=2,ensure_ascii=False),'```',
            '',f"结束原因：{row['session'].get('error') or row['session'].get('agent',{}).get('status') or row['status']}。"]
        if not row.get('frozen'):lines+=['最后反馈：','```json',json.dumps(row.get('last_feedback'),indent=2,ensure_ascii=False),'```']
    if error:lines+=['',f'停止：{error}']
    (root/'E5d.md').write_text('\n'.join(lines)+'\n')
    return result


def run(args):
    from pt_agent.paths import RESULTS_DIR,REPO
    root=args.root;ledger=RESULTS_DIR/'llm_calls.jsonl';os.environ['PT_RESULTS']=str(RESULTS_DIR)
    test_log=(root/'OFFLINE_TESTS-verified.log').read_text()
    if not test_log.rstrip().endswith('OK'):raise RuntimeError('Offline gate not passed')
    if (root/'RUN.json').exists():raise RuntimeError('This E5d run was already started; do not retry')
    before=len(calls(ledger))
    if before!=244:raise RuntimeError('E5d ledger start differs from approved 244')
    write(root/'RUN.json',dict(round='E5d',api_before=before,model_limit=10,total_limit=20,retries=0,order=['bert','qwen'],
        prompt_version='r8-v11',tune_rule='v8-6.24',criterion='a1-6.20',models={m:str(c['model']) for m,c in configurations().items()}))
    write(root/'OFFLINE_TESTS.json',dict(complete=True,tests=129,log=str(root/'OFFLINE_TESTS-verified.log'),llm_calls=0))
    error=None
    try:
        for model,c in configurations().items():
            session=root/model/'session'
            command=[sys.executable,'-B','-m','pt_agent.experiments.a1_transformer_agent','--part','session',
                '--prompt-version','r8-v11','--tune-rule','v8-6.24','--placement','program-plan','--model',model,
                '--model-dir',str(c['model']),'--out',str(session),'--call-dir',str(root/'calls'),'--budget-path',str(ledger)]
            print('MODEL_SESSION='+model,flush=True)
            with (root/f'{model}-session.log').open('w') as log:
                status=subprocess.run(command,cwd=args.scripts_dir,stdout=log,stderr=subprocess.STDOUT)
            report=read(session/'REPORT.json') if (session/'REPORT.json').exists() else dict(complete=False,failure_class='code',error='Missing outer session report')
            if not report.get('complete'):
                if report.get('failure_class') in ('provider','resource'):
                    print('MODEL_STOP='+model+' '+str(report.get('error')),flush=True);continue
                raise RuntimeError(model+' session code stop: '+str(report.get('error')))
            if not report.get('frozen'):
                print('MODEL_NOT_FROZEN='+model,flush=True);continue
            for part in ('evaluate','cost'):
                label='report' if part=='evaluate' else part
                out=root/model/'acceptance'/label
                command=[sys.executable,'-B','-m','pt_agent.experiments.a1_transformer_agent','--part',part,
                    '--prompt-version','r8-v11','--tune-rule','v8-6.24','--placement','program-plan','--model',model,
                    '--model-dir',str(c['model']),'--program',str(session/'program.ptdsl'),
                    '--frozen',str(session/'FROZEN.json'),'--root',str(root/model/'acceptance'),'--out',str(out)]
                print('ACCEPTANCE='+model+' '+part,flush=True)
                status=subprocess.run(command,cwd=args.scripts_dir)
                if status.returncode or not read(out/'REPORT.json').get('complete'):raise RuntimeError(model+' '+part+' failed')
            out=root/model/'acceptance/pars';out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
            command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',str(Path(__file__)),
                '--scripts-dir',str(args.scripts_dir),'--root',str(root),'--native','--model',model,'--model-dir',str(c['model']),
                '--program',str(session/'program.ptdsl'),'--out',str(out)]
            print('ACCEPTANCE='+model+' pars',flush=True)
            with (out/'run.log').open('w') as log:status=subprocess.run(command,cwd=args.scripts_dir,stdout=log,stderr=subprocess.STDOUT)
            report=read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False,error='No pars report')
            report.update(exit=status.returncode,seconds=time.monotonic()-began)
            write(out/'REPORT.json',report);register(model,'pars',out,report)
            if status.returncode or not report.get('complete'):raise RuntimeError(model+' pars failed')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';write(root/'STOP.json',dict(error=error,api_after=len(calls(ledger))))
    result=summarize(root,error)
    output=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
    jobs=[line for line in output.splitlines() if len(line.split())>=2 and line.split()[1] in ('nix','nix-shell','hecate-opt')]
    result['final_processes']=jobs
    write(root/'FINAL_CHECKS.json',dict(api_before=before,api_after=result['api_after'],api_delta=result['api_delta'],
        namespace_calls=result['namespace_calls'],accounting_equal=result['accounting_equal'],within_limits=result['within_limits'],native_processes=jobs))
    write(root/'E5d.json',result)
    register('all','summary',root,result)
    print('E5D_DONE='+str(root/'E5d.json')+' API='+str(result['api_after'])+' ERROR='+str(error),flush=True)
    return 1 if error or jobs or not result['accounting_equal'] or not result['within_limits'] else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scripts-dir',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--native',action='store_true')
    p.add_argument('--model');p.add_argument('--model-dir',type=Path);p.add_argument('--program',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args();sys.path.insert(0,str(args.scripts_dir))
    if args.prepare:prepare(args.root)
    elif args.native:native(args)
    else:raise SystemExit(run(args))
