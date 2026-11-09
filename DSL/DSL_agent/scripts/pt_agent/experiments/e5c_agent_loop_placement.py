"""E5c A/C: offline session replay and pars acceptance; this entry has no API path."""

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..paths import REPO, RESULTS_DIR


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n')


def sources():
    return dict(
        bert=dict(model=RESULTS_DIR/'A1-bert-prepare-v2-a620-20261030/model',
                  review=RESULTS_DIR/'A1-a620-20261030/regrade/bert/REEVALUATION.json',
                  sessions=['A1-bert-session-20261029T055416-4780']),
        qwen=dict(model=RESULTS_DIR/'A1-qwen-prepare-v2-20261028T135355-6529/model',
                  review=RESULTS_DIR/'A1-r121window-20261030/regrade/qwen/REEVALUATION.json',
                  sessions=['A1-qwen-session-20261029T082910-a199', 'A1-qwen-session-20261029T090457-4f95',
                            'A1-qwen-session-20261029T092305-1f29', 'A1-qwen-session-20261029T094842-01c5',
                            'A1-qwen-session-20261029T102252-2f9b']))


def statement_changes(before, after):
    """Report changed statement fields from parsed text, without executing either program."""
    def statements(source):
        result = {}
        for function in ast.parse(source).body:
            if not isinstance(function, ast.FunctionDef):continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
                if not isinstance(node.value,ast.Call):continue
                call=node.value
                values={keyword.arg:keyword.value for keyword in call.keywords}
                values['op']=call.func
                for i,arg in enumerate(call.args):values[f'input[{i}]']=arg
                result[(function.name,node.targets[0].id)]=(node.lineno,values)
        return result
    old,new=statements(before),statements(after)
    changes=[]
    for key in sorted(set(old)|set(new)):
        a,b=old.get(key,(None,{})),new.get(key,(None,{}))
        fields=[]
        for field in sorted(set(a[1])|set(b[1])):
            left,right=a[1].get(field),b[1].get(field)
            if (ast.dump(left) if left else None)!=(ast.dump(right) if right else None):
                fields.append(dict(field=field,old=ast.unparse(left) if left else None,
                                   new=ast.unparse(right) if right else None))
        if fields:changes.append(dict(function=key[0],statement=key[1],old_line=a[0],new_line=b[0],fields=fields))
    return changes


def native_inside(args):
    """E5b driver flow: export, verified plan, bytecode trace, pars compile, statistics, prune."""
    import gc
    import resource
    from .. import toolchain
    from ..cost import estimate, release_estimate_caches, plan_latency
    from ..dsl.parser import parse
    from ..export import export_package
    from ..models.info import ModelInfo
    from .u4_relative_acceptance import native_entry
    resource.setrlimit(resource.RLIMIT_AS, (6*1024**3, 6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    toolchain.setup_env()
    program = parse(args.program.read_text())
    if (program.header.refresh, program.header.placement) != ('agent', 'program-plan'):
        raise ValueError('E5c native acceptance requires the frozen program-plan source')
    info = ModelInfo.load(args.model_dir)
    result = dict(complete=False, model=args.model, api_calls=0, program=str(args.program),
                  flow='E5b driver export/bytecode trace/pars compile', limits=toolchain.TRUSTED_LIMITS,
                  hecate_opt=str(toolchain.host_paths()['hecate_opt']), boot_window=toolchain.boot_window_rule(),
                  preflight=dict(skipped=True, reason='compiler placement only'),
                  managed=dict(skipped=True, reason='compiler placement only'))
    try:
        predicted = estimate(program, info.graph.weights)
        result['predicted'] = {key:predicted[key] for key in ('boots_pred','latency_pred_us','refresh_plan')}
        release_estimate_caches()
        package = export_package(args.program.read_text(), program, info.graph.weights, args.out/'export',
            model_info=info, prompt_version='r8-v10', run_info=dict(program_path=args.program, run_dir=args.out))
        weights = args.out/'export/weights'/(program.name+'.npz')
        saved = read(package/'config/refresh_plan.json')
        result['export'] = dict(package=str(package), weights=str(weights), boots=saved['boots'],
            calls=saved['calls'], timing=saved['timing'], replay=saved['replay'], variant=saved['placement'],
            compile_argv=read(package/'config/compile_argv.json'))
        del info, saved, predicted
        release_estimate_caches()
        gc.collect()
        write(args.out/'REPORT.json', result)
        directory = args.out/'export'
        result['trace'] = native_entry(package, weights, directory, 'compiler', 'trace', directory/'trace')
        bytecode = directory/'trace/trace_model.mlirbc'
        result['bytecode_bytes'] = bytecode.stat().st_size
        write(args.out/'REPORT.json', result)
        result['compile'] = native_entry(package, weights, directory, 'compiler', 'compile', directory/'compiled',
                                         input_path=bytecode)
        result['stats'] = toolchain.plan_stats(directory/'compiled', program.name)
        result['plan_latency'] = plan_latency(directory/'compiled'/f'lowered._hecate_{program.name}.runtime-plan.json')
        result['boots'] = dict(predicted=result['predicted']['boots_pred'], exported=result['export']['boots'],
                               compiled=result['stats']['boots'])
        result['three_way_equal'] = len(set(result['boots'].values())) == 1
        result['complete'] = result['three_way_equal']
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        for name in ('trace.log','compile.log'):
            path = args.out/'export'/name
            if path.exists():
                result[name.replace('.', '_')+'_tail'] = path.read_text(errors='replace')[-4000:]
        raise
    finally:
        result['pruned'] = {name: toolchain.prune(args.out/'export'/name)
                            for name in ('trace','compiled') if (args.out/'export'/name).exists()}
        write(args.out/'REPORT.json', result)


def a1_part(model, config, part, out, root, *, review=None, original=None, program=None):
    command = [sys.executable, '-B', '-m', 'pt_agent.experiments.a1_transformer_agent', '--part', part,
        '--model', model, '--prompt-version', 'r8-v10', '--tune-rule', 'v8-6.24', '--placement', 'program-plan',
        '--model-dir', str(config['model']), '--root', str(root), '--out', str(out)]
    if review:
        command += ['--response-file', str(review)]
    if original:
        command += ['--resume-session', str(original)]
    if program:
        command += ['--program', str(program), '--frozen', str(program.parent/'FROZEN.json')]
    subprocess.run(command, cwd=REPO/'DSL/DSL_agent/scripts', check=True)
    report = read(out/'REPORT.json')
    if not report.get('complete'):
        raise RuntimeError(f'{model} {part} incomplete: {out}')
    return report


def native(model, config, program, out):
    from ..registry import append, git_rev, new_run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '3600', '--',
        '-m', 'pt_agent.experiments.e5c_agent_loop_placement', '--inside-native', '--model', model,
        '--model-dir', str(config['model']), '--program', str(program), '--out', str(out)]
    with (out/'run.log').open('w') as log:
        status = subprocess.run(command, cwd=REPO/'DSL/DSL_agent/scripts', stdout=log, stderr=subprocess.STDOUT)
    report = read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False)
    report.update(exit=status.returncode, seconds=time.monotonic()-started)
    write(out/'REPORT.json', report)
    append(dict(run_id=new_run_id('E5c',model,'pars'), round='E5c', model=model, mode='pars', git_rev=git_rev(),
                metrics=report, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))
    if status.returncode or not report.get('complete'):
        raise RuntimeError(f'{model} pars acceptance failed: {out}')
    return report


def summarize(root, complete, error=None):
    result = dict(complete=complete, error=error, api_calls=0,
                  label='事后回放：LLM 输出固定，精调和选择使用 program-plan；report 有开发诊断历史', models={})
    e4c=RESULTS_DIR/'E4c-compile-burden-20261105/E4c.json'
    if e4c.exists():
        result['e4c_placement_times']=[{key:row.get(key) for key in ('program','placement','placement_s','compile_s','outcome')}
                                      for row in read(e4c)['rows'] if row.get('placement_s') is not None]
    for model, config in sources().items():
        directory = root/model
        row = {}
        recost_path = directory/'recost/REEVALUATION.json'
        if recost_path.exists():
            row['cost_changes'] = read(recost_path)['cost_changes']
        sessions = []
        for session in sorted(directory.glob('replay-session-*')):
            phases = [read(p) for p in sorted((session/'steps').glob('*/REPORT.json'))]
            sessions.append(dict(directory=str(session), steps=phases))
            if (session/'FROZEN.json').exists():
                row['frozen'] = read(session/'FROZEN.json')
                old = RESULTS_DIR/'A1-a624-20261102'/model/'replay-session-01/program.ptdsl'
                new_lines=(session/'program.ptdsl').read_text().splitlines()
                old_lines=old.read_text().splitlines()
                row['statement_changes']=statement_changes(old.read_text(),(session/'program.ptdsl').read_text())
                row['changed_lines']=[dict(line=i+1, old=old_lines[i] if i<len(old_lines) else None,
                                          new=new_lines[i] if i<len(new_lines) else None)
                    for i in range(max(len(old_lines),len(new_lines)))
                    if (old_lines[i] if i<len(old_lines) else None)!=(new_lines[i] if i<len(new_lines) else None)]
        row['sessions']=sessions
        for part in ('report','cost','pars'):
            path=directory/'acceptance'/part/'REPORT.json'
            if path.exists():row[part]=read(path)
        old_native=RESULTS_DIR/'A1-a624-20261102'/model/'acceptance-replay-session-01/compiler/REPORT.json'
        if old_native.exists():row['a624_compiler']=read(old_native)
        e5b=RESULTS_DIR/'E5b-program-placement-20261106'/model/'REPORT.json'
        row['e5b']=read(e5b)
        times=dict(llm_original_s=0., estimate_s=0., validation_s=0., plan_s=0., tune_s=0.,
                   feedback_estimate_s=0., feedback_validation_s=0., tune_estimate_s=0.,
                   tune_validation_s=0., confirm_s=0., pars_s=(row.get('pars',{}).get('compile') or {}).get('seconds'))
        used_calls=set()
        for session in sessions:
            for step in session['steps']:
                timing=step.get('timing',{})
                for key in ('estimate_s','validation_s','plan_s'):times[key]+=timing.get(key,0.)
                if step.get('step')=='evaluate':
                    times['feedback_estimate_s']+=timing.get('estimate_s',0.)
                    times['feedback_validation_s']+=timing.get('validation_s',0.)
                if step.get('step')=='tune':
                    times['tune_s']+=timing.get('total_s',0.)
                    times['tune_estimate_s']+=timing.get('estimate_s',0.)
                    times['tune_validation_s']+=timing.get('validation_s',0.)
                if step.get('step')=='confirm':times['confirm_s']+=timing.get('total_s',0.)
            trace=Path(session['directory'])/'agent_trace.jsonl'
            if trace.exists():
                events=[json.loads(line) for line in trace.read_text().splitlines()]
                times['llm_original_s']+=sum(event['seconds'] for event in events)
                used_calls.update(event['model_call'] for event in events if 'model_call' in event)
        if recost_path.exists():
            for key in ('estimate_s','validation_s','plan_s'):times[key]+=read(recost_path).get('timing',{}).get(key,0.)
            times['feedback_estimate_s']+=read(recost_path).get('timing',{}).get('estimate_s',0.)
        times['final_plan_s']=sum(row.get('pars',{}).get('export',{}).get('timing',{}).values())
        row['recost_cases']=len(row.get('cost_changes',[]))
        row['replayed_calls']=len(used_calls)
        times['feedback_consumed_estimate_s']=sum(
            item['timing']['estimate_s'] for item in row.get('cost_changes',[]) if item['call'] in used_calls)
        row['timing']=times
        row['outcome']='frozen' if row.get('frozen') else 'not_frozen_or_pending'
        if complete and not row.get('frozen'):
            row.update(outcome='not_frozen',reason='Saved sessions completed without a confirmation selection')
        elif error and not sessions:
            row.update(outcome='not_started',reason=error)
        result['models'][model]=row
    lines=['# E5c：program-plan 代价下的固定响应回放', '', result['label'], '',
           '耗时字段为实测秒数。tune_estimate_s/tune_validation_s 包含在 tune_s 中，plan_s 包含在 estimate_s 中；复用验证的阶段计0。', '',
           '反馈重算包含全部保存请求；feedback_consumed_estimate_s 仅汇总实际回放消费的请求。llm_original_s 引用这些旧响应的原始耗时，本轮API为0。', '',
           '| 模型 | A1-a624 depth-dp 自举 | 同程序 E5b 自举 | E5c pars 自举 | report100 一致率 / 误差 | 冻结来源 |',
           '|---|---:|---:|---:|---|---|']
    for model,row in result['models'].items():
        frozen=row.get('frozen',{})
        metrics=row.get('report',{}).get('engines',{}).get('tensor',{})
        confirmation=frozen.get('confirm',{}).get('selected') or {}
        model_call=confirmation.get('iteration',frozen.get('iteration'))
        sessions=row.get('sessions',[])
        if frozen and sessions:
            review=read(root/model/'recost/REEVALUATION.json')
            for session in sessions:
                if str(Path(frozen['program']).parent)==session['directory'] and session['steps']:
                    original=session['steps'][0]['original_session']
                    model_call=next((case['call'] for case in review['cases']
                                     if case['session']==original and case['iteration']==frozen['iteration']),model_call)
        row['frozen_model_call'] = model_call
        lines.append(f"| {model} | {row.get('e5b',{}).get('compiler_estimate',{}).get('boots_pred')} | "
            f"{row.get('e5b',{}).get('boots',{}).get('compiled')} | {row.get('pars',{}).get('boots',{}).get('compiled')} | "
            f"{metrics.get('agree')}/{metrics.get('total')} / {metrics.get('logit_rel_err')} | {frozen.get('origin')} / {model_call} |")
        lines += ['', f'## {model} 耗时与改动', '', '```json', json.dumps(row['timing'],indent=2), '```',
                  '', '逐 case 成本、步骤耗时和程序差异见 E5c.json。']
    lines += ['', '## 编译器放置耗时对照（引用已有记录）', '',
              'M7：BERT、Qwen optimized 放置均在 3600 秒内未完成；depth-dp 编译与同程序 E5b 数据见上表。',
              '', '```json',json.dumps(result.get('e4c_placement_times',[]),ensure_ascii=False,indent=2),'```']
    if error:lines += ['', '停止原因：'+error]
    write(root/'E5c.json', result)
    (root/'E5c.md').write_text('\n'.join(lines)+'\n')
    return result


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    os.environ['PT_RESULTS']=str(RESULTS_DIR)
    error=None
    ledger=RESULTS_DIR/'llm_calls.jsonl'
    before=len(ledger.read_text().splitlines()) if ledger.exists() else None
    write(root/'RUN.json',dict(round='E5c',api_calls=0,api_before=before,
          scope='A/C execution; B code and offline tests only',sources=sources()))
    try:
        for model,config in sources().items():
            a1_part(model,config,'log-recost',root/model/'recost',root/model,review=config['review'])
        # Preserve the A1-a624 replay order: Qwen's sessions first, then BERT.
        for model in ('qwen','bert'):
            config=sources()[model]
            frozen=None
            for index,name in enumerate(config['sessions'],1):
                session=root/model/f'replay-session-{index:02d}'
                while True:
                    checkpoint=session/'SESSION.json'
                    phase=read(checkpoint)['phase'] if checkpoint.exists() else 'evaluate'
                    if phase=='done':break
                    step=len(list((session/'steps').glob('*')))+1
                    total_cases=sum(case['session']==str(RESULTS_DIR/name)
                                    for case in read(root/model/'recost/REEVALUATION.json')['cases'])
                    if step>2*total_cases+1:
                        raise RuntimeError('Saved session step budget exhausted')
                    a1_part(model,config,'log-step',session/'steps'/f'{step:03d}-{phase}',session,
                            review=root/model/'recost/REEVALUATION.json',original=RESULTS_DIR/name)
                    summarize(root,False)
                if (session/'FROZEN.json').exists():
                    frozen=session/'program.ptdsl'
                    break
            if frozen:
                for part,name in (('evaluate','report'),('cost','cost')):
                    a1_part(model,config,part,root/model/'acceptance'/name,root/model/'acceptance',program=frozen)
                native(model,config,frozen,root/model/'acceptance/pars')
                summarize(root,False)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    result=summarize(root,error is None,error)
    after=len(ledger.read_text().splitlines()) if ledger.exists() else None
    active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            comm=(entry/'comm').read_text().strip()
            if comm in ('nix','nix-shell','hecate-opt'):active.append(dict(pid=int(entry.name),comm=comm))
        except OSError:pass
    checks=dict(api_before=before,api_after=after,api_unchanged=before==after,native_processes=active,
                complete=result['complete'])
    write(root/'FINAL_CHECKS.json',checks)
    if before!=after or active:
        raise RuntimeError('API accounting or final native process check failed')
    if error:raise RuntimeError(error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--inside-native',action='store_true')
    parser.add_argument('--model',choices=('bert','qwen'))
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--program',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.inside_native:native_inside(args)
    elif args.root:run(args.root)
    else:parser.error('Provide --root (outer replay) or --inside-native')


if __name__=='__main__':
    main()
