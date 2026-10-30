"""U1 stage entry for normalized Transformer programs and external evaluation."""

from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np


def inside(args):
    from .models.info import ModelInfo
    from .data.samples import StoredSplits
    from .analyzer import calibrate,reference_forward
    from .planners.rule import plan
    from .dsl.parser import parse
    from .dsl.typecheck import check
    from .cost import estimate
    from .transformer_validation import execute
    from .metrics import compare
    from .run import _write
    from . import toolchain
    toolchain.setup_env()
    info=ModelInfo.load(args.model_info);graph=info.graph
    version=getattr(args,'prompt_version','r8-v7')
    samples=StoredSplits(Path(args.model_info).parent/'samples',info,getattr(args,'prepare_split',None))
    out=Path(args.out);report=dict(round='U4',complete=False,stage=None,llm_calls=0,report_images=0,
                                  program_path=str(out/'program.ptdsl'),task=info.task)
    try:
        arrays,labels=samples.get('calibration')
        calibration_path=out/'calibration.json'
        calibration=(json.loads(calibration_path.read_text()) if calibration_path.exists() else calibrate(graph,arrays))
        _write(calibration_path,calibration)
        program=None
        stages=list(args.stages)
        if getattr(args,'planner','rule')=='agent' and 'test' in stages:
            stages.remove('test')
            position=next((i for i,s in enumerate(stages) if s in ('export','compile')),len(stages))
            stages.insert(position,'test')
        for stage in stages:
            report['stage']=stage;_write(out/'REPORT.json',report)
            if stage=='plan':
                if getattr(args,'planner','rule')=='agent':
                    from .planners.transformer_session import ModelBudgetClient,run_session,output_tokens
                    from .llm import LLMClient
                    client=getattr(args,'agent_client',None)
                    if client is None:
                        client=ModelBudgetClient(LLMClient(args.budget_path,model='deepseek-flash',
                            reasoning_effort='high',max_tokens=output_tokens(version),max_total_calls=450),
                            getattr(args,'model_call_dir',out/'calls'),model=graph.name,prompt_version=version)
                    agent=run_session(info,calibration,samples,client,out/'session',
                        max_iters=getattr(args,'agent_iters',8),tuner=getattr(args,'tuner','on')=='on',
                        tuner_budget_s=getattr(args,'tuner_budget_s',2400),
                        tune_policy=getattr(args,'tune_policy','new_best'),prompt_version=version)
                    report['agent']=agent;report['llm_calls']=getattr(client,'calls',agent['llm_calls'])
                    _write(out/'REPORT.json',report)
                    if agent['status']!='feasible':raise RuntimeError('Agent status: '+agent['status'])
                    source=agent['best_program']
                    _write(out/'FROZEN.json',json.loads((out/'session/FROZEN.json').read_text()))
                elif getattr(args,'softmax_budget',False):
                    from .planners.transformer_rule import plan as token_plan
                    source=token_plan(graph,calibration,slots=16384,softmax_budget=True)
                else:source=plan(graph,calibration,slots=16384)
                (out/'program.ptdsl').write_text(source)
                program=parse(source)
            elif stage=='check':
                program=program or parse((out/'program.ptdsl').read_text())
                if version in ('r8-v8','r8-v9','r8-v10'):
                    from .dsl.interface import require_interface
                    report['interface']=require_interface(program,info)
                checked=check(program,{k:v.shape for k,v in graph.weights.items()},
                              **(dict(prompt_version=version) if version in ('r8-v8','r8-v9','r8-v10') else {}))
                report['check']=dict(ok=checked.ok,diags=[asdict(d) for d in checked.diags])
                if not checked.ok:raise RuntimeError('token DSL typecheck failed')
            elif stage=='mock':
                probe={k:v[:4] for k,v in arrays.items()}
                exact=execute(program,graph.weights,probe,engine='mock',exact=True)
                expected=reference_forward(graph,probe)[graph.outputs[0]]
                report['exact_max_abs_diff']=float(np.max(np.abs(exact-expected)))
                if np.any(np.abs(exact-expected)>1e-9*(1+np.abs(expected))):
                    raise RuntimeError('token exact-mode mathematical check differs')
                predicted=execute(program,graph.weights,arrays,engine='tensor')
                reference=reference_forward(graph,arrays)[graph.outputs[0]]
                metric=dict(values=reference,task=info.task)
                if info.task=='causal_lm':metric.update(lengths=arrays['attention_mask'].sum(1).astype(int),selected_last=True)
                report['calibration_metrics']=compare(metric,predicted,labels)
            elif stage=='count':
                report['cost']=estimate(program,graph.weights)
            elif stage=='export':
                from .export import export_package
                package=export_package((out/'program.ptdsl').read_text(),program,graph.weights,out/'export',
                    **(dict(model_info=info,prompt_version=version) if version in ('r8-v8','r8-v9','r8-v10') else {}))
                report['export']=dict(package=str(package),weights=str(out/'export/weights'/(program.name+'.npz')))
            elif stage=='compile':
                if getattr(args,'planner','rule')=='agent':
                    from .experiments.a1_transformer_agent import native_stage
                    report['native']=native_stage(Path(args.model_info).parent,out/'program.ptdsl',out,
                        **(dict(prompt_version=version) if version in ('r8-v8','r8-v9','r8-v10') else {}))
                    _write(out/'REPORT.json',report)
                    continue
                from .run import _compile_export
                if 'export' not in report:raise ValueError('compile requires export')
                report.update(_compile_export(report['export']))
                if not report.get('compile') or report['compile']['exit']!=0:raise RuntimeError('token native compilation failed')
                if report['plan']['boots']!=report['cost']['boots_pred']:raise RuntimeError('predicted/native bootstrap count differs')
                report['pruned']=toolchain.prune(out/'export/compiled')
            elif stage=='test' and getattr(args,'planner','rule')=='agent':
                from .experiments.a1_transformer_agent import evaluate_frozen
                report['report']=evaluate_frozen(info,calibration,samples,out/'program.ptdsl',out/'report',
                                                  frozen=out/'FROZEN.json',prompt_version=version)
            elif stage in ('validate','test','emit'):
                raise ValueError('token held-out acceptance uses the U4 finite experiment entry')
            _write(out/'REPORT.json',report)
        report.update(complete=True,stage='complete')
        _write(out/'REPORT.json',report)
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report.update(complete=False,error=f'{type(exc).__name__}: {exc}')
        _write(out/'REPORT.json',report)
        return 1
