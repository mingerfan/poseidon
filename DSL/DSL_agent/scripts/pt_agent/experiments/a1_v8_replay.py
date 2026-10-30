"""A1 §6.5 finite offline diagnostics; no report access or provider calls."""

from dataclasses import asdict
import json
from pathlib import Path

from ..run import _write
from ..dsl.interface import require_interface
from ..dsl.parser import parse
from ..dsl.typecheck import check
from ..planners.agent_tools import build_context, evaluate, validate
from ..planners.transformer_session import json_safe


def context(args):
    from ..models.info import ModelInfo
    from .a1_transformer_agent import samples_for, read
    info=ModelInfo.load(args.model_dir)
    ctx=build_context(info,read(args.model_dir/'calibration.json'),samples=samples_for(args.model_dir,info))
    ctx.prompt_version='r8-v8'
    return ctx


def replay(args):
    ctx=context(args);rows=[];offset=0
    for session in args.legacy_sessions:
        trace={row['iteration']:row for row in map(json.loads,(session/'agent_trace.jsonl').read_text().splitlines())}
        replies=sorted((session/'sources').glob('iteration-*.ptdsl'),key=lambda p:int(p.stem.split('-')[-1]))
        for path in replies:
            iteration=int(path.stem.split('-')[-1]);number=offset+iteration
            before=trace.get(iteration,{}).get('fb',{}).get('stage','exception_input_binding')
            fb=evaluate(path.read_text(),ctx,engine_profile='tensor')
            _write(args.out/f'call-{number:03d}.json',json_safe(dict(source=str(path),feedback=fb)))
            row=dict(call=number,session=str(session),iteration=iteration,source=str(path),
                     old_stage=before,new_stage=fb['stage'],feasible=fb['feasible'],
                     errors=fb.get('errors'),exact=fb.get('exact'),validation=fb.get('validation'))
            rows.append(row)
            _write(args.out/'REPLAY.json',json_safe(dict(complete=False,rows=rows,llm_calls=0,report_images=0)))
            print(f"REPLAY call={number} {before}->{fb['stage']} feasible={fb['feasible']}",flush=True)
            if fb.get('stop'):raise RuntimeError('A1 §6.7 '+fb['stop'])
        offset+=max(trace.keys()|{int(path.stem.split('-')[-1]) for path in replies})
    if len(rows)!=26:raise ValueError(f'Expected the retained 26 programs, got {len(rows)}')
    result=dict(complete=True,rows=rows,programs=len(rows),llm_calls=0,report_images=0)
    _write(args.out/'REPLAY.json',json_safe(result))
    return json_safe(result)


def tune_diagnostic(args):
    from ..planners.tuner import tune
    ctx=context(args);program=parse(args.program.read_text())
    interface=require_interface(program,ctx.model_info)
    checked=check(program,ctx.weight_shapes,graph=ctx.graph,prompt_version='r8-v8')
    if not checked.ok:raise ValueError('Development diagnostic typecheck failed: '+str(checked.diags))
    before=evaluate(args.program.read_text(),ctx,engine_profile='tensor')
    _write(args.out/'BEFORE.json',json_safe(before))
    if before.get('stop'):raise RuntimeError('A1 §6.7 '+before['stop'])
    result=tune(program,ctx,budget_s=2400,max_sweeps=1,engine_profile='tensor')
    (args.out/'program.ptdsl').write_text(result.source)
    record={key:value for key,value in asdict(result).items() if key!='program'}
    record.update(complete=True,development_only=True,source=str(args.program),interface=interface,
                  llm_calls=0,report_images=0)
    _write(args.out/'TUNER.json',json_safe(record))
    return json_safe(record)


def baselines(args):
    from ..domain_criteria import require_analytic_coverage
    ctx=context(args)
    sources={'R0':args.r0_program,'R1':args.program}
    interfaces={name:require_interface(parse(path.read_text()),ctx.model_info) for name,path in sources.items()}
    result=dict(complete=False,model=args.model,split='heldout',rows={},interfaces=interfaces,llm_calls=0,report_images=0)
    for name,path in sources.items():
        program=parse(path.read_text())
        checked=check(program,ctx.weight_shapes,graph=ctx.graph,prompt_version='r8-v8')
        if not checked.ok:raise ValueError(name+' baseline typecheck failed: '+str(checked.diags))
        captured=[]
        measured=validate(program,ctx,engine='tensor',output_callback=captured.append)
        import numpy as np
        denominator=float(np.max(np.abs(ctx.val_exact_logits)))
        measured['logit_rel_err']=(float(np.max(np.abs(captured[0]-ctx.val_exact_logits)))/denominator
                                   if denominator and np.isfinite(captured[0]).all() else None)
        result['rows'][name]=dict(program=str(path),validation=measured,interface=interfaces[name])
        _write(args.out/'BASELINES.json',json_safe(result))
        print(f"BASELINE {name} feasible={measured['feasible']} calibrated={measured['max_u_calibrated']} analytic={measured['max_u_analytic']}",flush=True)
        require_analytic_coverage(measured)
    result['complete']=True
    _write(args.out/'BASELINES.json',json_safe(result))
    return json_safe(result)


def requests(args):
    from ..llm import Config
    from ..planners.prompts import build_system_prompt, initial_message
    ctx=context(args)
    text=build_system_prompt(ctx)
    # Summary contains original public names; generic prompt contains no target weights.
    overlap=[name for name in ctx.weights if name in text]
    if overlap:raise ValueError('Target weight names found outside graph summary: '+repr(overlap))
    body=Config(model='deepseek-flash',reasoning_effort='high',max_tokens=65536,timeout_seconds=600).settings()
    body['messages']=[dict(role='system',content=text),dict(role='user',content=initial_message(ctx))]
    _write(args.out/'first-request.pretty.json',body)
    return dict(complete=True,request=str(args.out/'first-request.pretty.json'),model=args.model,
                prompt_version='r8-v8',endpoint='https://api.deepseek.com/chat/completions',
                max_tokens=65536,max_model_calls=40,max_total_calls=450,global_start=166,
                budget_namespace=[args.model,'r8-v8'],llm_calls=0,report_images=0)
