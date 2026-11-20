"""Inner fresh Rule63 phases: explicit paths, no results registry or API access."""
import gc
import os
from pathlib import Path
import shutil
import sys

from .r10_paper_metrics import StageRecorder, run_process, write_json


def read(path):
    import json
    return json.loads(Path(path).read_text())


def program_inputs(args):
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from .r14_common_worker import check_program_signature
    info=ModelInfo.load(args.model_info);program=parse(args.program.read_text())
    checked=check(program,{name:value.shape for name,value in info.graph.weights.items()},graph=info.graph)
    if not checked.ok or not check_program_signature(info,program)['passed']:
        raise ValueError('Fresh Rule63 type/interface check failed')
    if program.header.refresh!='compiler' or program.header.placement!='depth-dp':
        raise ValueError('Preserve original Rule63 compiler-refresh depth header')
    acts=[op for op in program.ops if op.kind=='act']
    if not acts or any(op.attrs.get('method')!='cheb' or op.attrs.get('deg')!=63 for op in acts):
        raise ValueError('Every original activation must retain Chebyshev63')
    return info,program


def generate(args,rec,result):
    import numpy as np
    from ..models.info import ModelInfo
    from ..planners.rule import plan
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..tensor_sim import run_tensor
    from .r14_common_worker import check_program_signature
    from .r17_margin_validation_worker import _count_candidate,_packed_logits,check_math
    with rec.span('restore_fresh_model_and_calibration'):
        info=ModelInfo.load(args.model_info);calibration=read(args.calibration)
        if info.source.get('factory')!='pt_agent.experiments.r13_common_models:'+args.model:
            raise ValueError('Fresh model source identity changed')
        if any(value.dtype!=np.dtype('float64') for value in info.graph.weights.values()):
            raise ValueError('Original fresh graph weights must retain float64')
    options=dict(deg=63,margin=1.5,slots=16384,placement='depth-dp',refresh='compiler',act_method='cheb')
    with rec.span('deterministic_rule63_program'):
        text=plan(info.graph,calibration,**options);args.program.write_text(text);program=parse(text)
        checked=check(program,{name:value.shape for name,value in info.graph.weights.items()},graph=info.graph)
        signature=check_program_signature(info,program)
        if not checked.ok or not signature['passed']:raise ValueError('Fresh rule type/interface check failed')
        result.update(rule_options=options,signature=signature,program=str(args.program),rule_initialization_runs=1)
    with rec.span('fresh_raw_Count'):
        result['raw_Count']=_count_candidate(program,info.graph.weights)
        if result['raw_Count']['stats']['counts']['boot']!=0:
            raise ValueError('Compiler-refresh raw Count must have no frontend bootstrap')
    with rec.span('fresh_three_training_Tensor_Mock_mathematics'):
        intake=read(args.case_root/'work/intake/phase_result.json')
        with np.load(intake['artifacts']['training_probes'],allow_pickle=False)as source:
            images=np.array(source['input'])
        if images.shape!=(3,3,32,32)or images.dtype!=np.dtype('float64'):
            raise ValueError('Three fresh original training probes required')
        checks=[]
        for i in range(3):
            tensor=run_tensor(program,info.graph.weights,images[i:i+1],cache=False,act_mode='program')
            packed=_packed_logits(program,info.graph.weights,images[i:i+1],act_mode='program')
            observation=check_math(tensor,packed);checks.append(dict(index=i,**observation))
            if not observation['passed']:raise ValueError('Fresh strict Tensor/Mock mathematics failed')
        result['training_math']=dict(samples=3,criterion='1e-12+1e-12*abs(reference)',checks=checks,
            scope='Approximate Tensor versus packed arithmetic on training inputs; no task-quality evaluation')
    gc.collect()


def emit(args,rec,result):
    from ..backends import EmitBackend
    from ..lower import lower
    with rec.span('restore_and_typecheck_fresh_rule'):
        info,program=program_inputs(args)
    with rec.span('trusted_unplaced_rule_emit'):
        backend=EmitBackend(program.header.slots,frontend=args.frontend,
            constants=args.emit_dir/'constants.npz',save_dir=args.emit_dir)
        lower(program,info.graph.weights,backend,act_mode='program')
        result['emit_stats']=backend.stats()
        if result['emit_stats']['counts']['boot']!=0:
            raise ValueError('Native compiler must own all Rule63 bootstrap placement')
        expected=read(args.case_root/'work/generation/phase_result.json')['raw_Count']['stats']
        if any(result['emit_stats'][key]!=expected[key]for key in('counts','ops_total','distinct_consts')):
            raise ValueError('Fresh raw Count and Emit statistics differ')
    with rec.span('trusted_emit_write'):
        text,constants=backend.render()
        estimate=len(text.encode())+sum(v.nbytes+1024 for v in constants.values())+1024**2
        result['emit_estimate_bytes']=estimate
        from .completion_metrics import artifact_bytes
        if(artifact_bytes(args.case_root)+estimate>args.case_gib*1024**3-1024**3
                or shutil.disk_usage(args.emit_dir).free-estimate<10*1024**3):
            raise RuntimeError('resource_limit: Emit would consume case allowance or free10GiB')
        del text,constants
        backend.write(args.emit_dir)
    gc.collect()


def native(args,rec,result):
    from .. import toolchain
    from .r29_local_compile_matrix import compiler_argv,plan_observation
    from .r30_common_model_e2e import diagnostic_compile_argv,compile_artifact_metadata
    host=toolchain.host_paths()
    env=dict(os.environ,HECATE=str(host['hecate']),HECATE_BUILD=str(host['hecate_build']),
             LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
    if args.worker_phase=='trace':
        argv=[sys.executable,'-B',str(args.emit_dir/'candidate_trace.py')];seconds=args.native_seconds
    else:
        system='dacapo-depth-dp'if args.placement=='depth'else'dacapo-optimized'
        argv=compiler_argv(read(args.metadata_template),system,binary=host['hecate_opt'],
            source=args.emit_dir/'candidate_trace.mlir',profile=args.profile,output=args.emit_dir/'lowered.mlir',
            debug_snapshots=False)
        argv=diagnostic_compile_argv(argv,elide_final_ir=args.diagnostic_elide_final_ir);seconds=args.native_seconds
    write_json(args.work_dir/'native-argv.json',argv)
    with rec.span('native_'+args.worker_phase):
        observation=run_process(argv,cwd=args.emit_dir,log=args.work_dir/'native.log',
            resources=args.work_dir/'native-resources.jsonl',timeout=seconds,mem_gib=24,fsize_gib=16,env=env)
        result['native_process']=observation
        if(observation['exit_code']or observation['timed_out']or observation['signal']is not None
                or observation['remaining_pids']):raise RuntimeError('Original native '+args.worker_phase+' failed')
    if args.worker_phase=='trace':
        p=args.emit_dir/'candidate_trace.mlir'
        if not p.is_file()or not p.stat().st_size:raise ValueError('Trace produced no complete MLIR')
        result['trace_bytes']=p.stat().st_size
    else:
        with rec.span('observe_actual_compiled_RuntimePlan'):
            plans=sorted(args.emit_dir.glob('lowered.*.runtime-plan.json'))
            if len(plans)!=1:raise ValueError('Exactly one actual RuntimePlan required')
            plan=read(plans[0]);result.update(plan_observation(plan,read(args.profile)))
            result['runtime_plan']=str(plans[0])
            result['compiler_artifacts']=compile_artifact_metadata(args.emit_dir,plans[0],plan,
                elided=args.diagnostic_elide_final_ir)


def run(args):
    from . import r30_common_model_e2e as r30
    from .r29_local_compile_matrix import native_metadata_commands
    r30._threads();args.work_dir.mkdir(parents=True,exist_ok=True)
    target=args.work_dir/'phase_result.json'
    if target.exists():raise ValueError('Preserve original phase result')
    rec=StageRecorder(args.work_dir/'events.jsonl')
    result=dict(round='COLD-RULE63',model=args.model,phase=args.worker_phase,status='running',error=None,
        api_calls=0,heldout_samples=0,confirm_samples=0,report_samples=0,CKKS_runs=0)
    write_json(target,result)
    try:
        with native_metadata_commands(read(args.metadata_template)):
            with rec.span('fresh_'+args.worker_phase):
                if args.worker_phase=='intake':r30._intake(args,rec,result)
                elif args.worker_phase=='calibration':r30._calibration(args,rec,result)
                elif args.worker_phase=='generation':generate(args,rec,result)
                elif args.worker_phase=='emit':emit(args,rec,result)
                else:native(args,rec,result)
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc)[:3000])
    finally:write_json(target,result)
    return int(result['status']!='ok')
