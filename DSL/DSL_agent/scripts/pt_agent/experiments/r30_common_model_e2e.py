"""Fresh original CNN intake -> training calibration -> Agent -> trusted compile.

Every measured stage belongs to this invocation. Existing source algorithms and
the compiler metadata recipe are reused; saved model/calibration/program results
and historical elapsed times are excluded. No rule initializes the Agent.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from types import SimpleNamespace

from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'R30'
MODE = 'fresh-common-model-agent-e2e'
FACTORY_PREFIX = 'pt_agent.experiments.r13_common_models:'
MODELS = ('resnet20_silu','vgg16_silu_avgpool','alexnet_silu_avgpool')
PHASES = ('intake','calibration','agent','placement','emit','trace','compile')
VARIANT = 'llm-first-generation+generic-deterministic-tuning'
NO_TUNER_VARIANT = 'llm-generation+tuner-disabled'
GIB = 1024**3
FORMAL = ('compiled_bootstraps','predicted_latency_us','CKKS_latency','accuracy_acceptance')
DIAGNOSTIC_ELIDE_OPTION = '--mlir-elide-elementsattrs-if-larger=16'


def compiler_output_variant(args):
    return ('diagnostic-elide-nonsplat-elements-over16'
            if getattr(args,'diagnostic_elide_final_ir',False) else 'full-final-mlir')


def diagnostic_compile_argv(argv, *, elide_final_ir=False):
    """Preserve the real export prefix and adapt only registered print flags."""
    if type(elide_final_ir) is not bool:
        raise ValueError('The diagnostic final-IR option must be a bool')
    result=list(argv)
    if not elide_final_ir:
        return result
    if (result.count('-o')!=1 or result.index('-o')+1>=len(result)
            or result[result.index('-o')+1] in ('-', '/dev/null')
            or any(value.startswith('--mlir-elide-elementsattrs-if-larger') for value in result)
            or any(value.startswith('--enable-debug-printer') for value in result)):
        raise ValueError('Diagnostic elision requires the unchanged real output prefix and no competing printer option')
    result.insert(result.index('-o'),DIAGNOSTIC_ELIDE_OPTION)
    return result


def compile_artifact_metadata(emit, plan_path, plan, *, elided=False):
    """Observe full executable artifacts without reading or checking blob contents."""
    emit=Path(emit);plan_path=Path(plan_path)
    def regular(path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size<=0:
            raise ValueError('A complete regular compiler artifact is required: '+str(path))
        return dict(path=str(path),bytes=path.stat().st_size)
    final=regular(emit/'lowered.mlir');raw_plan=regular(plan_path)
    if plan_path.parent!=emit or not plan_path.name.startswith('lowered.') or not plan_path.name.endswith('.runtime-plan.json'):
        raise ValueError('The RuntimePlan must keep the original lowered output prefix')
    bundle_path=plan_path.with_name(plan_path.name.removesuffix('.runtime-plan.json')+'.bundle')
    reference=plan.get('plaintext_bundle')
    required={row.get('payload',{}).get('content') for row in plan.get('initialization',[])
              if row.get('payload',{}).get('kind')=='bundle'}
    bundle=None
    if reference is not None:
        manifest_meta=regular(bundle_path/'manifest.json');manifest=read(bundle_path/'manifest.json')
        if (bundle_path.is_symlink() or manifest.get('bundle_format_version')!=1
                or manifest.get('version')!=1 or reference.get('version')!=1
                or manifest.get('bundle_id')!=reference.get('id')):
            raise ValueError('The full exported plaintext manifest contract differs')
        inventory={};total=0
        for row in manifest.get('blobs',[]):
            content=row.get('content');length=row.get('byte_length')
            if not isinstance(content,str) or ':' not in content or content in inventory:
                raise ValueError('A unique existing plaintext locator is required')
            leaf=content.split(':',1)[1]
            if not leaf or any(ch not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-' for ch in leaf):
                raise ValueError('Unsafe existing plaintext locator')
            metadata=regular(bundle_path/'data'/(leaf+'.bin'))
            if type(length) is not int or length<=0 or metadata['bytes']!=length:
                raise ValueError('The complete exported plaintext blob length differs')
            inventory[content]=length;total+=length
        if not required.issubset(inventory):
            raise ValueError('A referenced executable plaintext payload is missing')
        bundle=dict(path=str(bundle_path),manifest=manifest_meta,blob_count=len(inventory),
                    referenced_blob_count=len(required),blob_bytes=total,payload_values_read=0,
                    content_identifiers='Existing locator strings only; no digest generation or comparison')
    elif required:
        raise ValueError('Bundle initialization requires the full exported manifest')
    final.update(role='diagnostic text; executable consumers use the full RuntimePlan and bundle',
        nonsplat_elements_over16_elided=elided,constant_preserving_roundtrip=not elided,
        elision_placeholder='dense_resource<__elided__>' if elided else None)
    return dict(final_ir=final,runtime_plan=raw_plan,plaintext_bundle=bundle,
                executable_payloads_elided=False,
                printer_scope='MLIR global text printer; final output and other default-flag textual diagnostics',
                additional_integrity_checks=0)


def agent_variant(args):
    return VARIANT if getattr(args,'tuner',True) else NO_TUNER_VARIANT


def read(path):
    return json.loads(Path(path).read_text())


def _threads():
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'


def _intake(args, recorder, result):
    from . import r13_common_worker as original
    import torch
    torch.set_num_interop_threads(1)
    selected=SimpleNamespace(model=FACTORY_PREFIX+args.model, inputs=args.inputs,
                             selections=args.selections,work_dir=args.work_dir)
    result.update(artifacts={},import_completed=False)
    original.run(selected,recorder,result)
    if not all(result.get(name) is True for name in ('import_completed','real_probes_passed',
                                                    'graph_math_passed','roundtrip_passed','static_coverage')):
        raise ValueError('Fresh original intake did not pass every original check')


def _calibration(args, recorder, result):
    from ..models.info import ModelInfo
    from ..analyzer import calibrate
    from .r14_common_worker import _read_calibration, _check_calibration
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with recorder.span('restore_current_intake'):
        info=ModelInfo.load(args.model_info)
        if info.source.get('factory')!=FACTORY_PREFIX+args.model:
            raise ValueError('Fresh intake model identity changed')
    with recorder.span('read_1000_training_calibration'):
        arrays,labels,source=_read_calibration(args,info)
        if len(labels)!=1000 or source.get('split')!='train':
            raise ValueError('Calibration requires the original 1000 training selections')
        write_json(args.work_dir/'calibration-source.json',source)
        result['source']=source
    with recorder.span('calibrate_batch_one'):
        result['calibration_attempts']=1
        write_json(args.work_dir/'phase_result.json',result)
        values=calibrate(info.graph,arrays,batch=1)
        _check_calibration(info.graph,values)
        write_json(args.calibration,values)
        result.update(calibration_completed=True,calibration_samples=1000,calibration_batch=1,
                      calibration=str(args.calibration),rule_initialization_runs=0,
                      emit_runs=0,count_runs=0)
        del arrays,labels,values,info
        gc.collect()


def _agent(args, recorder, result):
    from .r28_common_agent import worker
    selected=SimpleNamespace(model=args.model,model_info=args.model_info,calibration=args.calibration,
        work_dir=args.work_dir,ledger=args.ledger,allowance=args.allowance,calls=args.calls,
        tuner_seconds=args.tuner_seconds,tuner=getattr(args,'tuner',True),memory_gib=args.memory_gib,
        prepare_only=False,resume_source=None,resume_result=None)
    result['variant']=agent_variant(args)
    result['tuner_enabled']=selected.tuner
    result['agent_worker_exit']=worker(selected)
    result['agent_result']=str(args.work_dir/'worker-result.json')
    if result['agent_worker_exit']:
        raise RuntimeError('Fresh Agent session failed; preserve its partial session and responses')


def _post_freeze(args, recorder, result):
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from .r14_common_worker import check_program_signature
    from .r29_local_compile_matrix import compiler_argv,plan_observation
    from .. import toolchain
    phase=args.worker_phase;post=args.post_dir;emit=post/'emit'
    if phase in ('placement','emit'):
        with recorder.span('restore_frozen_program_and_current_graph'):
            info=ModelInfo.load(args.model_info);program=parse(args.program.read_text())
            if info.source.get('factory')!=FACTORY_PREFIX+args.model:
                raise ValueError('Post-freeze model differs from the fresh intake')
            checked=check(program,{name:value.shape for name,value in info.graph.weights.items()},graph=info.graph)
            if not checked.ok or check_program_signature(info,program)['passed'] is not True:
                raise ValueError('Frozen program changed its original model interface')
            if program.header.refresh!='agent' or program.header.placement!='program-plan':
                raise ValueError('Frozen Agent program must retain program-plan placement')
        if phase=='placement':
            from ..plan import plan_program
            with recorder.span('frozen_program_placement'):
                plan,backend,observed=plan_program(program,info.graph.weights,profile='qwen48')
                write_json(post/'refresh-plan.json',plan)
                result.update(boots_pred=observed.boots,latency_pred_us=observed.latency_us,
                              refresh_plan=str(post/'refresh-plan.json'))
                del backend,observed
        else:
            from ..backends import EmitBackend,RefreshBackend
            from ..lower import lower
            with recorder.span('post_freeze_emit_lower'):
                plan=read(post/'refresh-plan.json')
                emitter=EmitBackend(program.header.slots,frontend=args.frontend,
                    constants=emit/'constants.npz',save_dir=emit)
                active=RefreshBackend(emitter,policy='plan',plan=plan,**plan['levels'])
                lower(program,info.graph.weights,active,act_mode='program')
                stats=emitter.stats();result['emit_stats']=stats
                if stats['counts']['boot']!=plan['boots']:
                    raise ValueError('Emitted bootstrap replay differs from the frozen placement')
            with recorder.span('post_freeze_emit_write'):
                text,constants=emitter.render()
                estimate=len(text.encode())+sum(value.nbytes+1024 for value in constants.values())+1024**2
                free=shutil.disk_usage(args.work_dir).free
                result['emit_estimate_bytes']=estimate
                if (artifact_bytes(args.case_root)+estimate>args.case_gib*GIB-GIB
                        or free-estimate<10*GIB):
                    raise RuntimeError('Emit estimate exceeds case allowance or the 10 GiB reserve')
                del text,constants
                emitter.write(emit)
                result['trace_source']=str(emit/'candidate_trace.py')
                del active,emitter
    elif phase in ('trace','compile'):
        host=toolchain.host_paths()
        env=dict(os.environ,LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'],
                 HECATE=str(host['hecate']),HECATE_BUILD=str(host['hecate_build']))
        if phase=='trace':
            argv=[sys.executable,'-B',str(emit/'candidate_trace.py')]
            seconds=args.trace_seconds
        else:
            argv=compiler_argv(read(args.metadata_template),'pars',binary=host['hecate_opt'],
                source=emit/'candidate_trace.mlir',profile=args.profile,output=emit/'lowered.mlir')
            argv=diagnostic_compile_argv(argv,elide_final_ir=getattr(args,'diagnostic_elide_final_ir',False))
            result['compiler_output_variant']=compiler_output_variant(args)
            seconds=args.compile_seconds
        write_json(args.work_dir/'native-argv.json',argv)
        with recorder.span('native_'+phase,command=argv):
            observation=run_process(argv,cwd=emit,log=args.work_dir/'native.log',
                resources=args.work_dir/'native-resources.jsonl',timeout=seconds,
                mem_gib=args.memory_gib,fsize_gib=8,env=env)
            result['process']=observation
            if (observation['exit_code'] or observation['timed_out'] or observation['signal'] is not None
                    or observation['remaining_pids']):
                raise RuntimeError('Native '+phase+' failed')
            if phase=='trace':
                path=emit/'candidate_trace.mlir'
                if not path.is_file() or not path.stat().st_size:raise ValueError('No complete trace MLIR')
                result['mlir']=dict(path=str(path),bytes=path.stat().st_size)
            else:
                plans=sorted(emit.glob('lowered.*.runtime-plan.json'))
                if len(plans)!=1:raise ValueError('Exactly one completed RuntimePlan is required')
                result['runtime_plan']=str(plans[0])
                compiled_plan=read(plans[0])
                result['compiler_output']=compile_artifact_metadata(emit,plans[0],compiled_plan,
                    elided=getattr(args,'diagnostic_elide_final_ir',False))
        if phase=='compile':
            with recorder.span('observe_compiled_bootstraps_and_profile_prediction'):
                observed=plan_observation(compiled_plan,read(args.profile))
                result.update(observed)
                if observed['compiled_bootstraps']!=read(post/'refresh-plan.json')['boots']:
                    raise ValueError('Compiled bootstrap count differs from the frozen placement')
    gc.collect()


def worker(args):
    _threads();args.work_dir.mkdir(parents=True,exist_ok=True)
    target=args.work_dir/'phase_result.json'
    if target.exists():raise ValueError('Preserve the previous phase result')
    recorder=StageRecorder(args.work_dir/'events.jsonl')
    result=dict(round=ROUND,model=args.model,phase=args.worker_phase,status='running',error=None,
                variant=agent_variant(args),tuner_enabled=getattr(args,'tuner',True),report_samples=0,keygen_runs=0,real_ckks_runs=0,
                compiler_output_variant=compiler_output_variant(args),
                diagnostic_elide_final_ir=getattr(args,'diagnostic_elide_final_ir',False),
                extra_integrity_checks=0,historical_elapsed_reused=False)
    write_json(target,result)
    try:
        from .r29_local_compile_matrix import native_metadata_commands
        with native_metadata_commands(read(args.metadata_template)):
            # R28 owns this phase's recorder and sequence in the same process.
            # A second StageRecorder would create duplicate PID:sequence IDs.
            if args.worker_phase=='agent':
                _agent(args,recorder,result)
            else:
                with recorder.span('fresh_'+args.worker_phase):
                    if args.worker_phase=='intake':_intake(args,recorder,result)
                    elif args.worker_phase=='calibration':_calibration(args,recorder,result)
                    else:_post_freeze(args,recorder,result)
        result['status']='ok'
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        write_json(target,result)
    return 0 if result['status']=='ok' else 1


def audit_agent(record, *, model, model_info, calibration, agent_dir):
    """Independent metadata/structural gate; no tensor or generated code execution."""
    if (record.get('status')!='ok' or record.get('frozen') is not True or record.get('model')!=model
            or Path(record.get('source_model_info','')).resolve()!=Path(model_info).resolve()
            or Path(record.get('source_calibration','')).resolve()!=Path(calibration).resolve()
            or (record.get('quality_acceptance')or{}).get('passed') is not True):
        raise ValueError('The current fresh Agent candidate lacks frozen quality acceptance')
    agent=record['agent'];selected=agent['confirm']['selected']
    rows=[row for row in agent['confirm']['evals']if (row['iteration'],row['origin'])==(selected['iteration'],selected['origin'])]
    if len(rows)!=1:raise ValueError('Frozen selection must have one matching confirmation row')
    row=rows[0]
    if (row['passed'] is not True or row['diverged'] or row['n_diverged']!=0 or row['total']!=1000
            or not all(math.isfinite(row[k])for k in ('acc_drop_pp','max_u_calibrated','max_u_analytic'))
            or row['acc_drop_pp']>.5 or row['max_u_calibrated']>.90 or row['max_u_analytic']>1):
        raise ValueError('The confirmation record fails the unchanged quality policy')
    info=read(model_info)
    session=read(Path(agent_dir)/'SESSION.json')
    if (info.get('source',{}).get('factory')!=FACTORY_PREFIX+model
            or info.get('threshold')!={'acc_drop_pp':.5}
            or session.get('phase')!='done' or session.get('failed')is not None
            or session.get('calls')!=record.get('actual_api_calls')
            or agent.get('llm_calls')!=session.get('calls') or record.get('recovered_api_calls',0)!=0):
        raise ValueError('Fresh model policy, completed session or actual call accounting differs')
    api=Path(agent_dir)/'api'
    requests=list(api.glob('*.request.json'));responses=list(api.glob('*.response.json'))
    if len(requests)!=len(responses) or len(requests)!=record['actual_api_calls']:
        raise ValueError('Every actual new provider call must retain its request and response')
    probes=record['strict_math_probes']
    if len(probes)!=3:raise ValueError('Three independent strict arithmetic probe pairs are required')
    for index,probe in enumerate(probes):
        if probe['sample_id']!='test:'+str(9000+index):raise ValueError('Arithmetic probe identity differs')
        for role in ('exact','approximate'):
            check=probe[role]
            if (any(check.get(name)is not True for name in ('passed','finite','dtype_ok','shape_ok'))
                    or check.get('criterion')!='1e-12+1e-12*abs(reference)'
                    or not math.isfinite(check['max_error_ratio']) or check['max_error_ratio']>1):
                raise ValueError('An independent strict arithmetic check failed')
    if any(record.get(name)!=0 for name in ('report_samples','compiler_runs','real_ckks_runs')):
        raise ValueError('Pre-freeze process accessed a report or measured backend')
    path=Path(record['selected_program']).resolve()
    if not path.is_relative_to(Path(agent_dir).resolve()) or not path.is_file():
        raise ValueError('Frozen program must originate in this fresh run')
    identity=lambda source:ast.dump(ast.parse(source),include_attributes=False)
    if identity(path.read_text())!=identity(selected['source']) or identity(agent['best_program'])!=identity(selected['source']):
        raise ValueError('Frozen program structure differs from the confirmed selection')
    return dict(status='quality_accepted',formal_frozen=True,quality_acceptance_passed=True,
                strict_math_passed=True,model=model,selected_program=str(path),confirmation=row,
                strict_math_probes=probes,actual_api_calls=record['actual_api_calls'],session_phase='done',
                saved_original_provider_replies=len(responses),
                scope='current invocation frozen program; compiler and CKKS measurements separate')


def phase_command(args, phase, work, *, remaining_seconds):
    from .. import paths
    bounds=dict(intake=1800,calibration=1800,agent=args.agent_seconds,placement=1800,emit=1800,
                trace=args.trace_seconds,compile=args.compile_seconds)
    seconds=min(bounds[phase],max(1,int(remaining_seconds)-180))
    inner=['-B','-m','pt_agent.experiments.r30_common_model_e2e','--worker-phase',phase,
        '--model',args.model,'--inputs',str(args.inputs),'--selections',str(args.selections),
        '--work-dir',str(work),'--case-root',str(args.out),'--model-info',str(args.out/'work/intake/model_info/model_info.json'),
        '--calibration',str(args.out/'work/calibration/calibration.json'),'--post-dir',str(args.out/'post-freeze'),
        '--program',str(args.out/'work/agent/program.ptdsl'),'--ledger',str(args.ledger),
        '--allowance',str(args.allowance),'--profile',str(args.profile),'--frontend',str(args.frontend),
        '--metadata-template',str(args.metadata_template),'--calls',str(args.calls),
        '--tuner-seconds',str(args.tuner_seconds),'--memory-gib',str(args.memory_gib),'--case-gib',str(args.case_gib),
        '--trace-seconds',str(min(args.trace_seconds,seconds)),'--compile-seconds',str(min(args.compile_seconds,seconds))]
    if not getattr(args,'tuner',True):inner+=['--no-tuner']
    if getattr(args,'diagnostic_elide_final_ir',False):inner+=['--diagnostic-elide-final-ir']
    return [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds+60),
            '--python',str(paths.INTAKE_PYTHON),'--',*inner],seconds+120


def _append(path, row):
    with Path(path).open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def archive_emit_constants_after_trace(result, case_root):
    """Archive this run's NPZ after clean tracing; record recovery before deletion.

    Compilation reads the self-contained dense-constant MLIR. Restore the NPZ
    with the recorded argv before replaying candidate_trace.py.
    """
    from .r34_archive_text import archive_one, archive_state

    def clean(observation):
        return (observation.get('exit_code') == 0 and observation.get('timed_out') is False
                and observation.get('signal') is None and observation.get('remaining_pids') == []
                and observation.get('signals_sent') == [])

    trace = result.get('phases', {}).get('trace') or {}
    emitted = result.get('phases', {}).get('emit') or {}
    observations = [row.get('observation') or {} for row in result.get('processes', [])
                    if row.get('phase') == 'trace']
    if (result.get('completed', {}).get('emit') != 1 or emitted.get('status') != 'ok'
            or result.get('completed', {}).get('trace') != 1 or trace.get('status') != 'ok'
            or len(observations) != 1 or not clean(observations[0])
            or not clean(trace.get('process') or {})
            or result.get('attempted', {}).get('compile') or result.get('completed', {}).get('compile')):
        raise ValueError('Archive only after a clean completed trace and before compiler entry')
    case_root = Path(case_root)
    post = case_root / 'post-freeze'
    emit = post / 'emit'
    source = emit / 'constants.npz'
    mlir = emit / 'candidate_trace.mlir'
    trace_source = emit / 'candidate_trace.py'
    if (Path(trace.get('mlir', {}).get('path', '')) != mlir
            or Path(emitted.get('trace_source', '')) != trace_source
            or not mlir.is_file() or mlir.is_symlink() or mlir.resolve() != mlir.absolute()
            or mlir.stat().st_size == 0 or trace.get('mlir', {}).get('bytes') != mlir.stat().st_size
            or not trace_source.is_file() or trace_source.is_symlink()
            or trace_source.resolve() != trace_source.absolute()):
        raise ValueError('Archive requires emitted source and recorded complete MLIR from this run')
    if (not source.is_file() or source.is_symlink() or source.resolve() != source.absolute()
            or source.stat().st_nlink != 1):
        raise ValueError('The Emit NPZ must remain an original regular case file with one link')
    target = source.with_name(source.name + '.gz')
    temporary = target.with_name(target.name + '.partial')
    recovery_path = post / 'emit-constants-recovery.json'
    if any(path.exists() or path.is_symlink() for path in (target, temporary, recovery_path)):
        raise ValueError('Preserve an earlier archive, partial output or recovery manifest')
    if shutil.disk_usage(case_root).free < 10 * GIB:
        raise RuntimeError('The 10 GiB disk reserve is required before archiving')

    def durable_recovery(value):
        write_json(recovery_path, value)
        with recovery_path.open('rb') as stream:
            os.fsync(stream.fileno())
        directory = os.open(str(post), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    recovery = dict(status='prepared', model=result.get('model'), source=str(source),
                    archive=str(target), temporary=str(temporary),
                    source_bytes=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                    compiled_input=str(mlir), compiled_input_bytes=mlir.stat().st_size,
                    restore_argv=['gzip', '-dk', '--', str(target)],
                    scope='Restore before replaying candidate_trace.py; compilation uses the completed MLIR',
                    extra_integrity_checks=0)
    durable_recovery(recovery)
    try:
        observed = archive_one(source, target, temporary)
    except BaseException as exc:
        recovery.update(status='error', error=f'{type(exc).__name__}: {exc}',
                        interrupted_state=archive_state(source, target, temporary))
        durable_recovery(recovery)
        raise
    recovery.update(status=observed['status'], observation=observed)
    durable_recovery(recovery)
    return dict(status='ok', archive=observed, recovery_record=str(recovery_path),
                restore_argv=recovery['restore_argv'], source_constants=str(source),
                compiled_input=str(mlir), trace_replay_requires_restore=observed['status'] == 'compressed',
                extra_integrity_checks=0)


def execute(args):
    started=time.perf_counter_ns()
    from .. import paths
    from ..autonomous_budget import call_records
    from ..registry import git_rev
    if args.out.exists():raise ValueError('Preserve prior end-to-end evidence')
    if not args.out.is_relative_to(paths.RESULTS_DIR.resolve()) or args.out==paths.RESULTS_DIR.resolve():
        raise ValueError('Output must be a new child of RESULTS_DIR')
    args.out.mkdir(parents=True);(args.out/'work').mkdir();(args.out/'post-freeze').mkdir()
    recorder=StageRecorder(args.out/'events.jsonl');before=len(call_records(args.ledger))
    result=dict(round=ROUND,model=args.model,mode=MODE,status='running',error=None,phases={},processes=[],
        attempted={},completed={},variant=agent_variant(args),tuner_enabled=getattr(args,'tuner',True),rule_initialization_runs=0,historical_elapsed_reused=False,
        original_intake_reused=False,training_calibration_reused=False,report_samples=0,
        parameter_group='original qwen48 compiler profile and CPU operator contract; same-backend control',
        keygen_runs=0,real_ckks_runs=0,extra_integrity_checks=0,
        api_before=before,api_after=None,api_delta=None,quality_audit=None,git_rev=git_rev(),
        completion_path=str(args.out/'completion.json'),**dict.fromkeys(FORMAL,None))
    result['archive_emit_constants_after_trace'] = getattr(args, 'archive_emit_constants_after_trace', False)
    result['diagnostic_elide_final_ir'] = getattr(args, 'diagnostic_elide_final_ir', False)
    result['compiler_output_variant'] = compiler_output_variant(args)
    stop=threading.Event();active=threading.Event();failure={}
    def monitor():
        while not stop.wait(1):
            try:
                size,free=artifact_bytes(args.out),shutil.disk_usage(args.out).free
                with (args.out/'artifact-samples.jsonl').open('a')as stream:
                    stream.write(json.dumps(dict(clock_ns=time.perf_counter_ns(),bytes=size,free_bytes=free))+'\n')
                if (getattr(args, 'archive_emit_constants_after_trace', False)
                        and (time.perf_counter_ns()-started)/1e9 > args.seconds):
                    raise TimeoutError('End-to-end allowance exhausted during optional archive')
                if size>args.case_gib*GIB or free<10*GIB:
                    raise RuntimeError('Case allowance or 10 GiB disk reserve exhausted')
            except BaseException as exc:
                failure['error']=type(exc).__name__+': '+str(exc)
                if active.is_set():os.kill(os.getpid(),signal.SIGTERM)
                return
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    attached=[]
    try:
        with recorder.span('fresh_end_to_end'):
            with recorder.span('source_preflight_and_snapshot'):
                from .r13_common_models import FACTORIES
                factory=FACTORY_PREFIX+args.model
                original=FACTORIES[factory]
                required=[args.inputs/'dataset.json',args.selections,args.frontend,args.profile,args.metadata_template,
                          *(Path(original[k])for k in ('source','checkpoint','inference_preprocessing_source'))]
                if any(not path.is_file()for path in required):raise FileNotFoundError('Missing original source prerequisite')
                if shutil.disk_usage(args.out).free<10*GIB:raise RuntimeError('10 GiB disk reserve is required')
                result['source']=dict(factory=factory,records=str(args.inputs),selections=str(args.selections),
                    original_sources={key:str(original[key])for key in ('source','checkpoint','inference_preprocessing_source')},
                    compiler_profile=str(args.profile),compiler_metadata_recipe=str(args.metadata_template))
                snapshot=args.out/'source';snapshot.mkdir()
                for source in (Path(__file__),Path(__file__).with_name('r28_common_agent.py'),
                               Path(__file__).with_name('r13_common_worker.py'),Path(__file__).with_name('r14_common_worker.py'),
                               Path(__file__).with_name('r29_local_compile_matrix.py'),Path(__file__).with_name('completion_metrics.py'),
                               Path(__file__).parents[1]/'autonomous_budget.py'):
                    shutil.copyfile(source,snapshot/source.name)
                # Capture the trusted implementation before any stage starts.
                # These are source copies, with no digest binding or comparison.
                package=Path(__file__).parents[1]
                source_files=sorted(path for path in package.rglob('*.py')
                                    if 'tests' not in path.relative_to(package).parts)
                for source in source_files:
                    destination=snapshot/'pt_agent'/source.relative_to(package)
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,destination)
                result['startup_source_snapshot_files']=len(source_files)
                shutil.copyfile(args.profile,snapshot/'compiler-profile.json')
                shutil.copyfile(args.metadata_template,snapshot/'compile-metadata-template.json')
            env={k:v for k,v in os.environ.items()if not k.startswith(('LD_','NIX_','__NIX_','HECATE_'))
                 and k not in ('HECATE','IN_NIX_SHELL','PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')}
            env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
            phases = list(PHASES)
            if getattr(args, 'archive_emit_constants_after_trace', False):
                phases.insert(phases.index('compile'), 'archive_emit_constants')
            for phase in phases:
                if failure:raise RuntimeError(failure['error'])
                remaining=args.seconds-(time.perf_counter_ns()-started)/1e9
                if remaining<=180:raise TimeoutError('End-to-end allowance exhausted')
                if phase == 'archive_emit_constants':
                    result['attempted'][phase] = 1
                    write_json(args.out/'progress.json', result)
                    def archive_interrupted(signum, _frame):
                        raise RuntimeError(failure.get('error', 'Optional archive interrupted by SIGTERM'))
                    previous_handler = signal.signal(signal.SIGTERM, archive_interrupted)
                    try:
                        with recorder.span(phase):
                            active.set()
                            try:
                                result['phases'][phase] = archive_emit_constants_after_trace(result, args.out)
                            finally:
                                active.clear()
                            if failure:
                                raise RuntimeError(failure['error'])
                            result['completed'][phase] = 1
                    finally:
                        signal.signal(signal.SIGTERM, previous_handler)
                    write_json(args.out/'progress.json', result)
                    continue
                work=args.out/'work'/phase;work.mkdir();(work/'tmp').mkdir()
                argv,seconds=phase_command(args,phase,work,remaining_seconds=remaining)
                result['attempted'][phase]=1;write_json(args.out/'progress.json',result)
                with recorder.span(phase,command=argv)as parent:
                    active.set()
                    try:
                        observed=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=work/'worker.log',
                            resources=work/'resources.jsonl',timeout=seconds,mem_gib=args.memory_gib,fsize_gib=8,
                            env=dict(env,TMPDIR=str(work/'tmp')))
                    finally:active.clear()
                    result['processes'].append(dict(phase=phase,argv=argv,observation=observed))
                    for event in read_events(work/'events.jsonl'):
                        if event.get('parent_id')is None:event['parent_id']=parent['event_id']
                        attached.append(event)
                    value=read(work/'phase_result.json')if(work/'phase_result.json').is_file()else None
                    result['phases'][phase]=value
                    if (observed['exit_code']or observed['timed_out']or observed['signal']is not None
                            or observed['remaining_pids']or not value or value['status']!='ok'):
                        raise RuntimeError('Fresh end-to-end stage failed: '+phase)
                    result['completed'][phase]=1
                if phase=='agent':
                    original=read(work/'worker-result.json')
                    if original['status']=='quality_rejected':
                        result.update(status='quality_rejected',error=None)
                        break
                    audit=audit_agent(original,model=args.model,
                        model_info=args.out/'work/intake/model_info/model_info.json',
                        calibration=args.out/'work/calibration/calibration.json',agent_dir=work)
                    audit['source_result']=str(work/'worker-result.json')
                    write_json(args.out/'post-run-quality-audit.json',audit)
                    result['quality_audit']=str(args.out/'post-run-quality-audit.json')
                if phase==args.stop_after:
                    result['status']='ok'if phase=='compile'else'prepared'
                    break
            if result['status']=='running':result['status']='ok'
            if result['status']=='ok':
                compiled=result['phases']['compile']
                result.update(compiled_bootstraps=compiled['compiled_bootstraps'],
                              predicted_latency_us=compiled['predicted_latency_us'])
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        stop.set();watcher.join(timeout=3)
        result['monitor_error']=failure.get('error')
        timing=summarize_events(read_events(args.out/'events.jsonl')+attached,termination_observed_ns=time.perf_counter_ns())
        write_json(args.out/'timing.json',timing)
        after=call_records(args.ledger);result.update(api_after=len(after),api_delta=len(after)-before,
            recorded_elapsed_seconds=(time.perf_counter_ns()-started)/1e9,
            artifact_bytes=0,artifact_bytes_scope='required files before completion marker')
        for _ in range(12):
            write_json(args.out/'result.json',result)
            size=artifact_bytes(args.out)
            if result['artifact_bytes']==size:break
            result['artifact_bytes']=size
        else:raise RuntimeError('Terminal artifact accounting did not stabilize')
        registry=paths.RESULTS_DIR/'registry.jsonl';now=datetime.now(timezone.utc).isoformat()
        for index,call in enumerate(after[before:],before+1):
            _append(registry,dict(run_id=args.out.name+'-api-'+str(index),round=ROUND,model=args.model,
                mode='llm-call',git_rev=result['git_rev'],metrics=dict(call,global_call=index),artifacts_dir=str(args.out),ts=call['ts']))
        metrics={key:result[key]for key in ('status','attempted','completed','variant','api_delta','compiled_bootstraps',
                 'predicted_latency_us','CKKS_latency','accuracy_acceptance','recorded_elapsed_seconds','completion_path')}
        _append(registry,dict(run_id=args.out.name,round=ROUND,model=args.model,mode=MODE,git_rev=result['git_rev'],
            metrics=metrics,artifacts_dir=str(args.out),ts=now))
        completion=persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=timing,required_artifact_bytes=result['artifact_bytes'])
        print(json.dumps(dict(status=result['status'],out=str(args.out),error=result['error'],
            complete_case_elapsed_seconds=completion['complete_case_elapsed_seconds'])),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--worker-phase',choices=PHASES)
    parser.add_argument('--stop-after',choices=('calibration','agent','compile'),default='compile')
    for name in ('out','inputs','selections','work-dir','case-root','model-info','calibration','post-dir','program',
                 'ledger','allowance','profile','frontend','metadata-template'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--seconds',type=int,default=28800)
    parser.add_argument('--agent-seconds',type=int,default=7200)
    parser.add_argument('--compile-seconds',type=int,default=7200)
    parser.add_argument('--trace-seconds',type=int,default=1800)
    parser.add_argument('--memory-gib',type=int,default=6)
    parser.add_argument('--case-gib',type=int,default=8)
    parser.add_argument('--archive-emit-constants-after-trace', action='store_true',
        help='Reversibly archive the generated NPZ after trace and before native compilation')
    parser.add_argument('--diagnostic-elide-final-ir',action='store_true',
        help='Print large non-splat elements as diagnostic placeholders; keep executable RuntimePlan/bundle complete')
    parser.add_argument('--calls',type=int,default=6)
    parser.add_argument('--tuner-seconds',type=int,default=600)
    parser.add_argument('--no-tuner',dest='tuner',action='store_false',default=True,
                        help='Disable numerical tuning for a new session; preserve evaluation, confirmation and strict mathematics')
    args=parser.parse_args(argv)
    if not(4<=args.memory_gib<=12 and 4<=args.case_gib<=32 and 1<=args.calls<=6
           and 0<args.compile_seconds<=7200 and 0<args.trace_seconds<=1800
           and args.agent_seconds>0 and args.seconds>180 and args.tuner_seconds>0):
        parser.error('Use finite stage limits, compile<=7200s, trace<=1800s, memory4..12GiB, case4..32GiB, calls1..6')
    if args.worker_phase:
        required=('inputs','selections','work_dir','case_root','model_info','calibration','post_dir','program',
                  'ledger','allowance','profile','frontend','metadata_template')
        if any(getattr(args,key)is None for key in required):parser.error('Every inner path is explicit argv')
        return worker(args)
    from .. import paths
    if args.out is None:parser.error('--out is required')
    defaults=dict(inputs=paths.WORK_RESULTS/'agent-cifar-offline-records-r1804/cifar10-records',
        selections=paths.WORK_RESULTS/'agent-resnet20-original-accuracy-r1805/frozen-selections.json',
        ledger=paths.RESULTS_DIR/'llm_calls.jsonl',allowance=paths.RESULTS_DIR/'autonomous-api-allowance.json',
        profile=paths.PROFILES_DIR/'qwen48/compiler-config.json',frontend=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py',
        metadata_template=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    for key,default in defaults.items():setattr(args,key,(getattr(args,key)or default).expanduser().resolve())
    args.out=args.out.expanduser().resolve()
    result=execute(args)
    return 0 if result['status']in('ok','prepared','quality_rejected')else 1


if __name__=='__main__':raise SystemExit(main())
