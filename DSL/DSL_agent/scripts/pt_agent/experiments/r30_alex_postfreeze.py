"""Independent placement/Emit/trace/archive/compile of accepted Alex continuation.

Only the existing trusted R30 worker implements numerical lowering and compiler
commands. The original failed invocation and its continuation remain untouched.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from types import FunctionType

from .. import paths
from . import r30_common_model_e2e as common
from . import r30_continue_alex_session as continuation
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from .r31_lattigo_runtime_plan import physical_memory_observation

ROUND='R30-ALEX-POSTFREEZE'
MODE='accepted-alex-continuation-trusted-postfreeze-compile'
MODEL=continuation.MODEL
FACTORY=continuation.FACTORY
MODULE='pt_agent.experiments.r30_alex_postfreeze'
PHASES=('placement','emit','trace','archive_emit_constants','compile')
BOUNDS=dict(placement=1800,emit=1800,trace=1800,compile=7200)
SECONDS=13560
GIB=1024**3
RESERVE=1024**2
ZERO=('api_calls','agent_runs','tuner_runs','new_intakes','calibration_runs','new_model_forwards',
      'confirm_samples','report_samples','keygen_runs','real_ckks_runs','extra_integrity_checks',
      'new_operator_identifier_digests','downloads','installations')


def read(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):raise ValueError('An object metadata record is required')
    return value


def same(left,right):
    return isinstance(left,(str,Path)) and Path(left).resolve()==Path(right).resolve()


def clean(observation):
    return (observation.get('exit_code')==0 and observation.get('timed_out')is False
        and observation.get('signal')is None and observation.get('signals_sent')==[]
        and observation.get('remaining_pids')==[])


def _regular(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size<=0:
        raise ValueError('A retained nonempty regular source is required: '+str(path))
    return str(path.resolve())


def minimum_free(args):
    value=getattr(args,'minimum_free_gib',10)
    if type(value)is not int or not 4<=value<=10:raise ValueError('Explicit disk reserve is an integer4 through10GiB')
    return value


def trusted_namespace(free_gib):
    """Adapt only two explicit storage guards in copied trusted functions."""
    minimum_free(type('Policy',(),{'minimum_free_gib':free_gib})())
    namespace=dict(vars(common));proof=dict(minimum_free_gib=free_gib,original_module_modified=False,
        storage_guard_replacements={},arithmetic_and_compiler_argv_changed=False)
    if free_gib==10:return namespace,proof
    tree=ast.parse(Path(common.__file__).read_text())
    namespace['_POSTFREEZE_FREE_GIB']=free_gib
    for name in ('_post_freeze','archive_emit_constants_after_trace'):
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name==name)
        replacements=[]
        class Storage(ast.NodeTransformer):
            def visit_BinOp(self,node):
                if (isinstance(node.op,ast.Mult)and isinstance(node.left,ast.Constant)and node.left.value==10
                        and isinstance(node.right,ast.Name)and node.right.id=='GIB'):
                    replacements.append(name)
                    return ast.copy_location(ast.BinOp(left=ast.Name(id='_POSTFREEZE_FREE_GIB',ctx=ast.Load()),
                        op=ast.Mult(),right=ast.Name(id='GIB',ctx=ast.Load())),node)
                return self.generic_visit(node)
            def visit_Constant(self,node):
                if isinstance(node.value,str)and'10 GiB'in node.value:
                    return ast.copy_location(ast.Constant(node.value.replace('10 GiB',str(free_gib)+' GiB')),node)
                return node
        function=Storage().visit(function)
        if replacements!=[name]:raise ValueError('Expected exactly one original disk reserve in '+name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),common.__file__,'exec'),namespace)
        proof['storage_guard_replacements'][name]=1
    original=common.worker
    delegated=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    delegated.__kwdefaults__=original.__kwdefaults__;namespace['worker']=delegated
    return namespace,proof


def frozen_source(case):
    """Read successful continuation metadata/source, with no payload or ledger IO."""
    from ..planners.transformer_session import load_state
    from .r13_common_models import FACTORIES
    from .r30_frozen_cnn_report import official_normalization
    case=Path(case).resolve();record=read(case/'result.json');marker=read(case/'completion.json')
    if (record.get('round')!=continuation.ROUND or record.get('mode')!=continuation.MODE
            or record.get('model')!=MODEL or record.get('status')!='ok'
            or record.get('error')is not None or record.get('monitor_error')is not None
            or record.get('guard_trigger')is not None or record.get('incoming_signal')is not None
            or record.get('worker_attempts')!=1 or record.get('worker_completed')!=1
            or not clean(record.get('process')or{}) or record.get('cold_start')is not False
            or marker.get('status')!='ok' or marker.get('registry_persisted')is not True
            or marker.get('required_artifacts_persisted')is not True
            or any(record.get(key)!=0 for key in ('compiler_runs','real_ckks_runs','keygen_runs','report_samples'))):
        raise ValueError('A clean successful independently closed Alex continuation is required')
    source=record.get('source')or{};original=Path(source.get('original_case','')).resolve()
    if original==case or not same(source.get('model_info'),original/'work/intake/model_info/model_info.json'):
        raise ValueError('Retain the original fresh model intake path')
    if not same(source.get('calibration'),original/'work/calibration/calibration.json'):
        raise ValueError('Retain the original training calibration path')
    historical=read(original/'result.json');old_marker=read(original/'completion.json')
    if (historical.get('round')!='R30' or historical.get('model')!=MODEL
            or historical.get('status')!='error' or historical.get('tuner_enabled')is not True
            or historical.get('variant')!=common.VARIANT
            or historical.get('attempted')!=dict(intake=1,calibration=1,agent=1)
            or historical.get('completed')!=dict(intake=1,calibration=1)
            or old_marker.get('status')!='error' or old_marker.get('registry_persisted')is not True
            or old_marker.get('required_artifacts_persisted')is not True):
        raise ValueError('Retain the closed original failed invocation and its one unfinished Agent stage')
    names=('_post_freeze','worker','diagnostic_compile_argv')
    captured=original/'source/r30_common_model_e2e.py'
    if continuation._functions(captured,names)!=continuation._functions(common.__file__,names):
        raise ValueError('The original trusted post-freeze worker algorithm changed')
    processes=historical.get('processes')or[]
    if ([row.get('phase')for row in processes]!=['intake','calibration','agent']
            or any(row.get('observation',{}).get('remaining_pids')!=[] for row in processes)
            or any(not clean(row.get('observation')or{})for row in processes[:2])):
        raise ValueError('Original intake/calibration must be clean and every source child closed')
    new_calls=record.get('actual_new_api_calls');old_calls=record.get('original_api_calls')
    history=record.get('provider_history')or{}
    if (type(old_calls)is not int or type(new_calls)is not int or not 1<=old_calls<=6
            or not 0<=new_calls<=6-old_calls or source.get('initial_calls')!=old_calls
            or source.get('remaining_calls')!=6-old_calls
            or history.get('inherited_successful_calls')!=old_calls or history.get('actual_new_calls')!=new_calls
            or history.get('complete_provider_conversation')is not True
            or history.get('old_ledger_records_recreated')is not False):
        raise ValueError('Retain the original and additional successful calls within the original six-call policy')
    agent=case/'work/agent';worker=read(agent/'worker-result.json');state=load_state(agent/'SESSION.json')
    if not continuation.same_state(worker,record.get('worker')) or continuation.check_worker(worker,state,source,new_calls,agent_dir=agent)!='ok':
        raise ValueError('The retained successful worker and completed SESSION differ')
    if state.get('numeric_lock',False)is not False or worker.get('max_api_calls')!=6:
        raise ValueError('Preserve the original numeric policy and six-call limit')
    if (len(list((agent/'api').glob('*.request.json')))!=new_calls
            or len(list((agent/'api').glob('*.response.json')))!=new_calls
            or list((agent/'api').glob('*.error.json'))):
        raise ValueError('The continuation must retain its own successful response metadata')
    info_path=Path(source['model_info']);info=read(info_path);identity=info.get('source')or{}
    expected=FACTORIES[FACTORY]
    if (identity.get('factory')!=FACTORY or identity.get('type')!='torch'
            or info.get('task')!='classification' or info.get('metric')!='classification_top1'
            or info.get('threshold')!={'acc_drop_pp':.5}
            or info.get('split',{}).get('kind')!='legacy_cifar'
            or info.get('split',{}).get('report')!=list(range(8000))):
        raise ValueError('Preserve the original Alex model and CIFAR task/report partition')
    for field,key in [('original_source','source'),('checkpoint','checkpoint'),
                      ('inference_preprocessing_source','inference_preprocessing_source')]:
        if not same(identity.get(field),expected[key]):raise ValueError('Original model source differs: '+field)
        _regular(expected[key])
    norm=official_normalization(expected['inference_preprocessing_source'])
    if identity.get('normalization')!=norm or identity.get('preprocessing_provenance')!='official inference test':
        raise ValueError('Preserve official inference normalization')
    calibration=read(original/'work/calibration/phase_result.json')
    provenance=read(original/'work/calibration/calibration-source.json')
    indices=provenance.get('indices')or[]
    if (source.get('calibration_samples')!=1000 or calibration.get('calibration_completed')is not True
            or calibration.get('calibration_samples')!=1000 or calibration.get('calibration_batch')!=1
            or provenance.get('split')!='train' or provenance.get('count')!=1000 or len(indices)!=1000
            or any(type(index)is not int or not 0<=index<50000 for index in indices)
            or len(set(indices))!=1000 or provenance.get('normalization')!=norm):
        raise ValueError('Retain the original1000-training calibration and its official normalization')
    intake=read(original/'work/intake/phase_result.json');maths=intake.get('graph_math')or{}
    if (intake.get('model')!=MODEL or intake.get('status')!='ok'
            or any(intake.get(key)is not True for key in ('import_completed','real_probes_passed',
                'graph_math_passed','roundtrip_passed','static_coverage'))
            or maths.get('passed')is not True or len(maths.get('samples')or[])!=3
            or any(row.get('passed')is not True or row.get('finite')is not True for row in maths['samples'])):
        raise ValueError('Retain all original three-probe intake mathematics')
    probes=original/'work/intake/training_probes.npz';probe_source=original/'work/intake/training_probe_source.json'
    provenance=read(probe_source)
    if (provenance.get('split')!='train' or provenance.get('indices')!=[0,1,2]
            or provenance.get('sample_ids')!=['train:0','train:1','train:2']):
        raise ValueError('Saved original training probes differ')
    _regular(probes)
    selected=worker['agent']['confirm']['selected'];program=Path(worker['selected_program']).resolve()
    if program!=agent/'program.ptdsl':raise ValueError('Use the continuation selected program at its canonical saved path')
    decorator=next(node for node in ast.parse(program.read_text()).body if isinstance(node,ast.FunctionDef)).decorator_list
    if len(decorator)!=1:raise ValueError('One original frozen program header required')
    original_source=historical.get('source')or{}
    records=Path(original_source.get('records','')).resolve()
    if records!=(paths.WORK_RESULTS/'agent-cifar-offline-records-r1804/cifar10-records').resolve():
        raise ValueError('Preserve the original CIFAR record source')
    profile=Path(original_source.get('compiler_profile',''));_regular(profile)
    if read(profile)!=read(original/'source/compiler-profile.json'):
        raise ValueError('Retain the original compiler profile JSON semantics')
    template=Path(source.get('metadata_template',''));_regular(template)
    if not same(template,original/'source/compile-metadata-template.json'):
        raise ValueError('Use the original saved operator metadata template')
    fronts={str(Path(row['argv'][row['argv'].index('--frontend')+1]).resolve())
            for row in processes if '--frontend'in row.get('argv',[])}
    if len(fronts)!=1:raise ValueError('Original source must record one consistent trusted frontend path')
    frontend=Path(next(iter(fronts)));_regular(frontend)
    return dict(source_case=str(case),original_case=str(original),model=MODEL,factory=FACTORY,
        model_info=str(info_path),calibration=source['calibration'],program=str(program),agent_dir=str(agent),
        metadata_template=str(template),profile=str(profile),frontend=str(frontend),records=str(records),
        normalization=norm,probes=str(probes),probe_source=str(probe_source),report_count=8000,
        selected_iteration=selected['iteration'],selected_origin=selected['origin'],
        formal_frozen=True,strict_math_passed=True,confirmation=worker['agent']['confirm']['evals'],
        original_api_calls=old_calls,continuation_api_calls=new_calls,total_api_calls=state['calls'],
        tuner=True,tuner_budget_s=600,confirm_rule='c1',confirmation_repeated=False,
        cold_start=False,original_complete_seconds=old_marker.get('complete_case_elapsed_seconds'),
        continuation_complete_seconds=marker.get('complete_case_elapsed_seconds'),
        diagnostic_elide_final_ir=historical.get('diagnostic_elide_final_ir',False),
        original_r13_math_checks_reused=3,source_program_ast_matches=True,reference_criterion='1e-12+1e-12*abs(reference)',
        original_sources={key:str(expected[key])for key in ('source','checkpoint','inference_preprocessing_source')})


def phase_command(args,gate,phase,work,remaining):
    seconds=min(BOUNDS[phase],int(remaining)-180)
    if seconds<=0:raise TimeoutError('Post-freeze total allowance exhausted')
    argv=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds+60),
        '--python',str(args.python),'--','-B','-m',MODULE,'--worker-phase',phase,'--work-dir',str(work),
        '--case-root',str(args.out),'--post-dir',str(args.out/'post-freeze'),'--model-info',gate['model_info'],
        '--program',gate['program'],'--profile',gate['profile'],'--frontend',gate['frontend'],
        '--metadata-template',gate['metadata_template'],'--memory-gib',str(args.memory_gib),
        '--case-gib',str(args.case_gib),'--minimum-free-gib',str(minimum_free(args)),
        '--worker-seconds',str(seconds),'--trace-seconds',str(min(seconds,BOUNDS['trace'])),
        '--compile-seconds',str(min(seconds,BOUNDS['compile']))]
    if gate['diagnostic_elide_final_ir']:argv+=['--diagnostic-elide-final-ir']
    return argv,seconds+120


def worker(args):
    if args.worker_phase not in BOUNDS:raise ValueError('Only post-freeze phases are accepted')
    args.model=MODEL;args.tuner=True
    namespace,proof=trusted_namespace(minimum_free(args));expired=[False]
    seconds=getattr(args,'worker_seconds',None)or BOUNDS[args.worker_phase]
    if type(seconds)is not int or not 0<seconds<=BOUNDS[args.worker_phase]:raise ValueError('Use the original bounded phase wall allowance')
    def timeout(signum,frame):
        expired[0]=True;raise TimeoutError('Post-freeze worker wall allowance exhausted')
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds)
    # The existing worker installs native_metadata_commands before numerical imports.
    try:return namespace['worker'](args)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        target=getattr(args,'work_dir',Path('/nonexistent'))/'phase_result.json'
        if target.is_file():
            record=read(target);record.update(storage_adaptation=proof,worker_wall_seconds=seconds,
                worker_wall_timeout_observed=expired[0]);write_json(target,record)


def _register(out,result):
    row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=None,
        metrics={key:result.get(key)for key in ('status','attempted','completed','compiled_bootstraps',
            'predicted_latency_us','incoming_signal','guard_trigger','source_case')},
        artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):
            raise ValueError('Preserve the existing post-freeze registry identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def _terminal(out,result,budget):
    for _ in range(12):
        write_json(out/'result.json',result);size=artifact_bytes(out)
        if result.get('artifact_bytes')==size:
            if size+RESERVE>budget:raise RuntimeError('Terminal metadata/completion reserve exceeds case allowance')
            return
        result['artifact_bytes']=size
    raise RuntimeError('Terminal metadata size did not stabilize')


def execute(args):
    free_gib=minimum_free(args)
    gate=frozen_source(args.source_case)
    out=Path(args.out).resolve();args.out=out
    if (out.exists()or out==paths.RESULTS_DIR.resolve()or not out.is_relative_to(paths.RESULTS_DIR.resolve())
            or any(out.is_relative_to(Path(gate[key]))for key in ('source_case','original_case'))):
        raise ValueError('Use a fresh isolated output under RESULTS')
    started=time.perf_counter_ns();out.mkdir(parents=True);(out/'work').mkdir();(out/'post-freeze').mkdir()
    result=dict(round=ROUND,mode=MODE,model=MODEL,source_case=gate['source_case'],source=gate,status='running',error=None,
        attempted={},completed={},phases={},processes=[],incoming_signal=None,guard_stop_signal=None,
        timeout_stop_signal=None,wall_timeout_observed=False,phase_timeouts={},wall_timeout_error=None,
        guard_trigger=None,monitor_error=None,cold_start=False,feedback_to_agent=False,
        elapsed_scope='This post-freeze placement through compilation only; original and continuation clocks separate',
        compiled_bootstraps=None,predicted_latency_us=None,CKKS_latency=None,accuracy_acceptance=None,
        security_acceptance=None,artifact_bytes=None,completion_reserve_bytes=RESERVE,
        artifact_bytes_scope='required files before completion marker; final marker includes itself',
        limits=dict(seconds=args.seconds,memory_AS_gib=args.memory_gib,subtree_RSS_gib=14,
            minimum_available_gib=1.25,case_gib=args.case_gib,minimum_disk_free_gib=free_gib),**dict.fromkeys(ZERO,0))
    recorder=StageRecorder(out/'events.jsonl');stop=threading.Event();active=[None];attached=[];old={};lock=None
    def check():
        if (time.perf_counter_ns()-started)/1e9>args.seconds:
            result['wall_timeout_observed']=True;raise TimeoutError('Post-freeze outer wall allowance exhausted')
        current=out/'work'/active[0]/'resources.jsonl'if active[0]else out/'none'
        physical=physical_memory_observation(_latest_rss_sample(current),_available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
        size=artifact_bytes(out);free=shutil.disk_usage(out).free
        if physical['stop']or size+RESERVE>args.case_gib*GIB or free<free_gib*GIB:
            result['guard_trigger']=dict(physical=physical,artifact_bytes=size,disk_free_bytes=free)
            raise RuntimeError('Post-freeze physical/disk/artifact allowance exhausted')
    def monitor():
        while not stop.wait(.5):
            try:check()
            except BaseException as exc:
                field='wall_timeout_error'if isinstance(exc,TimeoutError)else'monitor_error'
                result[field]=type(exc).__name__+': '+str(exc)
                try:write_json(out/'guard.json',dict(trigger=result['guard_trigger'],error=result[field],
                    wall_timeout_observed=result['wall_timeout_observed']))
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
    def interrupted(signum,frame):
        key='timeout_stop_signal'if result['wall_timeout_observed']else'guard_stop_signal'if result['guard_trigger']or result['monitor_error']else'incoming_signal'
        result[key]=signum;raise InterruptedError('Post-freeze outer received signal'+str(signum))
    watcher=threading.Thread(target=monitor,daemon=True)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):old[sig]=signal.signal(sig,interrupted)
        lock=(paths.RESULTS_DIR/'.r30-alex-postfreeze.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with recorder.span('independent_postfreeze_compile'):
            with recorder.span('source_snapshot'):
                check();write_json(out/'source-preflight.json',gate);snapshot=out/'source';snapshot.mkdir()
                for path in (Path(__file__),Path(common.__file__),Path(continuation.__file__),
                             Path(__file__).with_name('r29_local_compile_matrix.py'),
                             Path(__file__).with_name('r30_continue_alex_session_worker.py')):
                    shutil.copyfile(path,snapshot/path.name)
                shutil.copyfile(gate['profile'],snapshot/'compiler-profile.json')
                shutil.copyfile(gate['metadata_template'],snapshot/'compile-metadata-template.json')
            watcher.start()
            env={k:v for k,v in os.environ.items()if not k.startswith(('LD_','NIX_','__NIX_','HECATE_'))
                 and k not in ('HECATE','IN_NIX_SHELL','PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')}
            env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
            for phase in PHASES:
                check();result['attempted'][phase]=1;write_json(out/'progress.json',result);active[0]=phase
                with recorder.span(phase)as parent:
                    if phase=='archive_emit_constants':
                        namespace,proof=trusted_namespace(free_gib)
                        result['phases'][phase]=namespace['archive_emit_constants_after_trace'](result,out)
                        result['phases'][phase]['storage_adaptation']=proof
                    else:
                        work=out/'work'/phase;work.mkdir();(work/'tmp').mkdir()
                        argv,seconds=phase_command(args,gate,phase,work,args.seconds-(time.perf_counter_ns()-started)/1e9)
                        observation=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=work/'worker.log',
                            resources=work/'resources.jsonl',timeout=seconds,mem_gib=args.memory_gib,fsize_gib=8,
                            env=dict(env,TMPDIR=str(work/'tmp')))
                        result['processes'].append(dict(phase=phase,argv=argv,observation=observation))
                        if observation.get('signal')is not None:
                            key='timeout_stop_signal'if result['wall_timeout_observed']else'guard_stop_signal'if result['guard_trigger']or result['monitor_error']else'incoming_signal'
                            result[key]=observation['signal']
                        for event in read_events(work/'events.jsonl'):
                            if event.get('parent_id')is None:event['parent_id']=parent['event_id']
                            attached.append(event)
                        target=work/'phase_result.json';record=read(target)if target.is_file()else None
                        result['phases'][phase]=record
                        layers=[]
                        if observation.get('timed_out')is True:layers.append('outer-R10')
                        if (record or{}).get('worker_wall_timeout_observed')is True:layers.append('worker-wall')
                        if (record or{}).get('process',{}).get('timed_out')is True:layers.append('native-R10')
                        if layers:result['phase_timeouts'][phase]=layers
                        if not clean(observation)or not record or record.get('status')!='ok':
                            raise RuntimeError('Post-freeze phase did not complete cleanly: '+phase)
                        if phase in ('trace','compile')and not clean(record.get('process')or{}):
                            raise RuntimeError('Native child did not complete cleanly: '+phase)
                    result['completed'][phase]=1
                active[0]=None;write_json(out/'progress.json',result)
            compiled=result['phases']['compile'];planned=result['phases']['placement']
            if compiled['compiled_bootstraps']!=planned['boots_pred']:
                raise ValueError('Current compiled bootstraps differ from current frozen placement')
            result.update(status='ok',compiled_bootstraps=compiled['compiled_bootstraps'],
                          predicted_latency_us=compiled['predicted_latency_us'])
            check()
    except BaseException as exc:
        result.update(status='timeout'if isinstance(exc,TimeoutError)or result['wall_timeout_observed']or result['phase_timeouts']else
            'resource_stopped'if result['guard_trigger']or result['monitor_error']else
            'interrupted'if result['incoming_signal']else'error',error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=2)
        for sig,handler in old.items():signal.signal(sig,handler)
        if lock is not None:lock.close()
        if active[0]and active[0]!='archive_emit_constants':
            target=out/'work'/active[0]/'phase_result.json'
            if target.is_file():result['phases'][active[0]]=read(target)
        result['timing']=summarize_events(read_events(out/'events.jsonl')+attached,time.perf_counter_ns())
        try:_terminal(out,result,args.case_gib*GIB)
        except BaseException as exc:
            result.update(status='artifact_budget_exceeded',error=type(exc).__name__+': '+str(exc));write_json(out/'result.json',result)
        _register(out,result)
        marker=persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        if marker['case_artifact_bytes']>args.case_gib*GIB:raise RuntimeError('Actual terminal case exceeded its allowance')
    return result


def compiled_source(compile_case,source_case=None):
    case=Path(compile_case).resolve();record=read(case/'result.json');marker=read(case/'completion.json')
    source=record.get('source')or{}
    if source_case is not None and not same(source_case,source.get('source_case')):
        raise ValueError('The compiled case belongs to a different frozen continuation')
    gate=frozen_source(source['source_case'])
    if (record.get('round')!=ROUND or record.get('mode')!=MODE or record.get('model')!=MODEL
            or record.get('status')!='ok' or record.get('error')is not None or record.get('guard_trigger')is not None
            or record.get('monitor_error')is not None or not continuation.same_state(record.get('source'),gate)
            or marker.get('status')!='ok' or marker.get('registry_persisted')is not True
            or marker.get('required_artifacts_persisted')is not True
            or any(record.get('attempted',{}).get(p)!=1 or record.get('completed',{}).get(p)!=1 for p in PHASES)
            or any(record.get(key)!=0 for key in ZERO)):
        raise ValueError('A complete unchanged independently compiled continuation is required')
    for phase in BOUNDS:
        rows=[row for row in record.get('processes',[])if row.get('phase')==phase]
        child=record.get('phases',{}).get(phase)or{}
        if len(rows)!=1 or not clean(rows[0].get('observation')or{})or child.get('status')!='ok':
            raise ValueError('Clean complete post-freeze phase missing: '+phase)
        if phase in ('trace','compile')and not clean(child.get('process')or{}):raise ValueError('Native child failed')
    phases=record['phases'];compiled=phases['compile']
    if (compiled.get('compiled_bootstraps')!=phases['placement'].get('boots_pred')
            or compiled.get('compiled_bootstraps')!=record.get('compiled_bootstraps')):
        raise ValueError('Compiled bootstrap inventory differs from this frozen placement')
    mlir=case/'post-freeze/emit/candidate_trace.mlir';retained=mlir if mlir.is_file()else mlir.with_name(mlir.name+'.gz')
    _regular(retained)
    argv=json.loads((case/'work/compile/native-argv.json').read_text())
    if not isinstance(argv,list)or str(mlir)not in argv:raise ValueError('Compilation must use this emitted frozen MLIR')
    gate.update(compile_case=str(case),compiled_input_logical_path=str(mlir),compiled_input_retained_path=str(retained),
        compiled_input_retention='raw'if retained==mlir else'reversible-gzip',clean_native_compile=True,
        session_phase='done',quality_audit=str(case/'source-preflight.json'),current_script_ast_identity_required=False,
        post_freeze_complete_seconds=marker.get('complete_case_elapsed_seconds'))
    return gate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--python',type=Path,default=paths.INTAKE_PYTHON)
    parser.add_argument('--seconds',type=int,default=SECONDS);parser.add_argument('--memory-gib',type=int,default=12)
    parser.add_argument('--case-gib',type=int,default=32)
    parser.add_argument('--minimum-free-gib',type=int,default=10)
    parser.add_argument('--worker-seconds',type=int)
    parser.add_argument('--worker-phase',choices=tuple(BOUNDS))
    for key in ('work-dir','case-root','post-dir','model-info','program','profile','frontend','metadata-template'):
        parser.add_argument('--'+key,type=Path)
    parser.add_argument('--trace-seconds',type=int,default=1800);parser.add_argument('--compile-seconds',type=int,default=7200)
    parser.add_argument('--diagnostic-elide-final-ir',action='store_true')
    args=parser.parse_args(argv)
    if not(180<args.seconds<=SECONDS and 4<=args.memory_gib<=12 and 4<=args.case_gib<=32
           and 4<=args.minimum_free_gib<=10
           and 0<args.trace_seconds<=1800 and 0<args.compile_seconds<=7200):parser.error('Use the fixed finite original post-freeze resource range')
    if args.worker_phase:
        if any(getattr(args,key)is None for key in ('work_dir','case_root','post_dir','model_info','program','profile','frontend','metadata_template')):
            parser.error('Every inner path is explicit')
        return worker(args)
    if args.source_case is None or args.out is None:parser.error('--source-case and --out required')
    result=execute(args);print(json.dumps(dict(status=result['status'],out=str(args.out),error=result['error'])))
    return 0 if result['status']=='ok'else 1


if __name__=='__main__':raise SystemExit(main())
