"""Source-bound common CNN reference, ideal-plan proof and one native execution.

The saved original train:0 is the sole sample. Preparation evaluates the current
frozen program once; native execution uses the independently checked prepared
binary. Original modules, model weights and compiler plans stay intact.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from types import FunctionType, SimpleNamespace

from .. import paths
from . import r31_fresh_cnn_input as fresh
from . import r31_lattigo_runtime_plan as runtime
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r31_lattigo_binding import NativeBinding, map_precision_plan

ROUND='R31-COMMON-CNN-NATIVE'
MODE='source-bound-frozen-common-cnn-single-train0'
MODULE='pt_agent.experiments.r31_common_cnn_native'
MODELS=('resnet20_silu','vgg16_silu_avgpool','alexnet_silu_avgpool')
GIB=1024**3
CASE_BYTES=32*GIB
FREE_BYTES=10*GIB
RESERVE=1024**2
NATIVE_SECONDS=10800
ZERO_FIELDS=('api_calls','agent_feedback','report_samples','confirm_samples','heldout_samples',
    'downloads','installations','extra_integrity_checks','compiler_runs','go_builds')


def read(path):return json.loads(Path(path).read_text())


def same(a,b):
    return isinstance(a,(str,Path)) and isinstance(b,(str,Path)) and Path(a).resolve()==Path(b).resolve()


def binding():
    return NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,
        bootstrap_normalization_log2=3,auxiliary_prime_bits=61,c2s_bits=52)


def _clone(module,**updates):
    namespace=dict(vars(module));namespace.update(updates)
    for key,value in tuple(namespace.items()):
        if isinstance(value,FunctionType) and value.__module__==module.__name__:
            fn=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            fn.__kwdefaults__=value.__kwdefaults__;namespace[key]=fn
    return namespace


def accepted_source(source_case,model,compile_case=None,quality_audit_case=None):
    """Use the original accepted cold source or the audited VGG continuation."""
    if model not in MODELS:raise ValueError('Choose one original common CNN')
    if compile_case is None:
        if quality_audit_case is not None:raise ValueError('An audit belongs to an explicit VGG compile case')
        from .r30_frozen_cnn_report import source_gate
        return source_gate(source_case,model),Path(source_case).resolve()
    if model!='vgg16_silu_avgpool' or quality_audit_case is None:
        raise ValueError('An explicit continuation compile requires its independent VGG quality audit')
    from .r30_vgg_postfreeze import compiled_source
    return compiled_source(compile_case,source_case,quality_audit_case),Path(compile_case).resolve()


def metadata_gate(gate,plan_path):
    """Bind the original source, normalized saved probes and compiled plan."""
    info=read(gate['model_info']);identity=info.get('source')or{};probes=read(gate['probe_source'])
    expected_factory='pt_agent.experiments.r13_common_models:'+gate['model']
    if (gate['factory']!=expected_factory or identity.get('factory')!=expected_factory
            or identity.get('normalization')!=gate['normalization']
            or info.get('task')!='classification' or info.get('metric')!='classification_top1'
            or info.get('threshold')!={'acc_drop_pp':.5}):
        raise ValueError('Original common-model factory, normalization and task policy differ')
    for key,original in [('original_source','source'),('checkpoint','checkpoint'),
                         ('inference_preprocessing_source','inference_preprocessing_source')]:
        if not same(identity.get(key),gate['original_sources'][original]):
            raise ValueError('Original source identity differs: '+key)
    if (probes.get('split')!='train' or probes.get('indices')!=[0,1,2]
            or probes.get('sample_ids')!=['train:0','train:1','train:2']
            or probes.get('normalization')!=gate['normalization']
            or probes!=identity.get('calibration_source')):
        raise ValueError('Retain the current intake training-probe provenance')
    if len(info.get('inputs',[]))!=1 or info['inputs'][0].get('shape')!=[1,3,32,32] or info['inputs'][0].get('dtype')!='float64':
        raise ValueError('The original float64 CIFAR input signature is required')
    compiled=read(Path(gate['compile_case'])/'result.json');phase=compiled.get('phases',{}).get('compile')or{}
    if (compiled.get('status')!='ok' or compiled.get('model')!=gate['model'] or phase.get('status')!='ok'
            or not same(phase.get('runtime_plan'),plan_path)):
        raise ValueError('The accepted actual compiler output must identify this RuntimePlan')
    inventory=fresh.check_plan(read(plan_path),phase.get('compiled_bootstraps'))
    return dict(plan=inventory,probe_source=probes,normalization=gate['normalization'])


def archive_source(manifest_path,compile_case,bundle):
    """Return only the reviewed canonical restoration command and metadata."""
    if manifest_path is None:
        if not bundle.is_dir():raise ValueError('An archived bundle requires its closed recovery manifest')
        return None
    manifest_path=Path(manifest_path).resolve();record=read(manifest_path)
    rows=[row for row in record.get('files',[])if row.get('kind')=='runtime_bundle']
    closed=read(manifest_path.parent/'result.json')
    saved=[row for row in closed.get('files',[])if row.get('kind')=='runtime_bundle']
    if (not same(record.get('case_result'),Path(compile_case)/'result.json') or len(rows)!=1 or len(saved)!=1
            or not same(rows[0].get('path'),bundle) or not same(saved[0].get('source'),bundle)
            or closed.get('status')!='ok' or saved[0].get('status')!='compressed'):
        raise ValueError('The closed archive must belong to this actual compiled bundle')
    archive=bundle.with_name(bundle.name+'.tar.gz')
    command=['tar','--extract','--gzip','--file',str(archive),'--directory',str(bundle.parent),
        '--no-same-owner','--no-same-permissions']
    if saved[0].get('restore_argv')!=command or not same(saved[0].get('archive'),archive) or not archive.is_file():
        raise ValueError('Use only the retained reviewed canonical bundle restore argv')
    return dict(manifest=str(manifest_path),restore_argv=command,inventory=rows[0]['inventory'],
        raw_bytes=saved[0]['raw_bytes'],extra_integrity_checks=0,archive_contents_read=False)


def source_gate(args):
    gate,compiled=accepted_source(args.source_case,args.model,args.compile_case,args.quality_audit_case)
    gate=dict(gate,compile_case=str(compiled))
    result=read(compiled/'result.json');plan=Path(result['phases']['compile']['runtime_plan']).resolve()
    canonical=compiled/'post-freeze/emit/lowered._hecate_golden.runtime-plan.json'
    if plan!=canonical:raise ValueError('Use the canonical current post-freeze compiler RuntimePlan')
    bundle=plan.with_name(plan.name.removesuffix('.runtime-plan.json')+'.bundle')
    template=Path(gate.get('metadata_template')or(Path(gate['source_case'])/'source/compile-metadata-template.json')).resolve()
    if not isinstance(read(template),list):raise ValueError('Use the original saved compiler argv template')
    metadata=metadata_gate(gate,plan)
    return dict(gate=gate,runtime_plan=str(plan),bundle=str(bundle),source_metadata=metadata,
        archive=archive_source(args.archive_manifest,compiled,bundle),metadata_template=str(template),
        layout_proof=None,sample_id='train:0',source_split='train',old_expected_values_used=False,
        feedback_to_agent=False,normalization_reapplied=False)


def _fresh_namespace(model):
    return _clone(fresh,MODEL=model,_metadata_gate=metadata_gate)


def input_worker(args):
    source=read(args.source_record);gate=source['gate']
    args.runtime_plan=Path(source['runtime_plan']);args.layout_proof=None
    for key in ('model_info','program','probes','probe_source'):setattr(args,key,Path(gate[key]))
    from .r29_local_compile_matrix import native_metadata_commands
    with native_metadata_commands(read(source['metadata_template'])):
        return _fresh_namespace(gate['model'])['worker'](args)


def math_worker(args):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    source=read(args.source_record);args.work_dir.mkdir(exist_ok=False)
    rec=StageRecorder(args.work_dir/'events.jsonl');result=dict(status='running',error=None,
        replay_attempts=0,replay_completed=0,math_passed=False,training_samples=1,model_forwards=0,
        bootstrap_observations=[],op_completed=None,op_counts=None)
    began=time.monotonic()
    def checkpoint():
        if time.monotonic()-began>900:raise TimeoutError('Original-plan math wall allowance exhausted')
        write_json(args.work_dir/'worker-result.json',result)
    try:
        import numpy as np
        from .r31_lattigo_math_worker import replay_plan,check_math
        plan=read(source['runtime_plan']);manifest=read(Path(source['bundle'])/'manifest.json')
        inventory=runtime.plan_inventory(plan,manifest)
        if inventory['op_counts']!=source['source_metadata']['plan']['op_counts']:
            raise ValueError('Current actual original-plan operation inventory differs')
        inp=args.input_prepare/'input.bin';expected=read(args.input_prepare/'expected.json')
        if expected.get('indices')!=list(range(10)) or expected.get('sample_id')!='train:0' or expected.get('old_expected_values_used') is not False:
            raise ValueError('Require the independently prepared current train:0 output layout')
        packed=np.fromfile(inp,dtype='<f8')
        if packed.shape!=(16384,) or not np.isfinite(packed).all():raise ValueError('Invalid original packed input')
        def observe(row):result['bootstrap_observations'].append(dict(row));checkpoint()
        result['replay_attempts']=1;checkpoint()
        with rec.span('current_original43_last_use_replay'):
            output,summary=replay_plan(plan,manifest,Path(source['bundle']),packed,bootstrap_observer=observe,checkpoint=checkpoint)
        summary['reference_scope']='Current frozen '+source['gate']['model']+'; train:0; ideal metadata operations only'
        result.update(replay_completed=1,op_completed=summary['execution_completed'],op_counts=summary['op_counts'],replay_summary=summary)
        if (summary['execution_completed']!=inventory['execution_count'] or len(summary['bootstrap_inputs'])!=inventory['bootstrap_count']):
            raise ValueError('Every actual current original-plan operation must complete')
        observed=output[expected['indices']][None,:];reference=np.asarray(expected['values'],dtype=np.float64)[None,:]
        compared=check_math(reference,observed);result.update(comparison=compared,decoded_values=observed.tolist(),math_passed=compared['passed'])
        if not compared['passed']:raise ValueError('Original compiled plan and current frozen Tensor reference differ')
        proof=dict(status='strict_math_passed',source_plan=source['runtime_plan'],current_program=source['gate']['program'],
            input=str(inp),reference=str(args.input_prepare/'expected.json'),source_factory=source['gate']['factory'],
            source_case=source['gate']['source_case'],compile_case=source['gate']['compile_case'],sample_ids=['train:0'],
            operation_count=inventory['execution_count'],ideal_bootstrap_count=inventory['bootstrap_count'],
            criterion=fresh.CRITERION,original_plan_modified=False,precision_mapping_applied=False)
        write_json(args.work_dir/'source-math-proof.json',proof);result.update(status='ok',source_math_proof=str(args.work_dir/'source-math-proof.json'))
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:write_json(args.work_dir/'worker-result.json',result)
    return result


def map_plan(source,work):
    original=read(source['runtime_plan']);manifest=read(Path(source['bundle'])/'manifest.json')
    inventory=runtime.plan_inventory(original,manifest)
    mapped,report=map_precision_plan(original,binding(),bootstrap_cache=32)
    checked=runtime.plan_inventory(mapped,manifest,binding=binding())
    if inventory['op_counts']!=checked['op_counts']:raise ValueError('Explicit binding changed original operations')
    target=Path(work)/'derived-plan.json';write_json(target,mapped)
    return dict(plan=str(target),report=report,inventory=checked,variant=binding().variant(32))


def component_prediction(source,profile_prediction):
    from .native_component_matrix import inventory,project,cell_key
    calibration=read(profile_prediction)
    if calibration.get('variant')!=binding().variant(32) or len(calibration.get('all39_measured_cells',[]))!=39:
        raise ValueError('Use the independent actual selected-group39-cell component profile')
    lookup={cell_key(row['cell']):row for row in calibration['all39_measured_cells']}
    try:
        counted=inventory(Path(source['runtime_plan']),lookup)
    except ValueError as exc:
        if 'does not cover cells' not in str(exc):raise
        return dict(status='missing_cells',engineering_component_point_seconds=None,missing=str(exc),
            formal_complete_prediction_seconds=None,profile_source=str(Path(profile_prediction).resolve()))
    return dict(status='covered_component_point',profile_source=str(Path(profile_prediction).resolve()),
        source_inventory=counted,**project(counted,None,calibration))


def native_command(prepared,prepared_case,source,preparation,out):
    from .r31_lattigo_profile39 import validate_prepared
    checked=validate_prepared(prepared,c2s_bits=52)
    if prepared.get('supports_input_ssa_check') is not True or prepared.get('supports_key_request_trace') is not True:
        raise ValueError('The actual prepared binary must support SSA and public key-request observations')
    binary=Path(checked['binary']).resolve();compiled_source=Path(prepared_case).resolve()/'client.go'
    if binary!=Path(prepared_case).resolve()/'work/r31-client' or not os.access(binary,os.X_OK) or not compiled_source.is_file():
        raise ValueError('Require the actual prepared binary and its compiled source snapshot')
    command=[str(binary),'-mode','execute','-plan',preparation['mapping']['plan'],'-bundle',source['bundle'],
        '-input',str(Path(preparation['out'])/'input-worker/input.bin'),'-expected',str(Path(preparation['out'])/'input-worker/expected.json'),
        '-allow-bootstrap','-bootstrap-cache','32','-precision-bits','50','-mod1-inv-degree','15',
        '-log-message-ratio','2','-bootstrap-normalization-log2','3','-auxiliary-prime-bits','61','-c2s-bits','52',
        '-check-input-ssa','-trace-key-requests']+runtime.checkpoint_arguments(prepared,1000)
    return command+['-output',str(out/'client.json'),'-events',str(out/'go-events.jsonl')]


def check_client(client,plan,expected,prepared):
    from .fresh_resnet_native_audit import recompute_ckks
    counts={}
    for row in plan['execution']:counts[row['op']]=counts.get(row['op'],0)+1
    ssa=client.get('input_ssa_observations')or[];boots=counts.get('boot',0)
    if (client.get('status')!='ok' or client.get('encrypted_plan_completed') is not True
            or any(client.get(key)!=len(plan['execution']) for key in ('compute_attempts','compute_completed'))
            or client.get('op_completed_by_kind')!=counts or client.get('op_attempts_by_kind')!=counts
            or client.get('keygen_api_attempts')!=client.get('keygen_api_completed')
            or any(client.get(key)!=boots for key in ('bootstrap_attempts','bootstrap_completed','native_bootstrap_api_attempts','native_bootstrap_api_completed'))
            or any(client.get(key)!=1 for key in ('encryption_attempts','encryption_completed','final_decryption_attempts','final_decryption_completed'))
            or client.get('intermediate_decryptions')!=0 or client.get('get_log_enabled') is not False
            or client.get('input_ssa_check_enabled') is not True or len(ssa)!=len(plan['execution'])
            or any(row.get('metadata_preserved') is not True or (row.get('ordinal'),row.get('op'),row.get('inputs'))!=(op['ordinal'],op['op'],op['inputs'])for row,op in zip(ssa,plan['execution']))):
        raise ValueError('Actual whole-plan counts, ordered SSA or final-only decryption contract differs')
    prior=prepared['client']
    for key in ('variant','precision_bits','c2s_bits','auxiliary_prime_bits','bootstrap_cache_capacity',
                'bootstrap_normalization_log2','bootstrap_input_divisor','residual_parameters','bootstrap_parameter_summary','bootstrap_precision_policy'):
        if client.get(key)!=prior.get(key):raise ValueError('Actual prepared public parameter metadata differs: '+key)
    observations=client.get('bootstrap_observations')or[]
    if (len(observations)!=boots or client.get('bootstrap_input_value_divisions')!=boots or client.get('bootstrap_output_integer_restores')!=boots
            or any(row.get('native_completed')is not True or row.get('input_ssa_preserved')is not True
                or row.get('input_value_divisor')!=8 or row.get('output_integer_multiplier')!=8
                or row.get('restored_output_level')!=5 for row in observations)):
        raise ValueError('Every actual native bootstrap must preserve input SSA and restore physical value')
    compared=recompute_ckks(client['final_selected_values'],expected['values'])
    if (not compared['passed'] or client.get('comparison',{}).get('passed') is not True
            or client['comparison'].get('absolute_tolerance')!=1e-5 or client['comparison'].get('relative_tolerance')!=1e-4):
        raise ValueError('Final original-unit CKKS output acceptance failed')
    return compared


def check_budget(out):
    if artifact_bytes(out)+RESERVE>CASE_BYTES or shutil.disk_usage(out).free<FREE_BYTES:
        raise RuntimeError('Common CNN artifact/free-disk allowance exhausted')


@contextmanager
def guard(out,result,active,*,native=False):
    stop=threading.Event();old={sig:signal.getsignal(sig)for sig in (signal.SIGTERM,signal.SIGINT)}
    def incoming(sig,frame):result['incoming_signal']=sig;raise RuntimeError('Incoming signal '+str(sig))
    for sig in old:signal.signal(sig,incoming)
    def watch():
        while not stop.wait(.25):
            try:
                check_budget(out)
                if native and active[0] is not None:
                    row=runtime.physical_memory_observation(runtime._latest_rss_sample(active[0]),runtime._available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
                    if row['stop']:result['physical_guard']=row;raise RuntimeError('Physical memory reserve exhausted')
            except BaseException as exc:
                result['guard_error']=type(exc).__name__+': '+str(exc)
                try:write_json(out/'guard-observation.json',dict(error=result['guard_error'],physical=result.get('physical_guard')))
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:yield
    finally:
        stop.set();thread.join(timeout=2)
        for sig,handler in old.items():signal.signal(sig,handler)


def _launch(result,recorder,out,name,argv,seconds,memory,active,env=None):
    check_budget(out);record=dict(stage=name,argv=[str(x)for x in argv]);result['processes'].append(record)
    active[0]=out/(name+'-resources.jsonl')
    try:
        with recorder.span(name):
            record['observation']=run_process(record['argv'],cwd=paths.REPO/'DSL/DSL_agent/scripts',
                log=out/(name+'.log'),resources=active[0],timeout=seconds,mem_gib=memory,fsize_gib=8,env=env)
    finally:active[0]=None
    obs=record['observation']
    if obs.get('exit_code')!=0 or obs.get('timed_out')or obs.get('signal')is not None or obs.get('remaining_pids'):
        raise RuntimeError('Bounded phase failed: '+name)
    check_budget(out)


def _terminal(out,result,recorder,started,revision):
    result['timing']=summarize_events(read_events(out/'events.jsonl'))
    result['recorded_pre_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
    result['metrics']={key:result.get(key)for key in ('status','model','phase','comparison','incoming_signal','guard_error','physical_guard')}
    result['metrics'].update(security_acceptance=None,formal_task_accuracy=None,formal_latency=None)
    result['artifact_bytes']=0
    for _ in range(20):
        write_json(out/'result.json',result);size=artifact_bytes(out)
        if size+RESERVE>CASE_BYTES:
            result.update(status='error',error='Terminal artifacts exceeded the reserved case allowance');result['metrics']['status']='error'
        if size==result['artifact_bytes']:break
        result['artifact_bytes']=size
    else:raise RuntimeError('Terminal metadata size did not converge')
    row=dict(run_id=out.name,round=ROUND,model=result['model'],mode=MODE,git_rev=revision,
        metrics=result['metrics'],artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):raise ValueError('Preserve the existing registry run identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
    if artifact_bytes(out)>CASE_BYTES:raise RuntimeError('Final case including completion exceeded allowance')


def prepare(args):
    out=args.out.resolve();source=source_gate(args)
    for protected in (Path(source['gate']['source_case']),Path(source['gate']['compile_case'])):
        if out==paths.RESULTS_DIR.resolve()or not out.is_relative_to(paths.RESULTS_DIR.resolve())or out.is_relative_to(protected)or protected.is_relative_to(out):
            raise ValueError('Use a separate fresh configured results case')
    out.mkdir(exist_ok=False);started=time.perf_counter_ns();rec=StageRecorder(out/'events.jsonl')
    from ..registry import git_rev
    revision=git_rev();result=dict(round=ROUND,mode=MODE,phase='prepare',model=args.model,out=str(out),
        status='running',error=None,source=source,processes=[],security_acceptance=None,formal_latency=None,
        limits=dict(input_worker_seconds=480,input_nix_seconds=540,input_outer_seconds=570,input_as_gib=4,
            math_seconds=900,math_outer_seconds=930,math_as_gib=4,case_bytes=CASE_BYTES,free_bytes=FREE_BYTES))
    result.update(dict.fromkeys(ZERO_FIELDS,0));active=[None]
    write_json(out/'source-record.json',source)
    for module in (sys.modules[__name__],fresh,runtime):shutil.copyfile(module.__file__,out/Path(module.__file__).name)
    try:
        with rec.span('source_bound_common_cnn_preparation'),guard(out,result,active):
            if not Path(source['bundle']).is_dir():
                if not args.restore_bundle or source['archive']is None:raise ValueError('Restore the reviewed canonical bundle explicitly before plan replay')
                _launch(result,rec,out,'restore-bundle',source['archive']['restore_argv'],600,2,active)
            manifest=read(Path(source['bundle'])/'manifest.json')
            if source['archive']:
                from .dacapo_native_plan_math import check_restored_bundle
                result['restored_bundle']=check_restored_bundle(dict(manifest=manifest),Path(source['bundle']),source['archive']['manifest'])
            command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','540','--python',str(args.intake_python),
                '--','-B','-m',MODULE,'input-worker','--source-record',str(out/'source-record.json'),'--work-dir',str(out/'input-worker')]
            _launch(result,rec,out,'input-reference',command,570,4,active)
            result['input_worker']=read(out/'input-worker/worker-result.json')
            if not _fresh_namespace(args.model)['_success'](result['input_worker']):raise ValueError('Current frozen Tensor reference did not complete')
            fresh.check_outputs(result['input_worker'],out/'input-worker')
            command=[str(args.numpy_python),'-B','-m',MODULE,'math-worker','--source-record',str(out/'source-record.json'),
                '--work-dir',str(out/'math-worker'),'--input-prepare',str(out/'input-worker')]
            _launch(result,rec,out,'original-plan-math',command,930,4,active)
            result['math_worker']=read(out/'math-worker/worker-result.json')
            if result['math_worker'].get('status')!='ok'or result['math_worker'].get('math_passed')is not True:
                raise ValueError('Original plan mathematical acceptance failed')
            result['mapping']=map_plan(source,out)
            result['prediction']=component_prediction(source,args.profile_prediction)
            write_json(out/'prediction-before-native.json',result['prediction'])
            result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        for key in ('input_worker','math_worker'):
            path=out/key.replace('_','-')/'worker-result.json'
            if path.is_file():result[key]=read(path)
        _terminal(out,result,rec,started,revision)
    return result


def execution_source(preparation_case,prepared_case):
    directory=Path(preparation_case).resolve();record=read(directory/'result.json');completion=read(directory/'completion.json')
    if (record.get('round')!=ROUND or record.get('mode')!=MODE or record.get('phase')!='prepare'or record.get('status')!='ok'
            or completion.get('status')!='ok'or completion.get('registry_persisted')is not True
            or completion.get('required_artifacts_persisted')is not True or record.get('out')!=str(directory)):
        raise ValueError('Require a closed accepted common-model input/math/mapping preparation')
    source=record['source'];gate=source['gate']
    current=source_gate(SimpleNamespace(source_case=Path(gate['source_case']),model=gate['model'],
        compile_case=Path(gate['compile_case'])if gate.get('quality_audit_case')else None,
        quality_audit_case=Path(gate['quality_audit_case'])if gate.get('quality_audit_case')else None,
        archive_manifest=Path(source['archive']['manifest'])if source['archive']else None))
    if current!=source:raise ValueError('Accepted current model/program/plan source changed after preparation')
    if not _fresh_namespace(gate['model'])['_success'](record['input_worker']):raise ValueError('Prepared reference counter contract differs')
    fresh.check_outputs(record['input_worker'],directory/'input-worker')
    proof=read(directory/'math-worker/source-math-proof.json');inventory=source['source_metadata']['plan']
    math_result=record.get('math_worker')or{}
    if (math_result.get('status')!='ok'or math_result.get('math_passed')is not True
            or math_result.get('replay_attempts')!=1 or math_result.get('replay_completed')!=1
            or math_result.get('op_completed')!=inventory['execution_count']or math_result.get('op_counts')!=inventory['op_counts']):
        raise ValueError('Require the complete independently accepted original-plan replay')
    if (proof.get('status')!='strict_math_passed'or not same(proof.get('source_plan'),source['runtime_plan'])
            or not same(proof.get('current_program'),gate['program'])or not same(proof.get('reference'),directory/'input-worker/expected.json')
            or not same(proof.get('input'),directory/'input-worker/input.bin')or not same(proof.get('source_case'),gate['source_case'])
            or not same(proof.get('compile_case'),gate['compile_case'])or proof.get('precision_mapping_applied')is not False
            or proof.get('source_factory')!=gate['factory']or proof.get('operation_count')!=inventory['execution_count']
            or proof.get('ideal_bootstrap_count')!=inventory['bootstrap_count']or proof.get('criterion')!=fresh.CRITERION
            or proof.get('sample_ids')!=['train:0']or proof.get('original_plan_modified')is not False):
        raise ValueError('The strict ideal mathematical proof must bind this current source and Tensor reference')
    mapped,report=map_precision_plan(read(source['runtime_plan']),binding(),bootstrap_cache=32)
    if not same(record['mapping']['plan'],directory/'derived-plan.json')or read(record['mapping']['plan'])!=mapped or record['mapping']['report']!=report:
        raise ValueError('The mapped plan must preserve this exact original program and parameter binding')
    if read(directory/'prediction-before-native.json')!=record['prediction']:raise ValueError('Preserve the independent prediction frozen before native execution')
    prior=read(Path(prepared_case)/'result.json')
    return record,prior


def execute(args):
    record,prior=execution_source(args.preparation_case,args.prepared);source=record['source'];out=args.out.resolve()
    if out.parent!=paths.RESULTS_DIR.resolve()or out.exists():raise ValueError('Use a fresh direct configured results child')
    command=native_command(prior,args.prepared,source,record,out)
    out.mkdir();started=time.perf_counter_ns();rec=StageRecorder(out/'events.jsonl');active=[None]
    from ..registry import git_rev
    revision=git_rev();result=dict(round=ROUND,mode=MODE,phase='execute',model=record['model'],status='running',error=None,
        source=source,preparation_case=str(args.preparation_case.resolve()),prepared=str(args.prepared.resolve()),processes=[],
        limits=dict(native_seconds=NATIVE_SECONDS,address_space_gib=24,subtree_rss_gib=14,minimum_available_gib=1.25,
            GOMEMLIMIT='11GiB',case_bytes=CASE_BYTES,free_bytes=FREE_BYTES,checkpoint_interval=1000),
        training_samples=1,security_acceptance=None,formal_task_accuracy=None,formal_latency=None)
    result.update(dict.fromkeys(ZERO_FIELDS,0))
    shutil.copyfile(Path(__file__),out/'entry-source.py');shutil.copyfile(args.prepared/'client.go',out/'compiled-client-source.go')
    write_json(out/'source-record.json',source);write_json(out/'prediction-before-native.json',record['prediction'])
    env={key:value for key,value in os.environ.items()if not key.startswith(('GO','LD_','NIX_','__NIX_'))}
    telemetry=out/'telemetry';telemetry.mkdir();(telemetry/'mode').write_text('off\n')
    env.update(GOMAXPROCS='1',GOMEMLIMIT='11GiB',TEST_TELEMETRY_DIR=str(telemetry),GOENV='off',GOTOOLCHAIN='local',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        with rec.span('single_current_frozen_plan_native'),guard(out,result,active,native=True):
            _launch(result,rec,out,'native',command,NATIVE_SECONDS,24,active,env)
            result['client']=read(out/'client.json')
            result['comparison']=check_client(result['client'],read(record['mapping']['plan']),read(Path(record['out'])/'input-worker/expected.json'),prior)
            result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        if (out/'client.json').is_file():result['client']=read(out/'client.json')
        _terminal(out,result,rec,started,revision)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='phase',required=True)
    prep=sub.add_parser('prepare')
    for key in ('source-case','out','profile-prediction'):prep.add_argument('--'+key,type=Path,required=True)
    prep.add_argument('--model',choices=MODELS,required=True)
    for key in ('compile-case','quality-audit-case','archive-manifest'):prep.add_argument('--'+key,type=Path)
    prep.add_argument('--restore-bundle',action='store_true');prep.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    prep.add_argument('--numpy-python',type=Path,default=paths.ORBIT_ROOT.parent/'orbit-venv/bin/python')
    native=sub.add_parser('execute')
    for key in ('preparation-case','prepared','out'):native.add_argument('--'+key,type=Path,required=True)
    for phase in ('input-worker','math-worker'):
        worker=sub.add_parser(phase)
        for key in ('source-record','work-dir'):worker.add_argument('--'+key,type=Path,required=True)
        if phase=='math-worker':worker.add_argument('--input-prepare',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.phase in ('input-worker','math-worker'):
        value=input_worker(args)if args.phase=='input-worker'else math_worker(args)
    else:value=prepare(args)if args.phase=='prepare'else execute(args)
    print(json.dumps(dict(status=value['status'],error=value.get('error')),allow_nan=False))
    return int(value['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
