"""Serial existing HEIR export/math/compile/Go/clear stages for one saved CNN.

Default preparation reads metadata and writes commands only. --run invokes the
existing complete phase drivers in sequence. A failed phase closes this case;
there is no retry, key generation, ciphertext execution or report phase.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_common_models import read_graph_contract, checked_saved_frontend_identity, checked_export_contract, checked_math_record
from .heir_cnn_math_check import validate_memory_gib
from .heir_initializer_probe import (durable_json,physical_observation,serial_lock,
    interruption_status,receive_signal,observe_termination)
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample

ROUND='HEIR-COMMON-PIPELINE'
MODE='serial-saved-common-model-native-compile-and-clear-math'
MODELS={'vgg16_silu_avgpool':'vgg16-silu-avgpool','alexnet_silu_avgpool':'alexnet-silu-avgpool'}
PHASE_SECONDS=dict(export=900,adapter_math=600,native_compile=7560,go_codegen=1200,clear_math=3000)
TOTAL_SECONDS=sum(PHASE_SECONDS.values())+600
GIB=1024**3
ZERO_FIELDS=('llm_calls','keygen_runs','encryptions','decryptions','real_ckks_attempts','real_ckks_completed',
             'heldout_samples','confirm_samples','report_samples','downloads','installations')


def read(path):
    result=json.loads(Path(path).read_text())
    if not isinstance(result,dict):raise ValueError('Object source metadata required')
    return result


def sources(model):
    intake=paths.RESULTS_DIR/('r13-common-'+MODELS[model]+'-20261115')
    frontend=paths.RESULTS_DIR/'r17a-margin-heldout-20261115'/model
    native=paths.RESULTS_DIR/'heir-resnet20-exact-selector-fixed-pool-native-20261115'
    return dict(intake_result=intake/'result.json',graph_dir=intake/'worker/model_info/graph',
        program=frontend/'worker/program.ptdsl',frontend_result=frontend/'result.json',
        probes=intake/'worker/training_probes.npz',probe_source=intake/'worker/training_probe_source.json',
        pipeline=native/'pass-pipeline.txt',native_parameters=native/'native-parameters.json')


def source_gate(model,source):
    original=read(source['intake_result'])
    if original.get('status')!='ok' or original.get('model')!='pt_agent.experiments.r13_common_models:'+model:
        raise ValueError('Original accepted R13 model differs')
    checked_saved_frontend_identity(read(source['frontend_result']),source['program'],model)
    contract=read_graph_contract(source['graph_dir'],model)
    probe=read(source['probe_source'])
    if (probe.get('split')!='train' or probe.get('indices')!=[0,1,2]
            or probe.get('sample_ids')!=['train:0','train:1','train:2'] or not source['probes'].is_file()):
        raise ValueError('Original three saved R13 training inputs required')
    pipeline=source['pipeline'].read_text().strip()
    prefix='--pass-pipeline='
    if not pipeline.startswith(prefix):raise ValueError('Accepted native pass-pipeline source required')
    pipeline=pipeline[len(prefix):]
    parameter_paths=list(dict.fromkeys(re.findall(r'parameter-file=([^\s}]+)',pipeline)))
    if (not parameter_paths or any(Path(name).resolve()!=source['native_parameters'].resolve() for name in parameter_paths)
            or 'layout-optimization{' in pipeline or 'secret-insert-mgmt-ckks{after-mul=true ' not in pipeline):
        raise ValueError('Original exact native compatibility pipeline differs')
    parameters=read(source['native_parameters'])
    if (parameters.get('logN')!=16 or len(parameters.get('Q',[]))!=11 or len(parameters.get('P',[]))!=3
            or parameters.get('logDefaultScale')!=45):raise ValueError('Original N16/Q11/P3/scale45 contract required')
    return dict(paths={key:str(value.resolve()) for key,value in source.items()},contract=contract,
        native_parameters=parameters,pipeline=pipeline,source_samples=['train:0','train:1','train:2'],
        weights_or_probe_payloads_read_by_source_gate=False,quality_selection=False)


def phase_directories(out):
    return {stage:Path(out)/'phases'/(Path(out).name+'-'+stage) for stage in PHASE_SECONDS}


def commands(args,source,directories,pipeline):
    adapter_memory_gib=validate_memory_gib(getattr(args,'adapter_memory_gib',2))
    common=['--common-model',args.model]
    exact=['--exact-spatial-mean-matvec'] if args.model=='vgg16_silu_avgpool' else []
    source_args=['--graph-dir',str(source['graph_dir']),'--program',str(source['program'])]
    probes=['--probes',str(source['probes']),'--probe-source',str(source['probe_source'])]
    module=lambda name:[sys.executable,'-B','-m','pt_agent.experiments.'+name]
    native=directories['native_compile'];export=directories['export'];adapter=directories['adapter_math'];go=directories['go_codegen']
    return dict(
        export=[str(args.export_python),'-B','-m','pt_agent.experiments.heir_cnn_export',*source_args,
            '--output',str(export),*common,*exact],
        adapter_math=module('heir_cnn_math_check')+['--supervise',*source_args,'--mlir',str(export/'model.mlir'),
            *probes,'--out',str(adapter),'--memory-gib',str(adapter_memory_gib),*common,*exact],
        native_compile=module('heir_cnn_preflight')+['--source',str(export/'model.mlir'),'--out',str(native),
            *common,'--explicit-pipeline',str(pipeline),'--heir-opt',str(args.heir_opt),
            '--pipeline','torch-linalg-to-ckks','--slots','16384','--management','greedy',
            '--compile-seconds','7200','--memory-gib','12'],
        go_codegen=module('heir_cnn_codegen')+['--input',str(native/'model-lattigo.mlir'),
            '--source-result',str(native/'result.json'),'--out',str(go),'--translate',str(args.translate),
            '--build','--go',str(args.go),'--cache-source',str(args.cache_source),'--memory-gib','4','--build-seconds','600'],
        clear_math=module('heir_native_clear_math')+['--native-result',str(native/'result.json'),
            '--go-result',str(go/'result.json'),'--adapter-result',str(adapter/'result.json'),*probes,
            '--go',str(args.go),'--out',str(directories['clear_math']),*common,'--execute',
            '--memory-gib','12','--rss-limit-gib','6','--execute-seconds','1800'])


def check_phase(stage,directories,source):
    directory=directories[stage];contract=source['contract'];model=contract['model']
    metadata=read(directories['export']/'export.json')
    checked_export_contract(metadata,contract)
    if metadata.get('exact_spatial_mean_matvec')!=(model=='vgg16_silu_avgpool'):
        raise ValueError('Original spatial mean representation differs')
    if stage=='export':
        if not (directory/'model.mlir').is_file():raise ValueError('Trusted export source missing')
        return dict(status='ok',metadata=str(directory/'export.json'),source_model=model)
    record=read(directory/'result.json')
    if record.get('status')!='ok' or record.get('model')!=model:raise ValueError('Completed phase/model differs')
    if any(record.get(name,0)!=0 for name in ZERO_FIELDS):raise ValueError('Phase accessed an excluded source or backend')
    if stage=='adapter_math':
        references=checked_math_record(record,contract,metadata)
        return dict(status='ok',record=str(directory/'result.json'),strict_probe_count=len(references))
    if stage=='native_compile':
        if record.get('model_compiled') is not True or record.get('common_model_contract')!=contract:
            raise ValueError('Native full original model compile contract differs')
        artifact=read(directory/'model-native-artifact.json')
    elif stage=='go_codegen':
        if (record.get('library_build_passed') is not True
                or Path(record.get('source_compiler_result','')).resolve()!=(directories['native_compile']/'result.json').resolve()
                or Path(record.get('input_mlir','')).resolve()!=(directories['native_compile']/'model-lattigo.mlir').resolve()):
            raise ValueError('Go build must use this case native compiler output')
        artifact=record['native_artifact']
    else:
        worker=record.get('worker') or {}
        if (worker.get('status')!='ok' or worker.get('native_math_attempts')!=3
                or worker.get('native_math_completed')!=3 or worker.get('math_probes_completed')!=3
                or any(worker.get(key)!=0 for key in ('keygen_runs','encryptions','bootstraps','decryptions'))):
            raise ValueError('Native clear must complete all three original inputs with zero crypto')
        samples=worker.get('samples') or []
        references=checked_math_record(read(directories['adapter_math']/'result.json'),contract,metadata)
        if [row.get('sample_id') for row in samples]!=source['source_samples']:raise ValueError('Native clear sample identities differ')
        maximum=ratio=0.
        for row,original in zip(samples,references):
            reference,observed=row.get('reference'),row.get('observed')
            if (reference!=original['reference'][0] or not isinstance(observed,list) or len(observed)!=contract['output_shape'][1]
                    or row.get('passed') is not True):raise ValueError('Original classifier output reference differs')
            for want,got in zip(reference,observed):
                if not math.isfinite(want) or not math.isfinite(got):raise ValueError('Nonfinite native clear output')
                error=abs(got-want);relative=error/(1e-12+1e-12*abs(want))
                maximum=max(maximum,error);ratio=max(ratio,relative)
                if relative>1:raise ValueError('Strict original mathematics criterion failed')
        return dict(status='ok',record=str(directory/'result.json'),strict_probe_count=3,outputs=3*contract['output_shape'][1],
            maximum_absolute_error=maximum,maximum_tolerance_ratio=ratio,criterion='1e-12+1e-12*abs(reference)')
    parameters=artifact.get('parameters') or []
    original=source['native_parameters']
    if (len(parameters)!=1 or any(parameters[0].get(key)!=original[value] for key,value in
            (('logN','logN'),('Q','Q'),('P','P'),('log_default_scale','logDefaultScale')))):
        raise ValueError('Native original parameter literal differs')
    return dict(status='ok',record=str(directory/'result.json'),static_bootstrap_operations=artifact['static_bootstrap_operations'])


def register(registry,run_id,model,mode,metrics,out):
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==run_id for line in stream if line.strip()):raise ValueError('Existing pipeline registry identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(dict(run_id=run_id,round=ROUND,model=model,mode=mode,
            git_rev=None,metrics=metrics,artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat()),allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def execute(args):
    validate_memory_gib(getattr(args,'adapter_memory_gib',2))
    with serial_lock(paths.RESULTS_DIR):return _execute(args)


def _execute(args):
    adapter_memory_gib=validate_memory_gib(getattr(args,'adapter_memory_gib',2))
    started=time.perf_counter_ns();out=args.out.resolve()
    if out.exists() or not out.is_relative_to(paths.RESULTS_DIR.resolve()) or out==paths.RESULTS_DIR.resolve():
        raise ValueError('New isolated model case under the results root required')
    out.mkdir(parents=True);(out/'phases').mkdir();(out/'phase-records').mkdir()
    recorder=StageRecorder(out/'events.jsonl');stop=threading.Event();failures=[];active=[None]
    deadline=time.monotonic()+TOTAL_SECONDS
    registry=args.registry or paths.RESULTS_DIR/'registry.jsonl'
    result=dict(round=ROUND,model=args.model,mode=MODE,status='running',error=None,failed_stage=None,
        attempted={stage:0 for stage in PHASE_SECONDS},completed={stage:0 for stage in PHASE_SECONDS},phases={},
        actual_model_quality_selection=False,original_libraries_modified=False,original_models_changed=False,
        security_acceptance=None,accuracy_acceptance=None,CKKS_latency=None,compiled_bootstraps=None,
        static_native_bootstraps=None,formal_matrix_complete=False,**{name:0 for name in ZERO_FIELDS},
        incoming_signal=None,timeout_signal=None,wall_timeout_observed=False,guard_trigger=None,guard_stop_signal=None,
        limits=dict(AS_gib=12,subtree_RSS_gib=14,minimum_host_available_gib=1.25,minimum_free_disk_gib=10,
            adapter_math_child_AS_gib=adapter_memory_gib,
            case_gib=args.case_gib,compile_seconds=7200,phase_outer_seconds=PHASE_SECONDS,total_seconds=TOTAL_SECONDS,
            clear_child_RSS_gib=6,clear_child_seconds=1800,concurrency=1,automatic_total='sum of finite phase bounds plus600seconds source/persistence allowance'))
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupted(signum,frame):receive_signal(result,signum)
    def check_disk():
        if artifact_bytes(out)>args.case_gib*GIB:
            result['guard_trigger']=dict(kind='artifact',allowance_bytes=args.case_gib*GIB)
            raise RuntimeError('Finite model case artifact allowance exhausted')
        if shutil.disk_usage(out).free<10*GIB:
            result['guard_trigger']=dict(kind='disk',minimum_free_bytes=10*GIB)
            raise RuntimeError('10GiB free disk reserve exhausted')
    def check():
        if failures:raise RuntimeError(failures[0])
        if time.monotonic()>deadline:
            result['wall_timeout_observed']=True;raise TimeoutError('Finite model pipeline wall allowance exhausted')
        observed=physical_observation(_latest_rss_sample(active[0]) if active[0] else None,
            _available_ram_bytes(),shutil.disk_usage(out).free)
        if observed['stop']:
            result['resource_guard']=observed;result['guard_trigger']=observed
            durable_json(out/'resource-guard.json',observed)
            raise RuntimeError('; '.join(observed['reasons']))
    def monitor():
        next_disk=0
        while not stop.wait(.1):
            try:
                check()
                if time.monotonic()>=next_disk:check_disk();next_disk=time.monotonic()+1
            except BaseException as exc:
                failures.append(type(exc).__name__+': '+str(exc))
                try:result['monitor_stop_reason']=failures[-1]
                finally:
                    try:os.kill(os.getpid(),signal.SIGTERM)
                    except OSError as signal_error:result['monitor_signal_error']=str(signal_error)
                return
    watcher=threading.Thread(target=monitor,daemon=True)
    for sig in old:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,TOTAL_SECONDS)
    try:
        check();check_disk();watcher.start()
        with recorder.span('source-and-command-preparation'):
            source=sources(args.model);checked=source_gate(args.model,source);result['source']=checked
            pipeline=out/'native.pipeline';pipeline.write_text(checked['pipeline']+'\n')
            directories=phase_directories(out);planned=commands(args,source,directories,pipeline)
            result['commands']={stage:dict(argv=command,outer_seconds=PHASE_SECONDS[stage],AS_gib=12,
                output=str(directories[stage])) for stage,command in planned.items()}
            shutil.copy2(Path(__file__),out/'driver.py');durable_json(out/'source.json',checked)
        result['status']='prepared';durable_json(out/'result.json',result)
        if args.run:
            env={key:value for key,value in os.environ.items() if key not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')
                and not key.startswith(('LD_','NIX_','__NIX_','HECATE_'))}
            env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
            for stage,command in planned.items():
                check();check_disk();phase_started=time.perf_counter_ns();result['attempted'][stage]=1
                metadata=out/'phase-records'/stage;metadata.mkdir()
                row=dict(stage=stage,status='running',argv=command,output=str(directories[stage]),observation=None,error=None)
                result['phases'][stage]=row;durable_json(out/'result.json',result)
                try:
                    with recorder.span(stage,argv=command):
                        active[0]=metadata/'resources.jsonl'
                        try:
                            observation=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=metadata/'worker.log',
                                resources=active[0],timeout=min(PHASE_SECONDS[stage],max(.01,deadline-time.monotonic())),
                                mem_gib=12,fsize_gib=8,env=env)
                            row['observation']=observation;observe_termination(result,observation)
                        except BaseException as exc:
                            row['supervision_exception']=type(exc).__name__+': '+str(exc)
                            row['last_resource_sample']=_latest_rss_sample(active[0]);raise
                        finally:active[0]=None
                        if (observation.get('exit_code')!=0 or observation.get('timed_out') is not False
                                or observation.get('signal') is not None or observation.get('remaining_pids')!=[]
                                or observation.get('signals_sent')!=[]):raise RuntimeError('Phase process did not complete cleanly')
                        check();check_disk();row['audit']=check_phase(stage,directories,checked)
                        row['status']='ok';result['completed'][stage]=1
                except BaseException as exc:
                    row.update(status=interruption_status(result),error=type(exc).__name__+': '+str(exc)[:3000])
                    result.update(status=row['status'],failed_stage=stage,error=row['error'])
                finally:
                    row['elapsed_before_persistence_seconds']=(time.perf_counter_ns()-phase_started)/1e9
                    durable_json(metadata/'result.json',row)
                    register(registry,out.name+'--phase-'+stage,args.model,MODE+'/'+stage,
                        dict(status=row['status'],attempted=1,completed=result['completed'][stage],audit=row.get('audit')),metadata)
                    timing=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns())
                    marker=persist_case_completion(metadata,round_label=ROUND,started_ns=phase_started,status=row['status'],
                        timing=timing,required_artifact_bytes=artifact_bytes(metadata))
                    row['complete_phase_seconds']=marker['complete_case_elapsed_seconds']
                    durable_json(out/'result.json',result)
                if row['status']!='ok':break
            else:
                result['status']='ok'
                result['static_native_bootstraps']=result['phases']['native_compile']['audit']['static_bootstrap_operations']
    except BaseException as exc:
        result.update(status=interruption_status(result),error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in old.items():signal.signal(sig,handler)
        result['resource_monitor_error']=failures[0] if failures else None
        if failures:result.update(status=interruption_status(result),error=failures[0])
        timing=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns());result['timing']=timing
        durable_json(out/'result.json',result)
        register(registry,out.name,args.model,MODE,{key:result[key] for key in ('status','failed_stage','attempted','completed',
            'static_native_bootstraps','security_acceptance','accuracy_acceptance')},out)
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=timing,
            required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',action='store_true',help='Execute each existing phase once; stop after its first failure')
    parser.add_argument('--case-gib',type=int,default=32);parser.add_argument('--registry',type=Path)
    parser.add_argument('--adapter-memory-gib',type=float,default=2,
        help='Only the supervised adapter-math child allowance, finite 2..12 GiB (default 2)')
    parser.add_argument('--export-python',type=Path,default=paths.WORKSPACE/'baselines/orbit-venv/bin/python')
    parser.add_argument('--heir-opt',type=Path,default=paths.WORKSPACE/'baselines/heir-2026.11.1-venv/bin/heir-opt')
    parser.add_argument('--translate',type=Path,default=paths.WORKSPACE/'baselines/heir-2026.11.1-venv/bin/heir-translate')
    parser.add_argument('--go',type=Path,default=paths.WORKSPACE/'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/go')
    parser.add_argument('--cache-source',type=Path,default=paths.RESULTS_DIR/'heir-single-ckks-r21-20261115/work')
    args=parser.parse_args(argv)
    try:validate_memory_gib(args.adapter_memory_gib)
    except ValueError as exc:parser.error(str(exc))
    if not 0<args.case_gib<=64:parser.error('Finite case allowance must be1..64GiB;10GiB free disk reserve remains fixed')
    result=execute(args)
    print(json.dumps({key:result[key] for key in ('status','model','failed_stage','attempted','completed','static_native_bootstraps')}))
    return 0 if result['status'] in ('prepared','ok') else 1


if __name__=='__main__':raise SystemExit(main())
