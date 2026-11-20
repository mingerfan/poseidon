"""Bounded native HEIR CNN integration stages, with untouched library sources."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .. import paths
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json


FORMAL = ('compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance')


def native_artifact_metadata(path):
    text=Path(path).read_text()
    parameters=[]
    pattern=(r'paramsLiteral = <logN = (\d+), Q = \[([^]]*)\], P = \[([^]]*)\], '
             r'logDefaultScale = (\d+)>')
    for logn,q,p,scale in re.findall(pattern,text):
        primes=lambda s:[int(v.strip()) for v in s.split(',') if v.strip()]
        qs,ps=primes(q),primes(p)
        parameters.append(dict(logN=int(logn),ring_degree=2**int(logn),Q=qs,P=ps,
                               Q_bits=[v.bit_length() for v in qs],P_bits=[v.bit_length() for v in ps],
                               log_default_scale=int(scale)))
    return dict(parameters=parameters,static_bootstrap_operations=len(re.findall(r'= lattigo\.ckks\.bootstrap\s',text)),
                runtime_bootstraps=None,security_validated=False,backend_executed=False,
                scope='Native generated compiler artifact; static operations and parameter literals only')


def compiler_command(tool, source, target, resources, *, slots=16384, management='greedy'):
    return [str(tool),'--mlir-disable-threading','--annotate-module=backend=lattigo scheme=ckks',
            f'--torch-linalg-to-ckks=min-slot-count={slots} ciphertext-management-style={management}',
            '--scheme-to-lattigo',str(source),'-o',str(target),
            '--ext-const-output-dir='+str(resources),'--ext-const-runtime-load-dir='+str(resources)]


def finalize_size(out,result):
    target=out/'result.json'
    for _ in range(20):
        write_json(target,result)
        size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if result.get('artifact_bytes')==size:return
        result['artifact_bytes']=size
    raise RuntimeError('Result metadata size did not stabilize')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--python',type=Path,default=paths.WORKSPACE/'baselines/orbit-venv/bin/python')
    p.add_argument('--heir-opt',type=Path,default=paths.WORKSPACE/'baselines/heir-2026.11.1-venv/bin/heir-opt')
    p.add_argument('--graph-dir',type=Path)
    p.add_argument('--program',type=Path)
    p.add_argument('--source',type=Path,help='Reuse a previously generated trusted MLIR for one new bounded compile')
    p.add_argument('--pipeline',choices=('torch-linalg-to-ckks','mlir-to-ckks','verify'),default='torch-linalg-to-ckks')
    p.add_argument('--slots',type=int,default=16384)
    p.add_argument('--management',choices=('greedy','orbit-ilp'),default='greedy')
    p.add_argument('--compile-seconds',type=int,default=300)
    p.add_argument('--memory-gib',type=float,default=4)
    p.add_argument('--print-ir',action='store_true')
    p.add_argument('--operator-suite',action='store_true')
    p.add_argument('--dump-pipeline',action='store_true')
    p.add_argument('--explicit-pipeline',type=Path,help='Reviewed textual HEIR native pass configuration')
    from .heir_common_models import COMMON_MODELS
    p.add_argument('--common-model',choices=COMMON_MODELS)
    args=p.parse_args(argv)
    if bool(args.graph_dir)!=bool(args.program):p.error('graph-dir and saved program must be supplied together')
    if args.source and args.graph_dir:p.error('source reuse and graph export are separate modes')
    if args.common_model and not (args.source or args.graph_dir):p.error('common-model requires original graph/program or trusted source reuse')
    out=args.out.absolute()
    out.mkdir(parents=True,exist_ok=False)
    (out/'tmp').mkdir()
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV') and not k.startswith(('LD_','NIX_','__NIX_'))}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(out/'tmp'),HEIR_CNN_SMALL_OUTPUT=str(out/'small'))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    result=dict(round='HEIR-CNN',model=args.common_model or ('resnet20_silu' if args.graph_dir else 'deterministic-small-cnn'),
                mode='native-heir-cnn-model-integration',status='running',error=None,
                official_release='v2026.11.01',approximation_source='R17 fixed margin1.50 degree63',
                approximation_selected_by_heir=False,native_management=args.management,
                compiler_pipeline=args.pipeline,minimum_slots=args.slots,processes=[],
                attempts=dict(tests=0,export=0,compile=0),completed=dict(tests=0,export=0,compile=0),
                model_compiled=False,minimal_operator_compiled=False,backend_executed=False,
                llm_calls=0,real_ckks_attempts=0,real_ckks_completed=0,keygen_runs=0,
                candidate_frozen=False,confirm_samples=0,report_samples=0,
                parameters_validated=False,artifact_bytes=None,**{n:None for n in FORMAL})
    recorder=StageRecorder(out/'events.jsonl')
    full_model=bool(args.graph_dir)
    def run(stage,command,seconds):
        if shutil.disk_usage(out).free<10*1024**3:
            raise RuntimeError('Preserve the 10GiB free-disk reserve before a new stage')
        result['attempts'][stage]+=1
        with recorder.span(stage,argv=command):
            observation=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/(stage+'.log'),
                                    resources=out/(stage+'-resources.jsonl'),timeout=seconds,mem_gib=args.memory_gib,env=env)
        result['processes'].append(dict(stage=stage,argv=command,observation=observation))
        write_json(out/'result.json',result)
        if observation['exit_code']!=0 or observation['timed_out'] or observation['signal'] is not None or observation['remaining_pids']:
            raise RuntimeError(stage+' failed; see retained log and process metadata')
        result['completed'][stage]+=1
    try:
        with recorder.span('heir-cnn-preflight'):
            if args.source:
                source=args.source.absolute()
                if not source.is_file():raise ValueError('Trusted input MLIR is absent')
                result['source']=str(source)
                metadata=source.parent/'export.json'
                if metadata.is_file():
                    result['export_record']=json.loads(metadata.read_text())
                    full_model=(result['export_record'].get('function')=='resnet20_silu'
                                and result['export_record'].get('graph_nodes')==74
                                and len(result['export_record'].get('approximations',[]))==19)
                    if full_model:result['model']='resnet20_silu'
                    if args.common_model:
                        from .heir_common_models import read_graph_contract,checked_export_contract
                        contract=read_graph_contract(result['export_record'].get('sources',{}).get('graph',''),args.common_model)
                        checked_export_contract(result['export_record'],contract)
                        from .heir_common_models import checked_saved_frontend_identity
                        saved_program=Path(result['export_record'].get('sources',{}).get('program',''))
                        checked_saved_frontend_identity(json.loads((saved_program.parent.parent/'result.json').read_text()),saved_program,args.common_model)
                        result['common_model']=args.common_model;result['common_model_contract']=contract
                        result['model']=args.common_model;full_model=True
                elif args.common_model:raise ValueError('Common-model source reuse requires original export metadata')
            elif args.graph_dir:
                saved=json.loads((args.program.parent.parent/'result.json').read_text())
                saved_worker=saved.get('worker') or {}
                if (saved.get('status')!='ok' or saved.get('round') not in ('R17','R17a')
                        or saved_worker.get('heldout_acceptance') is not True
                        or saved_worker.get('completed_samples')!=1000
                        or saved_worker.get('math_probes_passed') is not True
                        or saved_worker.get('rule_options')!={'deg':63,'margin':1.5,'slots':16384,'placement':'depth-dp','refresh':'compiler','act_method':'cheb'}
                        or Path(saved_worker.get('artifacts',{}).get('program','')).resolve()!=args.program.resolve()):
                    raise ValueError('Complete accepted fixed R17 frontend source is required')
                if args.common_model:
                    from .heir_common_models import read_graph_contract,checked_saved_frontend_identity
                    checked_saved_frontend_identity(saved,args.program,args.common_model)
                    result['common_model']=args.common_model
                    result['common_model_contract']=read_graph_contract(args.graph_dir,args.common_model)
                result['saved_frontend']=dict(result=str(args.program.parent.parent/'result.json'),
                                              round=saved['round'],heldout_samples=1000,
                                              candidate_frozen=saved_worker.get('candidate_frozen'),
                                              rule_options=saved_worker['rule_options'])
                export_command=[str(args.python),'-B','-m','pt_agent.experiments.heir_cnn_export','--graph-dir',str(args.graph_dir.absolute()),
                                '--program',str(args.program.absolute()),'--output',str(out/'export')]
                if args.common_model:export_command+=['--common-model',args.common_model]
                run('export',export_command,180)
                source=out/'export/model.mlir'
                result['export_record']=json.loads((out/'export/export.json').read_text())
                if args.common_model:
                    from .heir_common_models import checked_export_contract
                    checked_export_contract(result['export_record'],result['common_model_contract'])
            else:
                run('tests',[str(args.python),'-B','-m','unittest','discover','-s','pt_agent/tests','-p','test_heir_cnn_export.py','-v'],180)
                source=out/'small/small_cnn.mlir'
                result['export_record']=json.loads((out/'small/small_cnn_export.json').read_text())
            sources=([out/'small'/(n+'_only.mlir') for n in ('conv','bn','poly','shortcut','mean','linear')]
                     if args.operator_suite else [source])
            result['operator_results']=[]
            for index,source in enumerate(sources):
                label=source.stem if args.operator_suite else 'model'
                resources_dir=out/(label+'-resources')
                resources_dir.mkdir()
                command=compiler_command(args.heir_opt,source,out/(label+'-lattigo.mlir'),resources_dir,slots=args.slots,management=args.management)
                command=[c.replace('--torch-linalg-to-ckks=','--'+args.pipeline+'=') for c in command]
                if args.pipeline=='verify':
                    command=[str(args.heir_opt),'--mlir-disable-threading',str(source),'-o',str(out/(label+'-verified.mlir'))]
                if args.print_ir:command+=['--mlir-print-ir-before-all']
                if args.dump_pipeline:command+=['--dump-pass-pipeline']
                if args.explicit_pipeline:
                    native_pipeline=args.explicit_pipeline.read_text().strip().replace('${RESOURCES}',str(resources_dir))
                    parameter_files=[]
                    for parameter_file in dict.fromkeys(re.findall(r'parameter-file=([^\s}]+)',native_pipeline)):
                        original_parameter_file=Path(parameter_file)
                        parameter_literal=json.loads(original_parameter_file.read_text())
                        preserved=out/('native-parameters-'+str(len(parameter_files))+'.json')
                        shutil.copyfile(original_parameter_file,preserved)
                        native_pipeline=native_pipeline.replace('parameter-file='+parameter_file,'parameter-file='+str(preserved))
                        parameter_files.append(dict(source=str(original_parameter_file.absolute()),
                                                    preserved=str(preserved),literal=parameter_literal))
                    result['native_parameter_files']=parameter_files
                    (out/'explicit.pipeline').write_text(native_pipeline+'\n')
                    command=[str(args.heir_opt),'--mlir-disable-threading','--pass-pipeline='+native_pipeline,
                             str(source),'-o',str(out/(label+'-lattigo.mlir')),
                             '--ext-const-output-dir='+str(resources_dir),'--ext-const-runtime-load-dir='+str(resources_dir)]
                    result['explicit_pipeline_source']=str(args.explicit_pipeline.absolute())
                    result['native_compatibility_configuration']=dict(
                        base='same installed release native torch-linalg-to-ckks and scheme-to-lattigo',
                        changes=[] if 'layout-optimization{' in native_pipeline else
                        ['layout-optimization pass omitted after recorded native default crash'],
                        layout_optimization_enabled='layout-optimization{' in native_pipeline,
                        original_libraries_modified=False,default_heir_configuration=False)
                    if 'secret-insert-mgmt-ckks{after-mul=true ' in native_pipeline:
                        result['native_compatibility_configuration']['changes'].append(
                            'official greedy-modulus-switch-after-mul=true policy')
                    if parameter_files:
                        result['native_compatibility_configuration']['changes'].append(
                            'official parameter-file option separates ring degree from the packing slot count')
                try:
                    run('compile',command,args.compile_seconds)
                    result['operator_results'].append(dict(source=str(source),status='ok'))
                    if args.pipeline!='verify':
                        artifact=native_artifact_metadata(out/(label+'-lattigo.mlir'))
                        write_json(out/(label+'-native-artifact.json'),artifact)
                        result['operator_results'][-1]['native_artifact']=artifact
                except Exception as exc:
                    result['operator_results'].append(dict(source=str(source),status='error',error=str(exc)))
                    raise
                if args.operator_suite:
                    # The logs/resources belong to distinct attempted compiler processes.
                    (out/'compile.log').rename(out/(label+'-compile.log'))
                    (out/'compile-resources.jsonl').rename(out/(label+'-resources.jsonl'))
            result['status']='ok'
            result['mlir_verified']=True
            result['model_compiled']=full_model and args.pipeline!='verify'
            result['minimal_operator_compiled']=not full_model and args.pipeline!='verify'
    except BaseException as exc:
        result['status']='error';result['error']=type(exc).__name__+': '+str(exc)
    finally:
        write_json(out/'timing.json',summarize_events(read_events(out/'events.jsonl')))
        result['ts']=datetime.now(timezone.utc).isoformat()
        finalize_size(out,result)
        registry=paths.RESULTS_DIR/'registry.jsonl'
        record=dict(run_id=out.name,round='HEIR-CNN',model=result['model'],mode=result['mode'],git_rev='working-tree',
                    metrics={k:result[k] for k in ('status','model_compiled','minimal_operator_compiled',*FORMAL)},artifacts_dir=str(out),ts=result['ts'])
        with registry.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            f.seek(0)
            if any(json.loads(line).get('run_id')==out.name for line in f if line.strip()):
                raise ValueError('Existing registry identity must not be reused')
            f.seek(0,2);f.write(json.dumps(record)+'\n');f.flush()
            fcntl.flock(f,fcntl.LOCK_UN)
    print(json.dumps({k:result[k] for k in ('status','error','artifact_bytes','model_compiled','minimal_operator_compiled')}))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':sys.exit(main())
