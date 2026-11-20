"""Native HEIR Lattigo split-code emission and optional isolated offline library build."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from .. import paths
from .r10_paper_metrics import StageRecorder,read_events,run_process,summarize_events,write_json
from .heir_cnn_preflight import finalize_size,native_artifact_metadata


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--source-result',type=Path,help='Explicit successful compiler result or independent terminal audit retaining the original rejected parent result')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--translate',type=Path,default=paths.WORKSPACE/'baselines/heir-2026.11.1-venv/bin/heir-translate')
    p.add_argument('--build',action='store_true')
    p.add_argument('--go',type=Path,default=paths.WORKSPACE/'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/go')
    p.add_argument('--cache-source',type=Path,default=paths.RESULTS_DIR/'heir-single-ckks-r21-20261115/work')
    p.add_argument('--memory-gib',type=float,default=2)
    p.add_argument('--build-seconds',type=int,default=600)
    a=p.parse_args(argv)
    if not a.input.is_file():p.error('Compiled native Lattigo MLIR is required')
    out=a.out.absolute();out.mkdir(parents=True,exist_ok=False)
    module=out/'module';module.mkdir()
    utils=module/'heircnn_utils';utils.mkdir()
    for name in ('tmp','telemetry'):(out/name).mkdir()
    (out/'telemetry/mode').write_text('off\n')
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV') and not k.startswith(('LD_','NIX_','__NIX_','GO'))}
    env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'),GOTOOLCHAIN='local',GOPROXY='off',GOSUMDB='off',
               GOPATH=str(out/'gopath'),GOMODCACHE=str(out/'module-cache'),GOCACHE=str(out/'build-cache'),
               GOTMPDIR=str(out/'tmp'),TEST_TELEMETRY_DIR=str(out/'telemetry'),CGO_ENABLED='0',
               GOMAXPROCS='1',GOMEMLIMIT=str(int(max(.25,a.memory_gib-1)*1024))+'MiB')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    from .heir_native_clear_math import checked_native_source
    source_record=a.source_result or a.input.parent/'result.json'
    source_metadata,source_ir=checked_native_source(source_record)
    if source_ir.resolve()!=a.input.resolve():p.error('The native source record must identify this exact compiled artifact path')
    record=dict(round='HEIR-CNN-GO',mode='native-heir-cnn-lattigo-codegen',model=source_metadata['model'],status='running',error=None,
                input_mlir=str(a.input.absolute()),native_artifact=native_artifact_metadata(a.input),
                source_compiler_result=str(source_record.absolute()),
                attempts=dict(utils_emission=0,main_emission=0,library_build=0),completed=dict(utils_emission=0,main_emission=0,library_build=0),
                processes=[],library_build_passed=False,backend_executed=False,
                original_libraries_modified=False,downloads=0,installations=0,
                binary_runs=0,keygen_runs=0,real_ckks_attempts=0,real_ckks_completed=0,llm_calls=0,
                compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None,
                telemetry_mode='off',telemetry_directory=str(out/'telemetry'),artifact_bytes=None)
    record['go_runtime_environment']={key:env[key] for key in ('GOMAXPROCS','GOMEMLIMIT')}
    events=StageRecorder(out/'events.jsonl')
    def run(stage,cmd,seconds):
        if shutil.disk_usage(out).free<10*1024**3:raise RuntimeError('Preserve 10GiB free-disk reserve')
        record['attempts'][stage]+=1
        with events.span(stage,argv=cmd):
            obs=run_process(cmd,cwd=module,log=out/(stage+'.log'),resources=out/(stage+'-resources.jsonl'),
                            timeout=seconds,mem_gib=a.memory_gib,env=env)
        record['processes'].append(dict(stage=stage,argv=cmd,observation=obs))
        write_json(out/'result.json',record)
        if obs['exit_code']!=0 or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']:
            raise RuntimeError(stage+' failed; original outputs retained')
        record['completed'][stage]+=1
    try:
        with events.span('native-codegen-and-optional-library-build'):
            run('utils_emission',[str(a.translate),str(a.input.absolute()),'--mlir-disable-threading',
                                  '--emit-lattigo-preprocessing','--package-name=heircnn_utils','-o',str(utils/'heircnn_utils.go')],60)
            run('main_emission',[str(a.translate),str(a.input.absolute()),'--mlir-disable-threading',
                                 '--emit-lattigo-preprocessed','--extra-imports=heircnn/heircnn_utils',
                                 '--package-name=heircnn','-o',str(module/'heircnn.go')],60)
            original=a.cache_source/'module'
            mod=(original/'go.mod').read_text()
            (module/'go.mod').write_text('module heircnn\n'+mod[mod.index('\n')+1:])
            shutil.copyfile(original/'go.sum',module/'go.sum')
            if a.build:
                with events.span('private-cache-copy'):
                    shutil.copytree(a.cache_source/'module-cache',out/'module-cache')
                    shutil.copytree(a.cache_source/'build-cache',out/'build-cache')
                run('library_build',[str(a.go),'build','-p=1','-mod=readonly','-buildvcs=false','./...'],a.build_seconds)
                record['library_build_passed']=True
            record['status']='ok'
    except BaseException as exc:
        record['status']='error';record['error']=type(exc).__name__+': '+str(exc)
    finally:
        write_json(out/'timing.json',summarize_events(read_events(out/'events.jsonl')))
        record['ts']=datetime.now(timezone.utc).isoformat()
        finalize_size(out,record)
        entry=dict(run_id=out.name,round=record['round'],model=record['model'],mode=record['mode'],git_rev='working-tree',
                   metrics={k:record[k] for k in ('status','library_build_passed','compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance')},
                   artifacts_dir=str(out),ts=record['ts'])
        with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            if any(json.loads(l).get('run_id')==out.name for l in f if l.strip()):raise ValueError('Existing result identity')
            f.seek(0,2);f.write(json.dumps(entry)+'\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
    print(json.dumps({k:record[k] for k in ('status','error','library_build_passed','artifact_bytes')}))
    return 0 if record['status']=='ok' else 1


if __name__=='__main__':sys.exit(main())
