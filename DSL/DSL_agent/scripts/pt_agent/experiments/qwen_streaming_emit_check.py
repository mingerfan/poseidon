"""One small trusted-source/storage trace check, with no model or CKKS."""

import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import sys

from .r10_paper_metrics import StageRecorder,write_json


def prepare(args,recorder,result):
    import numpy as np
    from ..backends.streaming_emit import StreamingEmitBackend
    with recorder.span('emit_small_arithmetic'):
        backend=StreamingEmitBackend(16384,directory=args.work_dir/'emit',
            frontend=args.frontend/'python/hecate/hecate/expr.py',max_constant_bytes=1024*1024)
        a,b=backend.input(0),backend.input(1)
        pattern=np.zeros(16384,dtype=np.float64);pattern[0]=.25;pattern[17]=1.5;pattern[31]=-0.
        backend.output([a*backend.const(pattern)+b.rotate(1)])
        source=backend.write()
        result.update(status='ok',source=str(source.resolve()),emit_stats=backend.stats(),source_generation_completed=True)


def trace(args,recorder,result):
    previous={name:os.environ.get(name) for name in ('HECATE','PATH')}
    old_module=sys.modules.get('hecate_expr')
    cwd=Path.cwd()
    try:
        os.environ['HECATE']=str(args.frontend.resolve())
        with recorder.span('native_small_trace'):
            namespace=runpy.run_path(str(args.source),run_name='__main__')
            saved=Path(namespace['_saved_mlir']).resolve()
            if not saved.is_relative_to(args.source.parent.resolve()) or not saved.is_file() or not saved.stat().st_size:
                raise ValueError('The native frontend must save a nonempty MLIR artifact beside its generated source')
            if (namespace['_save_attempts']!=1 or namespace['_save_completed'] is not True
                    or namespace['_finalize_attempts']!=1 or namespace['_frontend_finalized'] is not True):
                raise ValueError('The generated source must save and finalize exactly once')
            result.update(status='ok',native_trace_completed=True,source=str(args.source.resolve()),
                mlir=dict(path=str(saved),size_bytes=saved.stat().st_size),
                save_attempts=namespace['_save_attempts'],save_completed=namespace['_save_completed'],
                finalize_attempts=namespace['_finalize_attempts'],frontend_finalized=namespace['_frontend_finalized'],
                source_reader_closed=namespace['C'].payload.closed,
                scope='one original-backend arithmetic trace with a streamed indexed plaintext constant',
                artifacts=[dict(path=str(path.resolve()),size_bytes=path.stat().st_size)
                           for path in args.source.parent.rglob('*') if path.is_file()])
    finally:
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
        if old_module is None:sys.modules.pop('hecate_expr',None)
        else:sys.modules['hecate_expr']=old_module
        os.chdir(cwd)
        result['environment_restored']=all(os.environ.get(name)==value for name,value in previous.items())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','trace'),required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--frontend',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args(argv);args.work_dir.mkdir(parents=True,exist_ok=True)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    result=dict(status='running',phase=args.phase,source_generation_completed=False,native_trace_completed=False,
        model_loads=0,sample_reads=0,api_calls=0,compiler_runs=0,keygen_runs=0,real_ckks_executions=0,
        full_model_native_trace_verified=False,full_model_native_trace_memory_bytes=None,
        compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None)
    recorder=StageRecorder(args.work_dir/'events.jsonl')
    try:
        with recorder.span('streaming_emit_small_check'):
            (prepare if args.phase=='prepare' else trace)(args,recorder,result)
    except BaseException as exc:result.update(status='error',error_type=type(exc).__name__,error=str(exc)[:2000])
    result['worker_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.work_dir/'worker-result.json',result)
    print(json.dumps({key:result[key] for key in ('status','phase','source_generation_completed','native_trace_completed','worker_maxrss_kib')}))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
