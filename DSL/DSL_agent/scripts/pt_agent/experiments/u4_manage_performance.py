"""U4 §7.9 finite standard-library management profiling; reuse retained IR."""

import argparse
from datetime import datetime, timezone
import faulthandler
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from ..paths import REPO, RESULTS_DIR


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def diagnose(args):
    from .. import toolchain
    from ..manage import manage, check, emit
    from ..manage.greedy import timing_log, timed_stage
    from ..manage.mlir_read import read, graph_backend, RAW_OPS
    from ..tests._u4_legacy_manage import compare_functions
    toolchain.setup_env()
    if args.model=='tests':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.test_u4_management_performance')
        outcome=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
        write(args.out/'REPORT.json',dict(round='U4',section='7.9.3.1',model='tests',
            complete=outcome.wasSuccessful(),tests=outcome.testsRun,failures=len(outcome.failures),
            errors=len(outcome.errors),skipped=len(outcome.skipped),llm_calls=0))
        return 0 if outcome.wasSuccessful() else 1
    profile=toolchain.management_profile()
    result=dict(round='U4',section='7.9.1',model=args.model,complete=False,
                llm_calls=0,ckks_calls=0,report_images=0,new_trace_calls=0,new_compile_calls=0)
    write(args.out/'REPORT.json',result)
    with timing_log(args.out/'trace-stages.json'):
        if args.model=='bert':
            configs=list((args.saved/'managed').glob('*/config/model.json'))
            if len(configs)!=1:raise ValueError('Expected one retained package configuration')
            package_config=configs[0].parent
            model=json.loads(configs[0].read_text())
            profile=json.loads((package_config/'management.json').read_text())['profile']
            bytecode=args.saved/'managed/trace/preprocessed.mlirbc'
            structure=args.saved/'managed/trace/preprocessed.structure.mlir'
            for source in (bytecode,structure):
                if not source.is_file():raise FileNotFoundError(source)
            shutil.copyfile(package_config/'compiler-config.json',args.out/'boot-compiler-config.json')
            command=toolchain.to_host_argv(toolchain.frontend_structure_argv('dacapo-depth-dp','qwen48',args.out),args.out)
            command[1]=str(bytecode)
            result['saved_inputs']=dict(bytecode=str(bytecode),structure=str(structure))
            with timed_stage('saved_bytecode_read'):
                result['bytecode_print']=toolchain.trusted_run(args.out,command,args.out/'bytecode-print.log',
                    env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
                if result['bytecode_print']['exit']:raise RuntimeError('Retained bytecode structure print failed')
            with timed_stage('saved_structure_read'):
                original=read(structure,profile,allowed=RAW_OPS,structure_only=True)
            with timed_stage('bytecode_structure_read'):
                restored=read(args.out/'preprocessed.structure.mlir',profile,allowed=RAW_OPS,structure_only=True)
                result['saved_ir_structure_comparison']=compare_functions(original,restored)
                if not result['saved_ir_structure_comparison']['equal']:
                    raise RuntimeError('Saved bytecode and structure graphs differ')
                del restored
                gc.collect()
            recorded=graph_backend(original,model['slots'])
            result['diagnostic_payloads']='structure-only; management diagnosis, no check/emission/execution'
        else:
            from ..models.info import ModelInfo
            from ..dsl.parser import parse
            from ..lower import lower
            from ..manage import RecordingBackend
            from .u4_managed_payloads import shared_preprocess
            row=next(r for r in json.loads(args.manifest.read_text())['cases'] if r['label']==args.case)
            with timed_stage('squeeze_recording'):
                info=ModelInfo.load(Path(row['info']))
                program=parse(Path(row['source']).read_text())
                recorded=RecordingBackend(program.header.slots,locations={op.id:op for op in program.ops})
                lower(program,info.graph.weights,recorded)
                del info
                gc.collect()
            with timed_stage('squeeze_preprocess'):
                shared,function,result['preprocess']=shared_preprocess(recorded,args.out/'preprocessed',profile,
                    name='squeeze_diagnostic',compressed=True)
                del recorded,function
                gc.collect()
                recorded=shared
        result['input_nodes']=len(recorded.nodes)
        result['input_ciphertexts']=len(recorded._inputs)
        write(args.out/'REPORT.json',result)
        with (args.out/'stacks.log').open('w') as stacks:
            faulthandler.dump_traceback_later(600,repeat=True,file=stacks)
            try:
                function=manage(recorded,profile,name='management_diagnostic')
            finally:
                faulthandler.cancel_dump_traceback_later()
        result.update(output_nodes=len(function.nodes),boots=sum(n.kind=='bootstrap' for n in function.nodes),
                      cuts=len(function.cuts),management_complete=True)
        if args.model=='squeeze':
            check(function)
            result['managed_ir']=str(emit(function,args.out/'managed.mlir',compressed=True))
            result['pruned']=dict(preprocessed=toolchain.prune(args.out/'preprocessed'),
                                  managed=toolchain.prune(args.out))
        result['complete']=True
        write(args.out/'REPORT.json',result)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--model',choices=('bert','squeeze','tests'),required=True)
    for name in ('out','saved','manifest'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--case',default='u3a-squeezenet_silu_avgpool')
    args=parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        try:return diagnose(args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            path=args.out/'REPORT.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            result.update(complete=False,error=f'{type(exc).__name__}: {exc}')
            write(path,result);write(args.out/'STOP.json',result)
            return 1
    fields=('out','saved','manifest','case')
    if args.launch:
        from ..tools.nixrun import main as nix_main
        sys.argv=[sys.argv[0],'--seconds','1200','--','-m',__spec__.name,'--inside','--model',args.model]
        for name in fields:
            if getattr(args,name) is not None:sys.argv+=['--'+name,str(getattr(args,name))]
        try:return nix_main()
        except subprocess.CalledProcessError as exc:
            stderr=exc.stderr or ''
            (args.out/'nix.stderr.txt').write_text(stderr)
            write(args.out/'STOP.json',dict(complete=False,stage='nix-launch',exit=exc.returncode,stderr=stderr))
            sys.stderr.write(stderr);return exc.returncode
    stamp=datetime.now(timezone.utc)
    run_id='u4-manage-diagnostic-'+args.model+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id;args.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B','-m',__spec__.name,'--launch','--model',args.model]
    for name in fields:
        if getattr(args,name) is not None:command+=['--'+name,str(getattr(args,name))]
    write(args.out/'argv.json',command)
    print('RUN='+str(args.out),flush=True);began=time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status=subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT)
    path=args.out/'REPORT.json'
    result=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    result.update(run_id=run_id,git_rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip(),
                  exit=status.returncode,seconds=time.monotonic()-began,diagnostic_wall_limit_seconds=1200)
    if status.returncode==124:result['diagnostic_timeout']=True
    write(path,result)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model=args.model,mode='management-diagnostic',git_rev=result['git_rev'],
        metrics=dict(complete=result['complete'],llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),exit=status.returncode,complete=result['complete'])),flush=True)
    return status.returncode


if __name__=='__main__':raise SystemExit(main())
