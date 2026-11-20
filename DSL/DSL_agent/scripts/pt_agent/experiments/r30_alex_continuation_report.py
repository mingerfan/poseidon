"""Private source-gate/argv adaptation of the original frozen CNN report.

All original report mathematics, sample order, access claim and quality policy
run unchanged. The unique access identity is the frozen continuation case.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
from types import FunctionType

from . import r30_alex_postfreeze as postfreeze
from . import r30_frozen_cnn_report as original


def report_namespace(compile_case):
    """Clone trusted Python functions with private globals; old module is intact."""
    compile_case=Path(compile_case).resolve()
    namespace=dict(vars(original));namespace['__spec__']=__spec__
    for name,value in tuple(namespace.items()):
        if isinstance(value,FunctionType)and value.__module__==original.__name__:
            clone=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__
            namespace[name]=clone
    def source_gate(source_case,model):
        if model!=postfreeze.MODEL:raise ValueError('Only the accepted original Alex continuation is supported')
        gate=postfreeze.compiled_source(compile_case,source_case)
        gate['continuation_report_adapter']=True
        gate['report_outcome_used_for_selection']=False
        return gate
    run=original.run_process
    def launch(argv,**kwargs):
        # Original execute constructs its existing worker CLI. Its own module
        # name is redirected to this entry, and one explicit path is appended.
        argv=list(argv)
        if '--compile-case'in argv:raise ValueError('The report compile path must be forwarded exactly once')
        argv+=['--compile-case',str(compile_case)]
        out=Path(argv[argv.index('--out')+1]);snapshot=out/'source';snapshot.mkdir(exist_ok=False)
        for path in (Path(__file__),Path(postfreeze.__file__),Path(original.__file__)):
            shutil.copyfile(path,snapshot/path.name)
        return run(argv,**kwargs)
    worker=namespace['worker']
    def accepted_worker(args):
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[key]='1'
        from .r29_local_compile_matrix import native_metadata_commands
        gate=source_gate(args.source_case,args.model)
        # The retained compile recipe is an argv JSON list, not a result object.
        with native_metadata_commands(json.loads(Path(gate['metadata_template']).read_text())):
            code=worker(args)
        target=args.out/'worker-result.json'
        if target.is_file():
            result=postfreeze.read(target)
            result.update(continuation_report_adapter=True,compile_case=str(compile_case),
                report_outcome_used_for_selection=False,new_operator_identifier_digests=0,
                new_operator_identifier_digest_scope='Existing saved operator metadata; payload format exporters are not run')
            original.write_json(target,result)
        return code
    namespace.update(source_gate=source_gate,run_process=launch,worker=accepted_worker)
    return namespace


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--compile-case',type=Path,required=True)
    args,remaining=parser.parse_known_args(argv)
    return report_namespace(args.compile_case)['main'](remaining)


if __name__=='__main__':raise SystemExit(main())
