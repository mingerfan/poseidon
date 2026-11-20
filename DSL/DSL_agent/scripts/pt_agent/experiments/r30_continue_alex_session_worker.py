"""Private mechanical hooks into the unchanged trusted R28 worker function."""
import argparse
import ast
import copy
import signal
from pathlib import Path

from . import r28_common_agent as common
from . import r30_continue_alex_session as hooks


def worker_namespace():
    """Only the two local resume imports differ; shared modules stay unchanged."""
    tree=ast.parse(Path(common.__file__).read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='worker')
    original=copy.deepcopy(function);replacements=[]
    class Adapt(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.level==1 and node.module=='r30_continue_common_session':
                if len(node.names)!=1 or node.names[0].asname is not None or node.names[0].name not in ('initialize_worker','verify_context'):
                    raise ValueError('Original worker resume interface changed')
                name=node.names[0].name;replacements.append(name)
                return ast.copy_location(ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],
                    value=ast.Attribute(value=ast.Name(id='_resume_hooks',ctx=ast.Load()),attr=name,ctx=ast.Load())),node)
            return node
    changed=Adapt().visit(function)
    if sorted(replacements)!=['initialize_worker','verify_context']:
        raise ValueError('Exactly the two original worker resume hooks are required')
    namespace=dict(vars(common));namespace['_resume_hooks']=hooks
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),common.__file__,'exec'),namespace)
    return namespace,dict(original_worker_AST=ast.dump(original,include_attributes=False),
        mechanical_resume_hook_names=sorted(replacements),common_module_globals_modified=False,
        algorithm_source='Unchanged trusted r28_common_agent.worker; two private local resume-hook assignments')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=(hooks.MODEL,),required=True)
    for name in ('work-dir','model-info','calibration','ledger','allowance','resume-session','resume-case','metadata-template'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--calls',type=int,choices=(6,),required=True)
    parser.add_argument('--tuner-seconds',type=int,choices=(600,),required=True)
    parser.add_argument('--memory-gib',type=int,required=True);parser.add_argument('--seconds',type=int,required=True)
    args=parser.parse_args(argv);args.tuner=True;args.prepare_only=False;args.resume_source=None;args.resume_result=None
    if not 4<=args.memory_gib<=12 or not 180<=args.seconds<=17880:parser.error('Use bounded original worker limits')
    hooks.restore_for_worker(args)  # Read-only admission before any model/framework import.
    def expire(signum,frame):raise TimeoutError('Alex continuation worker wall allowance exhausted')
    previous=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,args.seconds)
    try:
        from .r29_local_compile_matrix import native_metadata_commands
        with native_metadata_commands(hooks._read(args.metadata_template)):
            namespace,proof=worker_namespace()
            code=namespace['worker'](args)
            target=args.work_dir/'worker-result.json'
            if target.is_file():
                record=hooks._read(target)
                record.update(continuation_worker_reuse={key:value for key,value in proof.items()
                    if key!='original_worker_AST'},worker_wall_seconds=args.seconds,
                    worker_wall_timeout_observed=str(record.get('error','')).startswith('TimeoutError:'))
                common.write_json(target,record)
            return code
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':raise SystemExit(main())
