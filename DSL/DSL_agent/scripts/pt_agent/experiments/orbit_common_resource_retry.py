"""Explicit Orbit frontend address-space retry with physical-memory supervision.

Only private parser caps and the worker module target change. Original source
admission, model/Poly arithmetic, trace and TDAG acceptance remain delegated.
This entry performs one trace and one parse; optimization is a separate job.
"""
import ast
from contextlib import contextmanager
from inspect import isgeneratorfunction
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from types import FunctionType

from . import orbit_common_model_frontend as frontend
from . import orbit_common_model_worker as worker
from .r10_paper_metrics import write_json
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from .r31_lattigo_runtime_plan import physical_memory_observation

MODULE='pt_agent.experiments.orbit_common_resource_retry'
VARIANT='orbit-original-frontend-as-up-to24-rss14-v1'
GIB=1024**3


def _clone(module):
    namespace=dict(vars(module))
    def function(value):
        clone=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
        clone.__kwdefaults__=value.__kwdefaults__
        return clone
    for name,value in tuple(namespace.items()):
        if isinstance(value,FunctionType)and value.__module__==module.__name__:
            wrapped=getattr(value,'__wrapped__',None)
            if (isinstance(wrapped,FunctionType)and isgeneratorfunction(wrapped)
                    and wrapped.__module__==module.__name__
                    and value.__code__ is contextmanager(wrapped).__code__):
                # functools.wraps gives this stdlib helper the worker's module
                # label. Its globals must remain contextlib's; only the original
                # generator body uses the private worker hooks and constants.
                namespace[name]=contextmanager(function(wrapped))
            else:
                namespace[name]=function(value)
    return namespace


def _resource_main(module,namespace,*,inner):
    """One exact memory comparison changes; all original parser defaults stay."""
    tree=ast.parse(Path(module.__file__).read_text())
    node=next(value for value in tree.body if isinstance(value,ast.FunctionDef)and value.name=='main')
    changed=[]
    class MemoryCap(ast.NodeTransformer):
        def visit_Compare(self,value):
            if (len(value.comparators)==2 and isinstance(value.comparators[0],ast.Attribute)
                    and isinstance(value.comparators[0].value,ast.Name)
                    and value.comparators[0].value.id=='args' and value.comparators[0].attr=='memory_gib'
                    and isinstance(value.comparators[1],ast.Constant)and value.comparators[1].value==12):
                value.comparators[1]=(ast.Call(func=ast.Name(id='_trace_or_parse_cap',ctx=ast.Load()),
                    args=[ast.Attribute(value=ast.Name(id='args',ctx=ast.Load()),attr='phase',ctx=ast.Load())],keywords=[])
                    if inner else ast.Constant(24))
                changed.append('memory_gib_cap')
            return self.generic_visit(value)
        def visit_Constant(self,value):
            if isinstance(value.value,str):
                text=value.value.replace('12GiB memory','24GiB trace address space and4GiB parse')
                text=text.replace('12GiB memory','24GiB address space')
                return ast.copy_location(ast.Constant(text),value)
            return value
    node=MemoryCap().visit(node)
    if changed!=['memory_gib_cap']:raise ValueError('Require exactly the original single12GiB parser guard')
    namespace['_trace_or_parse_cap']=lambda phase:24 if phase=='trace'else 4
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),namespace)
    return namespace


def worker_namespace():
    namespace=_resource_main(worker,_clone(worker),inner=True)
    initial=namespace['_initial_result'];checkpoint=namespace['_checkpoint']
    def initial_result(phase):
        result=initial(phase);result['resource_adapter']=VARIANT
        return result
    def record_resources(args,result):
        result.update(resource_adapter=VARIANT,address_space_limit_gib=args.memory_gib,
            physical_RSS_guard_gib=14,host_available_reserve_gib=1.25,
            memory_limit_semantics='RLIMIT_AS is virtual address space; outer sampled physical memory is separate')
        return checkpoint(args,result)
    namespace.update(_initial_result=initial_result,_checkpoint=record_resources)
    return namespace


def _guard_observation(resources):
    return physical_memory_observation(_latest_rss_sample(resources),_available_ram_bytes(),
        rss_gib=14,minimum_available_gib=1.25)


def guarded_run(argv,*,state,run,**options):
    """R10 owns child cleanup; a physical-memory notification reaches its handler."""
    if '--phase'not in argv:return run(argv,**options)
    phase=argv[argv.index('--phase')+1]
    if phase not in ('trace','parse'):raise ValueError('Only the original frontend phases are supervised')
    resources=Path(options['resources']);directory=resources.parent
    stop=threading.Event();row=dict(phase=phase,AS_gib=options['mem_gib'],RSS_gib=14,
        host_available_reserve_gib=1.25,stop_observed=False,error=None,samples=0)
    state['phases'].append(row)
    def inspect():
        observation=_guard_observation(resources);row['samples']+=1
        row['latest']=observation
        with (directory/'physical-memory-resources.jsonl').open('a')as stream:
            stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(),phase=phase,**observation),allow_nan=False)+'\n')
            stream.flush()
        if observation['stop']:
            row.update(stop_observed=True,error='Physical memory allowance or host reserve exhausted')
            write_json(directory/'physical-memory-guard.json',row)
            raise RuntimeError(row['error'])
    inspect()  # Admission before any new child; a failure starts no experiment.
    def monitor():
        while not stop.wait(.5):
            try:inspect()
            except BaseException as exc:
                row.update(stop_observed=True,error=type(exc).__name__+': '+str(exc))
                try:write_json(directory/'physical-memory-guard.json',row)
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:return run(argv,**options)
    finally:
        stop.set();thread.join(timeout=2)
        write_json(directory/'physical-memory-summary.json',state)


def outer_namespace():
    namespace=_resource_main(frontend,_clone(frontend),inner=False)
    namespace['MODULE']=MODULE
    state=dict(variant=VARIANT,default_trace_AS_gib=12,maximum_trace_AS_gib=24,parse_AS_gib=4,
        RSS_gib=14,host_available_reserve_gib=1.25,phases=[],original_modules_modified=False,
        original_arithmetic_or_parameters_changed=False,automatic_retries=0)
    run=namespace['run_process'];terminal=namespace['_terminal'];snapshot=namespace['_snapshot']
    def monitored(argv,**options):return guarded_run(argv,state=state,run=run,**options)
    def persist(out,result):
        result['resource_adaptation']=state
        result['limits'].update(trace_virtual_AS_max_gib=24,physical_subtree_RSS_gib=14,minimum_available_gib=1.25,
            memory_limit_semantics='Per-process RLIMIT_AS with separate sampled physical subtree/host reserve guard')
        result['metrics']['resource_adaptation']=state
        if any(row['stop_observed']for row in state['phases']):
            result.update(status='stopped_on_error',resource_monitor_error='Physical-memory adapter stopped this attempt')
        return terminal(out,result)
    def sources(locations,out):
        records=snapshot(locations,out);target=Path(out)/'source/resource-adapter.py'
        shutil.copyfile(__file__,target)
        records.append(dict(source=str(Path(__file__).resolve()),snapshot=str(target),size_bytes=target.stat().st_size))
        write_json(Path(out)/'resource-adaptation.json',state)
        return records
    namespace.update(run_process=monitored,_terminal=persist,_snapshot=sources)
    return namespace


def main(argv=None):
    values=list(sys.argv[1:]if argv is None else argv)
    return (worker_namespace()if'--phase'in values else outer_namespace())['main'](values)


if __name__=='__main__':raise SystemExit(main())
