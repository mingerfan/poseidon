"""U4 §7.6/§7.7 BERT managed acceptance after the finite S1 regression."""

import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..paths import REPO,RESULTS_DIR
from .u4_managed_payloads import write


def states(plan):
    """Yield the exact S1 step-state records without retaining a second list."""
    values = {v['id']:v for v in plan['values']}
    def state(name):
        value = values[name]
        return {key:value.get(key) for key in ('kind','level','scale_log2','components','ntt')}
    for phase in ('initialization','execution','finalization'):
        for step in plan[phase]:
            yield dict(op=step.get('op') or step['kind'],inputs=[state(v) for v in step.get('inputs',[])],
                       output=state(step['output']) if 'output' in step else None,attrs=step.get('attrs',{}))


def _state_tuples(plan):
    """The exact five S1 value fields, shared between equal immutable states."""
    fields=('kind','level','scale_log2','components','ntt')
    shared={}
    values={}
    for value in plan['values']:
        state=tuple(value.get(key) for key in fields)
        values[value['id']]=shared.setdefault(state,state)
    for phase in ('initialization','execution','finalization'):
        for step in plan[phase]:
            yield (step.get('op') or step['kind'],tuple(values[v] for v in step.get('inputs',())),
                   values[step['output']] if 'output' in step else None,step.get('attrs',{}))


def _reference(path):
    """Release the first full JSON before loading the second; keep its states."""
    from .s0_pipeline_spike import stats
    plan=json.loads(path.read_text())
    summary=stats(plan)
    initialization=plan['initialization']
    records=list(_state_tuples(plan))
    del plan
    gc.collect()
    return summary,initialization,records


def compare_saved(saved,candidate):
    from itertools import zip_longest
    from .s0_pipeline_spike import stats
    original,initialization,records=_reference(saved)
    b=json.loads(candidate.read_text())
    current=stats(b)
    first = next((i for i,(x,y) in enumerate(zip_longest(records,_state_tuples(b))) if x!=y),None)
    missing=object()
    same_payloads = all(x==y for x,y in zip_longest(
        (s.get('payload') for s in initialization),(s.get('payload') for s in b['initialization']),
        fillvalue=missing))
    return dict(saved=str(saved),candidate=str(candidate),stats_equal=original==current,
                step_states_equal=first is None,first_step_difference=first,
                payload_descriptors_equal=same_payloads,equal=original==current and first is None and same_payloads)


def compare_complete(left,right):
    from itertools import zip_longest
    import numpy as np
    from .s0_pipeline_spike import stats,constants,bundle
    original,initialization,records=_reference(left)
    b=json.loads(right.read_text())
    current=stats(b)
    differences = [i for i,(x,y) in enumerate(zip_longest(records,_state_tuples(b))) if x!=y]
    equal_constants = [x is not None and y is not None and bool(np.array_equal(x,y))
        for x,y in zip_longest(constants(dict(initialization=initialization),left),constants(b,right))]
    blobs = []
    for p in (left,right):
        manifest = bundle(p)/'manifest.json'
        blobs.append(len(json.loads(manifest.read_text())['blobs']) if manifest.exists() else 0)
    return dict(original=original,candidate=current,step_states_equal=not differences,
        differing_step_indices=differences,constants_equal=all(equal_constants),constant_comparisons=equal_constants,
        bundle_constant_counts=blobs,stats_equal=original==current,old_plan=str(left),new_plan=str(right),
        equal=original==current and not differences and all(equal_constants) and blobs[0]==blobs[1])


def inside(args):
    import gc
    import resource
    import traceback
    from .. import toolchain
    from ..dsl.parser import parse
    from ..models.info import ModelInfo
    from ..export import export_package
    from ..manage import streams
    from ..manage.streams import resolve
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    result = dict(round='U4',section='7.6/7.7.4',model='bert',complete=False,llm_calls=0,
                  report_images=0,ckks_calls=0,program=str(args.program),modes={})
    def save():write(args.out/'REPORT.json',result)
    try:
        if args.ir_size_report is not None:
            sizing=json.loads(args.ir_size_report.read_text())
            if not sizing['complete'] or not all(c['equal'] for c in sizing['controls']):
                raise ValueError('BERT managed requires a completed IR size preflight')
            size=sizing['uncompressed_bytes']
            free=streams.admit(args.out,extra_bytes=size if size>streams.TEMP_INPUT_LIMIT else 0)
            result['ir_size_preflight']=dict(report=str(args.ir_size_report),uncompressed_bytes=size,
                route=sizing['route'],free_before_bytes=free)
        regression = json.loads(args.regression.read_text())
        if not regression['complete'] or regression['local_passed']!=19 or regression['server_pending']!=1:
            raise ValueError('§7.6.4 requires the completed 19 local cases and retained 1 server-pending case')
        result['regression'] = str(args.regression)
        numerical = json.loads(args.evaluation.read_text())
        if not numerical['complete'] or numerical['program']!=str(args.program):
            raise ValueError('BERT managed requires the same completed numerical program')
        info = ModelInfo.load(args.prepared/'model')
        text = args.program.read_text();program = parse(text)
        package = export_package(text,program,info.graph.weights,args.out/'managed')
        weights = args.out/'managed/weights'/(program.name+'.npz')
        del info
        gc.collect()
        result.update(package=str(package),weights=str(weights))
        toolchain.setup_env()
        from hecate_python_env import VENV
        directory = args.out/'managed'
        environment = toolchain.package_trace_env()
        environment['PYTHONPATH'] = str(REPO/'DSL/DSL_agent/scripts')+os.pathsep+environment['PYTHONPATH']
        base = [str(VENV/'bin/python'),'-B','-m','pt_agent.experiments.u4_native_continuation',
            '--package',str(package),'--weights',str(weights),'--mode','managed','--out',str(directory),
            '--hecate-opt',str(toolchain.host_paths()['hecate_opt'])]
        result['modes']['managed'] = record = {}
        record['trace'] = toolchain.trusted_run(directory,base+['--entry','trace','--directory',str(directory/'trace')],
            directory/'trace.log',env_extra=environment)
        save()
        if record['trace']['exit']:
            raise RuntimeError('BERT managed trace failed')
        managed_source=resolve(directory/'trace/managed.mlir')
        actual_size=streams.uncompressed_size(managed_source) if managed_source.suffix=='.gz' else managed_source.stat().st_size
        record['managed_ir_size']=dict(uncompressed_bytes=actual_size,
            route='stdin' if actual_size<=streams.TEMP_INPUT_LIMIT else 'temporary-file',
            estimate_equal=args.ir_size_report is None or actual_size==sizing['uncompressed_bytes'])
        streams.admit(directory,extra_bytes=actual_size if actual_size>streams.TEMP_INPUT_LIMIT else 0)
        save()
        management = json.loads((directory/'trace/management.json').read_text())
        expected = json.loads(args.diagnosis.read_text())['graphs']['frontend']
        record['preprocessed_depth_dp'] = dict(boots=management['boots'],cuts=management['cuts'],
            boots_equal=management['boots']==15399,cuts_equal=management['cuts']==expected['cuts'])
        if not record['preprocessed_depth_dp']['boots_equal'] or not record['preprocessed_depth_dp']['cuts_equal']:
            from itertools import zip_longest
            record['preprocessed_depth_dp']['first_boundary_difference'] = next(
                (dict(index=i,expected=a,actual=b) for i,(a,b) in enumerate(
                    zip_longest(expected['cuts'],management['cuts'])) if a!=b),None)
            save();raise RuntimeError('BERT managed preprocessed depth-dp differs from compiler')
        record['constant_sizes'] = json.loads((directory/'trace/constant_sizes.json').read_text())
        record['compile'] = toolchain.trusted_run(directory,
            base+['--entry','compile','--directory',str(directory/'compiled'),
                  '--input',str(resolve(directory/'trace/managed.mlir'))],directory/'compile.log',env_extra=environment)
        save()
        if record['compile']['exit']:
            raise RuntimeError('BERT managed compiler tail failed')
        name = program.name;suffix='lowered._hecate_'+name+'.runtime-plan.json'
        managed_plan = directory/'compiled'/suffix
        record['stats'] = toolchain.plan_stats(directory/'compiled',name)
        if record['stats']['boots']!=15399:
            raise RuntimeError('BERT managed compiled bootstrap count differs')
        record['pruned_trace'] = toolchain.prune(directory/'trace')
        record['pruned_compiled'] = toolchain.prune(directory/'compiled',keep=(suffix.removesuffix('.runtime-plan.json')+'.bundle',))
        save()
        # The §7.8 checkpoint reuses its saved plan. The following independent
        # §7.6 payload comparison needs the bundle that prior cleanup consumed;
        # recover it once from the retained original bytecode, without retracing.
        compiler = args.out/'compiler-reference';compiler.mkdir()
        config = package/'config/compiler-config.json'
        substitutions = dict(hecate_opt=str(toolchain.host_paths()['hecate_opt']),input=str(args.bytecode),
                             package=str(package),output_dir=str(compiler))
        command = [arg.format(**substitutions) for arg in json.loads((package/'config/compile_argv.json').read_text())]
        reference = result['modes']['compiler-reference'] = dict(reason='recover pruned constant bundle for full S1 payload comparison',
            input_bytecode=str(args.bytecode),new_trace_calls=0)
        reference['compile'] = toolchain.trusted_run(compiler,command,compiler/'compile.log',env_extra=environment)
        save()
        if reference['compile']['exit']:
            raise RuntimeError('BERT comparison-reference compile failed')
        reference_plan = compiler/suffix
        result['saved_compiler_comparison'] = compare_saved(args.compiler_plan,reference_plan)
        save()
        if not result['saved_compiler_comparison']['equal']:
            raise RuntimeError('Recovered reference differs from retained compiler RuntimePlan')
        reference['pruned'] = toolchain.prune(compiler,keep=(suffix.removesuffix('.runtime-plan.json')+'.bundle',))
        result['runtime_plan'] = compare_complete(reference_plan,managed_plan)
        save()
        if not result['runtime_plan']['equal']:
            raise RuntimeError('BERT compiler/managed full RuntimePlan comparison differs')
        result['final_prune'] = {str(p):toolchain.prune(p) for p in (compiler,directory/'compiled')}
        result['complete'] = True
        save()
    except Exception as exc:
        traceback.print_exc()
        result.update(complete=False,error=f'{type(exc).__name__}: {exc}')
        save();write(args.out/'STOP.json',result)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--launch',action='store_true')
    for name in ('out','prepared','program','evaluation','regression','diagnosis','compiler-plan','bytecode','ir-size-report'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    if args.inside:return inside(args)
    fields = ('out','prepared','program','evaluation','regression','diagnosis','compiler_plan','bytecode','ir_size_report')
    if args.launch:
        from ..tools.nixrun import main as nix_main
        sys.argv = [sys.argv[0],'--seconds','3600','--','-m',__spec__.name,'--inside']
        for key in fields:
            if getattr(args,key) is not None:sys.argv += ['--'+key.replace('_','-'),str(getattr(args,key))]
        try:return nix_main()
        except subprocess.CalledProcessError as exc:
            stderr=exc.stderr or '';(args.out/'nix.stderr.txt').write_text(stderr)
            write(args.out/'STOP.json',dict(complete=False,stage='nix-launch',exit=exc.returncode,stderr=stderr))
            sys.stderr.write(stderr);return exc.returncode
    stamp=datetime.now(timezone.utc);run_id='u4-bert-managed76-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id;args.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B','-m',__spec__.name,'--launch']
    for key in fields:
        if getattr(args,key) is not None:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
    write(args.out/'argv.json',command);print('RUN='+str(args.out),flush=True);started=time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status=subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT)
    path=args.out/'REPORT.json'
    result=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    if (args.out/'STOP.json').exists() and not path.exists():result.update(json.loads((args.out/'STOP.json').read_text()))
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    result.update(run_id=run_id,git_rev=rev,exit=status.returncode,seconds=time.monotonic()-started);write(path,result)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model='bert',mode='managed-shared-acceptance',git_rev=rev,
        metrics=dict(complete=result['complete'],llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=result['complete'],exit=status.returncode)),flush=True)
    return status.returncode


if __name__=='__main__':raise SystemExit(main())
