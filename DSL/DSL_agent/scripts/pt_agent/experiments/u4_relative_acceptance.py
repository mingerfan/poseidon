"""Finite U4 §11.8 costs and two-mode native acceptance of frozen programs."""

import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from unittest.mock import patch

from ..paths import REPO,RESULTS_DIR
from .u4_point_diagnosis import write


def source(args):
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    directory=getattr(args,'model_dir',None)
    info=ModelInfo.load(directory if directory is not None else args.prepared/'model')
    text=args.program.read_text()
    program=parse(text)
    if getattr(args,'prompt_version',None) in ('r8-v8','r8-v9','r8-v10'):
        from ..dsl.interface import require_interface
        require_interface(program,info)
    return info,text,program


def compare_management(args, recorded, profile, current, *, name):
    """Compare all Function fields against the retained pre-memory-fix source."""
    import importlib.util
    from ..manage.greedy import timed_stage
    from ..tests._u4_legacy_manage import compare_functions
    spec=importlib.util.spec_from_file_location('pt_agent.manage._memory_before',args.baseline_manager)
    before=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    with timed_stage('before_function_reference'):
        previous=before.manage(recorded,profile,name=name)
    comparison=compare_functions(previous,current)
    comparison['source']=str(args.baseline_manager)
    write(args.out/'FUNCTION_COMPARISON.json',comparison)
    if not comparison['equal']:raise RuntimeError('Memory correction changed the complete managed Function')
    del previous,before
    gc.collect()
    return comparison


def function_compare(args):
    """The three retained M3 programs; only a small member enables tracemalloc."""
    from .. import toolchain
    from ..dsl.parser import parse
    from ..lower import lower
    from ..manage import RecordingBackend,manage,check
    from ..manage.greedy import timed_stage
    from .u4_managed_payloads import shared_preprocess
    row=json.loads(args.case_report.read_text())['case']
    profile=toolchain.management_profile()
    toolchain.setup_env()
    with timed_stage('model_read'):
        if row.get('info'):
            from ..models.info import ModelInfo
            graph=ModelInfo.load(Path(row['info'])).graph
        else:
            from ..models.zoo import get
            graph=get(row['model'],args.models_root).import_graph()
        program=parse(Path(row['source']).read_text())
    with timed_stage('recording'):
        recorded=RecordingBackend(program.header.slots,locations={op.id:op for op in program.ops})
        lower(program,graph.weights,recorded)
        del graph
        gc.collect()
    with timed_stage('shared_preprocess'):
        prepared,function,preprocess=shared_preprocess(recorded,args.out/'shared',profile,
                                                      name=program.name,compressed=True)
        del recorded,function
        gc.collect()
    with timed_stage('management'):
        current=manage(prepared,profile,name=program.name)
    comparison=compare_management(args,prepared,profile,current,name=program.name)
    check(current)
    return dict(complete=True,case=row,program=row['source'],preprocess=preprocess,
                function_comparison=comparison,diagnostic=args.diagnostic,
                pruned=toolchain.prune(args.out/'shared'))


def plan_proof(args):
    """Independent retained comparator and tuple-state equivalence on real IR."""
    import importlib.util
    from itertools import zip_longest
    from ..manage.greedy import timed_stage
    from .u4_bert_managed import compare_saved,states,_state_tuples
    spec=importlib.util.spec_from_file_location('pt_agent.experiments._plan_before',args.baseline_comparator)
    previous=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    with timed_stage('reference_plan_comparator'):
        before=previous.compare_saved(args.saved_plan,args.candidate_plan)
    with timed_stage('compact_plan_comparator'):
        after=compare_saved(args.saved_plan,args.candidate_plan)
    if before!=after or not after['equal']:
        raise RuntimeError('Compact plan comparator changed its result fields or BERT plan differs')
    with timed_stage('real_plan_state_representation'):
        plan=json.loads(args.candidate_plan.read_text())
        keys=('kind','level','scale_log2','components','ntt')
        count=0
        for count,(old,new) in enumerate(zip_longest(states(plan),_state_tuples(plan)),1):
            expected=(old['op'],tuple(tuple(v[k] for k in keys) for v in old['inputs']),
                      tuple(old['output'][k] for k in keys) if old['output'] is not None else None,old['attrs'])
            if expected!=new:raise RuntimeError('Compact value/operation representation differs at '+str(count-1))
    return dict(complete=True,before=before,after=after,result_fields_equal=True,state_records=count,
                baseline_comparator=str(args.baseline_comparator))


def cost(args):
    from .. import cost as costs
    info,_,program=source(args)
    placement=[];original=costs.depth_dp
    def observed(*values,**kwargs):
        result=original(*values,**kwargs);placement.append(result);return result
    with patch.object(costs,'depth_dp',side_effect=observed):
        predicted=costs.estimate(program,info.graph.weights)
    if len(placement)!=1:raise RuntimeError('Expected one prediction placement')
    result=dict(complete=True,program=str(args.program),cost=predicted,cuts=placement[0].cuts)
    if args.old_cost is not None:
        saved=json.loads(args.old_cost.read_text())
        previous=saved['cost']
        result['old_cost']=str(args.old_cost)
        result['old_new']={key:dict(before=previous[key],after=predicted[key]) for key in
                           ('boots_pred','max_depth','latency_pred_us','barriers','const_bytes')}
    return result


def coverage(args):
    from ..data.samples import StoredSplits
    from ..transformer_validation import points,execute
    from .u4_point_diagnosis import distribution
    info,_,program=source(args)
    arrays,_=StoredSplits(args.prepared/'model/samples',info).get('heldout',20)
    descriptors=points(program)
    result=dict(complete=False,program=str(args.program),split='heldout',samples=len(next(iter(arrays.values()))),
                affects_rule=False,stages=[])
    for exact in (True,False):
        rows=[];stage=dict(engine='tensor',prefix='exact' if exact else 'all_approx',points=rows)
        result['stages'].append(stage)
        def observed(name,value,configuration):
            point=descriptors[len(rows)]
            rows.append(dict(role=point.role or 'act',line=point.op.lineno,key=point.key,
                             fit=point.attrs.get('fit','absolute'),**distribution(value,point.attrs['domain'])))
            write(args.out/'REPORT.json',result)
        execute(program,info.graph.weights,arrays,engine='tensor',exact=exact,observer=observed)
        if len(rows)!=len(descriptors):raise RuntimeError('Point coverage inventory differs')
    write(args.out/'REPORT.json',result)
    result['outside']=sum(p['outside'] for s in result['stages'] for p in s['points'])
    if result['outside']:raise RuntimeError('U4 §11.5 held-out point domain coverage failed')
    result['complete']=True
    return result


def trace_environment():
    from .. import toolchain
    environment=toolchain.package_trace_env()
    environment['PYTHONPATH']=str(REPO/'DSL/DSL_agent/scripts')+os.pathsep+environment['PYTHONPATH']
    return environment


def native_entry(package,weights,directory,mode,entry,output,*,input_path=None):
    from .. import toolchain
    from hecate_python_env import VENV
    command=[str(VENV/'bin/python'),'-B','-m','pt_agent.experiments.u4_native_continuation',
             '--package',str(package),'--weights',str(weights),'--mode',mode,'--out',str(directory),
             '--hecate-opt',str(toolchain.host_paths()['hecate_opt']),'--entry',entry,'--directory',str(output)]
    if input_path is not None:command+=['--input',str(input_path)]
    result=toolchain.trusted_run(directory,command,directory/(entry+'.log'),env_extra=trace_environment())
    if result['exit']:raise RuntimeError(mode+' '+entry+' failed: '+str(directory/(entry+'.log')))
    return result


def compiler(args):
    from .. import toolchain
    from ..export import export_package
    toolchain.setup_env()
    info,text,program=source(args)
    numerical=json.loads(args.evaluation.read_text())
    predicted=json.loads(args.cost_report.read_text())
    if not numerical['complete'] or numerical['program']!=str(args.program):
        raise ValueError('Native acceptance requires the same completed numerical program')
    package=export_package(text,program,info.graph.weights,args.out/'compiler',
                           **(dict(model_info=info,prompt_version=args.prompt_version)
                              if getattr(args,'prompt_version',None) in ('r8-v8','r8-v9','r8-v10') else {}))
    weights=args.out/'compiler/weights'/(program.name+'.npz')
    if predicted['cost']['const_bytes']>1.5*1024**3:
        from ..run import _server_job
        job=_server_job(args.out/'compiler',package,weights,mode='compiler')
        return dict(complete=False,server_job=str(job),reason='constant bytes exceed unchanged local allowance')
    del info;gc.collect()
    directory=args.out/'compiler'
    result=dict(complete=False,program=str(args.program),package=str(package),weights=str(weights),
                evaluation=str(args.evaluation),cost_report=str(args.cost_report))
    result['trace']=native_entry(package,weights,directory,'compiler','trace',directory/'trace')
    write(args.out/'REPORT.json',result)
    bytecode=directory/'trace/trace_model.mlirbc'
    if not bytecode.is_file():raise ValueError('Expected frontend bytecode')
    result['bytecode']=str(bytecode);result['bytecode_bytes']=bytecode.stat().st_size
    result['compile']=native_entry(package,weights,directory,'compiler','compile',directory/'compiled',input_path=bytecode)
    suffix='lowered._hecate_'+program.name+'.runtime-plan.json'
    result['runtime_plan']=str(directory/'compiled'/suffix)
    result['stats']=toolchain.plan_stats(directory/'compiled',program.name)
    log=(directory/'compile.log').read_text()
    matches=re.findall(r'Depth-DP bootstrap boundaries:\s*(\d+)',log)
    if len(matches)!=1:raise RuntimeError('Expected one compiler boundary count')
    result['boundaries']=int(matches[0])
    result['boots_equal']=result['stats']['boots']==predicted['cost']['boots_pred']
    result['boundaries_equal']=result['boundaries']==predicted['cost']['barriers']
    write(args.out/'REPORT.json',result)
    if not result['boots_equal'] or not result['boundaries_equal']:
        raise RuntimeError('Compiler bootstrap count/boundaries differ from prediction')
    result['pruned_compiled']=toolchain.prune(directory/'compiled',keep=(suffix.removesuffix('.runtime-plan.json')+'.bundle',))
    result['complete']=True
    return result


def preflight(args):
    from .. import toolchain
    from ..lower import lower
    from ..manage import RecordingBackend,manage,streams
    from ..manage.greedy import timed_stage
    from ..manage.earth_emit import constant_table,constant_sizes
    from .u4_managed_payloads import shared_preprocess
    from .u4_managed_ir_size import text_size
    from .u4_cse_diagnosis import frontend_summary
    toolchain.setup_env()
    with timed_stage('model_read'):
        info,_,program=source(args)
    compiler_report=json.loads(args.compiler_report.read_text())
    predicted=json.loads(args.cost_report.read_text())
    if not compiler_report['complete']:raise ValueError('Preflight requires completed compiler acceptance')
    profile=toolchain.management_profile()
    with timed_stage('recording'):
        recorded=RecordingBackend(program.header.slots,locations={op.id:op for op in program.ops})
        lower(program,info.graph.weights,recorded)
        del info
        gc.collect()
    with timed_stage('shared_preprocess'):
        if args.shared is not None:
            from ..manage.mlir_read import read,graph_backend,RAW_OPS
            constants={v.name:v.data for v in recorded._constants.values()}
            del recorded
            gc.collect()
            structure=streams.resolve(args.shared/'preprocessed.structure.mlir')
            function=read(structure,profile,allowed=RAW_OPS,constants=constants)
            recovered=graph_backend(function,program.header.slots)
            preprocessed=dict(reused=str(args.shared),constant_ops_checked=sum(n.kind=='constant' for n in function.nodes))
            del constants
        else:
            recovered,function,preprocessed=shared_preprocess(recorded,args.out/'shared',profile,name=program.name,compressed=True)
            del recorded
        gc.collect()
    result=dict(complete=False,program=str(args.program),compiler_report=str(args.compiler_report),
                shared=str(args.shared or args.out/'shared'),preprocess=preprocessed)
    with timed_stage('frontend_summary'):
        front=frontend_summary(function,program.header.slots,program)
    result['preprocessed_placement']=dict(boots=front['boots'],cuts=front['cuts'],boundaries=front['boundaries'],
        boots_equal=front['boots']==predicted['cost']['boots_pred'],cuts_equal=front['cuts']==predicted['cuts'])
    write(args.out/'REPORT.json',result)
    if not result['preprocessed_placement']['boots_equal'] or not result['preprocessed_placement']['cuts_equal']:
        raise RuntimeError('Shared frontend placement differs from prediction')
    del function,front
    gc.collect()
    with timed_stage('management'):
        managed=manage(recovered,profile,name=program.name)
    if args.baseline_manager is not None:
        result['function_comparison']=compare_management(args,recovered,profile,managed,name=program.name)
    del recovered
    gc.collect()
    with timed_stage('size_preflight'):
        constant_table(managed)
        size=text_size(managed,program.header.slots)
    free=streams.admit(args.out,extra_bytes=size if size>streams.TEMP_INPUT_LIMIT else 0)
    result.update(complete=True,uncompressed_bytes=size,route='stdin' if size<=streams.TEMP_INPUT_LIMIT else 'temporary-file',
                  free_bytes=free,managed_boots=sum(n.kind=='bootstrap' for n in managed.nodes),managed_cuts=managed.cuts,
                  compiler_boundaries=compiler_report['boundaries'],constant_sizes=constant_sizes(managed))
    if result['managed_boots']!=predicted['cost']['boots_pred'] or managed.cuts!=predicted['cuts']:
        raise RuntimeError('Managed preflight bootstrap count/cuts differ')
    # Bytecode and the intermediate compiler text have both been consumed.
    if not args.diagnostic and args.shared is None:
        result['compiler_trace_pruned']=toolchain.prune(Path(compiler_report['bytecode']).parent)
    return result


def managed(args):
    from .. import toolchain
    from ..export import export_package
    from ..lower import lower
    from ..manage import RecordingBackend,manage,emit,check,streams
    from ..manage.greedy import timed_stage
    from ..manage.mlir_read import read,graph_backend,RAW_OPS
    from .u4_bert_managed import compare_complete,compare_saved
    toolchain.setup_env()
    sizing=json.loads(args.size_report.read_text());predicted=json.loads(args.cost_report.read_text())
    original=json.loads(args.compiler_report.read_text())
    if not sizing['complete']:raise ValueError('Managed acceptance requires a completed IR size preflight')
    size=sizing['uncompressed_bytes'];streams.admit(args.out,extra_bytes=size if size>streams.TEMP_INPUT_LIMIT else 0)
    with timed_stage('model_read'):
        info,text,program=source(args)
    directory=args.out/'managed'
    with timed_stage('package_export'):
        package=export_package(text,program,info.graph.weights,directory,
                               **(dict(model_info=info,prompt_version=args.prompt_version)
                                  if getattr(args,'prompt_version',None) in ('r8-v8','r8-v9','r8-v10') else {}))
        weights=directory/'weights'/(program.name+'.npz')
    profile=toolchain.management_profile()
    with timed_stage('recording'):
        recorded=RecordingBackend(program.header.slots,locations={op.id:op for op in program.ops})
        lower(program,info.graph.weights,recorded)
        constants={v.name:v.data for v in recorded._constants.values()}
        del recorded,info
        gc.collect()
    with timed_stage('shared_read'):
        structure=streams.resolve(Path(sizing['shared'])/'preprocessed.structure.mlir')
        function=read(structure,profile,allowed=RAW_OPS,constants=constants)
        prepared=graph_backend(function,program.header.slots)
        del constants,function
        gc.collect()
    with timed_stage('management'):
        function=manage(prepared,profile,name=program.name)
    del prepared
    gc.collect()
    check(function)
    metrics={};ir=emit(function,directory/'managed.mlir',metrics=metrics,compressed=True)
    actual=streams.uncompressed_size(ir)
    result=dict(complete=False,program=str(args.program),size_report=str(args.size_report),
        package=str(package),weights=str(weights),compiler_reference_package=original['package'],
        managed_ir=dict(uncompressed_bytes=actual,preflight_bytes=size,equal=actual==size,emit=metrics),
        managed_boots=sum(n.kind=='bootstrap' for n in function.nodes),managed_cuts=function.cuts)
    write(args.out/'REPORT.json',result)
    if actual!=size:raise RuntimeError('Managed IR size differs from preflight')
    if result['managed_boots']!=predicted['cost']['boots_pred'] or function.cuts!=predicted['cuts']:
        raise RuntimeError('Managed bootstrap count/cuts differ from prediction')
    del function;gc.collect()
    with timed_stage('native_compile'):
        result['compile']=native_entry(package,weights,directory,'managed','compile',directory/'compiled',input_path=ir)
    suffix='lowered._hecate_'+program.name+'.runtime-plan.json'
    plan=directory/'compiled'/suffix
    with timed_stage('runtime_plan_stats'):
        result['stats']=toolchain.plan_stats(directory/'compiled',program.name)
    result['boots_equal']=result['stats']['boots']==original['stats']['boots']==predicted['cost']['boots_pred']
    result['boundaries_equal']=len(result['managed_cuts'])==original['boundaries']==predicted['cost']['barriers']
    with timed_stage('runtime_plan_compare'):
        result['runtime_plan']=compare_complete(Path(original['runtime_plan']),plan)
    result['runtime_plan_path']=str(plan)
    if args.saved_plan is not None:
        with timed_stage('saved_runtime_plan_compare'):
            result['saved_runtime_plan']=compare_saved(args.saved_plan,plan)
        if not result['saved_runtime_plan']['equal']:
            raise RuntimeError('BERT complete RuntimePlan differs from the retained section 11.8 result')
    write(args.out/'REPORT.json',result)
    if not result['boots_equal'] or not result['boundaries_equal'] or not result['runtime_plan']['equal']:
        raise RuntimeError('Three-way placement or full RuntimePlan comparison differs')
    result['pruned']={str(folder):toolchain.prune(folder) for folder in
                     (directory, directory/'compiled', Path(original['runtime_plan']).parent)}
    if not sizing.get('preprocess',{}).get('reused'):
        result['pruned_shared']=toolchain.prune(Path(sizing['shared']))
    result['complete']=True
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('coverage','cost','compiler','preflight','managed','function-compare','plan-proof'),required=True)
    parser.add_argument('--model',choices=('bert','qwen','resnet20'),required=True)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--diagnostic',action='store_true',help='U4 §7.10 diagnostic-only tracemalloc snapshots')
    path_keys=('prepared','program','evaluation','cost_report','old_cost','compiler_report','size_report',
               'baseline_manager','case_report','models_root','shared','saved_plan','candidate_plan','baseline_comparator')
    for key in ('out',*path_keys):
        parser.add_argument('--'+key.replace('_','-'),type=Path)
    args=parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        from ..manage.greedy import timing_log
        if args.diagnostic:
            if args.model in ('bert','qwen'):
                raise ValueError('U4 section 7.10.5 restricts tracemalloc to a small regression program')
            import tracemalloc
            tracemalloc.start()
        try:
            with timing_log(args.out/'memory-stages.json'):
                report=dict({'coverage':coverage,'cost':cost,'compiler':compiler,'preflight':preflight,
                             'managed':managed,'function-compare':function_compare,'plan-proof':plan_proof}[args.part](args))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            path=args.out/'REPORT.json';report=json.loads(path.read_text()) if path.exists() else {}
            report.update(complete=False,error=f'{type(exc).__name__}: {exc}');write(args.out/'STOP.json',report)
        write(args.out/'REPORT.json',report)
        return 0 if report['complete'] else 1
    stamp=datetime.now(timezone.utc);run_id='u4-'+args.model+'-relative-'+args.part+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id;args.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
             '-m',__spec__.name,'--inside','--part',args.part,'--model',args.model,'--out',str(args.out)]
    for key in path_keys:
        if getattr(args,key) is not None:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
    if args.diagnostic:command.append('--diagnostic')
    print('RUN='+str(args.out),flush=True);write(args.out/'argv.json',command);started=time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3630)
    path=args.out/'REPORT.json';report=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    report.update(round='U4',section='7.10',diagnostic=args.diagnostic,
                  memory_stages=str(args.out/'memory-stages.json'),run_id=run_id,model=args.model,part=args.part,git_rev=rev,
                  exit=status.returncode,seconds=time.monotonic()-started,llm_calls=0,report_images=0,ckks_calls=0)
    write(path,report)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model=args.model,mode='relative-'+args.part,git_rev=rev,
        metrics=dict(complete=report['complete'],llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=report['complete'],exit=status.returncode)),flush=True)
    return status.returncode


if __name__=='__main__':raise SystemExit(main())
