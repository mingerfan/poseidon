"""Actual Count/plan/replay and fixed heldout work; explicit paths only."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

from . import fig2_uniform_degree as outer
from .r10_paper_metrics import StageRecorder, write_json


def initial_result(model):
    return dict(round=outer.ROUND,mode=outer.MODE,model=model,status='running',error=None,
        source_checked=False,metadata_template_checked=False,candidate_frozen=False,variants=[],active_degree=None,
        model_restore_attempts=0,model_restore_completed=0,heldout_load_attempts=0,heldout_load_completed=0,
        reference_cache_load_attempts=0,reference_cache_load_completed=0,
        compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None,
        security_acceptance=None,**dict.fromkeys(outer.ZERO_FIELDS,0))


def new_variant(degree):
    result=dict(degree=degree,status='running',error=None,completed_samples=0,math_passed=False,
        all_finite=None,heldout_acceptance=None,task_passed=None,domain_passed=None,
        validation_metrics=None,max_u=None,domain_rows=None,planned_bootstraps=None,
        independent_native_projection=None)
    for key in outer.PAIRS:
        result[key+'_attempts']=0;result[key+'_completed']=0
    return result


def _checkpoint(args,result):
    if outer.artifact_bytes(args.work_dir.parent)+outer.RESERVE_BYTES>outer.CASE_BYTES:
        raise RuntimeError('The4GiB finite case allowance was exhausted')
    write_json(args.work_dir/'worker_result.json',result)


def run_variants(args,result,recorder,evaluate):
    for degree in outer.DEGREES:
        result['active_degree']=degree
        row=new_variant(degree);result['variants'].append(row)
        _checkpoint(args,result)
        try:
            with recorder.span('degree_variant',degree=degree):evaluate(row)
            if row['status'] not in ('accepted','quality_rejected'):
                raise RuntimeError('Variant lacks a completed quality decision')
        except BaseException as exc:
            row.update(status='error',error=type(exc).__name__+': '+str(exc)[:2500],
                validation_metrics=None,heldout_acceptance=None,task_passed=None,domain_passed=None)
            _checkpoint(args,result)
            raise
        _checkpoint(args,result)
    result['active_degree']=None


def domain_observer(program,sample_id,rows,np,maximum_u):
    """Reuse the original domain criterion with the actual degree in cfg."""
    acts=[op for op in program.ops if op.kind=='act'];seen=0
    def observer(name,data,cfg):
        nonlocal seen
        if seen>=len(acts):raise ValueError('Unexpected activation observation')
        op,row=acts[seen],rows[seen]
        expected=('cheb',op.attrs['deg'],*op.attrs['domain'])
        if name!=op.inputs[0] or cfg!=expected or row['op_id']!=op.id:
            raise ValueError('Activation observer order/configuration differs')
        if not isinstance(data,np.ndarray) or data.size==0 or not np.isfinite(data).all():
            raise ValueError('Nonfinite or empty activation input')
        low,high=float(data.min()),float(data.max())
        u=float(maximum_u(data,op.attrs['domain'],fn='silu',domain_source='calibrated'))
        if not math.isfinite(u):raise ValueError('Nonfinite activation coverage')
        if row['obs_min'] is None or low<row['obs_min']:row.update(obs_min=low,min_sample_id=sample_id)
        if row['obs_max'] is None or high>row['obs_max']:row.update(obs_max=high,max_sample_id=sample_id)
        if row['u_sample_id'] is None or u>row['max_u']:row.update(max_u=u,u_sample_id=sample_id)
        row['count']+=1;seen+=1
    def finish():
        if seen!=len(acts):raise ValueError('Missing original activation observations')
    return observer,finish


def _load_runtime(args,result,recorder):
    # native_metadata_commands is installed by main before these imports.
    from . import r17_margin_validation_worker as original
    original._initialize_torch()
    import numpy as np
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..tensor_sim import run_tensor
    from ..metrics import compare,passes
    from ..domain_criteria import maximum_u
    from ..plan import plan_program,verify_program
    from .r14_common_worker import check_program_signature
    result['model_restore_attempts']+=1;_checkpoint(args,result)
    with recorder.span('restore_original_saved_graph'):
        info=ModelInfo.load(args.model_info)
    result['model_restore_completed']+=1
    validation=original.validate_source(outer.read(args.r14_result),outer.read(args.r13_result),info,
        outer.read(args.calibration),outer.read(args.calibration_source))
    if not validation['passed']:raise ValueError('Original saved calibration contract failed: '+str(validation['errors']))
    for array in info.graph.weights.values():array.setflags(write=False)
    result['heldout_load_attempts']+=1;_checkpoint(args,result)
    with recorder.span('read_fixed_heldout_once'):
        arrays,labels,source=original._read_heldout(args,info)
    result['heldout_load_completed']+=1;result['sample_source']=source
    result['reference_cache_load_attempts']+=1;_checkpoint(args,result)
    with recorder.span('read_original_reference_cache'):
        with np.load(args.cached_outputs,allow_pickle=False) as saved:
            if set(saved.files)!={'reference','approximate','labels','indices'}:
                raise ValueError('Original reference cache keys differ')
            reference=saved['reference'];old_labels=saved['labels'];indices=saved['indices']
        if (reference.shape!=(1000,10) or reference.dtype!=np.dtype('float64')
                or not np.isfinite(reference).all() or not np.array_equal(labels,old_labels)
                or not np.array_equal(indices,np.arange(9000,10000,dtype=np.int64))):
            raise ValueError('Original heldout references/labels/indices differ')
        reference.setflags(write=False)
    result['reference_cache_load_completed']+=1
    return locals()


def _evaluate(args,result,row,recorder,runtime,base):
    np=runtime['np'];info=runtime['info'];original=runtime['original'];degree=row['degree']
    directory=args.work_dir/('cheb'+str(degree));directory.mkdir()
    text=outer.degree_variant(base,degree);(directory/'program.ptdsl').write_text(text)
    row['program']=str(directory/'program.ptdsl');row['ast_check']=outer.check_variant(base,text,degree)
    program=runtime['parse'](text)
    checked=runtime['check'](program,{name:value.shape for name,value in info.graph.weights.items()},graph=info.graph)
    signature=runtime['check_program_signature'](info,program)
    row.update(typecheck=dict(ok=checked.ok,diags=[asdict(item) for item in checked.diags]),signature=signature)
    if not checked.ok or not signature['passed']:raise ValueError('Original program type or signature failed')
    row['count_attempts']+=1;_checkpoint(args,result)
    with recorder.span('raw_Count',degree=degree):count=original._count_candidate(program,info.graph.weights)
    row['count_completed']+=1;row['raw_Count']=count
    row['plan_attempts']+=1;_checkpoint(args,result)
    with recorder.span('normalized_depth_dp_plan',degree=degree):
        plan,dag,placement=runtime['plan_program'](program,info.graph.weights,profile='qwen48',frontend_normalize=True)
        if plan['placement']!='depth-dp' or plan['candidates'] is not None:
            raise ValueError('Uniform experiment must use depth-DP without portfolio selection')
        row['normalized_DAG']=dag.stats();row['planned_bootstraps']=plan['boots']
        row['external_profile_objective_us']=placement.latency_us
        row['external_profile_objective_scope']='Original qwen48 external latency table; independent local native prediction remains null'
        write_json(directory/'refresh-plan.json',plan)
    row['plan_completed']+=1;row['replay_attempts']+=1;_checkpoint(args,result)
    with recorder.span('actual_Count_plan_replay',degree=degree):
        row['plan_replay']=runtime['verify_program'](program,info.graph.weights,plan,profile='qwen48')
    if row['plan_replay']['inner_boots']!=plan['boots']:raise ValueError('Plan Count replay differs')
    row['replay_completed']+=1
    del dag,placement,plan,count;gc.collect()
    rows=original._domain_rows(program);row['domain_rows']=rows
    actual=np.empty((1000,10),dtype=np.float64)
    with (directory/'samples.jsonl').open('w') as samples,(directory/'math.jsonl').open('w') as mathematics:
        for index in range(1000):
            sample_id='test:'+str(index+9000)
            image=runtime['arrays'][info.graph.inputs[0]][index:index+1]
            observer,finish=domain_observer(program,sample_id,rows,np,runtime['maximum_u'])
            row['tensor_attempts']+=1
            with recorder.span('tensor_program',degree=degree,sample_id=sample_id):
                logits=runtime['run_tensor'](program,info.graph.weights,image,cache=False,act_mode='program',observer=observer)
            row['tensor_completed']+=1;finish()
            if (logits.shape!=(1,10) or logits.dtype!=np.dtype('float64') or not np.isfinite(logits).all()):
                raise ValueError('Nonfinite or incompatible classifier logits')
            if index<3:
                row['mock_attempts']+=1
                with recorder.span('packed_Mock_program',degree=degree,sample_id=sample_id):
                    mock=original._packed_logits(program,info.graph.weights,image,act_mode='program')
                row['mock_completed']+=1;row['math_attempts']+=1
                check=original.check_math(logits,mock)
                mathematics.write(json.dumps(dict(sample_id=sample_id,Tensor=logits.tolist(),Mock=mock.tolist(),**check))+'\n');mathematics.flush()
                if not check['passed']:raise ValueError('Strict Tensor/Mock mathematics failed')
                row['math_completed']+=1
                if index==2:row['math_passed']=True
                del mock
            actual[index]=logits[0];row['completed_samples']+=1
            samples.write(json.dumps(dict(sample_id=sample_id,label=int(runtime['labels'][index]),reference=runtime['reference'][index].tolist(),observed=logits[0].tolist()))+'\n')
            if index%10==0:samples.flush();_checkpoint(args,result)
            del logits,observer,finish
    maximum=max(item['max_u'] for item in rows)
    covered=len(rows)==outer.MODEL_RUNS[args.model][2] and all(item['count']==1000 for item in rows)
    measured=runtime['compare'](dict(values=runtime['reference'],task=info.task),actual,runtime['labels'])
    row.update(validation_metrics=measured,task_passed=bool(runtime['passes'](measured,info.threshold)),
        domain_passed=bool(covered and maximum<=.90),domain_row_count=len(rows),max_u=maximum,
        all_finite=True,evaluation_completed=True)
    row['heldout_acceptance']=bool(row['math_passed'] and row['task_passed'] and row['domain_passed'])
    row['status']='accepted' if row['heldout_acceptance'] else 'quality_rejected'
    np.save(directory/'approximate.npy',actual,allow_pickle=False)
    write_json(directory/'result.json',row)


def run_model(args,recorder,result):
    source={key:getattr(args,key) for key in ('rule_result','r14_result','r13_result','model_info','calibration',
        'calibration_source','base_program','cached_outputs','records','metadata_template')}
    with recorder.span('source_gate'):
        result['source']=outer.source_gate(args.model,source);result['source_checked']=True
    runtime=_load_runtime(args,result,recorder);base=args.base_program.read_text()
    run_variants(args,result,recorder,lambda row:_evaluate(args,result,row,recorder,runtime,base))
    result['status']='ok'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=tuple(outer.MODEL_RUNS),required=True)
    for key in ('work-dir','rule-result','r14-result','r13-result','model-info','calibration','calibration-source',
                'base-program','cached-outputs','records','metadata-template'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args(argv);args.work_dir.mkdir(parents=True,exist_ok=True)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    result=initial_result(args.model);recorder=StageRecorder(args.work_dir/'events.jsonl')
    started=time.perf_counter_ns()
    def expire(signum,frame):raise TimeoutError('Uniform-degree worker1800-second allowance exhausted')
    old=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,outer.SECONDS)
    try:
        from .r29_local_compile_matrix import native_metadata_commands
        with recorder.span('read_recorded_compiler_argv'):
            template=outer.load_metadata_template(args.metadata_template)
            result['metadata_template_checked']=True
        with native_metadata_commands(template):
            run_model(args,recorder,result)
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc)[:2500])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        result['recorded_pre_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
        result['worker_ru_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(args.work_dir/'worker_result.json',result)
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
