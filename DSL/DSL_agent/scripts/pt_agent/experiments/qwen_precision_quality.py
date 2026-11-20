"""Training-prefix quality and independent packed arithmetic of a fixed Qwen rule.

The original six HF prefix states remain the reference.  Approximate outputs
are measured on those fixed states and never replace their original tokens.
No fitting, candidate selection, freezing, reporting split or crypto occurs.
"""

import argparse
import json
import os
from pathlib import Path
import resource

from .r10_paper_metrics import StageRecorder,write_json


def strict_math(reference,observed,np):
    a,b=np.asarray(reference),np.asarray(observed)
    if a.shape!=b.shape or a.dtype!=np.float64 or b.dtype!=np.float64:
        raise ValueError('Strict arithmetic requires matching original float64 shapes')
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    difference=np.abs(a-b)
    return dict(passed=bool(finite and np.all(difference<=1e-12+1e-12*np.abs(a))),
        finite=finite,elements=int(a.size),criterion='1e-12+1e-12*abs(reference)',
        max_absolute_difference=float(difference.max()) if finite else None,
        max_tolerance_ratio=float((difference/(1e-12+1e-12*np.abs(a))).max()) if finite else None)


def domain_record(value,attrs,np,mask=None):
    data=np.asarray(value)
    if data.dtype!=np.float64:
        raise ValueError('Approximation observations must retain float64')
    if mask is not None:
        data=data[np.asarray(mask,dtype=bool)]
    finite=bool(np.isfinite(data).all())
    lo,hi=attrs['domain']
    return dict(fn=attrs['fn'],method=attrs['method'],domain=[lo,hi],elements=int(data.size),
        all_finite=finite,observed_min=float(data.min()) if finite and data.size else None,
        observed_max=float(data.max()) if finite and data.size else None,
        contained=bool(finite and data.size and np.all((data>=lo)&(data<=hi))))


def source(args):
    count=json.loads(args.count_result.read_text());worker=count.get('worker') or {}
    if (count.get('status')!='ok' or count.get('model')!='Qwen2-0.5B'
            or worker.get('count_completed')!=1 or worker.get('diagnostic_typecheck_passed') is not True
            or worker.get('candidate_frozen') is not False
            or worker.get('decomposition_variant')!='unit-width-public-square-v2'
            or worker.get('newton_budget')!=7):
        raise ValueError('A completed, unfrozen original full-model precision Count is required')
    if (Path(count['source']['model_info']).resolve()!=args.model_info.resolve()
            or Path(count['source']['original_hf_reference']).resolve()!=args.reference.resolve()):
        raise ValueError('Original model and HF reference sources differ from Count')
    program=Path(worker['diagnostic_program'])
    if program.parent.resolve()!=(args.count_result.parent/'worker').resolve():
        raise ValueError('The fixed rule must remain in its original Count worker directory')
    calibration=json.loads(Path(count['source']['stable_calibration_result']).read_text())
    if (calibration.get('stable_bridge_validated') is not True
            or len(calibration.get('math_checks',[]))!=6
            or any(row.get('passed') is not True for row in calibration['math_checks'])
            or Path(calibration['source']['model_info']).resolve()!=args.model_info.resolve()
            or Path(calibration['source']['original_hf_reference']).resolve()!=args.reference.resolve()):
        raise ValueError('The preserved strict original six-state reference must pass')
    hf=json.loads(args.reference.read_text())
    trajectories=hf.get('trajectories',[])
    if (hf.get('status')!='ok' or hf.get('hf_forward_completed')!=6 or len(trajectories)!=3
            or any(not row['sample_id'].startswith('train:') or len(row['steps'])!=2 for row in trajectories)):
        raise ValueError('Exactly the original three real training trajectories are required')
    return count,program,hf


def arrays_for(info,row,np):
    ids=np.asarray(row['prefix_token_ids'],dtype=np.int64)
    mask=np.asarray(row['attention_mask'],dtype=np.float64)
    selection=np.asarray(row['sel'],dtype=np.float64)
    length=row['prefix_length']
    if (ids.shape!=(32,) or mask.shape!=(1,32) or selection.shape!=(1,32)
            or not 1<=length<=32 or not np.isin(mask,(0.,1.)).all()
            or not np.isin(selection,(0.,1.)).all() or mask.sum()!=length
            or not mask[0,:length].all() or mask[0,length:].any()
            or selection.sum()!=1 or selection[0,length-1]!=1):
        raise ValueError('The original fixed-prefix mask or last-token selector differs')
    embedding=info.graph.weights['original.model.embed_tokens.weight']
    if embedding.shape!=(151936,896) or ids.min()<0 or ids.max()>=len(embedding):
        raise ValueError('Original embedding or tokenizer vocabulary differs')
    return dict(inputs_embeds=np.asarray(embedding[ids])[None,:,:],attention_mask=mask,sel=selection)


def _load(args,result):
    from ..models.mapped_info import load_mapped_info
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    count,path,hf=source(args)
    info=load_mapped_info(args.model_info)
    program=parse(path.read_text())
    if (sum(op.kind=='rmsnorm' for op in program.ops)!=49
            or sum(op.kind=='softmax' for op in program.ops)!=24
            or sum(op.kind=='act' for op in program.ops)!=24):
        raise ValueError('The original24-layer normalization and activation occurrences must remain complete')
    checked=check(program,{key:value.shape for key,value in info.graph.weights.items()},graph=info.graph)
    if not checked.ok:
        raise ValueError('The saved fixed program no longer passes the original typecheck')
    result.update(source=dict(count_result=str(args.count_result.resolve()),program=str(path.resolve()),
        model_info=str(args.model_info.resolve()),original_hf_reference=str(args.reference.resolve())),
        weight_mapping=info.mapped_weights,threshold=dict(info.threshold),source_checked=True,
        decomposition_variant=count['worker']['decomposition_variant'],newton_budget=count['worker']['newton_budget'])
    return info,program,hf


def tensor_quality(args,recorder,result,np):
    from ..transformer_tensor import run,polynomial
    from ..metrics import compare,passes
    info,program,hf=_load(args,result)
    expected_points=sum(1 if op.kind in ('act','rmsnorm','layernorm') else
                        2+op.attrs.get('steps',0) if op.kind=='softmax' else 0 for op in program.ops)
    result.update(sample_rows=[],point_rows=[],softmax_rows=[],op_rows=[])
    for trajectory in hf['trajectories']:
        for original in trajectory['steps']:
            state=dict(sample_id=trajectory['sample_id'],step=original['step'])
            arrays=arrays_for(info,original,np)
            point_start=len(result['point_rows']);softmax_start=len(result['softmax_rows'])
            local_softmax={}

            def point(op,role,value,attrs,mask):
                measured=domain_record(value,attrs,np,mask)
                measured.update(state,op_id=op.id,line=op.lineno,role=role)
                result['point_rows'].append(measured)
                result['active_point']=dict(measured,completed=False)
                observed=polynomial(value,attrs,False)
                measured['polynomial_output_finite']=bool(np.isfinite(observed).all())
                result['active_point'].update(completed=True,polynomial_output_finite=measured['polynomial_output_finite'])
                if op.kind=='softmax' and role=='exp':
                    transformed=np.asarray(value)*(2.**op.attrs.get('steps',0))
                    selected=np.ones_like(transformed,dtype=bool) if mask is None else mask
                    hidden=np.where(selected,transformed,-np.inf)
                    if not selected.any(axis=-1).all():
                        raise ValueError('Every original attention row requires an effective key')
                    exact_exp=np.exp(hidden-hidden.max(axis=-1,keepdims=True))
                    local_softmax[op.id]=exact_exp/exact_exp.sum(axis=-1,keepdims=True)
                return observed

            def step(op,value):
                data=np.asarray(value);finite=bool(np.isfinite(data).all())
                result['op_rows'].append(dict(state,op_id=op.id,line=op.lineno,kind=op.kind,
                    shape=list(data.shape),finite=finite,
                    minimum=float(data.min()) if finite else None,maximum=float(data.max()) if finite else None))
                result['active_op']=dict(state,op_id=op.id,line=op.lineno,kind=op.kind,finite=finite)
                if op.kind=='softmax':
                    reference=local_softmax.pop(op.id)
                    result['softmax_rows'].append(dict(state,op_id=op.id,line=op.lineno,
                        finite=finite,max_absolute_difference=float(np.abs(data-reference).max()) if finite else None,
                        maximum_sum_error=float(np.abs(data.sum(-1)-1.).max()) if finite else None,
                        minimum_probability=float(data.min()) if finite else None,
                        reference='Independent stable exact softmax on the same public affine-transformed polynomial input'))
                if len(result['op_rows'])%64==0 or not finite:
                    write_json(args.work_dir/'worker-result.json',result)

            with recorder.span('fixed_tensor_program',**state):
                result['tensor_forward_attempts']+=1
                write_json(args.work_dir/'worker-result.json',result)
                with np.errstate(over='raise',invalid='raise',divide='raise'):
                    observed=run(program,info.graph.weights,arrays,point_evaluator=point,step_observer=step)
                if (len(result['point_rows'])-point_start!=expected_points
                        or len(result['softmax_rows'])-softmax_start!=24):
                    raise ValueError('Every selected nonlinear role and all24 softmax outputs must be observed')
                reference=np.load(args.reference.parent/original['logits_file'],allow_pickle=False,mmap_mode='r')
                if observed.dtype!=np.float64 or observed.shape!=reference.shape or not np.isfinite(observed).all():
                    raise ValueError('Fixed approximate logits must retain the original finite float64 shape')
                metrics=compare(dict(values=reference,task=info.task,lengths=[original['prefix_length']],selected_last=True),observed,None)
                filename='tensor-'+str(len(result['sample_rows']))+'-logits.npy'
                np.save(args.work_dir/filename,observed,allow_pickle=False)
                result['sample_rows'].append(dict(state,logits_file=filename,metrics=metrics,
                    policy_passed=passes(metrics,info.threshold),original_greedy_token=int(reference.argmax()),
                    approximate_greedy_token=int(observed.argmax()),max_absolute_difference=float(np.abs(observed-reference).max()),
                    domain_points=len(result['point_rows'])-point_start,
                    all_domains_contained=all(x['contained'] for x in result['point_rows'][point_start:]),
                    softmax_checks=len(result['softmax_rows'])-softmax_start,all_finite=True))
                result['tensor_forward_completed']+=1
                write_json(args.work_dir/'worker-result.json',result)
    if result['tensor_forward_completed']!=6:
        raise ValueError('The original six training-prefix states must each execute once')
    result.update(status='ok',training_prefix_measurement_completed=True,
        training_quality_passed=all(x['policy_passed'] and x['all_domains_contained'] for x in result['sample_rows']),
        all_finite=True,approximate_tokens_fed_back=False)


def mock_arithmetic(args,recorder,result,np):
    from .qwen_mock_progress import ObservedStreamingMockBackend,progress_records
    from ..kernels.compact_linear import compact_transformer_projections
    from ..lower import lower,pack_inputs,decode_outputs
    tensor=json.loads(args.tensor_result.read_text())
    if (tensor.get('status')!='ok' or tensor.get('phase')!='tensor'
            or tensor.get('tensor_forward_completed')!=6
            or Path(tensor['source']['count_result']).resolve()!=args.count_result.resolve()):
        raise ValueError('The same fixed program must first complete its six-state tensor measurement')
    info,program,hf=_load(args,result)
    trajectory=hf['trajectories'][0];row=trajectory['steps'][0];expected=tensor['sample_rows'][0]
    if expected['sample_id']!=trajectory['sample_id'] or expected['step']!=row['step']:
        raise ValueError('The independently selected original first training state differs')
    arrays=arrays_for(info,row,np)
    reference=np.load(args.tensor_result.parent/expected['logits_file'],allow_pickle=False,mmap_mode='r')
    backend=ObservedStreamingMockBackend(program.header.slots,batch=1)
    with recorder.span('single_original_state_packed_math',sample_id=trajectory['sample_id'],step=row['step']):
        result['mock_forward_attempts']=1
        write_json(args.work_dir/'worker-result.json',result)
        try:
            backend.bind(pack_inputs(program,[arrays[param.name] for param in program.params]))
            with progress_records(backend,program,args.work_dir,result),compact_transformer_projections():
                values,layouts=lower(program,info.graph.weights,backend)
            observed=decode_outputs([value.value for value in values],layouts)[0]
            result['mock_stats']=backend.stats()
            result['output_layout_blocks']=[dict(start=start,end=end,valid_size=block.size)
                for layout in layouts for start,end,block in layout.blocks]
            result['mock_sample']=dict(sample_id=trajectory['sample_id'],step=row['step'],
                prefix_length=row['prefix_length'],output_shape=list(observed.shape))
            expected_stats=json.loads(args.count_result.read_text())['worker']['count_stats']
            result['count_mock_schedule_matched']=all(result['mock_stats'][key]==expected_stats[key]
                for key in ('counts','ops_total','distinct_consts','rot_steps'))
            result['retained_constant_arrays']=sum(item.data is not None for item in backend._constants.values())
            result['constant_identity_entries']=len(backend._constant_objects)
            result['strict_tensor_mock_math']=strict_math(reference,observed,np)
            result['mock_forward_completed']=1
            result['tensor_reference']=str(args.tensor_result.resolve())
            np.save(args.work_dir/'mock-logits.npy',observed,allow_pickle=False)
        finally:
            backend.release_batch()
    if (not result['count_mock_schedule_matched'] or result['retained_constant_arrays']
            or result['constant_identity_entries']):
        raise ValueError('Independent packed arithmetic must retain the original counted schedule and streaming constants')
    if not result['strict_tensor_mock_math']['passed']:
        raise ValueError('Independent packed arithmetic differs under the original strict mathematical policy')
    result.update(status='ok',independent_packed_math_passed=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('tensor','mock'),required=True)
    for name in ('work-dir','count-result','model-info','reference'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tensor-result',type=Path)
    args=parser.parse_args(argv)
    if args.phase=='mock' and args.tensor_result is None:
        parser.error('--tensor-result is required for the independent packed comparison')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    if (args.work_dir/'worker-result.json').exists():
        raise FileExistsError('Preserve the previous quality diagnostic')
    result=dict(status='running',phase=args.phase,model='Qwen2-0.5B',candidate_frozen=False,
        tensor_forward_attempts=0,tensor_forward_completed=0,mock_forward_attempts=0,mock_forward_completed=0,
        source_checked=False,training_prefix_measurement_completed=False,training_quality_passed=None,
        independent_packed_math_passed=None,approximate_tokens_fed_back=False,
        heldout_samples=0,confirm_samples=0,report_samples=0,api_calls=0,
        approximation_selection_runs=0,coefficient_reconstruction_calls=None,
        coefficient_policy='Original deterministic evaluators reconstruct fixed selected coefficients; their call count is unmeasured',
        emit_runs=0,compiler_runs=0,keygen_runs=0,ckks_executions=0,
        full_ciphertext_model_binding_verified=False,compiled_bootstraps=None,predicted_latency=None,
        CKKS_latency=None,accuracy_acceptance=None,
        scope='Fixed original HF own-token training-prefix states; no approximate trajectory selection or formal accuracy acceptance')
    recorder=StageRecorder(args.work_dir/'events.jsonl')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    try:
        import numpy as np
        with recorder.span('qwen_fixed_precision_'+args.phase):
            (tensor_quality if args.phase=='tensor' else mock_arithmetic)(args,recorder,result,np)
    except BaseException as exc:
        result.update(status='error',error_type=type(exc).__name__,error=str(exc)[:3000])
    result['worker_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.work_dir/'worker-result.json',result)
    print(json.dumps({key:result[key] for key in ('status','phase','tensor_forward_completed','mock_forward_completed')}))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':
    raise SystemExit(main())
